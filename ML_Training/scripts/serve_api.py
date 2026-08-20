r"""Entry-point cho Epic AI-04 — chạy AI service (F05 · M08).

    .venv\Scripts\python.exe scripts/serve_api.py
    .venv\Scripts\python.exe scripts/serve_api.py --port 8080 --reload
    .venv\Scripts\python.exe scripts/serve_api.py --check

Mở tài liệu tương tác ở http://127.0.0.1:8000/docs

`--check` chạy một lượt tự kiểm qua HTTP mà KHÔNG cần mở cổng: dựng app trong
tiến trình, gọi cả ba endpoint, in kết quả. Dùng để xác nhận service lành mạnh
trước khi triển khai, hoặc trong CI.

Biến môi trường
----------------
    HFML_API_CORS_ORIGINS      danh sách origin, ngăn bằng dấu phẩy
    HFML_API_CHAT_TIMEOUT      giây, mặc định 60
    HFML_API_INFERENCE_TIMEOUT giây, mặc định 20
    HFML_API_WARM_UP           0 để bỏ nạp sẵn model
    HFML_ML01_SLUG             đổi artifact mà không sửa mã
    HFML_ML02_SLUG
    GEMINI_API_KEY             thiếu thì tầng LLM chạy chế độ template
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from hfml.api.v1 import API_PREFIX, API_SETTINGS, API_VERSION  # noqa: E402

_MARK = {"healthy": "✅", "degraded": "⚠️ ", "unhealthy": "❌"}


def self_check() -> int:
    """Gọi cả ba endpoint trong tiến trình, không mở cổng."""
    from fastapi.testclient import TestClient

    from analyze_profile import SAMPLE
    from hfml.api.v1 import create_app

    print("\n" + "=" * 72)
    print(f"AI-04 · TỰ KIỂM API v{API_VERSION}")
    print("=" * 72)

    with TestClient(create_app()) as client:
        report = client.get("/health").json()
        print(f"\nGET /health → {report['status'].upper()}")
        for name, item in report["components"].items():
            print(f"  {_MARK.get(item['status'], '·')} {name:<14} {item['detail'][:52]}")

        print(f"\nPOST {API_PREFIX}/inference")
        response = client.post(f"{API_PREFIX}/inference",
                               json={"household": SAMPLE})
        print(f"  → {response.status_code}")
        if response.status_code == 200:
            body = response.json()
            print(f"  ok={body['ok']} · {len(body['rules'])} quy tắc · "
                  f"{body['elapsed_ms']:.0f} ms")
            for name in ("ml01", "ml02"):
                part = body[name]
                print(f"  {name.upper()}: "
                      + (f"{part['label_vi']} ({part['probability']:.1%})"
                         if part["available"] else f"— [{part['reason_code']}]"))
        else:
            print(f"  {json.dumps(response.json(), ensure_ascii=False)[:200]}")

        print(f"\nPOST {API_PREFIX}/chat")
        response = client.post(f"{API_PREFIX}/chat", json={
            "household": SAMPLE,
            "question": "Sức khỏe tài chính của gia đình tôi thế nào?",
            "intent_code": "FINANCIAL_HEALTH_DIAGNOSIS"})
        print(f"  → {response.status_code}")
        if response.status_code == 200:
            answer = response.json()["answer"]
            print(f"  nguồn={answer['source']} · kiểm={answer['validated']} · "
                  f"intent={answer['intent_code']}")
            print(f"  {answer['text'][:200].replace(chr(10), ' ')}…")
        else:
            print(f"  {json.dumps(response.json(), ensure_ascii=False)[:200]}")

        # Một request sai, để thấy vỏ lỗi thống nhất.
        print(f"\nPOST {API_PREFIX}/inference  (cố ý sai)")
        response = client.post(f"{API_PREFIX}/inference",
                               json={"household": {"household_size": "bốn"}})
        body = response.json()
        print(f"  → {response.status_code} · error={body['error']} · "
              f"{len(body['details'])} chi tiết")

    print("\n" + "=" * 72)
    healthy = report["status"] != "unhealthy"
    print("  " + ("Service sẵn sàng." if healthy
                  else "Service KHÔNG phục vụ được — xem phần health ở trên."))
    print("=" * 72)
    return 0 if healthy else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true",
                        help="nạp lại khi sửa mã (chỉ dùng lúc dev)")
    parser.add_argument("--check", action="store_true",
                        help="tự kiểm rồi thoát, không mở cổng")
    args = parser.parse_args()

    if args.check:
        return self_check()

    import uvicorn

    print(f"\n{'=' * 72}")
    print(f"AI-04 · Household Finance AI Service   (API v{API_VERSION})")
    print(f"  Tài liệu : http://{args.host}:{args.port}/docs")
    print(f"  Health   : http://{args.host}:{args.port}/health")
    print(f"  Cấu hình : {json.dumps(API_SETTINGS.to_dict(), ensure_ascii=False)}")
    print("=" * 72 + "\n")

    uvicorn.run("hfml.api.v1.app:app", host=args.host, port=args.port,
                reload=args.reload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
