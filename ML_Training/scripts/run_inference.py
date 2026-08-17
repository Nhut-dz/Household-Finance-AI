r"""Entry-point cho Epic AI-03 — Python AI Inference Module (F05 · M07).

    .venv\Scripts\python.exe scripts/run_inference.py
    .venv\Scripts\python.exe scripts/run_inference.py --ask "Tôi vay được bao nhiêu?"
    .venv\Scripts\python.exe scripts/run_inference.py --analyze-only
    .venv\Scripts\python.exe scripts/run_inference.py --no-llm
    .venv\Scripts\python.exe scripts/run_inference.py --json

Chạy module qua đúng hai điểm vào công khai của nó — `analyze` và `chat`. Script
này KHÔNG import bất cứ thứ gì bên trong `hfml.pipeline` hay `hfml.llm`, và đó
là phép thử: nếu module đóng gói đúng thì chừng này là đủ để dùng.

Vì sao in `trace` chứ không chỉ in câu trả lời
-----------------------------------------------
Câu trả lời không cho biết bước nào đã chạy, bước nào bị bỏ, và bước nào làm
chậm cả request. `trace` trả lời đúng ba câu đó, nên khi có gì sai thì chỗ cần
nhìn là nó chứ không phải log.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from analyze_profile import SAMPLE                      # noqa: E402
from hfml.inference import SETTINGS, analyze, chat, health   # noqa: E402

_MARK = {"llm": "🟢", "llm_retry": "🟡", "template": "⚪", "out_of_scope": "🚫"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=None,
                        help="file JSON chứa hồ sơ")
    parser.add_argument("--ask", default="Sức khỏe tài chính của tôi thế nào?",
                        help="câu hỏi cho tầng LLM")
    parser.add_argument("--intent", default=None, help="mã intent từ chip")
    parser.add_argument("--analyze-only", action="store_true",
                        help="dừng ở Aggregation, không gọi LLM")
    parser.add_argument("--no-llm", action="store_true",
                        help="tắt tầng LLM — câu trả lời dựng từ template")
    parser.add_argument("--json", action="store_true", help="in JSON thô")
    args = parser.parse_args()

    if args.no_llm:
        SETTINGS.llm_enabled = False

    payload = (json.loads(args.input.read_text(encoding="utf-8"))
               if args.input else SAMPLE)

    result = (analyze(payload) if args.analyze_only
              else chat(payload, args.ask, intent_code=args.intent))

    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return 0 if result.ok else 1

    report = health()
    print("\n" + "=" * 72)
    print(f"AI-03 · INFERENCE MODULE   (schema {result.schema_version})")
    for name, item in report["models"].items():
        if item.get("loaded"):
            extra = (f" · ngưỡng {item['threshold']:.4f}"
                     + (" (ĐÃ GHI ĐÈ)" if item.get("threshold_overridden") else "")
                     if "threshold" in item else "")
            print(f"  ✅ {name}: {item['slug']} ({item['n_features']} feature){extra}")
        else:
            print(f"  ❌ {name}: {item['error']}")
    print(f"  LLM: {'bật' if report['llm']['enabled'] else 'TẮT'} · "
          f"{'sẵn sàng' if report['llm']['available'] else 'chưa cấu hình'}")
    print("=" * 72)

    print("\n--- Các bước đã chạy ---")
    for step in result.trace:
        mark = "✅" if step["ok"] else "❌"
        print(f"  {mark} {step['stage']:<14} {step['elapsed_ms']:>8.1f} ms"
              f"   {len(step['diagnostics'])} chẩn đoán")

    if result.errors:
        print("\n--- LỖI ---")
        for item in result.errors:
            print(f"  ❌ [{item.stage}/{item.code}] {item.message}"
                  + (f" ({item.field})" if item.field else ""))

    if result.warnings:
        print("\n--- Cảnh báo ---")
        for item in result.warnings:
            print(f"  ⚠️  [{item.stage}/{item.code}] {item.message[:90]}")

    if result.analysis:
        print(f"\n--- Phân tích (tổng quan: {result.analysis['overall_status']}) ---")
        for code, rule in result.analysis["rules"].items():
            summary = rule.get("details", {}).get("summary_vi", "")
            print(f"  {code} {rule.get('status', '—'):<12} {summary[:58]}")
        for name in ("ml01", "ml02"):
            part = result.analysis[name]
            if part.get("available"):
                print(f"  {name.upper()}: {part['label_vi']} "
                      f"({part['probability']:.1%})")
            else:
                print(f"  {name.upper()}: — [{part.get('reason_code')}]")

    if result.text:
        source = result.answer.get("source", "")
        print(f"\n--- Câu trả lời  {_MARK.get(source, '·')} {source} ---")
        print(result.text)
        check = result.answer.get("validation", {})
        status = {True: "đạt", False: "KHÔNG ĐẠT",
                  None: "không chạy được"}.get(check.get("valid"), "—")
        print(f"\n  [kiểm {status}"
              + (f" · số bịa: {', '.join(check['ungrounded_numbers'])}"
                 if check.get("ungrounded_numbers") else "") + "]")

    print("\n" + "=" * 72)
    print(f"  ok = {result.ok} · {len(result.errors)} lỗi · "
          f"{len(result.warnings)} cảnh báo · "
          f"{sum(s['elapsed_ms'] for s in result.trace):.0f} ms")
    print("=" * 72)
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
