r"""Entry-point cho Epic AI-01 — Inference Engine (F05 · M05).

    .venv\Scripts\python.exe scripts/analyze_profile.py
    .venv\Scripts\python.exe scripts/analyze_profile.py --input ho_so.json
    .venv\Scripts\python.exe scripts/analyze_profile.py --json

Chạy một hồ sơ xuyên suốt: chuẩn hoá → 5 rule → ML01 → ML02 → structured
result. Không có `--input` thì dùng hồ sơ mẫu.

`--json` in ra đúng cấu trúc mà Epic AI-02 sẽ nhận, không thêm chữ nào.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from hfml.logger import get_logger
from hfml.pipeline.orchestrator import analyze
from hfml.pipeline.predictor import model_status

log = get_logger(__name__)

#: Hồ sơ mẫu — một hộ có nhu cầu vay mua nhà, đủ dữ liệu cho cả ba tầng.
SAMPLE = {
    "representative_name": "Nguyễn Văn A",
    "birth_year": 1991,
    "residence": "TP. Hồ Chí Minh",
    "household_size": 4,
    "children_count": 2,
    "has_dependents": False,
    "average_monthly_income": 35_000_000,
    "average_monthly_expense": 17_000_000,
    "has_debt": True,
    "total_current_debt": 500_000_000,
    "monthly_debt_payment": 5_000_000,
    "has_savings": True,
    "savings_amount": 150_000_000,
    "assets": ["cash", "real_estate"],
    "financial_needs": ["home_loan"],
    "loan_application": {
        "borrower_age": 35,
        "gender": "male",
        "marital_status": "married",
        "children_count": 2,
        "education_level": "higher",
        "occupation": "office_staff",
        "employment_years": 8.5,
        "loan_amount": 1_400_000_000,
        "loan_term_months": 240,
        "monthly_payment": 12_000_000,
        "asset_price": 2_000_000_000,
        "loan_purpose": "buy_house",
        "previous_loan_count": 3,
        "late_payment_count": 1,
        "has_overdue_loan": False,
        "total_overdue_amount": 0,
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=None,
                        help="file JSON chứa hồ sơ")
    parser.add_argument("--json", action="store_true",
                        help="in structured result thô, không diễn giải")
    args = parser.parse_args()

    payload = (json.loads(args.input.read_text(encoding="utf-8"))
               if args.input else SAMPLE)

    result = analyze(payload)
    data = result.to_dict()

    if args.json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return 0 if data["ok"] else 1

    print("\n" + "=" * 72)
    print(f"AI-01 · INFERENCE ENGINE   (schema {data['schema_version']})")
    print("=" * 72)

    print("\n--- Trạng thái model ---")
    for name, status in model_status().items():
        if status["loaded"]:
            extra = (f" · ngưỡng {status['threshold']:.4f}"
                     if "threshold" in status else "")
            print(f"  ✅ {name}: {status['slug']} "
                  f"({status['n_features']} feature){extra}")
        else:
            print(f"  ❌ {name}: {status['error']}")

    summary = data["input_summary"]
    print("\n--- Đầu vào ---")
    if not summary.get("valid"):
        print(f"  ❌ Không hợp lệ ({summary.get('n_errors')} lỗi)")
    else:
        print(f"  {summary['representative_name']} · {summary['age']} tuổi · "
              f"hộ {summary['household_size']} người ({summary['children_count']} con)")
        print(f"  Thu {summary['monthly_income']:,.0f} · "
              f"chi {summary['monthly_expense']:,.0f} VNĐ/tháng")
        print(f"  Có thông tin khoản vay: "
              f"{'có' if summary['has_loan_application'] else 'không'}")

    if data["errors"]:
        print("\n--- LỖI ---")
        for item in data["errors"]:
            print(f"  ❌ [{item['code']}] {item['field']}: {item['message']}")

    if data["rules"]:
        print(f"\n--- Tầng quy tắc (tổng quan: {data['overall_status']}) ---")
        for code, rule in data["rules"].items():
            # Câu tóm tắt nằm ở `details.summary_vi`, không phải `message_vi`.
            summary = rule.get("details", {}).get("summary_vi", "")
            print(f"  {code} {rule.get('status', '—'):<12} {summary[:60]}")

    print("\n--- ML01 · Nhóm khuyến nghị tài chính ---")
    _print_model(data["ml01"])

    print("\n--- ML02 · Rủi ro khoản vay ---")
    _print_model(data["ml02"])

    if data["warnings"]:
        print("\n--- Cảnh báo (tầng LLM PHẢI nói ra) ---")
        for item in data["warnings"]:
            print(f"  ⚠️  [{item['code']}] {item['message']}")

    print(f"\n{'=' * 72}")
    print(f"  ok = {data['ok']} · {len(data['warnings'])} cảnh báo · "
          f"{len(data['errors'])} lỗi")
    print("  Structured result này là đầu vào của Epic AI-02.")
    print("=" * 72)
    return 0 if data["ok"] else 1


def _print_model(part: dict) -> None:
    if not part.get("available"):
        print(f"  — không có kết quả [{part.get('reason_code')}]")
        print(f"    {part.get('error')}")
        return

    confidence = part.get("confidence", {})
    print(f"  {part['label']} · {part['label_vi']} · "
          f"xác suất {part['probability']:.2%} ({confidence.get('description')})")
    for item in part.get("probabilities", []):
        print(f"    {item['label_vi']:<32} {item['probability']:.2%}")
    if confidence.get("low_confidence"):
        print("    ⚠️  kém tin cậy — xem phần cảnh báo")
    print(f"    model: {part['model_version']}")


if __name__ == "__main__":
    sys.exit(main())
