r"""Entry-point cho ML02 task 2 — Data Cleaning (F04 · M04).

    .venv\Scripts\python.exe scripts/clean_ml02.py
    .venv\Scripts\python.exe scripts/clean_ml02.py --rows 20000   # chạy thử
    .venv\Scripts\python.exe scripts/clean_ml02.py --no-bureau
    .venv\Scripts\python.exe scripts/clean_ml02.py --dry-run      # không ghi

Thành phẩm:

    data/interim/ml02/application_clean.csv.gz   dữ liệu sạch cho task 3
    data/interim/ml02/bureau_clean.csv.gz
    data/interim/ml02/cleaning_metadata.json     danh sách feature + nhật ký
    src/training/runs/ml02_cleaning/*.csv        bảng báo cáo từng bước

Dữ liệu đầu ra là "đã hết bẩn ở mức từng dòng", KHÔNG phải "sẵn sàng cho
model". Các bước còn lại đều CÓ HỌC nên bắt buộc nằm trong Pipeline và `fit`
chỉ trên tập train — script in ra danh sách đó ở cuối để khỏi ai quên.
"""
from __future__ import annotations

import argparse
import sys

from hfml.logger import get_logger
from hfml.ml.ml02_credit_risk.clean import (
    PIPELINE_STEPS_REMAINING,
    LeakageAuditFailed,
    build_clean_dataset,
    feature_columns,
    write_outputs,
)

log = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=None,
                        help="giới hạn số dòng application — CHỈ để chạy thử")
    parser.add_argument("--no-bureau", action="store_true",
                        help="bỏ qua bureau.csv (170 MB) cho nhanh")
    parser.add_argument("--dry-run", action="store_true",
                        help="chạy và in báo cáo nhưng không ghi file nào")
    args = parser.parse_args()

    try:
        app, bureau, reports = build_clean_dataset(
            nrows=args.rows, with_bureau=not args.no_bureau)
    except LeakageAuditFailed as exc:
        print("\n❌ DỪNG — kiểm toán rò rỉ dữ liệu KHÔNG đạt:\n")
        print(exc)
        return 1

    for name, report in reports.items():
        steps = report.steps_frame()
        print(f"\n=== {report.table} — {len(steps)} bước ===")
        print(steps[["name", "rows_before", "rows_after",
                     "cols_before", "cols_after"]].to_string(index=False))
        for step in report.steps:
            if step.detail:
                print(f"  · {step.name}: {step.detail}")

    app_report = reports["application"]

    print("\n=== Kiểm toán rò rỉ dữ liệu ===")
    for _, row in app_report.leakage.iterrows():
        print(f"  {'✅' if row['passed'] else '❌'} {row['check']}: {row['measured']}")

    if "bureau" in reports and not reports["bureau"].future_information.empty:
        print("\n=== bureau: thông tin đến SAU ngày nộp đơn ===")
        print(reports["bureau"].future_information.to_string(index=False))

    print("\n=== Kiểu dữ liệu (application, sau làm sạch) ===")
    print(app_report.dtypes["semantic"].value_counts().to_string())

    print("\n=== Quy tắc hợp lệ ===")
    violations = app_report.validation[app_report.validation["n_violations"] > 0]
    print(violations.to_string(index=False) if len(violations)
          else "  Không dòng nào vi phạm quy tắc nào.")

    top_missing = app_report.missing.head(8)
    print("\n=== 8 cột thiếu nhiều nhất (sau khi sentinel → NaN) ===")
    print(top_missing.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print(f"\n=== Kết quả ===")
    print(f"  application : {len(app):,} dòng × {app.shape[1]} cột "
          f"({len(feature_columns(app))} feature)")
    if bureau is not None:
        print(f"  bureau      : {len(bureau):,} dòng × {bureau.shape[1]} cột")

    print("\n=== CHƯA làm — các bước CÓ HỌC, phải nằm trong Pipeline ===")
    for name, learns in PIPELINE_STEPS_REMAINING:
        print(f"  · {name:<26} {learns}")
    print("  Chúng `fit` CHỈ trên tập train. Chạy trước rồi lưu kết quả là rò rỉ.")

    if args.dry_run:
        print("\n(--dry-run: không ghi file nào)")
        return 0

    if args.rows:
        print(f"\nCHÚ Ý: chạy với --rows {args.rows:,} nên KHÔNG ghi đè dữ liệu "
              "sạch chính thức. Bỏ --rows để sinh bản đầy đủ.")
        return 0

    print()
    for label, path in write_outputs(app, bureau, reports).items():
        print(f"  ghi → {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
