r"""Entry-point cho ML02 task 1 — Khám phá Home Credit Dataset (F04 · M04).

    .venv\Scripts\python.exe scripts/explore_ml02.py
    .venv\Scripts\python.exe scripts/explore_ml02.py --rows 20000   # chạy nhanh
    .venv\Scripts\python.exe scripts/explore_ml02.py --no-bureau    # bỏ bureau.csv

Script chỉ ĐIỀU PHỐI: đo bằng `hfml.ml.ml02_credit_risk.explore`, diễn đạt
bằng `...report`, rồi in phần tóm tắt ra terminal.

Thành phẩm:

    src/training/runs/ml02_eda/*.csv   bảng đo được, dùng lại ở task sau
    docs/ml02_eda.md                   báo cáo đọc được, commit vào git

Chạy `python` thay vì `.venv\Scripts\python.exe` sẽ hỏng: package `hfml` chỉ
được cài trong venv.
"""
from __future__ import annotations

import argparse
import sys

from hfml.logger import get_logger
from hfml.ml.ml02_credit_risk.explore import explore, write_tables
from hfml.ml.ml02_credit_risk.report import write_doc

log = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=None,
                        help="giới hạn số dòng đọc — CHỈ để chạy thử, "
                             "số ra không dùng cho báo cáo được")
    parser.add_argument("--no-bureau", action="store_true",
                        help="bỏ qua bureau.csv (170 MB) cho nhanh")
    parser.add_argument("--no-save", action="store_true",
                        help="không ghi runs/ và docs/")
    args = parser.parse_args()

    report = explore(nrows=args.rows, with_bureau=not args.no_bureau)

    target = report.target
    print(f"\n=== Nhãn TARGET trên {target['n_rows']:,} hồ sơ ===")
    print(f"  dương            {target['n_positive']:,} ({target['positive_rate']:.4%})")
    print(f"  scale_pos_weight {target['scale_pos_weight']:.2f}")
    print(f"  accuracy nếu đoán toàn 0: {target['majority_class_accuracy']:.4%}"
          "  ← lý do không dùng accuracy để chọn model")

    ranking = report.iv_ranking
    print(f"\n=== 15 cột mạnh nhất trong {len(ranking)} cột (Information Value) ===")
    print(ranking.head(15)[["column", "dtype", "missing_rate", "iv", "band"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n=== Cột mạnh nhất mà FORM KHÔNG lấy được — cái giá của bộ Rút gọn ===")
    print(report.unreachable.head(8)[["column", "missing_rate", "iv", "band"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    coverage = report.form_coverage
    mapped = coverage[coverage["home_credit"] != "—"]
    print(f"\n=== Form ↔ Home Credit ===")
    print(f"  {len(coverage)} trường form, {len(mapped)} ánh xạ được sang cột Home Credit")
    print(f"  tổng IV lấy được {mapped['iv'].sum():.4f} / {ranking['iv'].sum():.4f} "
          f"({mapped['iv'].sum() / ranking['iv'].sum():.1%})")

    if report.bureau_coverage:
        cov = report.bureau_coverage
        print(f"\n=== bureau.csv ===")
        print(f"  {cov['n_bureau_rows']:,} khoản vay của "
              f"{cov['n_customers_with_record']:,} khách hàng")
        print(f"  hồ sơ không có bản ghi: {cov['n_customers_without_record']:,} "
              f"({cov['share_without_record']:.2%})")
        print(f"  vỡ nợ  không bản ghi {cov['default_rate_without_record']:.4%}"
              f"  ·  có bản ghi {cov['default_rate_with_record']:.4%}")
        print(report.bureau_iv[["column", "iv", "band"]]
              .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n=== Phân phối feature tỉ lệ (§2.1 — bất biến đơn vị tiền tệ) ===")
    print(report.ratio_distribution[["feature", "p1", "p25", "p50", "p75", "p99"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    if args.no_save:
        print("\n(--no-save: không ghi file nào)")
        return 0

    if args.rows:
        # Ghi đè báo cáo bằng số của một mẫu nhỏ là cách âm thầm nhất để đưa
        # số sai vào báo cáo cuối — chặn ngay ở đây.
        print(f"\nCHÚ Ý: chạy với --rows {args.rows:,} nên KHÔNG ghi đè "
              "docs/ml02_eda.md. Bỏ --rows để sinh báo cáo chính thức.")
        return 0

    for path in write_tables(report):
        print(f"  ghi → {path}")
    print(f"  ghi → {write_doc(report)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
