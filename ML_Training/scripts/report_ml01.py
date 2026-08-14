"""Chấm test và vẽ lại toàn bộ hình báo cáo ML01 (F03 task 11–13).

    .venv\\Scripts\\python.exe scripts/report_ml01.py
    .venv\\Scripts\\python.exe scripts/report_ml01.py --rows 2000    # chạy nhanh

Bốn bước, gọi qua `hfml.ml.ml01_recommendation.report.build_ml01_report`:

    evaluate_on_test()          test_confusion.csv · test_per_class.csv
    compare_models()            model_comparison.csv
    feature_importance_report() feature_importance.csv
    generate_ml01_plots()       confusion_matrix_test.png · model_comparison.png
                                feature_importance.png · results_table.png

Các `scripts/train_*.py` đã tự chạy đúng chuỗi này sau khi train xong, nên
script này chỉ cần dùng khi muốn vẽ lại hình mà KHÔNG train lại — ví dụ sau
khi sửa code vẽ, hoặc khi đổi `--top-n`.

Chạy `python` thay vì `.venv\\Scripts\\python.exe` sẽ hỏng: package `hfml`
chỉ được cài trong venv.
"""
from __future__ import annotations

import argparse
import sys

from hfml.data.synthetic import PopulationParams
from hfml.ml.ml01_recommendation.report import build_ml01_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=None,
                        help="số hộ sinh ra (mặc định: theo PopulationParams)")
    parser.add_argument("--seed", type=int, default=None,
                        help="ghi đè random_seed của config")
    parser.add_argument("--n-splits", type=int, default=None,
                        help="số fold CV dùng khi phải backfill (mặc định: theo config)")
    parser.add_argument("--top-n", type=int, default=10,
                        help="số feature vẽ trong hình importance")
    args = parser.parse_args()

    params = PopulationParams(n=args.rows) if args.rows else None
    report = build_ml01_report(params, seed=args.seed, n_splits=args.n_splits,
                               top_n=args.top_n)

    print("\n=== Chỉ số trên tập test ===")
    print(report["test_summary"].to_string(index=False))

    print("\n=== So sánh CV với test ===")
    print(report["comparison"].to_string(index=False))

    print(f"\n=== Top {args.top_n} feature ===")
    print(report["importance_pivot"].head(args.top_n).to_string())

    print("\n=== Hình đã ghi ===")
    for name, path in report["figures"].items():
        print(f"{name:<20} → {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
