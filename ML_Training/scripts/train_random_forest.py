"""Chạy train Random Forest cho ML01 (F03 task 9).

    python scripts/train_random_forest.py                  # tham số mặc định
    python scripts/train_random_forest.py --rows 2000      # chạy nhanh khi thử
    python scripts/train_random_forest.py --no-save        # không ghi ra runs/

Script này CHỈ là entry-point: mọi logic nằm ở
`hfml.ml.ml01_recommendation.train`.

Output ghi vào `src/training/runs/` (`CONFIG.paths.runs`): model artifact,
metadata kèm cấu hình lần chạy, và một dòng trong `results.csv`.
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from hfml.data.synthetic import PopulationParams
from hfml.ml.ml01_recommendation.report import build_ml01_report
from hfml.ml.ml01_recommendation.train import train_random_forest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=None,
                        help="số hộ sinh ra (mặc định: theo PopulationParams)")
    parser.add_argument("--seed", type=int, default=None,
                        help="ghi đè random_seed của config")
    parser.add_argument("--no-save", action="store_true",
                        help="không ghi artifact/kết quả ra src/training/runs/")
    args = parser.parse_args()

    params = PopulationParams(n=args.rows) if args.rows else None
    result = train_random_forest(params, seed=args.seed,
                                 save=not args.no_save)

    metrics = result["validation_metrics"]
    rows = [
        {"chỉ số": name, "giá trị": metrics[name]}
        for name in ("accuracy", "macro_f1", "balanced_accuracy", "weighted_f1")
    ]
    print(f"\n=== Random Forest · chấm trên tập validation "
          f"({result['config']['n_validation_rows']:,} hộ) · "
          f"fit trên {result['config']['n_train_rows']:,} hộ train ===")
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n=== Cấu hình lần chạy ===")
    print(pd.DataFrame(result["config"].items(),
                       columns=["tham số", "giá trị"]).to_string(index=False))

    if "artifact" in result:
        print(f"\nArtifact  → {result['artifact']}")
        print(f"Kết quả   → {result['results_csv']}")

        report = build_ml01_report(params, seed=args.seed)
        print("\n=== Hình đã ghi ===")
        for name, path in report["figures"].items():
            print(f"{name:<20} → {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
