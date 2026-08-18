r"""Entry-point cho ML02 task 7 — Train Decision Tree (F04 · M04).

    .venv\Scripts\python.exe scripts/train_ml02_decision_tree.py
    .venv\Scripts\python.exe scripts/train_ml02_decision_tree.py --rows 20000
    .venv\Scripts\python.exe scripts/train_ml02_decision_tree.py --no-save

Train Decision Tree trên tập train của task 5, cả hai bộ feature, chấm trên
**validation**. Tập test vẫn khoá tới task 14.

⚠️ CHỈ Decision Tree. Bagging / Random Forest / XGBoost là task 8, 9, 10.

Thành phẩm:

    src/training/runs/ml02_models/ml02_decision_tree_<set>.joblib   artifact TRUNG GIAN
    src/training/runs/ml02_models/ml02_decision_tree_<set>.metadata.json
    src/training/runs/ml02_models/ml02_decision_tree_<set>.confusion.csv
    src/training/runs/results.csv                                  nối thêm dòng
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from hfml.logger import get_logger
from hfml.ml.evaluation.tracking import append_results
from hfml.ml.ml02_credit_risk.baseline import MEANINGFUL_LIFT
from hfml.ml.ml02_credit_risk.train import (
    decision_tree_params,
    load_training_data,
    results_frame,
    save_run,
    train_decision_tree,
)

log = get_logger(__name__)

SHOWN = ["pr_auc", "pr_auc_lift", "roc_auc", "f1_positive",
         "recall_positive", "precision_positive", "accuracy", "brier"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=None,
                        help="giới hạn số dòng train — CHỈ để chạy thử")
    parser.add_argument("--no-save", action="store_true",
                        help="chỉ in kết quả, không ghi file")
    args = parser.parse_args()

    data = load_training_data(nrows=args.rows)
    print(f"\n=== Task 7 · Decision Tree ===")
    print(f"  train {len(data.X_train):,} · validation {len(data.X_validation):,}")
    print(f"  siêu tham số: {decision_tree_params(len(data.X_train))}")
    print(f"  cân bằng lớp: class_weight='balanced' (task 4)")
    print("  Tập test KHÔNG được chạm.")

    models = train_decision_tree(data)

    table = results_frame(models)
    print("\n=== Chỉ số trên VALIDATION ===")
    print(table[["algo", "feature_set", "n_features", *SHOWN]].to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n=== Học thuộc tới đâu (PR-AUC train so với validation) ===")
    for model in models:
        print(f"  {model.feature_set:<8} train {model.metrics_train['pr_auc']:.4f} "
              f"→ validation {model.pr_auc:.4f}  (gap {model.overfit_gap:+.4f})")

    print("\n=== So với baseline task 6 ===")
    for model in models:
        lift = model.metrics_validation["pr_auc_lift"]
        dat = "✅" if lift >= MEANINGFUL_LIFT else "⚠️"
        print(f"  {dat} {model.feature_set:<8} PR-AUC {model.pr_auc:.4f} "
              f"= {lift:.2f}× mức đoán bừa (mốc {MEANINGFUL_LIFT}×)")

    for model in models:
        print(f"\n--- Confusion matrix · bộ {model.feature_set} (ngưỡng 0,5) ---")
        print(model.confusion_validation.to_string())

    if args.no_save or args.rows:
        print("\n(--no-save hoặc --rows: không ghi file nào)")
        return 0

    print()
    for model in models:
        for path in save_run(model).values():
            print(f"  ghi → {path}")
    print(f"  ghi → {append_results(table)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
