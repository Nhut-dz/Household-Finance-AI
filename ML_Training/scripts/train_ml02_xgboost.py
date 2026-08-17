r"""Entry-point cho ML02 task 10 — Train XGBoost (F04 · M04).

    .venv\Scripts\python.exe scripts/train_ml02_xgboost.py
    .venv\Scripts\python.exe scripts/train_ml02_xgboost.py --rows 20000
    .venv\Scripts\python.exe scripts/train_ml02_xgboost.py --no-save

Thuật toán cuối của bốn. Cân bằng lớp bằng `scale_pos_weight` tính từ RIÊNG
tập train, không phải `class_weight` — XGBoost không có tham số đó.

⚠️ Đây là task train cuối. So sánh và chọn model là task 11, 12, 14 — script
này KHÔNG kết luận thuật toán nào tốt nhất. Tập test vẫn khoá tới task 14.
"""
from __future__ import annotations

import argparse
import json
import sys

from hfml.logger import get_logger
from hfml.ml.evaluation.tracking import append_results
from hfml.ml.ml02_credit_risk.baseline import MEANINGFUL_LIFT
from hfml.ml.ml02_credit_risk.imbalance import scale_pos_weight_from
from hfml.ml.ml02_credit_risk.train import (
    ALGORITHMS,
    FEATURE_SETS,
    XGBOOST,
    XGBOOST_PARAMS,
    load_training_data,
    models_dir,
    results_frame,
    save_run,
    train_xgboost,
)

log = get_logger(__name__)

SHOWN = ["pr_auc", "pr_auc_lift", "roc_auc", "f1_positive",
         "recall_positive", "precision_positive", "accuracy", "brier"]


def saved_pr_auc(algo: str, feature_set: str) -> float | None:
    """PR-AUC đã ghi của một thuật toán, đọc từ metadata thay vì train lại."""
    path = models_dir() / f"ml02_{algo}_{feature_set}.metadata.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))["metrics_validation"]["pr_auc"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=None,
                        help="giới hạn số dòng train — CHỈ để chạy thử")
    parser.add_argument("--no-save", action="store_true",
                        help="chỉ in kết quả, không ghi file")
    args = parser.parse_args()

    data = load_training_data(nrows=args.rows)
    weight = scale_pos_weight_from(data.y_train)

    print("\n=== Task 10 · XGBoost ===")
    print(f"  train {len(data.X_train):,} · validation {len(data.X_validation):,}")
    for key, value in XGBOOST_PARAMS.items():
        print(f"  {key:<17}{value}")
    print(f"  scale_pos_weight {weight:.6f}  ← tính từ RIÊNG tập train (task 4)")
    print("  KHÔNG early stopping — nó cần dừng theo validation, mà đó chính là")
    print("  tập dùng để báo cáo và để chọn model ở task 12.")
    print("  Tập test KHÔNG được chạm.")

    models = train_xgboost(data)
    table = results_frame(models)

    print("\n=== Chỉ số trên VALIDATION ===")
    print(table[["algo", "feature_set", "n_features", *SHOWN]].to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n=== Học thuộc tới đâu (PR-AUC train so với validation) ===")
    for model in models:
        print(f"  {model.feature_set:<8} train {model.metrics_train['pr_auc']:.4f} "
              f"→ validation {model.pr_auc:.4f}  (gap {model.overfit_gap:+.4f})")

    print("\n=== Bốn thuật toán trên cùng một phép chia (PR-AUC validation) ===")
    print("  (bảng so sánh chính thức là task 12 — đây chỉ để đối chiếu nhanh)")
    for feature_set in FEATURE_SETS:
        parts = []
        for algo in ALGORITHMS:
            value = (next(m.pr_auc for m in models if m.feature_set == feature_set)
                     if algo == XGBOOST else saved_pr_auc(algo, feature_set))
            parts.append(f"{algo} {value:.4f}" if value is not None else f"{algo} —")
        print(f"  {feature_set:<8} " + " · ".join(parts))

    print("\n=== So với baseline task 6 ===")
    for model in models:
        lift = model.metrics_validation["pr_auc_lift"]
        print(f"  {'✅' if lift >= MEANINGFUL_LIFT else '⚠️'} "
              f"{model.feature_set:<8} PR-AUC {model.pr_auc:.4f} = {lift:.2f}× "
              f"mức đoán bừa (mốc {MEANINGFUL_LIFT}×)")

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
