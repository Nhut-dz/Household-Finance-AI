r"""Entry-point cho ML02 task 8 — Train Bagging Classifier (F04 · M04).

    .venv\Scripts\python.exe scripts/train_ml02_bagging.py
    .venv\Scripts\python.exe scripts/train_ml02_bagging.py --rows 20000
    .venv\Scripts\python.exe scripts/train_ml02_bagging.py --no-save

Cùng phép chia (task 5), cùng Pipeline feature (task 3), cùng tỉ số phạt
(task 4), cùng siêu tham số cây con (task 7) — chỉ khác ở chỗ lấy 50 mẫu
bootstrap rồi trung bình lại. Nhờ vậy chênh lệch so với task 7 đọc được đúng
là phần do giảm phương sai đem lại.

⚠️ CHỈ Bagging. Random Forest / XGBoost là task 9, 10. Tập test vẫn khoá.
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from hfml.config import CONFIG
from hfml.logger import get_logger
from hfml.ml.evaluation.tracking import append_results
from hfml.ml.ml02_credit_risk.baseline import MEANINGFUL_LIFT
from hfml.ml.ml02_credit_risk.train import (
    BAGGING,
    DECISION_TREE,
    N_ESTIMATORS,
    decision_tree_params,
    load_training_data,
    models_dir,
    results_frame,
    save_run,
    train_bagging,
)

log = get_logger(__name__)

SHOWN = ["pr_auc", "pr_auc_lift", "roc_auc", "f1_positive",
         "recall_positive", "precision_positive", "accuracy", "brier"]


def previous_pr_auc(feature_set: str) -> float | None:
    """PR-AUC của Decision Tree ở task 7, đọc từ metadata đã ghi.

    Đọc lại từ file thay vì train lại: train lại có nguy cơ ra số khác nếu một
    tham số nào đó lệch, và khi đó phần "Bagging hơn cây đơn bao nhiêu" sẽ so
    với một con số không phải con số đã báo cáo ở task 7.
    """
    import json

    path = models_dir() / f"ml02_{DECISION_TREE}_{feature_set}.metadata.json"
    if not path.exists():
        return None
    metadata = json.loads(path.read_text(encoding="utf-8"))
    return metadata["metrics_validation"]["pr_auc"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=None,
                        help="giới hạn số dòng train — CHỈ để chạy thử")
    parser.add_argument("--no-save", action="store_true",
                        help="chỉ in kết quả, không ghi file")
    args = parser.parse_args()

    data = load_training_data(nrows=args.rows)
    params = decision_tree_params(len(data.X_train))

    print("\n=== Task 8 · Bagging Classifier ===")
    print(f"  train {len(data.X_train):,} · validation {len(data.X_validation):,}")
    print(f"  n_estimators : {N_ESTIMATORS} (dùng chung với Random Forest task 9)")
    print(f"  cây con      : {params}")
    print("  cân bằng lớp : class_weight='balanced' trên CÂY CON "
          "(BaggingClassifier không có tham số này)")
    print("  max_features : 1.0 — ranh giới với Random Forest ở task 9")
    print("  Tập test KHÔNG được chạm.")

    models = train_bagging(data)
    table = results_frame(models)

    print("\n=== Chỉ số trên VALIDATION ===")
    print(table[["algo", "feature_set", "n_features", *SHOWN]].to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n=== Học thuộc tới đâu (PR-AUC train so với validation) ===")
    for model in models:
        print(f"  {model.feature_set:<8} train {model.metrics_train['pr_auc']:.4f} "
              f"→ validation {model.pr_auc:.4f}  (gap {model.overfit_gap:+.4f})")

    print("\n=== Bagging so với cây đơn (task 7) — phần do giảm phương sai ===")
    for model in models:
        truoc = previous_pr_auc(model.feature_set)
        if truoc is None:
            print(f"  {model.feature_set:<8} chưa có kết quả task 7 để đối chiếu")
            continue
        delta = model.pr_auc - truoc
        print(f"  {model.feature_set:<8} {truoc:.4f} → {model.pr_auc:.4f} "
              f"({delta:+.4f}, {delta / truoc:+.1%})")

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
