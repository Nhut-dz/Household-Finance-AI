r"""Entry-point cho ML02 task 9 — Train Random Forest (F04 · M04).

    .venv\Scripts\python.exe scripts/train_ml02_random_forest.py
    .venv\Scripts\python.exe scripts/train_ml02_random_forest.py --rows 20000
    .venv\Scripts\python.exe scripts/train_ml02_random_forest.py --no-save

Giữ nguyên mọi thứ của task 8 trừ `max_features='sqrt'` — thứ duy nhất phân
biệt Random Forest với Bagging. Nhờ vậy chênh lệch giữa hai dòng đọc được
chính xác là đóng góp của việc lấy mẫu feature tại mỗi lát cắt.

⚠️ CHỈ Random Forest. XGBoost là task 10. Tập test vẫn khoá tới task 14.
"""
from __future__ import annotations

import argparse
import json
import sys

from hfml.logger import get_logger
from hfml.ml.evaluation.tracking import append_results
from hfml.ml.ml02_credit_risk.baseline import MEANINGFUL_LIFT
from hfml.ml.ml02_credit_risk.train import (
    BAGGING,
    DECISION_TREE,
    N_ESTIMATORS,
    RF_MAX_FEATURES,
    decision_tree_params,
    load_training_data,
    models_dir,
    results_frame,
    save_run,
    train_random_forest,
)

log = get_logger(__name__)

SHOWN = ["pr_auc", "pr_auc_lift", "roc_auc", "f1_positive",
         "recall_positive", "precision_positive", "accuracy", "brier"]


def previous_pr_auc(algo: str, feature_set: str) -> float | None:
    """PR-AUC đã ghi của một thuật toán trước đó.

    Đọc từ metadata thay vì train lại: train lại có nguy cơ ra số khác nếu một
    tham số lệch, và khi đó phần "hơn bao nhiêu" so với một con số không phải
    con số đã báo cáo.
    """
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
    params = decision_tree_params(len(data.X_train))

    print("\n=== Task 9 · Random Forest ===")
    print(f"  train {len(data.X_train):,} · validation {len(data.X_validation):,}")
    print(f"  n_estimators : {N_ESTIMATORS} (giống task 8)")
    print(f"  cây          : min_samples_leaf={params['min_samples_leaf']}, "
          f"max_depth={params['max_depth']} (giống task 7, 8)")
    print(f"  max_features : {RF_MAX_FEATURES!r} ← THỨ DUY NHẤT khác Bagging")
    print("  cân bằng lớp : class_weight='balanced' (task 4)")
    print("  Tập test KHÔNG được chạm.")

    models = train_random_forest(data)
    table = results_frame(models)

    print("\n=== Chỉ số trên VALIDATION ===")
    print(table[["algo", "feature_set", "n_features", *SHOWN]].to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n=== Học thuộc tới đâu (PR-AUC train so với validation) ===")
    for model in models:
        print(f"  {model.feature_set:<8} train {model.metrics_train['pr_auc']:.4f} "
              f"→ validation {model.pr_auc:.4f}  (gap {model.overfit_gap:+.4f})")

    print("\n=== Random Forest so với Bagging (task 8) — phần do lấy mẫu feature ===")
    for model in models:
        truoc = previous_pr_auc(BAGGING, model.feature_set)
        if truoc is None:
            print(f"  {model.feature_set:<8} chưa có kết quả task 8 để đối chiếu")
            continue
        delta = model.pr_auc - truoc
        print(f"  {model.feature_set:<8} {truoc:.4f} → {model.pr_auc:.4f} "
              f"({delta:+.4f}, {delta / truoc:+.1%})")

    print("\n=== Ba thuật toán đã train, trên cùng một phép chia ===")
    for feature_set in ("reduced", "full"):
        parts = []
        for algo in (DECISION_TREE, BAGGING):
            value = previous_pr_auc(algo, feature_set)
            parts.append(f"{algo} {value:.4f}" if value else f"{algo} —")
        hien_tai = next(m for m in models if m.feature_set == feature_set)
        parts.append(f"random_forest {hien_tai.pr_auc:.4f}")
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
