r"""Entry-point cho ML02 task 6 — Xây dựng baseline (F04 · M04).

    .venv\Scripts\python.exe scripts/baseline_ml02.py
    .venv\Scripts\python.exe scripts/baseline_ml02.py --no-save

Fit `DummyClassifier` trên tập train của task 5, chấm trên **validation**, ghi
lại làm mốc cho task 7–12.

⚠️ KHÔNG train và KHÔNG tối ưu model chính nào. Tập test vẫn khoá — chỉ mở ở
task 14.

Thành phẩm:

    src/training/runs/ml02_baseline/metrics.csv          chỉ số hai hàng
    src/training/runs/ml02_baseline/confusion_*.csv      confusion matrix
    src/training/runs/ml02_baseline/baseline_metadata.json
    src/training/runs/results.csv                        nối thêm hai dòng
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

import pandas as pd

from hfml.config import CONFIG
from hfml.logger import get_logger
from hfml.ml.evaluation.tracking import append_results
from hfml.ml.ml02_credit_risk.baseline import (
    MEANINGFUL_LIFT,
    baseline_of,
    evaluate_baselines,
    expected_random_pr_auc,
    metrics_frame,
)
from hfml.ml.ml02_credit_risk.clean import TARGET_COLUMN, load_clean_application
from hfml.ml.ml02_credit_risk.split import load_split

log = get_logger(__name__)

#: Chỉ số đưa vào bảng in ra terminal, theo thứ tự quan trọng giảm dần.
SHOWN = ["pr_auc", "pr_auc_lift", "roc_auc", "f1_positive",
         "recall_positive", "precision_positive", "accuracy", "brier"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-save", action="store_true",
                        help="chỉ in báo cáo, không ghi file")
    args = parser.parse_args()

    df = load_clean_application()
    split = load_split()

    y_train = split.apply(df, "train")[TARGET_COLUMN]
    y_validation = split.apply(df, "validation")[TARGET_COLUMN]

    print(f"\n=== Baseline — fit trên train ({len(y_train):,}), "
          f"chấm trên validation ({len(y_validation):,}) ===")
    print("  Tập test KHÔNG được chạm ở task này.")

    results = evaluate_baselines(y_train, y_validation)
    table = metrics_frame(results)

    print()
    print(table[["name", *SHOWN]].to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))

    for result in results:
        print(f"\n--- {result.name} ---")
        print(f"  {result.role}")
        print(result.confusion.to_string())

    baseline = baseline_of(results)
    expected = expected_random_pr_auc(y_validation)

    print("\n=== Đọc bảng cho đúng ===")
    print(f"  Sàn của PR-AUC là TỈ LỆ DƯƠNG = {expected:.4f}, KHÔNG phải 0,5.")
    print(f"  PR-AUC đo được của baseline    = {baseline.pr_auc:.4f} "
          f"(lệch {abs(baseline.pr_auc - expected):.4f} so với lý thuyết)")
    print(f"  ROC-AUC của baseline           = {baseline.metrics['roc_auc']:.4f} "
          "(≈ 0,5 như mong đợi)")
    print(f"\n  → Task 7–10: model đạt PR-AUC ≥ {expected * MEANINGFUL_LIFT:.4f} "
          f"(lift ≥ {MEANINGFUL_LIFT}) mới coi là học được từ feature.")

    majority = next(r for r in results if r.name != baseline.name)
    print(f"\n  Hàng tham chiếu `most_frequent`: accuracy "
          f"{majority.metrics['accuracy']:.4%} nhưng recall lớp dương "
          f"{majority.metrics['recall_positive']:.4f}")
    print("  → Đây là lý do chọn model bằng PR-AUC chứ không phải accuracy.")

    if args.no_save:
        print("\n(--no-save: không ghi file nào)")
        return 0

    out_dir = CONFIG.paths.runs / "ml02_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)

    print()
    path = out_dir / "metrics.csv"
    table.to_csv(path, index=False, encoding="utf-8")
    print(f"  ghi → {path}")

    for result in results:
        path = out_dir / f"confusion_{result.name}.csv"
        result.confusion.to_csv(path, encoding="utf-8")
        print(f"  ghi → {path}")

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "task": "ML02 task 6 — Xây dựng baseline",
        "random_seed": CONFIG.random_seed,
        "evaluated_on": "validation",
        "test_set_touched": False,
        "n_train": int(len(y_train)),
        "n_validation": int(len(y_validation)),
        "official_baseline": baseline.name,
        "selection_metric": "pr_auc",
        "pr_auc_floor_is_base_rate": expected,
        "results": table.to_dict("records"),
    }
    path = out_dir / "baseline_metadata.json"
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    print(f"  ghi → {path}")

    # Nối vào sổ chung của F07 task 1 để task 12 dựng bảng so sánh từ một nguồn.
    rows = pd.DataFrame([{
        "task": "ml02",
        "algo": r.name,
        "feature_set": "—",       # DummyClassifier không đọc feature nào
        "random_seed": CONFIG.random_seed,
        "n_rows": int(len(y_train)),
        "n_splits": 1,            # holdout, KHÔNG K-Fold
        "split": "validation",
        "note": r.role,
        **r.metrics,
    } for r in results])
    print(f"  ghi → {append_results(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
