r"""Entry-point cho ML02 task 14 — Chọn model tốt nhất (F04 · M04).

    .venv\Scripts\python.exe scripts/select_ml02.py
    .venv\Scripts\python.exe scripts/select_ml02.py --no-save

⚠️ Đây là task ĐẦU TIÊN được chạm tập test — và chỉ sau khi quyết định đã chốt.
Ba bước, không được đảo thứ tự:

    1. CHỌN model    chỉ bằng bằng chứng validation (task 11, 12, 13)
    2. CHỐT cấu hình hiệu chuẩn xác suất → chọn ngưỡng, cũng trên validation
    3. MỞ tập test   đúng một lần, để BÁO CÁO chứ không để chọn

Export artifact là task 15 — script này KHÔNG ghi model.

Thành phẩm: `src/training/runs/ml02_selection/` — `decision.json`,
`final_metrics.csv`, `test_confusion.csv`.
"""
from __future__ import annotations

import argparse
import sys

import joblib
import numpy as np
import pandas as pd

from hfml.config import CONFIG
from hfml.logger import get_logger
from hfml.ml.ml02_credit_risk.compare import SELECTION_METRIC, output_dir as compare_dir
from hfml.ml.ml02_credit_risk.evaluate import artifact_path
from hfml.ml.ml02_credit_risk.select import (
    ALERT_RATE,
    CALIBRATION_METHOD,
    FinalReport,
    calibrate,
    calibration_gap,
    choose_threshold,
    decide,
    evaluate_on_test,
    load_all_splits,
    threshold_at_alert_rate,
    write_selection,
)
from hfml.ml.evaluation.metrics import binary_metrics

log = get_logger(__name__)

SHOWN = ["pr_auc", "pr_auc_lift", "roc_auc", "f1_positive",
         "recall_positive", "precision_positive", "brier", "accuracy"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-save", action="store_true",
                        help="chỉ in báo cáo, không ghi file")
    args = parser.parse_args()

    comparison = pd.read_csv(compare_dir() / "comparison.csv")
    pairwise = pd.read_csv(compare_dir() / "pairwise_vs_leader.csv")

    # ---- Bước 1: CHỌN, chỉ bằng validation --------------------------------
    print("\n" + "=" * 72)
    print("BƯỚC 1 · CHỌN MODEL — chỉ bằng bằng chứng validation")
    print("=" * 72)
    decision = decide(comparison, pairwise)
    print(f"\n  Model được chọn: {decision.deploy_slug}")
    print(f"  Á quân          : {decision.runner_up}")
    print("\n  Lý do:")
    for index, reason in enumerate(decision.reasons, 1):
        print(f"    {index}. {reason}")

    splits = load_all_splits()
    X_val, y_val = splits["validation"]
    X_test, y_test = splits["test"]

    pipeline = joblib.load(
        artifact_path(decision.algo, decision.deploy_feature_set))

    # ---- Bước 2: CHỐT cấu hình, vẫn trên validation ------------------------
    print("\n" + "=" * 72)
    print("BƯỚC 2 · CHỐT CẤU HÌNH — hiệu chuẩn rồi chọn ngưỡng, trên validation")
    print("=" * 72)

    proba_raw = pipeline.predict_proba(X_val)[:, 1]
    gap_raw = calibration_gap(np.asarray(y_val).astype(int), proba_raw)

    print(f"\n  Hiệu chuẩn bằng {CALIBRATION_METHOD}, fit trên validation.")
    calibrated = calibrate(pipeline, X_val, y_val)
    proba_cal = calibrated.predict_proba(X_val)[:, 1]
    gap_cal = calibration_gap(np.asarray(y_val).astype(int), proba_cal)

    print(f"    gap TRƯỚC hiệu chuẩn : {gap_raw:+.4f}  (model nói quá)")
    print(f"    gap SAU  hiệu chuẩn  : {gap_cal:+.4f}")
    print(f"    Brier trước → sau    : "
          f"{binary_metrics(y_val, proba_raw)['brier']:.4f} → "
          f"{binary_metrics(y_val, proba_cal)['brier']:.4f}")

    threshold, rule = choose_threshold(y_val, proba_cal)
    decision.threshold = threshold
    decision.threshold_rule = rule

    alt = threshold_at_alert_rate(proba_cal, ALERT_RATE)
    print(f"\n  Ngưỡng LOW_RISK / HIGH_RISK = {threshold:.4f}")
    print(f"    Quy tắc: {rule}")
    print(f"    (Không phải 0,5 — với tỉ lệ nền 8,07% thì 0,5 xếp gần như mọi")
    print(f"     hồ sơ vào LOW_RISK.)")
    print(f"\n  Điểm vận hành thứ hai: rà soát {ALERT_RATE:.0%} hồ sơ rủi ro nhất")
    print(f"    ứng với ngưỡng {alt:.4f}")

    validation_metrics = binary_metrics(y_val, proba_cal, threshold=threshold)
    print("\n  Chỉ số trên validation (sau hiệu chuẩn, tại ngưỡng đã chốt):")
    for key in SHOWN:
        print(f"    {key:<22}{validation_metrics[key]:.4f}")

    # ---- Bước 3: MỞ tập test ----------------------------------------------
    print("\n" + "=" * 72)
    print("BƯỚC 3 · MỞ TẬP TEST — đúng một lần, sau khi đã chốt")
    print("=" * 72)
    print(f"\n  {len(X_test):,} hồ sơ chưa từng được chạm ở task 1–13.")

    result = evaluate_on_test(decision, calibrated, X_test, y_test)
    test_metrics = result["metrics"]

    print("\n  Chỉ số trên TEST:")
    print(f"    {'chỉ số':<22}{'validation':>12}{'test':>12}{'chênh':>10}")
    for key in SHOWN:
        delta = validation_metrics[key] - test_metrics[key]
        print(f"    {key:<22}{validation_metrics[key]:>12.4f}"
              f"{test_metrics[key]:>12.4f}{delta:>+10.4f}")
    print(f"    {'calibration_gap':<22}{gap_cal:>12.4f}"
          f"{result['calibration_gap']:>12.4f}"
          f"{gap_cal - result['calibration_gap']:>+10.4f}")

    print(f"\n  Confusion matrix trên test (ngưỡng {threshold:.4f}):")
    print(result["confusion"].to_string())

    # Bộ FULL trên test — CHỈ để báo cáo §7.2, không deploy được.
    reference_metrics: dict = {}
    reference_path = artifact_path(decision.algo, decision.reference_feature_set)
    if reference_path.exists():
        reference = joblib.load(reference_path)
        reference_cal = calibrate(reference, X_val, y_val)
        reference_proba = reference_cal.predict_proba(X_test)[:, 1]
        reference_metrics = binary_metrics(
            y_test, reference_proba, threshold=threshold)
        print(f"\n  Bộ {decision.reference_feature_set} trên test "
              f"(CHỈ để báo cáo §7.2, KHÔNG deploy được):")
        print(f"    PR-AUC {reference_metrics['pr_auc']:.4f} so với "
              f"{test_metrics['pr_auc']:.4f} của bộ triển khai "
              f"({reference_metrics['pr_auc'] - test_metrics['pr_auc']:+.4f})")

    report = FinalReport(
        decision=decision,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        validation_calibration_gap=gap_cal,
        test_calibration_gap=result["calibration_gap"],
        test_confusion=result["confusion"],
        reference_test_metrics=reference_metrics,
    )

    print(f"\n  Chênh lệch tổng quát hoá (PR-AUC validation − test): "
          f"{report.generalisation_gap:+.4f}")
    print("    Lớn nghĩa là việc chọn model đã bám vào validation.")

    print("\n" + "=" * 72)
    print("  ⚠️ CHƯA export. Artifact được ghi ở task 15.")
    print("=" * 72)

    if args.no_save:
        print("\n(--no-save: không ghi file nào)")
        return 0

    print()
    for name, path in write_selection(report).items():
        print(f"  ghi → {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
