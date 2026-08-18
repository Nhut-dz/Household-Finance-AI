r"""Entry-point cho ML02 task 11 — Đánh giá model (F04 · M04).

    .venv\Scripts\python.exe scripts/evaluate_ml02.py
    .venv\Scripts\python.exe scripts/evaluate_ml02.py --no-save

Nạp 8 artifact của task 7–10 và đo đầy đủ trên **validation**. KHÔNG train lại.

⚠️ Task này ĐÁNH GIÁ, chưa xếp hạng và chưa chọn:

    task 11 (đây)   đo đầy đủ từng model
    task 12         so sánh, xác định model tốt nhất
    task 14         chốt model + kiểm trên test

Bảng in ra giữ nguyên thứ tự nạp, KHÔNG sắp xếp theo chỉ số nào.

Thành phẩm: `src/training/runs/ml02_evaluation/` — 6 bảng + metadata.
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from hfml.logger import get_logger
from hfml.ml.ml02_credit_risk.evaluate import (
    ALERT_RATES,
    METRIC_ORDER,
    REPORTING_THRESHOLD,
    calibration_table,
    capture_table,
    confusion_long,
    evaluate_all,
    metrics_table,
    threshold_sweep,
    write_evaluation,
)
from hfml.ml.ml02_credit_risk.train import load_training_data

log = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-save", action="store_true",
                        help="chỉ in báo cáo, không ghi file")
    args = parser.parse_args()

    data = load_training_data()
    print(f"\n=== Task 11 · Đánh giá trên VALIDATION ({len(data.X_validation):,} hồ sơ) ===")
    print("  Nạp artifact của task 7–10, KHÔNG train lại.")
    print("  Tập test KHÔNG được chạm.")

    evaluations = evaluate_all(data)

    print(f"\n=== Nhóm 1+2 · Chỉ số chính (ngưỡng báo cáo {REPORTING_THRESHOLD}) ===")
    print("  Thứ tự giữ nguyên, KHÔNG sắp xếp — xếp hạng là task 12.")
    table = metrics_table(evaluations)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n  accuracy đứng CUỐI bảng vì nó không được cầm lái (§7.3):")
    print("  đoán toàn 0 đã đạt 91,93%.")

    print("\n=== Confusion matrix · 4 ô đếm được ===")
    print("  false_negative = ca vỡ nợ bị BỎ LỌT — đắt nhất trong bài toán này.")
    print(confusion_long(evaluations).to_string(index=False))

    print("\n=== Nhóm 3 · Hiệu chuẩn (gap = model nói − thực tế) ===")
    calibration = calibration_table(evaluations)
    summary = (calibration.groupby(["algo", "feature_set"], sort=False)["gap"]
               .agg(gap_trung_binh="mean", gap_lon_nhat="max").reset_index())
    brier = table.set_index(["algo", "feature_set"])["brier"]
    summary["brier"] = [brier.loc[(a, f)] for a, f in
                        zip(summary["algo"], summary["feature_set"])]
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n  gap DƯƠNG ở mọi model là điều PHẢI xảy ra: class_weight/")
    print("  scale_pos_weight đẩy xác suất lớp dương lên trên tỉ lệ nền thật.")
    print("  Đây chính là lý do §7.4 yêu cầu hiệu chuẩn TRƯỚC khi đặt ngưỡng.")

    print(f"\n=== Nhóm 4 · Soi k% hồ sơ rủi ro nhất thì bắt được bao nhiêu ===")
    capture = capture_table(evaluations)
    pivot = capture.pivot_table(index=["algo", "feature_set"],
                                columns="alert_rate", values="capture_rate",
                                sort=False)
    print(pivot.to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"\n  Cột = tỉ lệ hồ sơ đưa vào rà soát ({', '.join(f'{r:.0%}' for r in ALERT_RATES)}).")
    print("  Ô = phần ca vỡ nợ bắt được. Cách đọc gần vận hành nhất, và KHÔNG")
    print("  cần chọn ngưỡng — chỉ cần một ngân sách rà soát.")

    print("\n=== Quét ngưỡng — NGUYÊN LIỆU cho task 14, không chọn gì ===")
    sweep = threshold_sweep(evaluations)
    best_full = sweep[sweep["feature_set"] == "full"]
    print(best_full.pivot_table(index="threshold", columns="algo",
                                values="f1_positive", sort=False)
          .to_string(float_format=lambda v: f"{v:.4f}"))
    print("\n  (F1 lớp dương theo ngưỡng, bộ FULL. Ngưỡng LOW_RISK/HIGH_RISK")
    print("  chốt ở task 14 SAU khi hiệu chuẩn — chọn trên xác suất chưa hiệu")
    print("  chuẩn thì con số ngưỡng không mang ý nghĩa xác suất nào.)")

    if args.no_save:
        print("\n(--no-save: không ghi file nào)")
        return 0

    print()
    for name, path in write_evaluation(evaluations).items():
        print(f"  ghi → {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
