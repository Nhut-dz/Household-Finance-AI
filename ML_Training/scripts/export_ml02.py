r"""Entry-point cho ML02 task 15 — Export model (F04 · M04).

    .venv\Scripts\python.exe scripts/export_ml02.py
    .venv\Scripts\python.exe scripts/export_ml02.py --no-save

Đóng gói model đã chốt ở task 14 thành artifact dùng được ở tầng inference,
rồi **nạp lại và kiểm** — "đã export" mà không nạp lại thử thì chỉ nghĩa là
"đã ghi ra một file".

Artifact tự chứa bốn phần: Pipeline feature (task 3) + model đã train
(task 10) + lớp hiệu chuẩn (task 14) + ngưỡng nghiệp vụ (task 14).

Thành phẩm:

    src/training/runs/ml02_xgboost_reduced_vfinal.joblib
    src/training/runs/ml02_xgboost_reduced_vfinal.metadata.json
    src/training/runs/ml02_selection/export_verification.json
"""
from __future__ import annotations

import argparse
import json
import sys

import joblib
import numpy as np

from hfml.config import CONFIG
from hfml.data.quality import load_manifest
from hfml.logger import get_logger
from hfml.ml.ml02_credit_risk.evaluate import artifact_path
from hfml.ml.ml02_credit_risk.export import (
    HIGH_RISK,
    LOW_RISK,
    RISK_LABELS_VI,
    Ml02CreditRiskModel,
    export,
    verify_export,
)
from hfml.ml.ml02_credit_risk.select import (
    calibrate,
    load_all_splits,
    output_dir as selection_dir,
)

log = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-save", action="store_true",
                        help="dựng artifact và kiểm nhưng không ghi ra runs/")
    args = parser.parse_args()

    decision_path = selection_dir() / "decision.json"
    if not decision_path.exists():
        print(f"❌ Chưa có {decision_path}. Chạy task 14 trước.")
        return 1
    decision = json.loads(decision_path.read_text(encoding="utf-8"))

    if decision.get("exported"):
        print("  (decision.json đã ghi exported=true — export lại sẽ ghi đè)")

    print("\n" + "=" * 72)
    print("TASK 15 · EXPORT MODEL")
    print("=" * 72)
    print(f"\n  Model chốt ở task 14 : {decision['selected_model']}")
    print(f"  Thuật toán           : {decision['algorithm']}")
    print(f"  Bộ feature           : {decision['deploy_feature_set']}")
    print(f"  Ngưỡng               : {decision['threshold']['value']:.4f}")
    print(f"  Hiệu chuẩn           : {decision['calibration']['method']} "
          f"(fit trên {decision['calibration']['fitted_on']})")

    # ---- Dựng lại artifact cuối -------------------------------------------
    splits = load_all_splits()
    X_val, y_val = splits["validation"]
    X_test, y_test = splits["test"]

    pipeline = joblib.load(
        artifact_path(decision["algorithm"], decision["deploy_feature_set"]))

    print("\n  Dựng lại lớp hiệu chuẩn trên validation…")
    calibrated = calibrate(pipeline, X_val, y_val)

    feature_names = list(
        pipeline.named_steps["features"].transform(X_val.head(1)).columns)

    model = Ml02CreditRiskModel(
        calibrated=calibrated,
        feature_names=feature_names,
        threshold=decision["threshold"]["value"],
        algo=decision["algorithm"],
        feature_set=decision["deploy_feature_set"],
    )

    print(f"\n  Artifact : {model.slug}")
    print(f"  Feature  : {len(feature_names)} cột, đúng thứ tự")
    print(f"  Nhãn     : {model.classes_} "
          f"({' · '.join(f'{k}={v}' for k, v in RISK_LABELS_VI.items())})")

    # ---- Kiểm hành vi trước khi ghi ---------------------------------------
    sample = X_test.head(2_000)
    proba = model.risk_probability(sample)
    labels = model.predict(sample)

    print("\n  Kiểm hành vi trên 2.000 hồ sơ test:")
    print(f"    xác suất  : min {proba.min():.4f} · trung vị "
          f"{np.median(proba):.4f} · max {proba.max():.4f}")
    print(f"    {HIGH_RISK:<10}: {(labels == HIGH_RISK).sum():,} hồ sơ "
          f"({(labels == HIGH_RISK).mean():.1%})")
    print(f"    {LOW_RISK:<10}: {(labels == LOW_RISK).sum():,} hồ sơ "
          f"({(labels == LOW_RISK).mean():.1%})")

    tai_nguong_05 = (proba >= 0.5).sum()
    print(f"\n    Nếu dùng ngưỡng 0,5 mặc định của sklearn: chỉ "
          f"{tai_nguong_05:,} hồ sơ được gắn {HIGH_RISK}")
    print(f"    → đó là lý do `predict()` phải gói ngưỡng "
          f"{model.threshold:.4f} vào artifact.")

    print("\n  Một bản ghi mẫu cho tầng llm:")
    for record in model.explain(sample.head(1)):
        for key, value in record.items():
            print(f"    {key:<16}{value}")

    if args.no_save:
        print("\n(--no-save: không ghi file nào)")
        return 0

    # ---- Ghi ---------------------------------------------------------------
    try:
        manifest = load_manifest()
    except FileNotFoundError:
        manifest = None
        log.warning("Chưa có dataset_manifest.json — bỏ trống data_version.")

    print()
    written = export(model, decision, manifest)
    for name, path in written.items():
        print(f"  ghi → {path}")

    # ---- Nạp lại và kiểm ---------------------------------------------------
    print("\n" + "=" * 72)
    print("KIỂM TRA NẠP LẠI")
    print("=" * 72)
    result = verify_export(model.slug, sample, proba)

    checks = [
        ("nạp lại được", result["loaded"]),
        ("xác suất trùng khít", result["proba_matches"]),
        ("thứ tự feature khớp metadata", result["feature_names_match"]),
        ("predict áp đúng ngưỡng đã chốt", result["threshold_applied"]),
        ("ngưỡng KHÔNG phải 0,5", result["threshold_is_not_half"]),
    ]
    print()
    for name, passed in checks:
        print(f"  {'✅' if passed else '❌'} {name}")
    print(f"\n  lệch xác suất lớn nhất: {result['max_proba_diff']:.3e}")
    print(f"  nhãn trả về: {result['labels_returned']}")

    path = selection_dir() / "export_verification.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    print(f"\n  ghi → {path}")

    # Cập nhật decision.json: quyết định của task 14 nay đã được export.
    decision["exported"] = True
    decision["exported_slug"] = model.slug
    decision_path.write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  ghi → {decision_path} (exported=true)")

    if not all(passed for _, passed in checks):
        print("\n❌ Có phép kiểm không đạt — KHÔNG dùng artifact này.")
        return 1

    print("\n" + "=" * 72)
    print("  F04 hoàn tất. Tích hợp vào hệ thống là bước riêng:")
    print(f"    ML_Training/src/hfml/api/main.py cần đổi")
    print(f"      ML02_SLUG            = {model.slug!r}")
    print(f"      ML02_RISK_THRESHOLD  = {model.threshold:.4f}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
