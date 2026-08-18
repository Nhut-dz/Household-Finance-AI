r"""Entry-point cho ML02 task 3 — Feature Engineering (F04 · M04).

    .venv\Scripts\python.exe scripts/build_features_ml02.py
    .venv\Scripts\python.exe scripts/build_features_ml02.py --rows 20000
    .venv\Scripts\python.exe scripts/build_features_ml02.py --dry-run

Đọc dữ liệu đã làm sạch ở task 2, dựng hai bộ feature, `fit` Pipeline **chỉ
trên tập train**, rồi ghi Pipeline đã fit cùng metadata.

Thành phẩm:

    src/training/runs/ml02_features/<set>_pipeline.joblib   dùng lại y hệt khi inference
    src/training/runs/ml02_features/<set>_features.csv      danh sách feature trước/sau
    src/training/runs/ml02_features/feature_catalog.csv     mô tả + công thức
    src/training/runs/ml02_features/feature_metadata.json   metadata cho task 4

⚠️ Script CÓ chia train/test, nhưng KHÔNG train model. Chia ở đây vì Pipeline
bắt buộc `fit` chỉ trên train — fit trên toàn bộ dữ liệu rồi mới chia là rò rỉ
(PLAN.md §4.4), và đó chính là thứ task này phải chứng minh là không xảy ra.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from hfml.config import CONFIG
from hfml.logger import get_logger
from hfml.ml.ml02_credit_risk.clean import (
    INVALID_ROW_FLAG,
    TARGET_COLUMN,
    load_clean_application,
    load_clean_bureau,
)
from hfml.ml.ml02_credit_risk.features import (
    ENGINEERED_FEATURES,
    FULL_ONLY_FEATURES,
    REDUCED_FEATURES,
    absolute_money_columns,
    aggregate_bureau,
    build_feature_pipeline,
    split_features_and_target,
)

log = get_logger(__name__)

FEATURE_SETS = ("reduced", "full")


def feature_catalog() -> pd.DataFrame:
    """Bảng mô tả feature sinh thêm — đưa thẳng vào báo cáo được."""
    return pd.DataFrame([
        {"feature": name, "description": desc, "formula": formula,
         "in_reduced": in_reduced}
        for name, desc, formula, in_reduced in ENGINEERED_FEATURES
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=None,
                        help="giới hạn số hồ sơ — CHỈ để chạy thử")
    parser.add_argument("--dry-run", action="store_true",
                        help="chạy và in báo cáo nhưng không ghi file")
    args = parser.parse_args()

    app = load_clean_application()
    if args.rows:
        app = app.head(args.rows)
    bureau = load_clean_bureau()

    log.info("Gộp bureau: %d dòng → một dòng mỗi khách", len(bureau))
    aggregates = aggregate_bureau(bureau)

    X_all, y_all = split_features_and_target(app)

    # Bỏ dòng bất hợp lệ khỏi RIÊNG tập train. Task 2 cố ý chỉ gắn cờ chứ
    # không bỏ, để tập test giữ nguyên phân bố thật — lúc chạy thật hồ sơ bất
    # hợp lệ vẫn cứ đến, và chỉ số phải phản ánh điều đó.
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all,
        test_size=CONFIG.training["test_size"],
        stratify=y_all,
        random_state=CONFIG.random_seed,
    )
    if INVALID_ROW_FLAG in app.columns:
        keep = app.loc[X_train.index, INVALID_ROW_FLAG] == 0
        dropped = int((~keep).sum())
        X_train, y_train = X_train.loc[keep], y_train.loc[keep]
        print(f"Bỏ {dropped} dòng bất hợp lệ khỏi RIÊNG tập train "
              f"(tập test giữ nguyên {len(X_test):,} dòng).")

    print(f"\nTrain {len(X_train):,} · Test {len(X_test):,} "
          f"· tỉ lệ dương train {y_train.mean():.4%} / test {y_test.mean():.4%}")

    results: dict[str, dict] = {}
    artifacts: dict[str, object] = {}

    for feature_set in FEATURE_SETS:
        pipeline = build_feature_pipeline(
            feature_set=feature_set, bureau_aggregates=aggregates)

        # `fit_transform` trên TRAIN, `transform` trên TEST. Không có đường
        # nào để thống kê của test lọt vào phép biến đổi.
        transformed_train = pipeline.fit_transform(X_train, y_train)
        transformed_test = pipeline.transform(X_test)

        names_out = list(transformed_train.columns)
        money = absolute_money_columns(names_out)

        results[feature_set] = {
            "n_features_in": X_train.shape[1],
            "n_features_out": len(names_out),
            "feature_names": names_out,
            "absolute_money_columns": money,
            "n_nan_after": int(transformed_train.isna().sum().sum()),
            "n_inf_after": int(
                transformed_train.select_dtypes("number")
                .isin([float("inf"), float("-inf")]).sum().sum()),
            "test_columns_match": names_out == list(transformed_test.columns),
        }
        artifacts[feature_set] = pipeline

        print(f"\n=== Bộ {feature_set.upper()} ===")
        print(f"  vào  : {X_train.shape[1]} cột")
        print(f"  ra   : {len(names_out)} feature")
        print(f"  NaN còn lại      : {results[feature_set]['n_nan_after']}")
        print(f"  inf còn lại      : {results[feature_set]['n_inf_after']}")
        print(f"  thứ tự cột test khớp train: "
              f"{results[feature_set]['test_columns_match']}")
        if money:
            print(f"  ⚠️ còn {len(money)} cột tiền tuyệt đối: {money[:6]}"
                  f"{' …' if len(money) > 6 else ''}")
        else:
            print("  ✅ không còn cột tiền tuyệt đối nào")
        if feature_set == "reduced":
            print(f"  feature: {', '.join(names_out)}")

    print("\n=== Feature sinh thêm ở task này ===")
    print(feature_catalog().to_string(index=False))

    print(f"\nBộ RÚT GỌN {len(REDUCED_FEATURES)} feature · "
          f"chỉ có ở bộ FULL: {', '.join(FULL_ONLY_FEATURES)}")

    if args.dry_run or args.rows:
        print("\n(--dry-run hoặc --rows: không ghi file nào)")
        return 0

    out_dir = CONFIG.paths.runs / "ml02_features"
    out_dir.mkdir(parents=True, exist_ok=True)

    print()
    for feature_set, pipeline in artifacts.items():
        path = out_dir / f"{feature_set}_pipeline.joblib"
        joblib.dump(pipeline, path)
        print(f"  ghi → {path}")

        frame = pd.DataFrame({"feature": results[feature_set]["feature_names"]})
        frame["position"] = range(len(frame))
        frame_path = out_dir / f"{feature_set}_features.csv"
        frame.to_csv(frame_path, index=False, encoding="utf-8")
        print(f"  ghi → {frame_path}")

    catalog_path = out_dir / "feature_catalog.csv"
    feature_catalog().to_csv(catalog_path, index=False, encoding="utf-8")
    print(f"  ghi → {catalog_path}")

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "task": "ML02 task 3 — Feature Engineering",
        "random_seed": CONFIG.random_seed,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "target_column": TARGET_COLUMN,
        "feature_sets": results,
        "reduced_features": list(REDUCED_FEATURES),
        "full_only_features": list(FULL_ONLY_FEATURES),
    }
    metadata_path = out_dir / "feature_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  ghi → {metadata_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
