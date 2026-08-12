"""In "phễu" tiền xử lý: mỗi bước của Pipeline cắt bao nhiêu cột (F01 task 14).

    python scripts/show_preprocessing.py                 # bộ Full
    python scripts/show_preprocessing.py --rows 50000    # chạy nhanh trên mẫu

Bảng in ra dùng thẳng cho báo cáo: nó trả lời "bạn xử lý dữ liệu thế nào?"
bằng con số thay vì bằng mô tả.

Quan trọng: pipeline được `fit` CHỈ trên tập train (chia stratify theo TARGET),
đúng như lúc train thật — nếu fit trên toàn bộ dữ liệu thì con số in ra sẽ
không phải con số của model thật.
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from hfml.config import CONFIG
from hfml.data import loader
from hfml.data.preprocessing.pipeline import build_preprocessing_pipeline, feature_names
from hfml.logger import get_logger, log_run_context

log = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=None,
                        help="giới hạn số dòng đọc (mặc định: toàn bộ 307.511)")
    parser.add_argument("--encoding", default="ordinal", choices=["ordinal", "onehot"])
    parser.add_argument("--scaling", default="none",
                        choices=["none", "standard", "minmax", "robust"])
    args = parser.parse_args()

    log_run_context(log)

    if not loader.resolve(loader.PRIMARY_FILE).exists():
        log.error("Chưa có dataset. Xem %s", loader._DOWNLOAD_HINT)
        return 1

    df = loader.load_application_train(nrows=args.rows)
    y = df[loader.TARGET_COLUMN]
    X = df.drop(columns=[loader.TARGET_COLUMN])

    X_train, X_test, y_train, _ = train_test_split(
        X, y, test_size=CONFIG.training["test_size"],
        stratify=y, random_state=CONFIG.random_seed)

    pipeline = build_preprocessing_pipeline(
        encoding=args.encoding, scaling=args.scaling)
    pipeline.fit(X_train, y_train)

    # Chạy lại từng bước để đếm cột còn lại sau mỗi bước.
    rows: list[dict] = []
    current = X_train
    previous = current.shape[1]
    rows.append({"bước": "đầu vào", "cột": previous, "thay đổi": ""})
    for name, step in pipeline.steps:
        current = step.transform(current)
        delta = current.shape[1] - previous
        rows.append({
            "bước": name,
            "cột": current.shape[1],
            "thay đổi": f"{delta:+d}" if delta else "0",
        })
        previous = current.shape[1]

    print(f"\n=== Phễu tiền xử lý — fit trên {len(X_train):,} dòng train ===")
    print(pd.DataFrame(rows).to_string(index=False))

    names = feature_names(pipeline)
    print(f"\nFeature cuối cùng: {len(names)}")
    print("Cờ _MISSING giữ lại:",
          [n for n in names if n.endswith("_MISSING")] or "không có")

    transformed = pipeline.transform(X_test)
    print(f"\nÁp lên test ({len(X_test):,} dòng): {transformed.shape[1]} cột, "
          f"còn thiếu {int(transformed.isna().to_numpy().sum())} ô")
    assert list(transformed.columns) == names, "thứ tự feature train ≠ test"
    print("Thứ tự feature train và test khớp nhau.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
