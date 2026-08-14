"""Entry-point chính thức cho phần TRAIN của ML01 — 4 thuật toán (F03 task 7–10).

    .venv\\Scripts\\python.exe scripts/train_ml01.py
    .venv\\Scripts\\python.exe scripts/train_ml01.py --rows 2000    # chạy nhanh
    .venv\\Scripts\\python.exe scripts/train_ml01.py --no-save      # không ghi runs/

Script chỉ ĐIỀU PHỐI: nó gọi lần lượt bốn hàm train đã có trong
`hfml.ml.ml01_recommendation.train`, không tự cài đặt lại thuật toán nào.
Mỗi hàm tự lo phần việc của nó — tách 80/20, CV 5-fold trên tập train, fit
model cuối trên toàn bộ tập train — nên chạy qua đây hay gọi từng hàm riêng
đều cho cùng một kết quả.

Phạm vi dừng ở TRAIN. Không đánh giá trên test, không so sánh, không feature
importance, không chọn model, không export — mỗi việc đó có đường chạy riêng.

Chạy `python` thay vì `.venv\\Scripts\\python.exe` sẽ hỏng: package `hfml`
chỉ được cài trong venv.
"""
from __future__ import annotations

import argparse
import inspect
import sys

import pandas as pd

from hfml.data.synthetic import PopulationParams
from hfml.logger import get_logger
from hfml.ml.ml01_recommendation.report import build_ml01_report
from hfml.ml.ml01_recommendation.train import (
    BAGGING,
    DECISION_TREE,
    RANDOM_FOREST,
    XGBOOST,
    train_bagging,
    train_decision_tree,
    train_random_forest,
    train_xgboost,
)

log = get_logger(__name__)

#: Thứ tự chạy = thứ tự task trong PLAN. Giữ nguyên để log đọc từ trên xuống
#: khớp với thứ tự task 7 → 10.
TRAINERS = (
    (DECISION_TREE, train_decision_tree),
    (BAGGING, train_bagging),
    (RANDOM_FOREST, train_random_forest),
    (XGBOOST, train_xgboost),
)


def _call(trainer, **kwargs):
    """Gọi hàm train, chỉ truyền những tham số mà chính nó nhận.

    Bốn hàm không cùng chữ ký: `train_decision_tree` (task 7) không có
    `save`/`runs_dir` vì task 7 được giao trước khi chốt quy ước ghi output
    vào `src/training/runs/`. Lọc tham số ở đây để entry-point chạy được cả
    bốn mà không phải sửa code của task 7.
    """
    accepted = inspect.signature(trainer).parameters
    return trainer(**{name: value for name, value in kwargs.items()
                      if name in accepted})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=None,
                        help="số hộ sinh ra (mặc định: theo PopulationParams)")
    parser.add_argument("--seed", type=int, default=None,
                        help="ghi đè random_seed của config")
    parser.add_argument("--n-splits", type=int, default=None,
                        help="số fold CV (mặc định: theo config)")
    parser.add_argument("--no-save", action="store_true",
                        help="không ghi artifact/kết quả ra src/training/runs/")
    args = parser.parse_args()

    params = PopulationParams(n=args.rows) if args.rows else None
    rows = []

    for algo, trainer in TRAINERS:
        log.info("── Train %s ──", algo)
        result = _call(trainer, params=params, seed=args.seed,
                       n_splits=args.n_splits, save=not args.no_save)
        metrics = result["cv_metrics"]
        rows.append({
            "thuật toán": algo,
            "CV accuracy": metrics["accuracy"],
            "CV macro-F1": metrics["macro_f1"],
            "sd giữa fold": metrics["macro_f1_std"],
            "artifact": (result["artifact"].name if "artifact" in result
                         else "— không ghi —"),
        })

    table = pd.DataFrame(rows)
    print(f"\n=== ML01 · CV trên tập train ({len(table)} thuật toán) ===")
    print(table.to_string(index=False))

    # Nêu ra ngay chỗ nào không sinh artifact. Im lặng ở đây thì mãi sau mới
    # phát hiện thiếu file, lúc đã cần dùng tới nó.
    unsaved = [row["thuật toán"] for row in rows if row["artifact"].startswith("—")]
    if unsaved and not args.no_save:
        print(f"\nCHÚ Ý: không có artifact cho {', '.join(unsaved)} — "
              "hàm train tương ứng chưa ghi output.")

    if not args.no_save:
        report = build_ml01_report(params, seed=args.seed, n_splits=args.n_splits)
        print("\n=== Chỉ số trên tập test ===")
        print(report["test_summary"].to_string(index=False))
        print("\n=== Hình đã ghi ===")
        for name, path in report["figures"].items():
            print(f"{name:<20} → {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
