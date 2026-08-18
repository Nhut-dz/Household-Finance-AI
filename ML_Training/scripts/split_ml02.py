r"""Entry-point cho ML02 task 5 — Chia train/validation/test (F04 · M04).

    .venv\Scripts\python.exe scripts/split_ml02.py
    .venv\Scripts\python.exe scripts/split_ml02.py --no-save

Chia 70/15/15 phân tầng theo nhãn, kiểm chứng, rồi ghi danh sách `SK_ID_CURR`
của từng tập để task 6 → 15 dùng chung đúng một bộ.

**KHÔNG dùng K-Fold Cross-Validation.** Mỗi thuật toán fit đúng một lần trên
train và chấm đúng một lần trên validation. Tập test khoá lại, chỉ mở ở task 14.

⚠️ KHÔNG train model nào ở đây.

Thành phẩm:

    src/training/runs/ml02_split/split_assignment.csv   SK_ID_CURR → tập
    src/training/runs/ml02_split/split_metadata.json    tỉ lệ, seed, phân bố
    src/training/runs/ml02_split/distribution.csv       phân bố nhãn ba tập
    src/training/runs/ml02_split/verification.csv       5 phép kiểm
"""
from __future__ import annotations

import argparse
import sys

from hfml.config import CONFIG
from hfml.logger import get_logger
from hfml.ml.ml02_credit_risk.clean import TARGET_COLUMN, load_clean_application
from hfml.ml.ml02_credit_risk.imbalance import scale_pos_weight_from
from hfml.ml.ml02_credit_risk.split import (
    distribution_table,
    load_split,
    save_split,
    split_train_val_test,
    verify_split,
)

log = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-save", action="store_true",
                        help="chỉ in báo cáo, không ghi file")
    args = parser.parse_args()

    df = load_clean_application()
    split = split_train_val_test(df)

    print(f"\n=== Chia 70/15/15 trên {len(df):,} hồ sơ "
          f"(seed {split.seed}, phân tầng theo {TARGET_COLUMN}) ===")
    table = distribution_table(df, split)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    if split.n_invalid_excluded_from_train:
        print(f"\n  Đã loại {split.n_invalid_excluded_from_train} dòng bất hợp lệ "
              "khỏi RIÊNG tập train (validation/test giữ nguyên).")

    print("\n=== Kiểm chứng ===")
    checks = verify_split(df, split)
    for _, row in checks.iterrows():
        print(f"  {'✅' if row['passed'] else '❌'} {row['check']}: {row['measured']}")

    print("\n=== scale_pos_weight tính trên RIÊNG tập train (task 4) ===")
    for name in ("train", "validation", "test"):
        subset = split.apply(df, name)
        print(f"  {name:<11} {scale_pos_weight_from(subset[TARGET_COLUMN]):.4f}")
    print("  Task 7–10 phải dùng con số của TRAIN. Số đo trên toàn bộ dataset")
    print("  (11,3872 ở task 4) chỉ để báo cáo.")

    print("\n=== Phương pháp ===")
    print("  Holdout hai lần cắt, phân tầng — KHÔNG K-Fold Cross-Validation.")
    print("  Hệ quả phải nói rõ ở task 12: mỗi chỉ số là MỘT điểm đo, không có")
    print("  độ lệch giữa các fold, nên chênh vài phần nghìn giữa hai model")
    print("  không quy chiếu được về độ nhiễu.")

    if not checks["passed"].all():
        print("\n❌ Có phép kiểm không đạt — KHÔNG ghi phép chia này.")
        return 1

    if args.no_save:
        print("\n(--no-save: không ghi file nào)")
        return 0

    print()
    written = save_split(split, df)
    out_dir = CONFIG.paths.runs / "ml02_split"
    table.to_csv(out_dir / "distribution.csv", index=False, encoding="utf-8")
    checks.to_csv(out_dir / "verification.csv", index=False, encoding="utf-8")
    for path in list(written.values()) + [out_dir / "distribution.csv",
                                          out_dir / "verification.csv"]:
        print(f"  ghi → {path}")

    # Nạp lại ngay để chứng minh file dùng được, không chỉ ghi được.
    reloaded = load_split()
    assert len(reloaded.train_ids) == len(split.train_ids)
    print(f"\n  Nạp lại kiểm tra: train {len(reloaded.train_ids):,} · "
          f"validation {len(reloaded.validation_ids):,} · "
          f"test {len(reloaded.test_ids):,} ✅")
    return 0


if __name__ == "__main__":
    sys.exit(main())
