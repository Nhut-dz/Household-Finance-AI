r"""Entry-point cho ML02 task 13 — Phân tích feature importance (F04 · M04).

    .venv\Scripts\python.exe scripts/importance_ml02.py
    .venv\Scripts\python.exe scripts/importance_ml02.py --builtin-only
    .venv\Scripts\python.exe scripts/importance_ml02.py --no-save

Ba cách đo cho 8 model: built-in (impurity), permutation trên PR-AUC, và SHAP.

⚠️ Kết quả là CHẨN ĐOÁN, không phải bước chọn feature. Permutation và SHAP đo
trên validation; dùng chúng để bỏ bớt feature rồi train lại và chấm lại trên
chính tập đó là RÒ RỈ. Feature set giữ nguyên sau task này.

Thành phẩm: `src/training/runs/ml02_importance/` — 4 bảng + metadata, kèm hình
`feature_importance.png` (bỏ bằng `--no-plots`).
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from hfml.logger import get_logger
from hfml.ml.evaluation.plots_ml02 import generate_importance_plots
from hfml.ml.ml02_credit_risk.importance import (
    N_REPEATS,
    PERMUTATION_SCORING,
    SAMPLE_SIZE,
    TOP_N,
    analyse_all,
    rank_comparison,
    write_importance,
)
from hfml.ml.ml02_credit_risk.train import load_training_data

log = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--builtin-only", action="store_true",
                        help="bỏ permutation và SHAP (chạy nhanh)")
    parser.add_argument("--models", nargs="*", default=None,
                        help="giới hạn model, dạng algo:feature_set")
    parser.add_argument("--no-save", action="store_true",
                        help="chỉ in báo cáo, không ghi file")
    parser.add_argument("--no-plots", action="store_true",
                        help="ghi CSV nhưng không vẽ hình")
    args = parser.parse_args()

    only = None
    if args.models:
        only = [tuple(spec.split(":")) for spec in args.models]

    data = load_training_data()
    print("\n=== Task 13 · Phân tích feature importance ===")
    print(f"  Đo trên {SAMPLE_SIZE:,} hồ sơ lấy từ validation.")
    print(f"  Permutation dùng chỉ số {PERMUTATION_SCORING} × {N_REPEATS} lần —")
    print("  đo bằng accuracy thì cột nào cũng 'không quan trọng', vì bỏ hết")
    print("  feature vẫn được 91,93% nhờ đoán toàn lớp âm.")
    print("  Tập test KHÔNG được chạm.")

    results = analyse_all(
        data, only=only,
        with_permutation=not args.builtin_only,
        with_shap=not args.builtin_only)

    for result in results:
        print(f"\n{'=' * 70}")
        print(f"{result.algo} · bộ {result.feature_set}")
        print("=" * 70)

        print(f"\n  Top {TOP_N // 2} theo built-in (impurity):")
        for _, row in result.builtin.head(TOP_N // 2).iterrows():
            print(f"    {row['feature']:<32} {row['importance']:.4f}")

        if not result.permutation.empty:
            print(f"\n  Top {TOP_N // 2} theo permutation (PR-AUC tụt bao nhiêu):")
            for _, row in result.permutation.head(TOP_N // 2).iterrows():
                print(f"    {row['feature']:<32} {row['importance']:+.4f} "
                      f"± {row['std']:.4f}")
            am = result.permutation[result.permutation["importance"] < 0]
            if len(am):
                print(f"    ({len(am)} feature có giá trị ÂM — xáo trộn chúng làm")
                print("     model TỐT LÊN, tức chúng chỉ đang thêm nhiễu)")

        if not result.shap.empty:
            print(f"\n  Top {TOP_N // 2} theo SHAP (trung bình |giá trị|):")
            for _, row in result.shap.head(TOP_N // 2).iterrows():
                print(f"    {row['feature']:<32} {row['importance']:.4f}")

        comparison = rank_comparison(result)
        if not comparison.empty and "rank_spread" in comparison:
            bat_dong = comparison.nlargest(3, "rank_spread")
            print("\n  Ba feature BA CÁCH ĐO BẤT ĐỒNG nhất (chênh thứ hạng):")
            for _, row in bat_dong.iterrows():
                parts = [f"{c.replace('rank_', '')} #{int(row[c])}"
                         for c in comparison.columns if c.startswith("rank_")
                         and c not in ("rank_mean", "rank_spread")]
                print(f"    {row['feature']:<32} " + " · ".join(parts))

    print(f"\n{'=' * 70}")
    print("Đọc bảng cho đúng")
    print("=" * 70)
    print("  built-in THIÊN VỊ cột nhiều giá trị: một cột liên tục có hàng nghìn")
    print("  điểm cắt khả dĩ nên gần như luôn tìm được lát cắt 'có vẻ tốt', còn")
    print("  cột nhị phân chỉ có một. Thiên vị này KHÔNG tự lộ ra ở bảng.")
    print("  permutation đo đúng đóng góp vào NĂNG LỰC DỰ BÁO — tin được hơn.")
    print("\n  ⚠️ Đây là CHẨN ĐOÁN. Không dùng bảng này để bỏ feature rồi train")
    print("  lại: validation khi ấy đã tham gia quyết định feature nào tồn tại.")
    print("  Chọn feature có giám sát nằm TRONG Pipeline, chỉ thấy tập train.")

    if args.no_save:
        print("\n(--no-save: không ghi file nào)")
        return 0

    print()
    for name, path in write_importance(results).items():
        print(f"  ghi → {path}")

    if args.no_plots:
        print()
        print("(--no-plots: không vẽ hình nào)")
        return 0

    # Vẽ NGAY SAU khi ghi CSV. Hàm vẽ đọc lại chính những file vừa ghi ở trên,
    # nên hình không bao giờ dựng được từ số của lần chạy trước.
    for name, path in generate_importance_plots(top_n=TOP_N).items():
        print(f"  vẽ  → {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
