r"""Entry-point cho ML02 task 12 — So sánh model (F04 · M04).

    .venv\Scripts\python.exe scripts/compare_ml02.py
    .venv\Scripts\python.exe scripts/compare_ml02.py --resamples 200   # chạy nhanh
    .venv\Scripts\python.exe scripts/compare_ml02.py --no-save

Xếp hạng 8 model trên validation và trả lời câu hỏi mà task 9 để ngỏ: chênh
lệch giữa hai model là thật hay chỉ là nhiễu. Vì task 5 bỏ K-Fold nên không có
độ lệch giữa fold — dùng **bootstrap cặp đôi** trên tập validation thay thế.

⚠️ Task này XẾP HẠNG, chưa chốt và chưa export:

    task 12 (đây)   xếp hạng, chỉ ra model dẫn đầu
    task 14         chốt model + kiểm trên test
    task 15         export

Thành phẩm: `src/training/runs/ml02_comparison/` — 4 bảng + metadata, kèm hình
`model_comparison.png` (bỏ bằng `--no-plots`).
"""
from __future__ import annotations

import argparse
import sys

from hfml.logger import get_logger
from hfml.ml.evaluation.plots_ml02 import generate_comparison_plots
from hfml.ml.ml02_credit_risk.compare import (
    CONFIDENCE,
    N_BOOTSTRAP,
    SELECTION_METRIC,
    build_tables,
    leaders,
    write_comparison,
)
from hfml.ml.ml02_credit_risk.evaluate import evaluate_all
from hfml.ml.ml02_credit_risk.train import FEATURE_SETS, load_training_data

log = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resamples", type=int, default=N_BOOTSTRAP,
                        help=f"số lần bootstrap (mặc định {N_BOOTSTRAP})")
    parser.add_argument("--no-save", action="store_true",
                        help="chỉ in báo cáo, không ghi file")
    parser.add_argument("--no-plots", action="store_true",
                        help="ghi CSV nhưng không vẽ hình")
    args = parser.parse_args()

    data = load_training_data()
    evaluations = evaluate_all(data)

    print(f"\n=== Task 12 · So sánh trên VALIDATION ({len(data.X_validation):,} hồ sơ) ===")
    print(f"  Chỉ số chọn model: {SELECTION_METRIC} (§7.3 — accuracy không cầm lái)")
    print("  Tập test KHÔNG được chạm.")

    # Tính MỘT LẦN rồi dùng cho cả phần in lẫn phần ghi. Bootstrap 1.000 lần
    # trên 8 model là phần đắt nhất của task này; tính hai lần là gấp đôi thời
    # gian chạy mà không được gì.
    print(f"\n  Đang bootstrap {args.resamples} lần cho 8 model…")
    tables = build_tables(evaluations, args.resamples)

    table = tables["comparison"]
    print("\n=== Bảng xếp hạng (trong TỪNG bộ feature) ===")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n  Xếp riêng từng bộ vì FULL và RÚT GỌN là hai bài toán triển khai")
    print("  khác nhau (§7.2) — xếp chung thì bộ deploy được không bao giờ hiện ra.")

    print(f"\n=== Chênh lệch có thật hay là nhiễu? "
          f"(bootstrap {args.resamples} lần, tin cậy {CONFIDENCE:.0%}) ===")
    print("  Task 5 bỏ K-Fold nên không có độ lệch giữa fold để quy chiếu.")
    print("  Bootstrap trên tập validation trả lời được câu đó.\n")

    intervals = tables["confidence_interval"]
    print(intervals.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n=== Model dẫn đầu so với phần còn lại (bootstrap CẶP ĐÔI) ===")
    print("  Cặp đôi vì hai model chấm trên CÙNG 46.127 hồ sơ: phần dao động do")
    print("  tập validation tác động lên cả hai và tự triệt tiêu khi lấy hiệu.")
    for feature_set in FEATURE_SETS:
        leader_pairs = tables["pairwise_vs_leader"]
        pairwise = leader_pairs[leader_pairs["feature_set"] == feature_set]
        if pairwise.empty:
            continue
        print(f"\n  --- bộ {feature_set} ---")
        for _, row in pairwise.iterrows():
            ket_luan = ("PHÂN BIỆT ĐƯỢC" if row["distinguishable"]
                        else "chưa phân biệt được với nhiễu")
            print(f"    {row['model_a']} − {row['model_b']}: "
                  f"{row['diff']:+.4f} "
                  f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] "
                  f"thắng {row['win_rate']:.1%} lần → {ket_luan}")

    print("\n=== Cặp ĐỨNG LIỀN NHAU — nơi khoảng cách hẹp nhất ===")
    print("  Trả lời câu task 9 để ngỏ: Random Forest thua Bagging 0,0018 ở bộ")
    print("  full — thật hay nhiễu?")
    for feature_set in FEATURE_SETS:
        all_adjacent = tables["pairwise_adjacent"]
        adjacent = all_adjacent[all_adjacent["feature_set"] == feature_set]
        if adjacent.empty:
            continue
        print(f"\n  --- bộ {feature_set} ---")
        for _, row in adjacent.iterrows():
            ket_luan = ("PHÂN BIỆT ĐƯỢC" if row["distinguishable"]
                        else "⚠️ CHƯA phân biệt được với nhiễu")
            print(f"    hạng {row['rank_a']} vs {row['rank_b']}: "
                  f"{row['model_a'].replace('ml02_', '')} − "
                  f"{row['model_b'].replace('ml02_', '')} = {row['diff']:+.4f} "
                  f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] → {ket_luan}")

    print("\n=== Full vs Rút gọn — phân tích tính khả thi triển khai (§7.2) ===")
    delta = tables["feature_set_delta"]
    print(delta.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n  `gap` = cái giá của việc form KHÔNG thu được EXT_SOURCE_1/2/3.")

    print("\n=== Model dẫn đầu từng bộ ===")
    for _, row in leaders(table).iterrows():
        print(f"  {row['feature_set']:<8} {row['algo']:<14} "
              f"PR-AUC {row['pr_auc']:.4f}")
    print("\n  ⚠️ 'Dẫn đầu' KHÔNG phải 'được chọn'. Task 14 mới chốt, và phải cân")
    print("  nhắc thêm hiệu chuẩn, mức học thuộc và khả năng triển khai —")
    print("  những thứ một cột PR-AUC không nói ra.")

    if args.no_save:
        print("\n(--no-save: không ghi file nào)")
        return 0

    print()
    for name, path in write_comparison(
            evaluations, args.resamples, tables=tables).items():
        print(f"  ghi → {path}")

    if args.no_plots:
        print()
        print("(--no-plots: không vẽ hình nào)")
        return 0

    # Vẽ NGAY SAU khi ghi CSV. Hàm vẽ đọc lại chính những file vừa ghi ở trên,
    # nên hình không bao giờ dựng được từ số của lần chạy trước.
    for name, path in generate_comparison_plots().items():
        print(f"  vẽ  → {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
