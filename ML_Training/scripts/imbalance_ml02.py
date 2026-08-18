r"""Entry-point cho ML02 task 4 — Xử lý mất cân bằng lớp (F04 · M04).

    .venv\Scripts\python.exe scripts/imbalance_ml02.py
    .venv\Scripts\python.exe scripts/imbalance_ml02.py --no-save

Đo mức mất cân bằng trên dữ liệu đã làm sạch ở task 2, chốt phương án cho bốn
thuật toán, ghi lại lý do.

⚠️ KHÔNG train model nào, và KHÔNG chia tập — việc chia là task 5. Con số
`scale_pos_weight` in ra đây là **số tham chiếu để báo cáo**; số đem đi train
phải tính lại trên riêng tập train ở task 7–10 bằng `scale_pos_weight_from()`.

Thành phẩm:

    src/training/runs/ml02_imbalance/summary.csv     số đo
    src/training/runs/ml02_imbalance/strategy.csv    thuật toán → cơ chế
    src/training/runs/ml02_imbalance/rejected.csv    phương án đã loại + lý do
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

import pandas as pd

from hfml.config import CONFIG
from hfml.logger import get_logger
from hfml.ml.ml02_credit_risk.clean import TARGET_COLUMN, load_clean_application
from hfml.ml.ml02_credit_risk.imbalance import (
    ALGORITHMS,
    measure_imbalance,
    rejected_table,
    strategy_table,
)

log = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-save", action="store_true",
                        help="chỉ in báo cáo, không ghi file")
    args = parser.parse_args()

    y = load_clean_application()[TARGET_COLUMN]
    report = measure_imbalance(y)

    print("\n=== Mức mất cân bằng của TARGET ===")
    print(f"  số hồ sơ                     {report.n_rows:,}")
    print(f"  dương (khó khăn trả nợ)      {report.n_positive:,}")
    print(f"  âm                           {report.n_negative:,}")
    print(f"  tỉ lệ dương                  {report.positive_rate:.4%}")
    print(f"  scale_pos_weight             {report.scale_pos_weight:.4f}")
    print(f"  accuracy nếu đoán toàn 0     {report.majority_class_accuracy:.4%}"
          "   ← lý do chọn model bằng PR-AUC, không phải accuracy")

    print("\n=== Phương án: HỌC CÓ TRỌNG SỐ, không lấy mẫu lại ===")
    table = strategy_table(y)
    print(table.to_string(index=False))
    print(f"\n  Cả {len(ALGORITHMS)} thuật toán nhận cùng tỉ số phạt "
          f"{report.scale_pos_weight:.4f} — đã kiểm: class_weight='balanced' "
          "cho tỉ số trọng số trùng khít scale_pos_weight tới 6 chữ số.")

    print("\n=== Phương án đã cân nhắc rồi loại ===")
    for _, row in rejected_table().iterrows():
        print(f"  ✗ {row['strategy']}")
        print(f"      {row['reason']}")

    print("\n=== Điểm rò rỉ phải canh ===")
    print("  Tỉ lệ dương là một THỐNG KÊ của dữ liệu. Con số 11,3872 ở trên tính")
    print("  trên toàn bộ dataset nên CHỈ dùng để báo cáo. Số đem đi train phải")
    print("  tính lại trên riêng tập train (task 5 chia, task 7–10 gọi")
    print("  `scale_pos_weight_from(y_train)`). Với class_weight='balanced' thì")
    print("  sklearn tự tính từ đúng `y` truyền vào fit(), nên không có cửa rò rỉ.")

    if args.no_save:
        print("\n(--no-save: không ghi file nào)")
        return 0

    out_dir = CONFIG.paths.runs / "ml02_imbalance"
    out_dir.mkdir(parents=True, exist_ok=True)

    print()
    summary = pd.DataFrame([report.to_dict()])
    for name, frame in (("summary", summary),
                        ("strategy", table),
                        ("rejected", rejected_table())):
        path = out_dir / f"{name}.csv"
        frame.to_csv(path, index=False, encoding="utf-8")
        print(f"  ghi → {path}")

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "task": "ML02 task 4 — Xử lý class imbalance",
        "strategy": "cost_sensitive_learning",
        "resampling_used": False,
        "reference_scale_pos_weight": report.scale_pos_weight,
        "note": "Số tham chiếu đo trên toàn bộ dataset. Số dùng khi train phải "
                "tính lại trên riêng tập train.",
        "imbalance": report.to_dict(),
    }
    path = out_dir / "imbalance_metadata.json"
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    print(f"  ghi → {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
