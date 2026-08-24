r"""Vẽ lại toàn bộ hình báo cáo ML02 mà KHÔNG chạy lại task nào (F04).

    .venv\Scripts\python.exe scripts/plots_ml02.py
    .venv\Scripts\python.exe scripts/plots_ml02.py --top-n 20
    .venv\Scripts\python.exe scripts/plots_ml02.py --only evaluation,selection

Sáu hình, ghi vào đúng thư mục chứa CSV đã sinh ra chúng:

    ml02_evaluation/   precision_recall_curve.png · roc_curve.png
                       threshold_analysis.png                      (task 11)
    ml02_comparison/   model_comparison.png                        (task 12)
    ml02_importance/   feature_importance.png                      (task 13)
    ml02_selection/    confusion_matrix_test.png                   (task 14)

Bốn script `evaluate_ml02.py`, `compare_ml02.py`, `importance_ml02.py`,
`select_ml02.py` đã TỰ vẽ phần hình của mình ngay sau khi ghi CSV, nên script
này chỉ cần khi muốn vẽ lại mà không chạy lại task — ví dụ sau khi sửa mã vẽ,
hoặc khi đổi `--top-n`.

Script KHÔNG tính lại chỉ số nào và KHÔNG chạm tập test: nó chỉ đọc các file
CSV/JSON đã có trong `src/training/runs/ml02_*/`. Thiếu file đầu vào thì nó
dừng và nói rõ phải chạy task nào trước, chứ không vẽ ra một hình rỗng.

Chạy `python` thay vì `.venv\Scripts\python.exe` sẽ hỏng: package `hfml`
chỉ được cài trong venv.
"""
from __future__ import annotations

import argparse
import sys

from hfml.ml.evaluation.plots_ml02 import (
    generate_comparison_plots,
    generate_evaluation_plots,
    generate_importance_plots,
    generate_selection_plots,
)

#: Tên nhóm → (task, hàm vẽ). Thứ tự ở đây là thứ tự phụ thuộc của F04.
GROUPS = {
    "evaluation": ("11", lambda top_n: generate_evaluation_plots()),
    "comparison": ("12", lambda top_n: generate_comparison_plots()),
    "importance": ("13", lambda top_n: generate_importance_plots(top_n=top_n)),
    "selection": ("14", lambda top_n: generate_selection_plots()),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default=None,
                        help=f"chỉ vẽ các nhóm này ({', '.join(GROUPS)})")
    parser.add_argument("--top-n", type=int, default=15,
                        help="số feature trong hình importance (mặc định 15)")
    args = parser.parse_args()

    chosen = list(GROUPS)
    if args.only:
        chosen = [name.strip() for name in args.only.split(",") if name.strip()]
        unknown = [name for name in chosen if name not in GROUPS]
        if unknown:
            raise SystemExit(
                f"Nhóm không có: {', '.join(unknown)}\n"
                f"  chọn trong: {', '.join(GROUPS)}")

    figures: dict[str, str] = {}
    for name in chosen:
        task, draw = GROUPS[name]
        print(f"\n── task {task} · {name} ──")
        # Không bắt lỗi ở đây: thiếu CSV đầu vào là tình huống PHẢI dừng, và
        # `FileNotFoundError` của tầng vẽ đã nói rõ thiếu file nào.
        for key, path in draw(args.top_n).items():
            print(f"  vẽ → {path}")
            figures[key] = str(path)

    print(f"\n=== Đã vẽ {len(figures)} hình ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
