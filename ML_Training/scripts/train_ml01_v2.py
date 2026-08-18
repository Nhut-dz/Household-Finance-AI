r"""Huấn luyện ML01 trên tập đã redesign (F03 · redesign 17/08/2026).

    .venv\Scripts\python.exe scripts/train_ml01_v2.py
    .venv\Scripts\python.exe scripts/train_ml01_v2.py --rows 40000
    .venv\Scripts\python.exe scripts/train_ml01_v2.py --no-export

Chỉ chạy SAU KHI `validate_ml01_dataset.py` báo đủ 5/5 tiêu chí. Huấn luyện
trên một tập mà nhãn suy được bằng một phép so sánh thì mọi con số sau đó đều
vô nghĩa, nên script này kiểm lại điều kiện đó trước khi bắt đầu.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from hfml.config import CONFIG                                    # noqa: E402
from hfml.logger import get_logger                                # noqa: E402
from hfml.ml.ml01_recommendation import features as feature_mod   # noqa: E402
from hfml.ml.ml01_recommendation import train_v2                  # noqa: E402
from hfml.ml.ml01_recommendation.dataset import build_dataset     # noqa: E402
from hfml.ml.ml01_recommendation.scoring import GROUPS            # noqa: E402
from hfml.ml.registry import save_model                           # noqa: E402

log = get_logger(__name__)

#: Điều kiện tối thiểu của tập dữ liệu — khớp với `validate_ml01_dataset.py`.
MIN_CLASS_SHARE = 0.10


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=30_000)
    parser.add_argument("--seed", type=int, default=CONFIG.random_seed)
    parser.add_argument("--no-export", action="store_true",
                        help="chạy thử, không ghi artifact")
    args = parser.parse_args()

    print("\n" + "=" * 74)
    print("ML01 v2 — huấn luyện trên tập đã redesign")
    print("=" * 74)

    dataset = build_dataset(n_rows=args.rows, seed=args.seed)
    summary = dataset.summary()

    print(f"\n--- Tập dữ liệu ---")
    print(f"  {summary['n_rows']:,} hồ sơ")
    for name, share in sorted(summary["class_share"].items()):
        count = summary["class_counts"][name]
        print(f"    {name:<14} {count:>6,}  {share:>6.1%}")
    print(f"  {summary['share_margin_below_0.02']:.1%} hồ sơ nằm sát biên "
          f"(chênh điểm nhất/nhì < 0,02)")

    if summary["min_class_share"] < MIN_CLASS_SHARE:
        print(f"\n  ❌ Lớp nhỏ nhất chỉ {summary['min_class_share']:.1%} — "
              f"dưới {MIN_CLASS_SHARE:.0%}. Dừng, không huấn luyện.")
        return 1

    X = feature_mod.build_features(dataset.observed)
    y = dataset.labels

    print(f"\n--- Feature ---")
    print(f"  {X.shape[1]} cột · {X.isna().sum().sum()} ô thiếu")

    model, report = train_v2.run(X, y)

    print(f"\n--- So sánh model ({train_v2.N_FOLDS}-fold CV trên "
          f"{report.n_train:,} hồ sơ train) ---")
    print(f"  {'model':<16}{'CV macro-F1':>16}{'±std':>9}"
          f"{'Test macro-F1':>15}{'Accuracy':>11}")
    for name, scores in report.cv_scores.items():
        mark = " ←" if name == report.selected else ""
        test_f1 = (f"{report.test_scores['macro_f1']:.4f}"
                   if name == report.selected else "—")
        test_acc = (f"{report.test_scores['accuracy']:.4f}"
                    if name == report.selected else "—")
        print(f"  {name:<16}{scores['macro_f1_mean']:>16.4f}"
              f"{scores['macro_f1_std']:>9.4f}{test_f1:>15}{test_acc:>11}{mark}")

    print(f"\n--- Model được chọn: {report.selected} ---")
    print(f"  {report.selection_reason}")

    print(f"\n--- Chỉ số trên test ({report.n_test:,} hồ sơ, chạm MỘT lần) ---")
    for key, value in report.test_scores.items():
        print(f"  {key:<18} {value:.4f}")

    print(f"\n--- Confusion matrix ---")
    print(f"  {'thực \\ đoán':<16}" + "".join(f"{g[:9]:>11}" for g in GROUPS))
    for name, row in zip(GROUPS, report.confusion):
        print(f"  {name:<16}" + "".join(f"{v:>11,}" for v in row))

    print(f"\n--- Báo cáo theo lớp ---")
    for line in report.report_text.splitlines():
        print("  " + line)

    if report.importance:
        print(f"\n--- Feature importance (10 cao nhất) ---")
        for name, value in list(report.importance.items())[:10]:
            bar = "█" * max(1, int(value * 60))
            print(f"  {name:<24}{value:>7.1%}  {bar}")

    if args.no_export:
        print("\n  (--no-export: không ghi artifact)")
        return 0

    path = save_model(model, metrics={"cv": report.cv_scores,
                                      "test": report.test_scores},
                      extra={"selection": report.to_dict(),
                             "dataset": summary,
                             "n_features": X.shape[1],
                             "label_design": "multi-dimensional scoring v2",
                             "limitation": (
                                 "Nhãn là synthetic — sinh từ hàm chấm điểm "
                                 "trên dân số mô phỏng, chưa có ground truth "
                                 "thực tế theo thời gian.")})
    print(f"\n--- Đã ghi artifact ---\n  {path}")
    print(f"  slug = {model.slug}")

    report_path = CONFIG.paths.runs / "ml01_v2_report.json"
    report_path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"  {report_path}")

    print("\n" + "=" * 74)
    print("  Nhắc: nhãn là synthetic. Model chứng minh được khả năng học trên")
    print("  tập mô phỏng, CHƯA chứng minh được khả năng dự đoán hành vi tài")
    print("  chính thực tế của hộ gia đình.")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
