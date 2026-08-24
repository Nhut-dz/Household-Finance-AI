"""F04 — kiểm tra sáu biểu đồ báo cáo ML02 (`evaluation/plots_ml02.py`).

Cùng tinh thần với `test_plots.py` của ML01: test không chấm "hình có đẹp
không" — mắt người làm việc đó. Chỗ test đứng canh là những thứ hỏng âm thầm:

    - hàm vẽ tự ý TÍNH LẠI thay vì đọc kết quả đã lưu
    - thiếu file đầu vào mà vẫn cho ra một hình rỗng
    - ghi sai thư mục con (hình task 11 rơi vào thư mục task 14)
    - bảng màu trôi khỏi bộ đã qua validator
    - màu bám theo THỨ HẠNG thay vì theo thuật toán

Hai cái cuối quan trọng hơn vẻ ngoài. Bảng màu 4 slot đã được
`validate_palette.js --pairs all` chứng nhận; ai đó đổi một mã hex mà không
chạy lại validator thì hình vẫn vẽ ra bình thường, chỉ có điều người mù màu
không đọc được nữa. Còn nếu màu bám theo thứ hạng thì chỉ cần lần train sau
đổi ngôi là mọi hình trong báo cáo đổi màu hết, và người đọc so hai bản báo
cáo với nhau sẽ hiểu sai.
"""
from __future__ import annotations

import json

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

from hfml.ml.evaluation.plots_ml02 import (  # noqa: E402
    ALGO_COLOR,
    COMPARE_SUBDIR,
    EVAL_SUBDIR,
    IMPORTANCE_SUBDIR,
    SELECTION_SUBDIR,
    SERIES_ML02,
    generate_ml02_plots,
    plot_confusion_matrix_test,
    plot_feature_importance,
    plot_model_comparison,
    plot_precision_recall_curve,
    plot_roc_curve,
    plot_threshold_analysis,
)

ALGOS = ("decision_tree", "bagging", "random_forest", "xgboost")
FEATURE_SETS = ("reduced", "full")
FEATURES = ("dti", "age_years", "credit_term_implied", "bureau_loan_count",
            "employment_years", "bureau_has_overdue")
BASE_RATE = 0.08

#: Hình → thư mục con phải ghi vào. Đây chính là yêu cầu "lưu vào folder
#: artifact tương ứng", nên nó được viết thành test chứ không chỉ nằm trong
#: docstring.
EXPECTED_SUBDIR = {
    "precision_recall_curve": EVAL_SUBDIR,
    "roc_curve": EVAL_SUBDIR,
    "threshold_analysis": EVAL_SUBDIR,
    "model_comparison": COMPARE_SUBDIR,
    "feature_importance": IMPORTANCE_SUBDIR,
    "confusion_matrix_test": SELECTION_SUBDIR,
}


@pytest.fixture
def runs(tmp_path):
    """Thư mục `runs/` giả lập với đúng các file mà sáu hàm vẽ đọc."""
    for subdir in (EVAL_SUBDIR, COMPARE_SUBDIR, IMPORTANCE_SUBDIR,
                   SELECTION_SUBDIR):
        (tmp_path / subdir).mkdir(parents=True)

    def score(index: int, feature_set: str) -> float:
        # Bộ full luôn nhỉnh hơn, đúng như dữ liệu thật (§7.2).
        return 0.14 + index * 0.01 + (0.06 if feature_set == "full" else 0.0)

    curves, metrics, sweep, comparison, intervals = [], [], [], [], []
    importance = {"builtin": [], "permutation": [], "shap": []}

    for index, algo in enumerate(ALGOS):
        for feature_set in FEATURE_SETS:
            pr_auc = score(index, feature_set)

            for step in range(20):
                fraction = step / 19
                curves.append({"algo": algo, "feature_set": feature_set,
                               "curve": "pr", "x": fraction,
                               "y": 1.0 - fraction * (1.0 - BASE_RATE)})
                curves.append({"algo": algo, "feature_set": feature_set,
                               "curve": "roc", "x": fraction,
                               "y": min(1.0, fraction ** 0.6)})

            row = {"algo": algo, "feature_set": feature_set, "pr_auc": pr_auc,
                   "pr_auc_lift": pr_auc / BASE_RATE, "roc_auc": 0.64 + index * 0.02,
                   "f1_positive": 0.2, "recall_positive": 0.5,
                   "precision_positive": 0.13, "brier": 0.2,
                   "balanced_accuracy": 0.6, "accuracy": 0.7}
            metrics.append(row)
            comparison.append({**row, "rank": len(ALGOS) - index})
            intervals.append({"algo": algo, "feature_set": feature_set,
                              "pr_auc": pr_auc, "ci_low": pr_auc - 0.008,
                              "ci_high": pr_auc + 0.008, "ci_width": 0.016})

            for step in range(15):
                threshold = 0.05 + step * 0.05
                sweep.append({
                    "algo": algo, "feature_set": feature_set,
                    "threshold": threshold,
                    # Đỉnh F1 nằm ở chỗ khác nhau tuỳ model — hình phải hiện
                    # ra được điều đó, nên fixture phải có điều đó.
                    "f1_positive": 0.25 - abs(threshold - (0.3 + index * 0.1)) * 0.2,
                    "recall_positive": max(0.0, 1.0 - threshold),
                    "precision_positive": min(1.0, threshold * 0.4),
                    "accuracy": 0.7})

            for rank, feature in enumerate(FEATURES, start=1):
                shared = {"algo": algo, "feature_set": feature_set,
                          "feature": feature, "rank": rank}
                importance["builtin"].append({**shared, "importance": 1 / (rank + 1)})
                importance["permutation"].append(
                    # Feature cuối mang giá trị ÂM: xáo nó làm model tốt lên.
                    {**shared, "importance": 0.05 / rank - 0.01, "std": 0.002})
                importance["shap"].append({**shared, "importance": 0.4 / rank})

    evaluation_dir = tmp_path / EVAL_SUBDIR
    pd.DataFrame(curves).to_csv(evaluation_dir / "curves.csv", index=False)
    pd.DataFrame(metrics).to_csv(evaluation_dir / "metrics.csv", index=False)
    pd.DataFrame(sweep).to_csv(evaluation_dir / "threshold_sweep.csv", index=False)

    compare_dir = tmp_path / COMPARE_SUBDIR
    pd.DataFrame(comparison).to_csv(compare_dir / "comparison.csv", index=False)
    pd.DataFrame(intervals).to_csv(
        compare_dir / "confidence_interval.csv", index=False)

    for name, rows in importance.items():
        pd.DataFrame(rows).to_csv(
            tmp_path / IMPORTANCE_SUBDIR / f"{name}.csv", index=False)

    selection_dir = tmp_path / SELECTION_SUBDIR
    pd.DataFrame(
        [[33_712, 8_691], [1_975, 1_749]],
        index=pd.Index(["0 · trả nợ bình thường", "1 · khó khăn trả nợ"],
                       name="thật"),
        columns=["0 · trả nợ bình thường", "1 · khó khăn trả nợ"],
    ).to_csv(selection_dir / "test_confusion.csv")
    (selection_dir / "decision.json").write_text(
        json.dumps({"selected_model": "ml02_xgboost_reduced",
                    "threshold": {"value": 0.1259}}, ensure_ascii=False),
        encoding="utf-8")
    return tmp_path


# ------------------------------------------------------------- sinh hình

def test_every_plot_lands_in_its_own_task_directory(runs):
    """Sáu hình, mỗi hình nằm cùng thư mục với CSV đã sinh ra nó."""
    produced = generate_ml02_plots(runs, top_n=5)
    assert set(produced) == set(EXPECTED_SUBDIR)
    for name, path in produced.items():
        assert path.exists(), name
        assert path.parent == runs / EXPECTED_SUBDIR[name], name
        assert path.suffix == ".png", name
        assert path.stat().st_size > 5_000, f"{name} trông như hình rỗng"


@pytest.mark.parametrize("plot", [
    plot_precision_recall_curve, plot_roc_curve, plot_threshold_analysis,
    plot_model_comparison, plot_feature_importance, plot_confusion_matrix_test])
def test_each_plot_can_write_to_an_explicit_path(runs, plot, tmp_path):
    out = tmp_path / "custom.png"
    assert plot(runs, out=out) == out
    assert out.exists()


# ---------------------------------------- đọc kết quả, KHÔNG tính lại

@pytest.mark.parametrize("plot,needed,subdir", [
    (plot_precision_recall_curve, "curves.csv", EVAL_SUBDIR),
    (plot_roc_curve, "curves.csv", EVAL_SUBDIR),
    (plot_threshold_analysis, "threshold_sweep.csv", EVAL_SUBDIR),
    (plot_model_comparison, "comparison.csv", COMPARE_SUBDIR),
    (plot_confusion_matrix_test, "test_confusion.csv", SELECTION_SUBDIR),
])
def test_missing_input_fails_loudly(runs, plot, needed, subdir):
    """Thiếu kết quả thì phải báo lỗi, tuyệt đối không tự chấm lại model.

    Hình rỗng còn tệ hơn lỗi: nó vẫn vào được báo cáo.
    """
    (runs / subdir / needed).unlink()
    with pytest.raises(FileNotFoundError, match=needed):
        plot(runs)


def test_plots_do_not_touch_the_saved_results(runs):
    """Vẽ hình là chỉ ĐỌC — không sửa file kết quả nào."""
    before = {path.name: path.read_bytes() for path in runs.rglob("*.csv")}
    generate_ml02_plots(runs, top_n=5)
    after = {path.name: path.read_bytes() for path in runs.rglob("*.csv")}
    assert before == after


# ------------------------------------------------ chịu được task còn thiếu

def test_threshold_plot_works_before_task_14_has_run(runs):
    """Task 11 vẽ được khi chưa có `decision.json`.

    Đường dọc "ngưỡng đã chốt" là trang trí. Bắt task 11 phải có kết quả task
    14 mới vẽ được là dựng ngược thứ tự phụ thuộc của F04 — và trong lần chạy
    pipeline đầu tiên thì file đó chưa tồn tại.
    """
    (runs / SELECTION_SUBDIR / "decision.json").unlink()
    out = plot_threshold_analysis(runs)
    assert out.exists() and out.stat().st_size > 5_000


def test_importance_plot_works_with_builtin_only(runs):
    """`importance_ml02.py --builtin-only` không ghi permutation/shap.

    Khi ấy hình phải rút còn một panel chứ không dựng hai panel rỗng.
    """
    (runs / IMPORTANCE_SUBDIR / "permutation.csv").unlink()
    (runs / IMPORTANCE_SUBDIR / "shap.csv").unlink()
    out = plot_feature_importance(runs, top_n=5)
    assert out.exists() and out.stat().st_size > 5_000


# ------------------------------------------------------------ bảng màu

def test_categorical_palette_matches_the_validated_set():
    """Bốn slot đã qua `validate_palette.js --pairs all` trên nền `#fcfcfb`.

    Slot 4 KHÔNG phải yellow của bảng gốc: yellow cạnh orange rớt sàn thị lực
    thường (ΔE 13,7 < 15). Violet là ứng viên duy nhất qua được cả sáu check.
    """
    assert SERIES_ML02 == ("#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7")


def test_colour_follows_the_algorithm_not_the_ranking():
    """Màu gắn chết vào thuật toán.

    Nếu màu sinh theo thứ tự xuất hiện trong bảng đang vẽ thì một lần train
    đổi ngôi là cả tập báo cáo đổi màu, và hai bản báo cáo cạnh nhau không so
    được nữa.
    """
    assert ALGO_COLOR == dict(zip(ALGOS, SERIES_ML02))
    assert len(set(ALGO_COLOR.values())) == len(ALGO_COLOR), "hai model trùng màu"
