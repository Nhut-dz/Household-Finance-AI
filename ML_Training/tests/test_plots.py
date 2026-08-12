"""F03 — kiểm tra ba biểu đồ báo cáo ML01 (`evaluation/plots.py`).

Test hình vẽ không kiểm được "hình có đẹp không" — mắt người làm việc đó, và
bước đó nằm ngoài bộ test. Chỗ test đứng ra canh là những thứ hỏng âm thầm:

    - hàm vẽ tự ý TÍNH LẠI thay vì đọc kết quả đã lưu
    - thiếu file đầu vào mà vẫn cho ra một hình rỗng
    - ghi sai thư mục
    - bảng màu trôi khỏi bộ đã qua validator

Cái cuối quan trọng hơn vẻ ngoài: bảng màu đã được `validate_palette.js`
chứng nhận đạt cả tách CVD lẫn sàn thị lực thường. Ai đó đổi một mã hex mà
không chạy lại validator thì hình vẫn vẽ ra bình thường, chỉ có điều người
mù màu không đọc được nữa.
"""
from __future__ import annotations

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

from hfml.ml.evaluation.plots import (  # noqa: E402
    SEQUENTIAL_STEPS,
    SERIES,
    generate_ml01_plots,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_model_comparison,
)
from hfml.ml.ml01_recommendation.labeler import ORDERED_GROUPS  # noqa: E402

LABELS = [group.value for group in ORDERED_GROUPS]
ALGOS = ("decision_tree", "bagging", "random_forest", "xgboost")


@pytest.fixture
def runs(tmp_path):
    """Thư mục runs/ giả lập với đúng ba file mà các hàm vẽ đọc."""
    confusion = []
    for index, algo in enumerate(ALGOS):
        for row, label in enumerate(LABELS):
            counts = {other: 10 + index for other in LABELS}
            counts[label] = 100 + row          # đường chéo trội
            confusion.append({"thật": label, **counts, "algo": algo})
    pd.DataFrame(confusion).to_csv(tmp_path / "test_confusion.csv", index=False)

    pd.DataFrame([
        {"algo": algo,
         "cv_accuracy": 0.90 + i / 100, "test_accuracy": 0.89 + i / 100,
         "gap_accuracy": 0.01,
         "cv_macro_f1": 0.88 + i / 100, "test_macro_f1": 0.87 + i / 100,
         "gap_macro_f1": 0.01,
         "cv_balanced_accuracy": 0.87, "test_balanced_accuracy": 0.86,
         "gap_balanced_accuracy": 0.01,
         "cv_macro_f1_std": 0.005, "fit_seconds": 1.0}
        for i, algo in enumerate(ALGOS)
    ]).to_csv(tmp_path / "model_comparison.csv", index=False)

    importance = []
    for algo in ("decision_tree", "random_forest", "xgboost"):
        for rank, feature in enumerate(
                ["savings_amount", "monthly_debt_payment", "age",
                 "household_size", "has_debt"], start=1):
            importance.append({
                "algo": algo, "source": "refit", "rank": rank,
                "feature": feature, "importance": 1.0 / (rank + 1),
            })
    pd.DataFrame(importance).to_csv(tmp_path / "feature_importance.csv", index=False)
    return tmp_path


# ------------------------------------------------------------- sinh hình

def test_every_plot_lands_in_the_runs_directory(runs):
    produced = generate_ml01_plots(runs, top_n=5)
    assert set(produced) == {"confusion_matrix", "model_comparison",
                             "feature_importance"}
    for name, path in produced.items():
        assert path.exists(), name
        assert path.parent == runs, name
        assert path.suffix == ".png", name
        assert path.stat().st_size > 5_000, f"{name} trông như hình rỗng"


@pytest.mark.parametrize("plot", [
    plot_confusion_matrix, plot_model_comparison, plot_feature_importance])
def test_each_plot_can_write_to_an_explicit_path(runs, plot, tmp_path):
    out = tmp_path / "custom.png"
    assert plot(runs, out=out) == out
    assert out.exists()


# ---------------------------------------- đọc kết quả, KHÔNG tính lại

@pytest.mark.parametrize("plot,needed", [
    (plot_confusion_matrix, "test_confusion.csv"),
    (plot_model_comparison, "model_comparison.csv"),
    (plot_feature_importance, "feature_importance.csv"),
])
def test_missing_input_fails_loudly(runs, plot, needed):
    """Thiếu kết quả thì phải báo lỗi, tuyệt đối không tự train lại.

    Hình rỗng còn tệ hơn lỗi: nó vẫn vào được báo cáo.
    """
    (runs / needed).unlink()
    with pytest.raises(FileNotFoundError, match=needed):
        plot(runs)


def test_plots_do_not_touch_the_saved_results(runs):
    """Vẽ hình là chỉ ĐỌC — không sửa file kết quả nào."""
    before = {path.name: path.read_bytes()
              for path in runs.glob("*.csv")}
    generate_ml01_plots(runs, top_n=5)
    after = {path.name: path.read_bytes() for path in runs.glob("*.csv")}
    assert before == after


# ------------------------------------------------------------ bảng màu

def test_categorical_palette_matches_the_validated_set():
    """Ba slot đã qua `validate_palette.js --pairs all` trên nền `#fcfcfb`.

    Đổi hex mà không chạy lại validator thì hình vẫn vẽ ra, chỉ có điều
    người mù màu hết phân biệt được các series.
    """
    assert SERIES == ("#2a78d6", "#eb6834", "#1baf7a")


def test_sequential_ramp_is_single_hue_and_monotone():
    """Thang tuần tự phải MỘT màu, sáng → tối.

    Rainbow cho một đại lượng liên tục làm người đọc thấy ranh giới ở chỗ
    không hề có ranh giới.
    """
    def luminance(hex_colour: str) -> float:
        r, g, b = (int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5))
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    steps = [luminance(step) for step in SEQUENTIAL_STEPS]
    assert steps == sorted(steps, reverse=True), "thang không đơn điệu sáng→tối"

    # Một màu: kênh blue trội ở mọi bước.
    for step in SEQUENTIAL_STEPS:
        r, g, b = (int(step[i:i + 2], 16) for i in (1, 3, 5))
        assert b > r and b > g, f"{step} không thuộc dải blue"


def test_confusion_plot_accepts_a_subset_of_models(runs):
    """Chọn ít model hơn vẫn phải ra hình — dùng khi báo cáo chỉ cần model cuối."""
    out = plot_confusion_matrix(runs, algos=("xgboost",))
    assert out.exists() and out.stat().st_size > 5_000
