"""Test xử lý mất cân bằng lớp của ML02 (`hfml.ml.ml02_credit_risk.imbalance`).

Hai bất biến quan trọng nhất được canh ở đây:

    · Bốn thuật toán phải nhận CÙNG một tỉ số phạt, dù dùng hai cơ chế khác
      tên (`class_weight` và `scale_pos_weight`). Lệch nhau thì bảng so sánh
      task 12 không còn công bằng, mà điều đó không lộ ra ở đâu cả.
    · Không bước nào được sinh thêm hay bỏ bớt dòng. Lấy mẫu lại là cửa rò rỉ
      và là thứ phá hiệu chuẩn xác suất, mà hiệu chuẩn là yêu cầu bắt buộc
      của ML02.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.utils.class_weight import compute_class_weight

from hfml.ml.ml02_credit_risk.imbalance import (
    ALGORITHMS,
    BALANCED,
    IMBALANCE_MECHANISM,
    REJECTED_STRATEGIES,
    estimator_params,
    imbalance_params,
    measure_imbalance,
    rejected_table,
    scale_pos_weight_from,
    strategy_table,
)


def labels(n: int = 10_000, positive_rate: float = 0.08) -> pd.Series:
    """Nhãn nhị phân với tỉ lệ dương định trước, không ngẫu nhiên."""
    n_positive = int(n * positive_rate)
    return pd.Series([1] * n_positive + [0] * (n - n_positive))


# ------------------------------------------------------------------- đo lường
def test_measures_the_known_imbalance():
    report = measure_imbalance(labels(10_000, 0.08))

    assert report.n_rows == 10_000
    assert report.n_positive == 800
    assert report.n_negative == 9_200
    assert report.positive_rate == pytest.approx(0.08)
    assert report.scale_pos_weight == pytest.approx(9_200 / 800)


def test_majority_accuracy_shows_why_accuracy_is_useless():
    """Mốc này là lý do chọn model bằng PR-AUC.

    Một model không học gì đạt 92% accuracy. Không có con số đó trong báo cáo
    thì "accuracy 92%" nghe như một kết quả tốt.
    """
    report = measure_imbalance(labels(10_000, 0.08))

    assert report.majority_class_accuracy == pytest.approx(0.92)


def test_single_class_labels_raise_instead_of_returning_infinity():
    """Nhãn một lớp → chia cho 0. Trả `inf` là để một tập hỏng chảy tiếp."""
    with pytest.raises(ValueError, match="một lớp"):
        measure_imbalance(pd.Series([0] * 100))

    with pytest.raises(ValueError, match="một lớp"):
        measure_imbalance(pd.Series([1] * 100))


def test_empty_labels_raise():
    with pytest.raises(ValueError, match="rỗng"):
        measure_imbalance(pd.Series([], dtype=int))


# ------------------------------------------- hai cơ chế, cùng một tỉ số phạt
def test_balanced_class_weight_equals_scale_pos_weight():
    """Điểm mấu chốt của cả task: hai cơ chế khác tên cho CÙNG mức phạt.

    Không có đẳng thức này thì XGBoost và ba thuật toán cây của sklearn đang
    học trên hai mức phạt khác nhau, và bảng so sánh ở task 12 so nhầm — mà
    không có gì trong bảng để lộ ra điều đó.
    """
    y = labels(10_000, 0.08)

    weights = compute_class_weight(BALANCED, classes=np.array([0, 1]), y=y)
    ty_so = weights[1] / weights[0]

    assert ty_so == pytest.approx(scale_pos_weight_from(y), rel=1e-9)


def test_every_algorithm_gets_a_mechanism_and_the_same_ratio():
    y = labels(10_000, 0.08)
    table = strategy_table(y)

    assert set(table["algorithm"]) == set(ALGORITHMS)
    assert table["effective_ratio"].nunique() == 1
    assert all(IMBALANCE_MECHANISM[a] for a in ALGORITHMS)


def test_three_sklearn_trees_use_balanced_class_weight():
    y = labels(1_000, 0.10)

    for algo in ("decision_tree", "random_forest"):
        assert imbalance_params(algo, y) == {"class_weight": BALANCED}


def test_xgboost_gets_a_numeric_scale_pos_weight():
    y = labels(10_000, 0.08)

    params = imbalance_params("xgboost", y)

    assert set(params) == {"scale_pos_weight"}
    assert params["scale_pos_weight"] == pytest.approx(9_200 / 800)


def test_bagging_puts_the_weight_on_its_child_estimator():
    """`BaggingClassifier` KHÔNG có tham số `class_weight`.

    Truyền vào sẽ `TypeError`; tệ hơn là nếu bị nuốt trong `**kwargs` thì model
    train mất cân bằng trong khi bảng cấu hình vẫn ghi là đã cân bằng. Trọng số
    thuộc về cây con.
    """
    y = labels(1_000, 0.10)

    assert imbalance_params("bagging", y) == {}
    assert estimator_params("bagging") == {"class_weight": BALANCED}


def test_only_bagging_has_child_estimator_params():
    for algo in ALGORITHMS:
        if algo != "bagging":
            assert estimator_params(algo) == {}


def test_bagging_child_weight_matches_the_others():
    """Cây con của Bagging phải nhận đúng mức phạt như Decision Tree đơn lẻ.

    Nếu không thì Bagging đang so với ba thuật toán kia trên một sân khác.
    """
    y = labels(10_000, 0.08)

    assert (estimator_params("bagging")["class_weight"]
            == imbalance_params("decision_tree", y)["class_weight"])


def test_unknown_algorithm_is_rejected():
    """Sai tên thuật toán phải báo lỗi, không trả về dict rỗng âm thầm.

    Dict rỗng nghĩa là "không có tham số cân bằng" — trùng đúng với trường hợp
    hợp lệ của `bagging`, nên nuốt lỗi ở đây sẽ cho ra một model train mất cân
    bằng mà không ai biết.
    """
    with pytest.raises(ValueError, match="F04"):
        imbalance_params("catboost", labels(100, 0.1))


# --------------------------------------------------- không lấy mẫu lại
def test_no_strategy_changes_the_number_of_rows():
    """Toàn bộ task này chỉ trả về THAM SỐ, không đụng vào dữ liệu.

    Không có hàm nào nhận `X` và trả về `X` khác kích thước — đó chính là điều
    làm cho phương án này không thể tạo rò rỉ kiểu SMOTE-trước-khi-chia-tập.
    """
    y = labels(1_000, 0.08)

    for algo in ALGORITHMS:
        params = imbalance_params(algo, y)
        assert isinstance(params, dict)
        assert "sampling_strategy" not in params
        assert "sampler" not in params


def test_resampling_alternatives_are_documented_with_reasons():
    """Phải trả lời được "sao không dùng SMOTE?" bằng lý do cụ thể."""
    table = rejected_table()
    names = set(table["strategy"])

    assert "SMOTE" in names
    assert {"Random oversampling", "Random undersampling"} <= names
    assert all(len(reason) > 40 for reason in table["reason"])
    assert len(REJECTED_STRATEGIES) >= 4


# ------------------------------------------------------------ chống rò rỉ
def test_weight_computed_on_train_differs_from_whole_dataset():
    """Trọng số suy từ tỉ lệ dương, mà đó là một THỐNG KÊ của dữ liệu.

    Test này khẳng định `scale_pos_weight_from()` thật sự phụ thuộc tập được
    truyền vào. Nếu hai tập khác phân bố mà cho cùng một số thì hàm chẳng đọc
    dữ liệu, và ràng buộc "tính trên riêng train" chỉ là hình thức.
    """
    toan_bo = labels(10_000, 0.08)
    train_lech = labels(6_000, 0.05)

    assert scale_pos_weight_from(toan_bo) != scale_pos_weight_from(train_lech)


def test_scale_pos_weight_is_computed_not_hardcoded():
    """Đổi tỉ lệ dương thì con số phải đổi theo, không phải hằng số 11,39."""
    assert scale_pos_weight_from(labels(1_000, 0.50)) == pytest.approx(1.0)
    assert scale_pos_weight_from(labels(1_000, 0.20)) == pytest.approx(4.0)
