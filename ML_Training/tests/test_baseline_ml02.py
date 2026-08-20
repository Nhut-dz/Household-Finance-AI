"""Test baseline và bộ chỉ số nhị phân của ML02 (task 6).

Bài test quan trọng nhất file này là `test_pr_auc_floor_is_the_base_rate_not_half`.
Nó canh một sự thật mà nếu hiểu sai thì **cả bảng kết quả bị đọc ngược**: sàn
của PR-AUC là tỉ lệ dương (0,0807), không phải 0,5. Một model đạt PR-AUC 0,20
là gấp 2,5 lần ngẫu nhiên, chứ không phải "kém hơn ngẫu nhiên".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hfml.ml.evaluation.metrics import binary_confusion, binary_metrics
from hfml.ml.ml02_credit_risk.baseline import (
    BASELINE_NAME,
    MAJORITY_NAME,
    baseline_of,
    evaluate_baselines,
    expected_random_pr_auc,
    fit_baseline,
    metrics_frame,
)


def labels(n: int = 20_000, positive_rate: float = 0.08) -> pd.Series:
    n_positive = int(n * positive_rate)
    values = np.array([1] * n_positive + [0] * (n - n_positive))
    np.random.default_rng(0).shuffle(values)
    return pd.Series(values)


# ------------------------------------------------- sàn của PR-AUC
def test_pr_auc_floor_is_the_base_rate_not_half():
    """Sàn PR-AUC = tỉ lệ dương. ROC-AUC mới có sàn 0,5.

    Đây là chỗ đọc nhầm nhiều nhất ở bài toán mất cân bằng. Đoán bừa cho
    PR-AUC ≈ 0,08 chứ không phải 0,5, nên thiếu hàng baseline trong bảng thì
    người đọc dễ kết luận ngược hoàn toàn.
    """
    y = labels(50_000, 0.08)
    rng = np.random.default_rng(1)
    ngau_nhien = rng.uniform(size=len(y))       # xác suất hoàn toàn vô nghĩa

    metrics = binary_metrics(y, ngau_nhien)

    assert metrics["pr_auc"] == pytest.approx(0.08, abs=0.01)
    assert metrics["roc_auc"] == pytest.approx(0.50, abs=0.01)
    assert metrics["pr_auc_lift"] == pytest.approx(1.0, abs=0.15)


def test_pr_auc_lift_says_how_many_times_better_than_random():
    """`pr_auc_lift` là tỉ số với sàn, để khỏi ai phải nhẩm."""
    y = labels(20_000, 0.10)
    # Xác suất trùng khít nhãn → PR-AUC = 1,0, lift = 1/0,10 = 10.
    metrics = binary_metrics(y, y.astype(float))

    assert metrics["pr_auc"] == pytest.approx(1.0)
    assert metrics["pr_auc_lift"] == pytest.approx(10.0, rel=0.02)


def test_expected_random_pr_auc_equals_the_positive_rate():
    assert expected_random_pr_auc(labels(10_000, 0.08)) == pytest.approx(0.08)


# --------------------------------------------- chỉ số riêng lớp dương
def test_metrics_report_the_positive_class_not_a_macro_average():
    """Macro làm mờ mất việc model bỏ rơi hoàn toàn lớp dương.

    Model đoán toàn 0 có macro-recall ~0,5 — nghe không tệ. `recall_positive`
    của nó là 0,0, và đó mới là con số nói đúng chuyện gì đang xảy ra.
    """
    y = labels(10_000, 0.08)
    toan_bo_la_0 = np.zeros(len(y))

    metrics = binary_metrics(y, toan_bo_la_0)

    assert metrics["recall_positive"] == 0.0
    assert metrics["precision_positive"] == 0.0
    assert metrics["f1_positive"] == 0.0
    # …trong khi accuracy vẫn rất cao.
    assert metrics["accuracy"] == pytest.approx(0.92, abs=0.005)


def test_accuracy_is_reported_but_is_visibly_useless():
    """Accuracy có trong bảng nhưng không được cầm lái (§7.3)."""
    y = labels(10_000, 0.08)

    metrics = binary_metrics(y, np.zeros(len(y)))

    assert metrics["accuracy"] > 0.90
    assert metrics["pr_auc"] < 0.15, "PR-AUC mới phản ánh đúng model vô dụng"


def test_perfect_probabilities_give_perfect_metrics():
    y = labels(5_000, 0.10)

    metrics = binary_metrics(y, y.astype(float))

    assert metrics["pr_auc"] == pytest.approx(1.0)
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["recall_positive"] == pytest.approx(1.0)
    assert metrics["brier"] == pytest.approx(0.0)


def test_threshold_only_affects_label_based_metrics():
    """Đổi ngưỡng KHÔNG được làm đổi PR-AUC và ROC-AUC.

    Hai chỉ số đó xếp hạng theo xác suất nên không phụ thuộc ngưỡng. Nếu chúng
    đổi theo ngưỡng thì phép tính sai ở đâu đó, và chỉ số chọn model của cả
    ML02 mất căn cứ.
    """
    y = labels(10_000, 0.08)
    rng = np.random.default_rng(2)
    proba = np.clip(y * 0.4 + rng.uniform(size=len(y)) * 0.6, 0, 1)

    a = binary_metrics(y, proba, threshold=0.5)
    b = binary_metrics(y, proba, threshold=0.2)

    assert a["pr_auc"] == pytest.approx(b["pr_auc"])
    assert a["roc_auc"] == pytest.approx(b["roc_auc"])
    # Còn recall thì phải đổi — hạ ngưỡng bắt được nhiều ca dương hơn.
    assert b["recall_positive"] > a["recall_positive"]


def test_confusion_matrix_orients_truth_as_rows():
    y = pd.Series([0, 0, 1, 1])
    proba = np.array([0.1, 0.9, 0.2, 0.8])      # một FP, một FN

    matrix = binary_confusion(y, proba)

    assert matrix.index.name == "thật"
    assert matrix.columns.name == "dự đoán"
    assert matrix.to_numpy().sum() == 4
    assert matrix.iloc[0, 1] == 1               # thật 0, đoán 1
    assert matrix.iloc[1, 0] == 1               # thật 1, đoán 0


# ------------------------------------------------------------- baseline
def test_stratified_baseline_lands_on_the_theoretical_floor():
    """Baseline đo được phải khớp PR-AUC lý thuyết.

    Lệch nhiều nghĩa là baseline đang làm gì đó ngoài dự kiến, hoặc tập
    validation không còn giữ đúng tỉ lệ nền.
    """
    y_train, y_val = labels(30_000, 0.08), labels(10_000, 0.08)

    baseline = baseline_of(evaluate_baselines(y_train, y_val))

    assert baseline.pr_auc == pytest.approx(expected_random_pr_auc(y_val), abs=0.01)
    assert baseline.metrics["roc_auc"] == pytest.approx(0.5, abs=0.02)


def test_most_frequent_reference_shows_the_accuracy_trap():
    """Hàng tham chiếu phải cho accuracy cao NHƯNG recall lớp dương bằng 0."""
    y_train, y_val = labels(30_000, 0.08), labels(10_000, 0.08)

    results = evaluate_baselines(y_train, y_val)
    majority = next(r for r in results if r.name == MAJORITY_NAME)

    assert majority.metrics["accuracy"] > 0.90
    assert majority.metrics["recall_positive"] == 0.0


def test_official_baseline_is_the_stratified_one():
    """`most_frequent` là tham chiếu, KHÔNG phải mốc để so."""
    results = evaluate_baselines(labels(5_000), labels(2_000))

    assert baseline_of(results).name == BASELINE_NAME
    assert baseline_of(results).strategy == "stratified"


def test_baseline_never_reads_any_feature():
    """Baseline không đọc `X` — đó là điều làm nó thành mốc.

    Mọi thứ một model thật hơn được baseline đều là phần do feature đóng góp.
    Test truyền hai bộ `X` khác hẳn nhau và khẳng định kết quả không đổi.
    """
    y_train, y_val = labels(10_000, 0.08), labels(4_000, 0.08)

    _, a = fit_baseline(y_train, y_val, seed=42)
    _, b = fit_baseline(y_train, y_val, seed=42)

    np.testing.assert_array_equal(a, b)


def test_baseline_is_reproducible_with_the_same_seed():
    y_train, y_val = labels(10_000), labels(4_000)

    _, a = fit_baseline(y_train, y_val, seed=42)
    _, b = fit_baseline(y_train, y_val, seed=42)
    _, c = fit_baseline(y_train, y_val, seed=7)

    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_positive_column_is_found_by_class_label_not_position():
    """Cột lớp dương lấy theo `classes_`, không đoán vị trí.

    Đoán "cột thứ hai" đúng trong hầu hết trường hợp và sai im lặng khi thứ tự
    lớp khác đi — lúc đó mọi xác suất bị đảo và PR-AUC ra ~1 − giá trị thật.
    """
    y_train, y_val = labels(10_000, 0.30), labels(4_000, 0.30)

    _, proba = fit_baseline(y_train, y_val)

    # Xác suất trung bình phải xấp xỉ tỉ lệ dương, không phải tỉ lệ âm.
    assert proba.mean() == pytest.approx(0.30, abs=0.03)


def test_metrics_frame_has_one_row_per_baseline():
    results = evaluate_baselines(labels(5_000), labels(2_000))

    table = metrics_frame(results)

    assert len(table) == 2
    assert set(table["name"]) == {BASELINE_NAME, MAJORITY_NAME}
    assert {"pr_auc", "recall_positive", "accuracy"} <= set(table.columns)
    assert table["role"].str.len().gt(20).all()


def test_baseline_lookup_fails_loudly_when_absent():
    with pytest.raises(ValueError, match="baseline"):
        baseline_of([])
