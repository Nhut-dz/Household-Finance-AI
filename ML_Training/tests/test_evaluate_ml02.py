"""Test tầng đánh giá của ML02 (task 11).

Hai bất biến quan trọng nhất được canh ở đây, và cả hai đều là ràng buộc về
PHẠM VI chứ không phải về phép tính:

    · Task 11 KHÔNG xếp hạng và KHÔNG chọn model — đó là task 12 và 14. Trộn
      hai việc thì phần đánh giá bị rút gọn thành "cái nào PR-AUC cao nhất",
      mà đó chính là chỗ bỏ sót học thuộc, hiệu chuẩn lệch, hay recall cao đổi
      bằng precision thấp.
    · Quét ngưỡng là NGUYÊN LIỆU, không phải quyết định. Hàm quét không được
      trả về ngưỡng nào cả.
"""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from hfml.ml.ml02_credit_risk import evaluate as evaluate_module
from hfml.ml.ml02_credit_risk.evaluate import (
    ALERT_RATES,
    METRIC_ORDER,
    REPORTING_THRESHOLD,
    SWEEP_THRESHOLDS,
    ModelEvaluation,
    calibration_table,
    capture_table,
    confusion_long,
    curve_table,
    metrics_table,
    threshold_sweep,
)
from hfml.ml.evaluation.metrics import binary_confusion, binary_metrics


def make_evaluation(algo: str = "xgboost", feature_set: str = "full",
                    n: int = 5_000, strength: float = 2.0,
                    seed: int = 0) -> ModelEvaluation:
    """Một kết quả đánh giá giả, có tín hiệu THẬT giữa xác suất và nhãn.

    `strength` càng lớn thì model càng tách được hai lớp — dùng để dựng hai
    model mạnh yếu khác nhau mà không phải train gì.
    """
    rng = np.random.default_rng(seed)
    truth = rng.binomial(1, 0.08, size=n)
    proba = np.clip(
        rng.beta(2, 8, size=n) + truth * strength * 0.1, 0.001, 0.999)
    return ModelEvaluation(
        algo=algo, feature_set=feature_set,
        y_true=truth, y_proba=proba,
        metrics=binary_metrics(truth, proba, threshold=REPORTING_THRESHOLD),
        confusion=binary_confusion(truth, proba, threshold=REPORTING_THRESHOLD),
    )


@pytest.fixture
def evaluations() -> list[ModelEvaluation]:
    return [
        make_evaluation("decision_tree", "reduced", strength=1.0, seed=1),
        make_evaluation("decision_tree", "full", strength=1.5, seed=2),
        make_evaluation("xgboost", "reduced", strength=2.0, seed=3),
        make_evaluation("xgboost", "full", strength=3.0, seed=4),
    ]


# ------------------------------------------- task 11 không xếp hạng, không chọn
def test_evaluation_module_never_ranks_or_selects():
    """Xếp hạng là task 12, chọn model là task 14.

    Có một hàm `best_model()` ở đây thì task 12–14 dễ gọi nó và bỏ qua phần
    đánh giá độc lập, mà đó mới là chỗ ràng buộc "không dùng test để chọn".
    """
    names = {n.lower() for n in dir(evaluate_module) if not n.startswith("_")}

    assert not any("best" in n or "select" in n or "rank" in n or "winner" in n
                   for n in names), names


def test_metrics_table_keeps_the_input_order(evaluations):
    """KHÔNG sắp xếp theo chỉ số — sắp xếp là hành vi của bước so sánh.

    Bảng đánh giá tự sắp theo PR-AUC thì nó ngầm biến thành bảng xếp hạng, và
    người đọc sẽ dừng ở dòng đầu.
    """
    table = metrics_table(evaluations)

    assert list(table["algo"]) == [e.algo for e in evaluations]
    assert list(table["feature_set"]) == [e.feature_set for e in evaluations]
    assert not table["pr_auc"].is_monotonic_decreasing or len(table) < 3


def test_threshold_sweep_returns_material_not_a_decision(evaluations):
    """Quét ngưỡng cho thấy chỉ số biến thiên thế nào, không chọn ngưỡng.

    Chọn ngưỡng phải làm SAU khi hiệu chuẩn (task 14) — chọn trên xác suất
    chưa hiệu chuẩn thì con số ngưỡng không mang ý nghĩa xác suất nào.
    """
    sweep = threshold_sweep(evaluations)

    assert set(sweep["threshold"]) == set(SWEEP_THRESHOLDS)
    # Mỗi model xuất hiện ở MỌI ngưỡng — không có ngưỡng nào được ưu tiên.
    for algo, group in sweep.groupby("algo"):
        assert len(group) == len(SWEEP_THRESHOLDS) * \
            len({e.feature_set for e in evaluations if e.algo == algo})

    # Trả về BẢNG, không phải một con số ngưỡng.
    assert isinstance(sweep, pd.DataFrame)


def test_sweep_thresholds_are_dense_where_the_real_one_will_be():
    """Tỉ lệ nền 8,07% nên ngưỡng thật sẽ nằm ở vùng THẤP.

    Quét thưa ở đó thì bảng không dùng được cho task 14.
    """
    thap = [t for t in SWEEP_THRESHOLDS if t <= 0.5]
    cao = [t for t in SWEEP_THRESHOLDS if t > 0.5]

    assert len(thap) > len(cao)


# ------------------------------------------------------ bảng chỉ số chính
def test_accuracy_is_last_in_the_metric_order():
    """accuracy đứng CUỐI vì nó không được cầm lái (§7.3).

    Thứ tự cột là thứ tự người ta đọc. Để accuracy trước PR-AUC là mời người
    đọc kết luận bằng con số sai.
    """
    assert METRIC_ORDER[0] == "pr_auc"
    assert METRIC_ORDER[-1] == "accuracy"


def test_metrics_table_reports_every_required_metric(evaluations):
    """Năm chỉ số bắt buộc phải có mặt: ROC-AUC, PR-AUC, F1, recall lớp 1."""
    table = metrics_table(evaluations)

    assert {"pr_auc", "roc_auc", "f1_positive",
            "recall_positive", "precision_positive"} <= set(table.columns)
    assert len(table) == len(evaluations)


def test_confusion_names_the_four_cells(evaluations):
    """Bốn ô phải có TÊN, không phải chỉ số hàng/cột.

    `false_negative` đọc ra ngay là số ca vỡ nợ bị bỏ lọt — thứ đắt nhất trong
    bài toán này. Một ma trận 2×2 không tên thì phải nhớ quy ước hàng/cột mới
    đọc được, và nhớ nhầm là đảo hai loại lỗi cho nhau.
    """
    table = confusion_long(evaluations)

    assert {"true_negative", "false_positive",
            "false_negative", "true_positive"} <= set(table.columns)
    for _, row in table.iterrows():
        total = (row["true_negative"] + row["false_positive"]
                 + row["false_negative"] + row["true_positive"])
        assert total == 5_000


# ---------------------------------------------------------- hiệu chuẩn
def test_calibration_table_measures_the_gap_direction(evaluations):
    """`gap` = model nói − thực tế. Dấu của nó có nghĩa, không được lấy trị tuyệt đối.

    Dương = model NÓI QUÁ, và với class_weight/scale_pos_weight thì nói quá là
    điều phải xảy ra. Lấy |gap| thì mất đúng thông tin đó.
    """
    table = calibration_table(evaluations)

    assert {"mean_predicted", "observed_rate", "gap"} <= set(table.columns)
    reconstructed = table["mean_predicted"] - table["observed_rate"]
    np.testing.assert_allclose(table["gap"], reconstructed, atol=1e-12)


def test_calibration_detects_a_systematically_overconfident_model():
    """Model nói quá phải cho gap dương — phép đo phải bắt được, không chỉ chạy."""
    rng = np.random.default_rng(0)
    truth = rng.binomial(1, 0.08, size=10_000)
    # Xác suất bị đẩy lên gấp ~4 lần tỉ lệ thật.
    proba = np.clip(0.08 * 4 + rng.normal(0, 0.05, size=10_000), 0.01, 0.99)

    evaluation = ModelEvaluation(
        algo="a", feature_set="full", y_true=truth, y_proba=proba,
        metrics=binary_metrics(truth, proba),
        confusion=binary_confusion(truth, proba))

    assert calibration_table([evaluation])["gap"].mean() > 0.15


# -------------------------------------------------- theo ngân sách rà soát
def test_capture_rate_grows_with_the_review_budget(evaluations):
    """Soi nhiều hơn thì bắt được nhiều hơn — nếu không, phép sắp xếp sai."""
    table = capture_table(evaluations)

    for (algo, feature_set), group in table.groupby(["algo", "feature_set"]):
        rates = group.sort_values("alert_rate")["capture_rate"].to_numpy()
        assert (np.diff(rates) >= -1e-9).all(), (algo, feature_set)


def test_capture_lift_is_above_one_for_a_model_with_signal(evaluations):
    """Model có tín hiệu phải bắt được nhiều hơn soi ngẫu nhiên.

    lift = tỉ lệ bắt được ÷ tỉ lệ soi. Bằng 1,0 nghĩa là sắp xếp theo rủi ro
    chẳng hơn gì sắp xếp ngẫu nhiên.
    """
    manh = [e for e in evaluations if e.algo == "xgboost" and e.feature_set == "full"]
    table = capture_table(manh)

    assert (table["lift"] > 1.0).all()


def test_capture_uses_every_configured_alert_rate(evaluations):
    table = capture_table(evaluations)

    assert set(table["alert_rate"]) == set(ALERT_RATES)


def test_capture_ranks_by_risk_descending():
    """Phải soi hồ sơ RỦI RO CAO trước. Sắp xếp ngược là đo ngược hoàn toàn."""
    truth = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    proba = np.array([0.1, 0.1, 0.2, 0.2, 0.9, 0.9, 0.9, 0.8, 0.8, 0.8])
    evaluation = ModelEvaluation(
        algo="a", feature_set="full", y_true=truth, y_proba=proba,
        metrics=binary_metrics(truth, proba),
        confusion=binary_confusion(truth, proba))

    table = capture_table([evaluation], alert_rates=(0.5,))

    # Soi 5/10 hồ sơ rủi ro cao nhất → bắt được 5/6 ca dương.
    assert table.iloc[0]["n_caught"] == 5
    assert table.iloc[0]["precision_at_k"] == pytest.approx(1.0)


# ------------------------------------------------------------ đường cong
def test_curve_table_holds_both_pr_and_roc(evaluations):
    table = curve_table(evaluations)

    assert set(table["curve"]) == {"pr", "roc"}
    for (algo, feature_set, curve), group in table.groupby(
            ["algo", "feature_set", "curve"]):
        assert len(group) > 10, (algo, feature_set, curve)


def test_curve_table_is_thinned_to_stay_readable(evaluations):
    """46.127 hồ sơ cho hàng chục nghìn điểm — ghi hết ra CSV là vô ích."""
    table = curve_table(evaluations)

    for _, group in table.groupby(["algo", "feature_set", "curve"]):
        assert len(group) <= evaluate_module.CURVE_POINTS


def test_roc_curve_starts_and_ends_at_the_corners(evaluations):
    table = curve_table(evaluations)
    roc = table[(table["curve"] == "roc")
                & (table["algo"] == "xgboost")
                & (table["feature_set"] == "full")]

    assert roc["x"].min() == pytest.approx(0.0)
    assert roc["x"].max() == pytest.approx(1.0)


# ----------------------------------------------------------------- ghi file
def test_written_metadata_declares_the_scope_limits(tmp_path, monkeypatch, evaluations):
    """Metadata phải nói rõ: chưa xếp hạng, chưa chạm test."""
    import json

    from hfml.config import CONFIG
    from hfml.ml.ml02_credit_risk.evaluate import write_evaluation

    monkeypatch.setattr(CONFIG.paths, "runs", tmp_path)
    written = write_evaluation(evaluations)
    metadata = json.loads(written["metadata"].read_text(encoding="utf-8"))

    assert metadata["evaluated_on"] == "validation"
    assert metadata["test_set_touched"] is False
    assert metadata["ranking_done_here"] is False
    assert metadata["selection_metric"] == "pr_auc"


def test_all_six_tables_are_written(tmp_path, monkeypatch, evaluations):
    from hfml.config import CONFIG
    from hfml.ml.ml02_credit_risk.evaluate import write_evaluation

    monkeypatch.setattr(CONFIG.paths, "runs", tmp_path)
    written = write_evaluation(evaluations)

    assert {"metrics", "confusion", "calibration", "capture_by_alert_rate",
            "threshold_sweep", "curves"} <= set(written)
    assert all(path.exists() for path in written.values())
