"""Test bước chọn model cuối của ML02 (task 14).

Task 14 là task đầu tiên được chạm tập test, nên bất biến quan trọng nhất ở đây
là về **THỨ TỰ**: chọn xong mới được mở test. Đảo thứ tự thì con số test không
còn là ước lượng độc lập mà thành một chỉ số đã được tối ưu gián tiếp — dạng rò
rỉ không để lại dấu vết nào trong mã.

Ba nhóm được canh:

    · `decide()` không nhận tập test        → không có gì để lỡ tay
    · `evaluate_on_test()` đòi ngưỡng đã chốt → không chốt xong không mở được
    · ngưỡng KHÔNG được là 0,5              → tỉ lệ nền 8,07%
"""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

from hfml.ml.ml02_credit_risk import select as select_module
from hfml.ml.ml02_credit_risk.select import (
    CALIBRATION_METHOD,
    DEPLOY_FEATURE_SET,
    REFERENCE_FEATURE_SET,
    Decision,
    FinalReport,
    calibrate,
    calibration_gap,
    choose_threshold,
    decide,
    evaluate_on_test,
    threshold_at_alert_rate,
)


def comparison_frame() -> pd.DataFrame:
    """Bảng so sánh giả, đúng shape mà task 12 sinh ra."""
    return pd.DataFrame([
        {"algo": "xgboost", "feature_set": "reduced", "pr_auc": 0.1711, "rank": 1},
        {"algo": "bagging", "feature_set": "reduced", "pr_auc": 0.1608, "rank": 2},
        {"algo": "random_forest", "feature_set": "reduced", "pr_auc": 0.1505, "rank": 3},
        {"algo": "xgboost", "feature_set": "full", "pr_auc": 0.2533, "rank": 1},
        {"algo": "bagging", "feature_set": "full", "pr_auc": 0.2311, "rank": 2},
    ])


def pairwise_frame(distinguishable: bool = True) -> pd.DataFrame:
    return pd.DataFrame([{
        "feature_set": "reduced",
        "model_a": "ml02_xgboost_reduced",
        "model_b": "ml02_bagging_reduced",
        "diff": 0.0102, "ci_low": 0.0045, "ci_high": 0.0151,
        "win_rate": 1.0, "distinguishable": distinguishable,
    }])


# ------------------------------------------------------ thứ tự bắt buộc
def test_decide_never_receives_a_test_set():
    """Muốn nhìn test lúc chọn thì phải sửa chữ ký hàm — tức phải cố ý."""
    params = set(inspect.signature(decide).parameters)

    assert not any("test" in p for p in params), params


def test_evaluate_on_test_refuses_without_a_locked_threshold():
    """Chưa chốt ngưỡng mà đã mở test thì con số test tham gia vào việc chọn.

    Khi đó nó thôi là ước lượng độc lập, và không có gì trong bảng kết quả để
    lộ ra điều đó.
    """
    chua_chot = Decision(algo="xgboost", deploy_feature_set="reduced",
                         reference_feature_set="full")

    assert chua_chot.threshold is None
    with pytest.raises(ValueError, match="Chưa chốt ngưỡng"):
        evaluate_on_test(chua_chot, None, pd.DataFrame(), pd.Series(dtype=int))


def test_selection_module_does_not_export_anything():
    """Export là task 15. Có hàm export ở đây thì task 14 dễ làm luôn."""
    names = {n.lower() for n in dir(select_module) if not n.startswith("_")}

    assert not any("export" in n or "dump" in n for n in names), names


# ---------------------------------------------------------- chọn model
def test_deploy_set_is_the_reduced_one_not_the_full_one():
    """§7.2: bộ FULL có `EXT_SOURCE_1/2/3` mà form KHÔNG thu được.

    Chọn bộ full làm model triển khai là chọn một model không chạy được trong
    sản phẩm — mà bảng chỉ số của nó lại đẹp hơn, nên đó là cái bẫy thật.
    """
    assert DEPLOY_FEATURE_SET == "reduced"
    assert REFERENCE_FEATURE_SET == "full"


def test_decision_picks_the_leader_of_the_deploy_set():
    decision = decide(comparison_frame(), pairwise_frame())

    assert decision.algo == "xgboost"
    assert decision.deploy_feature_set == "reduced"
    assert decision.deploy_slug == "ml02_xgboost_reduced"
    assert decision.runner_up == "bagging"


def test_decision_records_why_not_just_what():
    """Lý do phải ghi ra — "PR-AUC cao nhất" một mình không phải một lập luận."""
    decision = decide(comparison_frame(), pairwise_frame())

    assert len(decision.reasons) >= 3
    assert any("PR-AUC" in r for r in decision.reasons)
    assert any("PHÂN BIỆT ĐƯỢC" in r for r in decision.reasons)
    assert any("EXT_SOURCE" in r for r in decision.reasons)


def test_decision_flags_an_indistinguishable_lead():
    """Dẫn đầu bằng khoảng nằm trong sai số KHÔNG phải lý do để chọn.

    Vẫn chọn, nhưng phải ghi cảnh báo — không được trình bày như một khoảng
    cách chắc chắn.
    """
    decision = decide(comparison_frame(), pairwise_frame(distinguishable=False))

    assert any("CHƯA phân biệt" in r for r in decision.reasons)


def test_decision_fails_loudly_without_the_deploy_set():
    chi_co_full = comparison_frame()
    chi_co_full = chi_co_full[chi_co_full["feature_set"] == "full"]

    with pytest.raises(ValueError, match="reduced"):
        decide(chi_co_full, pairwise_frame())


def test_empty_comparison_is_rejected():
    with pytest.raises(ValueError, match="task 12"):
        decide(pd.DataFrame(), pairwise_frame())


# ------------------------------------------------------------ hiệu chuẩn
def _fitted_model(n: int = 4_000):
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"x": rng.normal(size=n), "z": rng.normal(size=n)})
    y = pd.Series(rng.binomial(1, 1 / (1 + np.exp(-(X["x"] * 2 - 2.5)))))
    model = DecisionTreeClassifier(
        max_depth=4, class_weight="balanced", random_state=42).fit(X, y)
    return model, X, y


def test_calibration_uses_isotonic_not_the_sigmoid_default():
    """Sigmoid giả định lệch hiệu chuẩn có dạng hàm logistic.

    Lệch ở đây do TRỌNG SỐ LỚP gây ra nên không có lý do gì mang dạng đó.
    Isotonic chỉ giả định đơn điệu — đúng thứ ta biết chắc.
    """
    assert CALIBRATION_METHOD == "isotonic"


def test_calibration_removes_the_overconfidence(monkeypatch):
    """`class_weight='balanced'` làm model nói quá; hiệu chuẩn phải kéo về 0.

    Đây là phép kiểm quan trọng nhất của bước này: không kéo được gap về gần 0
    thì ngưỡng chọn sau đó vẫn không mang ý nghĩa xác suất.
    """
    model, X, y = _fitted_model()
    truth = np.asarray(y).astype(int)

    gap_truoc = calibration_gap(truth, model.predict_proba(X)[:, 1])
    calibrated = calibrate(model, X, y)
    gap_sau = calibration_gap(truth, calibrated.predict_proba(X)[:, 1])

    assert gap_truoc > 0.10, "model chưa hiệu chuẩn phải nói quá"
    assert abs(gap_sau) < abs(gap_truoc) / 3


def test_calibration_does_not_retrain_the_model():
    """`FrozenEstimator` giữ nguyên model đã train.

    Để `CalibratedClassifierCV` tự train lại thì model cuối khác model đã được
    so sánh ở task 12, và cả bảng so sánh mất hiệu lực.
    """
    model, X, y = _fitted_model()
    truoc = model.predict_proba(X)[:, 1].copy()

    calibrate(model, X, y)

    np.testing.assert_array_equal(model.predict_proba(X)[:, 1], truoc)


def test_calibration_gap_sign_means_overconfidence():
    """Dấu của gap có nghĩa — dương là nói quá. Lấy |gap| thì mất thông tin đó."""
    truth = np.array([0] * 90 + [1] * 10)
    noi_qua = np.array([0.5] * 100)
    noi_thieu = np.array([0.01] * 100)

    assert calibration_gap(truth, noi_qua, n_bins=2) > 0
    assert calibration_gap(truth, noi_thieu, n_bins=2) < 0


# ------------------------------------------------------------- ngưỡng
def test_threshold_is_far_below_the_naive_half():
    """Với tỉ lệ nền 8,07%, ngưỡng 0,5 xếp gần như mọi hồ sơ vào LOW_RISK."""
    rng = np.random.default_rng(1)
    truth = rng.binomial(1, 0.08, size=20_000)
    proba = np.clip(0.08 + truth * 0.15 + rng.normal(0, 0.05, 20_000), 0.001, 0.999)

    threshold, _ = choose_threshold(truth, proba)

    assert 0.0 < threshold < 0.5


def test_threshold_rule_admits_its_own_limitation():
    """F1 coi bỏ lọt một ca vỡ nợ và gắn nhãn sai một hồ sơ tốt là ĐẮT NHƯ NHAU.

    Tín dụng thật thì không. Quy tắc phải nói ra điều đó, nếu không con số
    ngưỡng sẽ được đọc như một tối ưu thật.
    """
    rng = np.random.default_rng(2)
    truth = rng.binomial(1, 0.08, size=5_000)
    proba = np.clip(0.08 + truth * 0.2 + rng.normal(0, 0.05, 5_000), 0.001, 0.999)

    _, rule = choose_threshold(truth, proba)

    assert "F1" in rule
    assert "chi phí" in rule.lower() or "model_card" in rule


def test_alert_rate_threshold_reviews_the_requested_share():
    """Ngưỡng theo ngân sách: rà soát k% thì đúng ~k% hồ sơ vượt ngưỡng."""
    proba = np.linspace(0, 1, 10_000)

    threshold = threshold_at_alert_rate(proba, rate=0.10)

    assert (proba >= threshold).mean() == pytest.approx(0.10, abs=0.01)


# ------------------------------------------------------- báo cáo cuối
def test_generalisation_gap_compares_validation_with_test():
    report = FinalReport(
        decision=Decision(algo="xgboost", deploy_feature_set="reduced",
                          reference_feature_set="full", threshold=0.13),
        validation_metrics={"pr_auc": 0.1696},
        test_metrics={"pr_auc": 0.1714})

    assert report.generalisation_gap == pytest.approx(-0.0018)


def test_written_record_states_the_order_of_operations(tmp_path, monkeypatch):
    """Bản ghi phải khẳng định test được mở SAU khi chốt, và chưa export."""
    import json

    from hfml.config import CONFIG
    from hfml.ml.ml02_credit_risk.select import write_selection

    monkeypatch.setattr(CONFIG.paths, "runs", tmp_path)
    report = FinalReport(
        decision=Decision(algo="xgboost", deploy_feature_set="reduced",
                          reference_feature_set="full", threshold=0.1303,
                          threshold_rule="F1 lớn nhất", reasons=["a", "b", "c"]),
        validation_metrics={"pr_auc": 0.1696},
        test_metrics={"pr_auc": 0.1714},
        validation_calibration_gap=-0.0004,
        test_calibration_gap=-0.0009)

    record = json.loads(
        write_selection(report)["decision"].read_text(encoding="utf-8"))

    assert record["selected_using"] == "validation only"
    assert record["test_opened_after_decision"] is True
    assert record["exported"] is False
    assert record["threshold"]["chosen_on"] == "validation"
    assert record["threshold"]["value"] == pytest.approx(0.1303)
    assert record["calibration"]["fitted_on"] == "validation"