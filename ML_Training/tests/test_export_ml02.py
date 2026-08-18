"""Test bước export của ML02 (task 15).

Bài test quan trọng nhất là `test_predict_uses_the_locked_threshold_not_half`.
Đo trên dữ liệu thật: sau hiệu chuẩn, xác suất cao nhất trong 2.000 hồ sơ test
chỉ **0,3478** — nên nếu `predict()` cắt ở 0,5 thì **KHÔNG hồ sơ nào** được gắn
`HIGH_RISK`. Model vẫn chạy, vẫn trả xác suất, chỉ là không phân loại gì cả.
Gói ngưỡng vào artifact là cách duy nhất để tầng gọi không thể quên nó.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.tree import DecisionTreeClassifier

from hfml.config import CONFIG
from hfml.ml.ml02_credit_risk.export import (
    HIGH_RISK,
    LOW_RISK,
    RISK_LABELS,
    RISK_LABELS_VI,
    VERSION,
    Ml02CreditRiskModel,
    build_metadata,
    export,
    verify_export,
)

FEATURES = ["dti", "credit_income_ratio", "age_years"]


def toy_model(threshold: float = 0.13) -> tuple[Ml02CreditRiskModel, pd.DataFrame]:
    """Model nhỏ đã hiệu chuẩn, đủ để kiểm hợp đồng của artifact."""
    rng = np.random.default_rng(0)
    n = 3_000
    X = pd.DataFrame({name: rng.normal(size=n) for name in FEATURES})
    y = pd.Series(rng.binomial(1, 1 / (1 + np.exp(-(X["dti"] * 2 - 2.5)))))

    inner = DecisionTreeClassifier(
        max_depth=4, class_weight="balanced", random_state=42).fit(X, y)
    calibrated = CalibratedClassifierCV(
        FrozenEstimator(inner), method="isotonic").fit(X, y)

    return Ml02CreditRiskModel(
        calibrated=calibrated, feature_names=FEATURES,
        threshold=threshold), X


def decision_record() -> dict:
    return {
        "selection_metric": "pr_auc",
        "reasons": ["PR-AUC cao nhất", "khoảng cách phân biệt được"],
        "threshold": {"value": 0.1303, "rule": "F1 lớn nhất",
                      "caveat": "F1 coi hai loại lỗi đắt như nhau"},
        "calibration": {"method": "isotonic", "fitted_on": "validation"},
        "metrics_validation": {"pr_auc": 0.1696},
        "metrics_test": {"pr_auc": 0.1714},
    }


# ------------------------------------------------- ngưỡng gói vào artifact
def test_predict_uses_the_locked_threshold_not_half():
    """`predict()` phải cắt ở ngưỡng đã chốt, KHÔNG phải 0,5 của sklearn.

    Đo trên dữ liệu thật: sau hiệu chuẩn, xác suất cao nhất trong 2.000 hồ sơ
    test chỉ 0,3478. Cắt ở 0,5 thì KHÔNG hồ sơ nào được gắn HIGH_RISK — model
    vẫn chạy, vẫn trả xác suất, chỉ là không phân loại gì cả.
    """
    model, X = toy_model(threshold=0.13)

    proba = model.risk_probability(X)
    labels = model.predict(X)

    mong_doi = np.where(proba >= 0.13, HIGH_RISK, LOW_RISK)
    np.testing.assert_array_equal(labels, mong_doi)
    # Khác hẳn kết quả nếu dùng 0,5.
    assert (labels == HIGH_RISK).sum() != (proba >= 0.5).sum()


def test_threshold_travels_with_the_artifact(tmp_path):
    """Nạp lại artifact thì ngưỡng phải còn nguyên.

    Ngưỡng nằm ở tài liệu mà không nằm trong file thì sớm muộn có nơi triển
    khai quên áp nó.
    """
    model, X = toy_model(threshold=0.1303)
    export(model, decision_record(), directory=tmp_path)

    from hfml.ml.registry import load_model

    nap_lai = load_model(model.slug, tmp_path)

    assert nap_lai.threshold == pytest.approx(0.1303)
    np.testing.assert_array_equal(nap_lai.predict(X), model.predict(X))


def test_explain_reports_the_threshold_alongside_the_label():
    """Nói "rủi ro cao" mà không cho biết ngưỡng là khẳng định không kiểm chứng được."""
    model, X = toy_model()

    record = model.explain(X.head(1))[0]

    assert set(record) == {"label", "label_vi", "probability",
                           "threshold", "model_version"}
    assert record["threshold"] == model.threshold
    assert record["label"] in RISK_LABELS


# ------------------------------------------------------ nhãn nghiệp vụ
def test_labels_are_business_strings_not_zero_and_one():
    """Tầng api và llm đọc nhãn này ra cho người dùng.

    Để 0/1 thì mỗi nơi tự đặt tên một kiểu, và sớm muộn có nơi đảo ngược ý nghĩa.
    """
    model, _ = toy_model()

    assert model.classes_ == list(RISK_LABELS)
    assert RISK_LABELS == (LOW_RISK, HIGH_RISK)
    assert set(RISK_LABELS_VI) == set(RISK_LABELS)


def test_probability_column_one_is_the_high_risk_class():
    """Cột 1 của `predict_proba` phải là HIGH_RISK, khớp `classes_`.

    Đảo hai cột là lỗi im lặng nặng nhất có thể có: mọi hồ sơ rủi ro cao thành
    rủi ro thấp và ngược lại, mà đầu ra vẫn là xác suất hợp lệ trong [0, 1].
    """
    model, X = toy_model()

    proba = model.predict_proba(X)
    risky = model.risk_probability(X)

    assert proba.shape[1] == 2
    np.testing.assert_array_equal(proba[:, 1], risky)
    # Hồ sơ có xác suất cột 1 lớn nhất phải được gắn HIGH_RISK.
    assert model.predict(X.iloc[[int(np.argmax(risky))]])[0] == HIGH_RISK


# ----------------------------------------------------- không train lại
def test_exported_artifact_refuses_to_be_refit():
    """Train lại lên bản export thì mọi con số trong metadata mô tả model khác."""
    model, X = toy_model()

    with pytest.raises(NotImplementedError, match="task 14"):
        model.fit(X, pd.Series(np.zeros(len(X), dtype=int)))


# --------------------------------------------------------------- metadata
def test_metadata_carries_the_inference_contract():
    """Feature list đúng thứ tự + label mapping + ngưỡng — thiếu một là không dùng được."""
    model, _ = toy_model()

    metadata = build_metadata(model, decision_record())

    assert metadata["feature_names"] == FEATURES
    assert metadata["n_features"] == len(FEATURES)
    assert metadata["label_mapping"]["1"] == HIGH_RISK
    assert metadata["label_mapping"]["positive_class"] == HIGH_RISK
    assert metadata["threshold"]["value"] == pytest.approx(model.threshold)
    assert metadata["artifact_kind"] == "final_export"


def test_metadata_records_the_limitations():
    """Giới hạn sử dụng phải đi cùng model, không nằm riêng ở một tài liệu khác."""
    model, _ = toy_model()

    limitations = build_metadata(model, decision_record())["limitations"]

    assert len(limitations) >= 4
    assert any("tham khảo" in item.lower() for item in limitations)
    assert any("F1" in item for item in limitations)
    assert any("Home Credit" in item for item in limitations)


def test_metadata_includes_the_dataset_fingerprint():
    """Không có SHA-256 thì ba tháng sau không ai chứng minh được train trên file nào."""
    model, _ = toy_model()
    manifest = {"files": {"application_train": {
        "file": "application_train.csv", "sha256": "a" * 64}}}

    metadata = build_metadata(model, decision_record(), manifest)

    assert metadata["data_version"]["application_train.csv"] == "a" * 16


def test_metadata_survives_json_round_trip():
    """Metadata phải JSON hoá được — siêu tham số model dễ chứa kiểu lạ."""
    model, _ = toy_model()

    text = json.dumps(build_metadata(model, decision_record()), ensure_ascii=False)

    assert json.loads(text)["slug"] == model.slug


# ------------------------------------------------------------- slug
def test_slug_follows_the_project_convention():
    model, _ = toy_model()

    assert model.slug == f"ml02_xgboost_reduced_v{VERSION}"
    assert model.slug == "ml02_xgboost_reduced_vfinal"


# ------------------------------------------------------- nạp lại và kiểm
def test_verification_checks_all_four_failure_modes(tmp_path):
    """Bốn thứ hỏng theo bốn kiểu riêng — kiểm thiếu một là bỏ sót một kiểu."""
    model, X = toy_model()
    export(model, decision_record(), directory=tmp_path)
    expected = model.risk_probability(X)

    result = verify_export(model.slug, X, expected, directory=tmp_path)

    assert result["loaded"] is True
    assert result["proba_matches"] is True
    assert result["feature_names_match"] is True
    assert result["threshold_applied"] is True
    assert result["threshold_is_not_half"] is True
    assert result["max_proba_diff"] == pytest.approx(0.0)


def test_verification_detects_a_probability_mismatch(tmp_path):
    """Phép kiểm phải bắt được sai lệch, không chỉ chạy cho có."""
    model, X = toy_model()
    export(model, decision_record(), directory=tmp_path)

    sai = model.risk_probability(X) + 0.1
    result = verify_export(model.slug, X, sai, directory=tmp_path)

    assert result["proba_matches"] is False
    assert result["max_proba_diff"] > 0.05


def test_export_writes_both_files(tmp_path):
    model, _ = toy_model()

    written = export(model, decision_record(), directory=tmp_path)

    assert written["model"].exists()
    assert written["metadata"].exists()
    assert written["model"].suffix == ".joblib"


def test_export_does_not_write_into_the_real_runs_dir(tmp_path):
    """Test không được đè artifact thật — `directory` phải có tác dụng."""
    model, _ = toy_model()

    written = export(model, decision_record(), directory=tmp_path)

    assert tmp_path in written["model"].parents
    assert CONFIG.paths.runs not in written["model"].parents
