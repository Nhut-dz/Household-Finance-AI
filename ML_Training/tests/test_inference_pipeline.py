"""Test Epic AI-01 — Inference Engine (`hfml.pipeline`).

Bốn nhóm bất biến, mỗi nhóm ứng với một cách hệ thống hỏng mà **không báo lỗi**:

    · tự suy diễn dữ liệu thiếu   → rule vẫn tính, model vẫn trả xác suất, và
                                    không có gì lộ ra rằng đầu vào là bịa
    · quy đổi kỳ sai              → `dti` bị chia 12, hồ sơ nào cũng thành
                                    "gánh nặng trả nợ rất nhẹ"
    · bureau bị ghi đè            → mục C người dùng vừa khai bị thay bằng 0
    · một phần hỏng kéo cả kết quả → mất phần rule vì model thiếu artifact
"""
from __future__ import annotations

import json
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from hfml.data.schema import HouseholdProfile, LoanApplication
from hfml.pipeline import confidence as conf
from hfml.pipeline.adapters import (
    MONTHS_PER_YEAR,
    ML02_APPLICATION_COLUMNS,
    ML02_BUREAU_COLUMNS,
    to_ml01_frame,
    to_ml02_frame,
)
from hfml.pipeline.normalizer import normalize_input
from hfml.pipeline.orchestrator import RULE_CODES, SCHEMA_VERSION, analyze
from hfml.pipeline.predictor import (
    MISSING_INPUT,
    MODEL_UNAVAILABLE,
    PREDICTION_ERROR,
    PredictionResult,
    predict_ml01,
    predict_ml02,
)
from hfml.ml.ml01_recommendation.labeler import RAW_FEATURES


def household(**overrides) -> dict:
    base = dict(
        representative_name="Nguyễn Văn A", birth_year=1991, residence="TP.HCM",
        household_size=4, children_count=2, has_dependents=False,
        average_monthly_income=35_000_000, average_monthly_expense=17_000_000,
        has_debt=True, total_current_debt=500_000_000,
        monthly_debt_payment=5_000_000,
        has_savings=True, savings_amount=150_000_000,
        assets=["cash", "real_estate"], financial_needs=["saving"])
    base.update(overrides)
    return base


def loan(**overrides) -> dict:
    base = dict(
        borrower_age=35, gender="male", marital_status="married",
        children_count=2, education_level="higher", occupation="office_staff",
        employment_years=8.5, loan_amount=1_400_000_000, loan_term_months=240,
        monthly_payment=12_000_000, asset_price=2_000_000_000,
        loan_purpose="buy_house", previous_loan_count=3, late_payment_count=1,
        has_overdue_loan=False)
    base.update(overrides)
    return base


# ============================================================ task 1
def test_missing_required_field_stops_before_any_computation():
    """Thiếu trường bắt buộc → dừng, không chạy rule hay ML.

    Chạy tiếp với một chỗ trống nghĩa là rule tính trên số bịa và model trả
    xác suất trên số bịa — cả hai đều trông bình thường.
    """
    payload = household()
    del payload["average_monthly_income"]

    result = normalize_input(payload)

    assert not result.is_valid
    assert result.profile is None, "hồ sơ hỏng không được lọt ra ngoài"
    assert any("average_monthly_income" in i.field for i in result.errors)


def test_nothing_is_imputed_for_missing_optional_money():
    """Ba trường tiền có điều kiện: trống + cờ False = 0 ĐÃ BIẾT CHẮC.

    Đây không phải suy diễn — cờ `has_savings`/`has_debt` đã mang thông tin
    có/không. Nhưng cờ True mà để trống thì KHÔNG được điền 0.
    """
    khong_no = normalize_input(household(
        has_debt=False, total_current_debt=None, monthly_debt_payment=None))

    assert khong_no.is_valid
    assert khong_no.profile.total_current_debt == Decimal(0)

    co_no = normalize_input(household(has_debt=True, total_current_debt=None))
    assert not co_no.is_valid, "có nợ mà không khai dư nợ phải là LỖI"


def test_missing_birth_year_is_never_filled_with_a_default():
    """Thiếu năm sinh → tuổi là None, và ML01 từ chối chạy.

    Điền tuổi mặc định thì model vẫn trả về một nhóm trông hợp lý và không ai
    biết nó dựa trên tuổi bịa (§6.1c).
    """
    result = normalize_input(household(birth_year=None))

    assert result.is_valid
    assert result.summary()["age"] is None

    prediction = predict_ml01(result.profile, None)
    assert not prediction.available
    assert prediction.reason_code == MISSING_INPUT


def test_suspicious_data_warns_but_does_not_block():
    """Chi > thu chính là nhóm EMERGENCY của ML01.

    Chặn nó là chặn đúng đối tượng cần tư vấn nhất (§4.2).
    """
    result = normalize_input(household(average_monthly_expense=50_000_000))

    assert result.is_valid
    assert "expense_exceeds_income" in result.quality_flags
    assert any(w.code == "expense_exceeds_income" for w in result.warnings)


def test_broken_loan_block_does_not_kill_the_whole_request():
    """Khoản vay hỏng → mất ML02, hồ sơ vẫn chạy rule + ML01."""
    result = normalize_input(
        {**household(), "loan_application": loan(borrower_age=22,
                                                 employment_years=30)})

    assert result.is_valid, "hồ sơ hộ gia đình vẫn phải dùng được"
    assert not result.has_loan_data
    assert result.warnings, "phải nói ra là khoản vay bị bỏ qua"


def test_loan_block_is_backfilled_from_the_loan_form():
    """Khối vay của `HouseholdProfile` (F01) và `LoanApplication` (15/08) trùng dữ liệu.

    Hồ sơ chọn `home_loan` nhưng để dữ liệu ở `loan_application` thì **không
    thiếu gì** — chép sang là dời chỗ, không phải suy diễn.
    """
    payload = {**household(financial_needs=["home_loan"]),
               "loan_application": loan()}

    result = normalize_input(payload)

    assert result.is_valid
    assert result.profile.loan_amount == Decimal(1_400_000_000)


def test_user_entered_values_win_over_backfill():
    """Ghi đè giá trị người dùng đã khai là âm thầm đổi dữ liệu của họ."""
    payload = {**household(financial_needs=["home_loan"],
                           occupation="teacher", employment_years=3,
                           asset_price=1_000_000_000,
                           loan_amount=700_000_000, loan_term_months=120),
               "loan_application": loan()}

    result = normalize_input(payload)

    assert result.profile.loan_amount == Decimal(700_000_000)


# ============================================================ task 2
def test_ml01_frame_matches_the_trained_feature_contract():
    """Thứ tự cột sai là lỗi im lặng: model vẫn trả xác suất, chỉ là vô nghĩa."""
    profile = normalize_input(household()).profile

    frame = to_ml01_frame(profile, age=35)

    assert list(frame.columns) == list(RAW_FEATURES)
    assert len(frame) == 1


def test_ml01_frame_spreads_assets_into_six_flags():
    """`assets` là nhiều lựa chọn nên không one-hot được — phải trải multi-hot."""
    profile = normalize_input(household(assets=["cash", "gold"])).profile

    frame = to_ml01_frame(profile, age=35)

    assert bool(frame["has_asset_cash"].iloc[0])
    assert bool(frame["has_asset_gold"].iloc[0])
    assert not bool(frame["has_asset_vehicle"].iloc[0])


def test_ml02_frame_converts_money_to_the_annual_period():
    """Home Credit tính theo NĂM, form hỏi theo THÁNG.

    Phải nhân 12 cho CẢ thu nhập lẫn khoản trả. Quên nhân `AMT_ANNUITY` thì
    `dti` bị chia 12 và hồ sơ nào cũng thành "gánh nặng trả nợ rất nhẹ".
    """
    normalized = normalize_input({**household(), "loan_application": loan()})

    frame = to_ml02_frame(normalized.profile, normalized.loan)

    assert frame["AMT_INCOME_TOTAL"].iloc[0] == 35_000_000 * MONTHS_PER_YEAR
    assert frame["AMT_ANNUITY"].iloc[0] == 12_000_000 * MONTHS_PER_YEAR
    # `AMT_CREDIT` là tổng khoản vay, KHÔNG có kỳ nên không nhân.
    assert frame["AMT_CREDIT"].iloc[0] == 1_400_000_000
    # Tỉ lệ trả nợ phải giữ nguyên như tính theo tháng.
    assert (frame["AMT_ANNUITY"].iloc[0] / frame["AMT_INCOME_TOTAL"].iloc[0]
            == pytest.approx(12_000_000 / 35_000_000))


def test_ml02_frame_keeps_days_columns_negative():
    """Home Credit lưu số ngày TRƯỚC ngày nộp đơn nên `DAYS_*` luôn âm.

    Để dương thì `employment_ratio` vẫn ra số dương trông hợp lý, nhưng
    `age_years` thành âm — và không có gì báo.
    """
    normalized = normalize_input({**household(), "loan_application": loan()})

    frame = to_ml02_frame(normalized.profile, normalized.loan)

    assert frame["DAYS_BIRTH"].iloc[0] < 0
    assert frame["DAYS_EMPLOYED"].iloc[0] < 0


def test_ml02_frame_carries_the_form_credit_history():
    """Mục C của form phải vào được model — đây là nhóm feature mạnh thứ hai."""
    normalized = normalize_input({
        **household(),
        "loan_application": loan(previous_loan_count=5, late_payment_count=2,
                                 has_overdue_loan=True,
                                 total_overdue_amount=80_000_000)})

    frame = to_ml02_frame(normalized.profile, normalized.loan)

    assert frame["BUREAU_LOAN_COUNT"].iloc[0] == 5
    assert frame["BUREAU_OVERDUE_LOAN_COUNT"].iloc[0] == 2
    assert frame["BUREAU_HAS_OVERDUE"].iloc[0] == 1
    assert frame["BUREAU_TOTAL_OVERDUE"].iloc[0] == 80_000_000
    assert frame["BUREAU_NO_RECORD"].iloc[0] == 0


def test_overdue_count_cannot_exceed_the_number_of_loans():
    """Bureau không thể đếm nhiều khoản quá hạn hơn tổng số khoản đã có."""
    normalized = normalize_input({
        **household(),
        "loan_application": loan(previous_loan_count=2, late_payment_count=9)})

    frame = to_ml02_frame(normalized.profile, normalized.loan)

    assert frame["BUREAU_OVERDUE_LOAN_COUNT"].iloc[0] == 2


def test_never_borrowed_is_marked_not_guessed():
    """Chưa từng vay → `bureau_no_record = 1`, số năm lịch sử là NaN.

    Điền 0 năm là khẳng định họ vừa mở quan hệ tín dụng hôm nay.
    """
    normalized = normalize_input({
        **household(),
        "loan_application": loan(previous_loan_count=0, late_payment_count=0,
                                 has_overdue_loan=False)})

    frame = to_ml02_frame(normalized.profile, normalized.loan)

    assert frame["BUREAU_NO_RECORD"].iloc[0] == 1
    assert np.isnan(frame["BUREAU_HISTORY_YEARS"].iloc[0])


def test_ml02_frame_supplies_every_column_the_pipeline_reads():
    normalized = normalize_input({**household(), "loan_application": loan()})

    frame = to_ml02_frame(normalized.profile, normalized.loan)

    for column in ML02_APPLICATION_COLUMNS + ML02_BUREAU_COLUMNS:
        assert column in frame.columns, column


# ============================================================ task 7
def test_probabilities_outside_zero_one_are_rejected():
    """Tự sửa một vector xác suất sai là che mất chỗ hỏng ở tầng dưới."""
    with pytest.raises(conf.InvalidProbability):
        conf.validate_probabilities([0.5, 0.7])          # tổng ≠ 1
    with pytest.raises(conf.InvalidProbability):
        conf.validate_probabilities([-0.1, 1.1])
    with pytest.raises(conf.InvalidProbability):
        conf.validate_probabilities([np.nan, 1.0])


def test_ml01_low_confidence_uses_the_project_threshold():
    """Ngưỡng tin cậy đến từ `config.yaml`, không hardcode ở tầng này."""
    from hfml.config import CONFIG

    chac_chan = conf.check_ml01([0.90, 0.05, 0.03, 0.02])
    mong_manh = conf.check_ml01([0.30, 0.28, 0.22, 0.20])

    assert chac_chan.threshold == CONFIG.confidence_threshold
    assert not chac_chan.low_confidence
    assert mong_manh.low_confidence


def test_ml02_confidence_is_distance_to_the_boundary_not_the_probability():
    """Hai loại ngưỡng khác nhau — lẫn chúng là sai kép.

    Ngưỡng của ML02 là chỗ CẮT nhị phân (0,1303), không phải ngưỡng tin cậy.
    Xác suất 0,20 không hề "kém tin cậy" — nó là ước lượng rõ ràng rằng hồ sơ
    vượt ngưỡng. Đòi ML02 đạt 0,60 mới coi là chắc chắn thì với tỉ lệ nền
    8,07% sẽ không hồ sơ nào đạt.
    """
    sat_mep = conf.check_ml02(0.1310, decision_threshold=0.1303)
    xa_mep = conf.check_ml02(0.4500, decision_threshold=0.1303)

    assert sat_mep.low_confidence
    assert not xa_mep.low_confidence
    assert xa_mep.confidence > sat_mep.confidence


def test_descriptions_never_claim_certainty():
    """§8.2 guardrail 4: kết quả ML02 là ước lượng tham khảo, không phải cam kết."""
    cam_ket = {"sẽ", "chắc chắn", "nhất định", "khẳng định"}

    for p in (0.05, 0.25, 0.5, 0.7, 0.95):
        phrase = conf.describe(p)
        assert "khả năng" in phrase
        assert not any(word in phrase for word in cam_ket), phrase


# ============================================================ task 3, 6
def test_result_always_has_every_key_even_when_parts_fail():
    """Khoá thiếu rất dễ bị tầng trên đọc nhầm thành "không có rủi ro"."""
    payload = household()
    del payload["average_monthly_income"]

    data = analyze(payload).to_dict()

    for key in ("ok", "schema_version", "input_summary", "rules",
                "overall_status", "ml01", "ml02", "warnings", "errors"):
        assert key in data, key
    assert data["ml01"]["available"] is False
    assert data["ml02"]["available"] is False


def test_result_is_json_serialisable():
    """Epic AI-02 nhận JSON — kiểu lạ lọt vào là hỏng ở biên service."""
    data = analyze({**household(), "loan_application": loan()}).to_dict()

    assert json.loads(json.dumps(data, ensure_ascii=False))["ok"] is True


def test_all_five_rules_are_reported():
    data = analyze(household()).to_dict()

    assert list(data["rules"]) == list(RULE_CODES)
    assert all(data["rules"][code].get("status") for code in RULE_CODES)


def test_schema_version_is_declared():
    """Đổi shape mà không đổi số là cách chắc chắn nhất để AI-02 hỏng âm thầm."""
    assert analyze(household()).to_dict()["schema_version"] == SCHEMA_VERSION


# ============================================================ task 8
def test_missing_loan_data_is_not_reported_as_a_broken_model():
    """Phần lớn người dùng không khai khoản vay — đó là trạng thái BÌNH THƯỜNG.

    Nói "hệ thống không khả dụng" cho họ là nói sai; thứ họ cần nghe là "bạn
    chưa điền màn Thông tin khoản vay".
    """
    data = analyze(household()).to_dict()

    assert data["ok"] is True, "thiếu khoản vay không làm hỏng cả request"
    assert data["ml02"]["reason_code"] == MISSING_INPUT
    assert data["ml01"]["available"] is True


def test_rule_failure_does_not_take_down_the_models(monkeypatch):
    """Bốn nguồn kết quả độc lập — một cái hỏng không kéo ba cái kia."""
    from hfml.pipeline import orchestrator

    class Hong:
        def evaluate(self, *args, **kwargs):
            raise RuntimeError("giả lập rule engine hỏng")

    monkeypatch.setattr(orchestrator, "RuleEngine", Hong)
    data = analyze(household()).to_dict()

    assert data["rules"] == {}
    assert any(e["code"] == "rule_engine_error" for e in data["errors"])
    assert data["ml01"]["available"] is True, "ML01 vẫn phải chạy"


def test_model_failure_does_not_take_down_the_rules(monkeypatch):
    from hfml.pipeline import orchestrator

    def hong(*args, **kwargs):
        return PredictionResult.unavailable("ml01", "giả lập", PREDICTION_ERROR)

    monkeypatch.setattr(orchestrator, "predict_ml01", hong)
    data = analyze(household()).to_dict()

    assert data["ml01"]["reason_code"] == PREDICTION_ERROR
    assert len(data["rules"]) == len(RULE_CODES), "rule vẫn phải đủ 5"


def test_low_confidence_is_surfaced_not_swallowed(monkeypatch):
    """Im lặng nghĩa là người dùng đọc một phỏng đoán mong manh như kết luận chắc."""
    from hfml.pipeline import orchestrator

    def mong_manh(*args, **kwargs):
        return PredictionResult(
            model="ml01", available=True, label="GROWTH", label_vi="…",
            probability=0.31,
            confidence={"confidence": 0.31, "low_confidence": True,
                        "threshold": 0.60, "description": "khả năng thấp"})

    monkeypatch.setattr(orchestrator, "predict_ml01", mong_manh)
    data = analyze(household()).to_dict()

    codes = [w["code"] for w in data["warnings"]]
    assert "low_confidence" in codes
    message = next(w["message"] for w in data["warnings"]
                   if w["code"] == "low_confidence")
    assert "quy tắc" in message, "phải chỉ người đọc sang kết luận của rule"


def test_reason_codes_distinguish_the_four_failure_kinds():
    """Gộp bốn mã thành một là nói với người dùng rằng hệ thống hỏng."""
    assert MISSING_INPUT != MODEL_UNAVAILABLE != PREDICTION_ERROR
    assert PredictionResult.unavailable("ml01", "x").reason_code == MODEL_UNAVAILABLE
    assert PredictionResult.unavailable(
        "ml01", "x", MISSING_INPUT).reason_code == MISSING_INPUT


# ======================================================= không đụng training
def test_pipeline_never_fits_anything():
    """Inference KHÔNG được `fit` lại bất cứ transformer nào.

    Fit lại trên dữ liệu inference nghĩa là thống kê đến từ chính hồ sơ đang
    dự đoán — mỗi request cho một phép biến đổi khác, và không cái nào khớp
    với lúc train.
    """
    from pathlib import Path

    for name in ("normalizer", "adapters", "predictor", "orchestrator",
                 "confidence"):
        source = Path(f"src/hfml/pipeline/{name}.py").read_text(encoding="utf-8")
        assert ".fit(" not in source, f"{name}.py gọi .fit()"
        assert "fit_transform" not in source, f"{name}.py gọi fit_transform()"
