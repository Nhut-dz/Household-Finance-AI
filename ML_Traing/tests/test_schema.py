"""F01 task 4 — kiểm tra data contract của form đầu vào.

Mỗi test ở đây tương ứng một cách người dùng (hoặc backend) có thể gửi lên
dữ liệu sai. Cái đáng sợ không phải request lỗi — mà là request hợp lệ về
kiểu nhưng vô lý về nghĩa, chạy trót lọt tới tận khuyến nghị.
"""
from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError

from hfml.data.schema import (
    API_TO_DB_COLUMN,
    LOAN_TERM_CHOICES,
    LOAN_TRIGGER_NEEDS,
    OCCUPATION_TO_HOME_CREDIT,
    AssetType,
    DataQualityFlag,
    FinancialNeed,
    HouseholdProfile,
    OccupationType,
)


def profile(**overrides) -> HouseholdProfile:
    """Hồ sơ hợp lệ tối thiểu; test ghi đè đúng trường mình quan tâm."""
    base = dict(
        representative_name="Nguyễn Văn A",
        household_size=4,
        children_count=2,
        has_dependents=True,
        average_monthly_income=Decimal("30000000"),
        average_monthly_expense=Decimal("20000000"),
        has_debt=False,
        has_savings=False,
    )
    base.update(overrides)
    return HouseholdProfile(**base)


def loan_profile(**overrides) -> HouseholdProfile:
    """Hồ sơ có nhu cầu vay — phải khai đủ cả 5 ô của KHỐI VAY."""
    base = dict(
        financial_needs=[FinancialNeed.HOME_LOAN],
        occupation=OccupationType.OFFICE_STAFF,
        employment_years=Decimal("10"),
        asset_price=Decimal("2000000000"),
        loan_amount=Decimal("1400000000"),
        loan_term_months=240,
    )
    base.update(overrides)
    return profile(**base)


# ------------------------------------------------------------------ cơ bản

def test_minimal_valid_profile():
    p = profile()
    assert p.household_size == 4
    assert p.assets == [] and p.financial_needs == []
    assert not p.needs_loan_analysis


def test_extra_field_is_rejected():
    """Gửi thừa trường lạ phải báo lỗi, không được nuốt im lặng."""
    with pytest.raises(ValidationError, match="Extra inputs"):
        profile(monthly_income=Decimal("30000000"))   # tên cột DB, không phải tên API


@pytest.mark.parametrize("field_name", [
    "average_monthly_income", "average_monthly_expense", "savings_amount",
    "total_current_debt", "asset_price", "loan_amount",
])
def test_money_fields_reject_negative(field_name):
    """Kiểu `Money` gắn ge=0 qua Annotated — phải còn hiệu lực sau khi field
    ghi đè Field(title=...). Mất ràng buộc này thì thu nhập âm lọt qua."""
    with pytest.raises(ValidationError, match="greater_than_equal|greater than or equal"):
        profile(**{field_name: Decimal("-1")})


@pytest.mark.parametrize("year", [1899, date.today().year + 1])
def test_birth_year_out_of_range(year):
    with pytest.raises(ValidationError):
        profile(birth_year=year)


def test_name_length_limit():
    with pytest.raises(ValidationError):
        profile(representative_name="x" * 151)


def test_age_derived_from_birth_year():
    assert profile(birth_year=1990).age == date.today().year - 1990
    assert profile().age is None


def test_vietnamese_label_available_for_every_field():
    labels = HouseholdProfile.vietnamese_labels()
    assert labels["average_monthly_income"] == "Thu nhập trung bình tháng"
    assert set(labels) == set(HouseholdProfile.model_fields)


def test_every_enum_value_has_a_vietnamese_label():
    """Form dựng dropdown và tầng llm viết câu chữ đều lấy từ đây."""
    from hfml.data.schema import (
        ASSET_LABELS,
        FINANCIAL_NEED_LABELS,
        OCCUPATION_LABELS,
        AssetType,
        FinancialNeed,
    )

    assert set(OCCUPATION_LABELS) == set(OccupationType)
    assert set(ASSET_LABELS) == set(AssetType)
    assert set(FINANCIAL_NEED_LABELS) == set(FinancialNeed)
    for labels in (OCCUPATION_LABELS, ASSET_LABELS, FINANCIAL_NEED_LABELS):
        assert all(text.strip() for text in labels.values())


def test_loan_term_label_reads_in_years():
    from hfml.data.schema import loan_term_label

    assert loan_term_label(240) == "20 năm (240 tháng)"
    assert loan_term_label(12) == "1 năm (12 tháng)"


def test_api_to_db_mapping_points_at_real_fields():
    assert set(API_TO_DB_COLUMN) <= set(HouseholdProfile.model_fields)


def test_every_occupation_has_home_credit_mapping():
    """Thiếu ánh xạ nào là ML02 im lặng mất một nhóm nghề."""
    assert set(OCCUPATION_TO_HOME_CREDIT) == set(OccupationType)


# ------------------------------------------------------------ nhân khẩu

@pytest.mark.parametrize("size,children", [(4, 4), (2, 3), (1, 1)])
def test_children_must_be_fewer_than_household(size, children):
    with pytest.raises(ValidationError, match="phải nhỏ hơn"):
        profile(household_size=size, children_count=children)


def test_children_count_boundary_ok():
    assert profile(household_size=4, children_count=3).children_count == 3


# ------------------------------------------------------------------- nợ

def test_has_debt_requires_total_debt():
    with pytest.raises(ValidationError, match="Tổng dư nợ hiện tại"):
        profile(has_debt=True, monthly_debt_payment=Decimal("2000000"))


def test_has_debt_requires_monthly_payment():
    """Thiếu tiền trả nợ/tháng thì DTI không tính được → RB02, RB05, ML02 chết."""
    with pytest.raises(ValidationError, match="Số tiền trả nợ hàng tháng"):
        profile(has_debt=True, total_current_debt=Decimal("20000000"))


def test_no_debt_but_debt_amount_is_contradiction():
    with pytest.raises(ValidationError, match="Khai không có nợ"):
        profile(has_debt=False, total_current_debt=Decimal("20000000"))


def test_valid_debt_profile():
    p = profile(has_debt=True,
                total_current_debt=Decimal("20000000"),
                monthly_debt_payment=Decimal("2000000"))
    assert p.monthly_debt_payment == Decimal("2000000")


# -------------------------------------------------------------- tiết kiệm

def test_has_savings_requires_amount():
    with pytest.raises(ValidationError, match="Số tiền tiết kiệm"):
        profile(has_savings=True)


def test_no_savings_but_amount_is_contradiction():
    with pytest.raises(ValidationError, match="Khai không có tiết kiệm"):
        profile(has_savings=False, savings_amount=Decimal("50000000"))


# ----------------------------------------------------------------- mảng

def test_duplicate_assets_rejected():
    with pytest.raises(ValidationError, match="trùng lặp"):
        profile(assets=[AssetType.REAL_ESTATE, AssetType.REAL_ESTATE])


def test_duplicate_financial_needs_rejected():
    with pytest.raises(ValidationError, match="trùng lặp"):
        profile(financial_needs=[FinancialNeed.HOME_LOAN, FinancialNeed.HOME_LOAN])


# ------------------------------------------------------------ khoản vay

@pytest.mark.parametrize("need", sorted(LOAN_TRIGGER_NEEDS, key=lambda n: n.value))
def test_loan_needs_require_loan_fields(need):
    with pytest.raises(ValidationError, match="thông tin khoản vay"):
        profile(financial_needs=[need])


def test_other_need_does_not_require_loan_fields():
    p = profile(financial_needs=[FinancialNeed.SAVING])
    assert not p.needs_loan_analysis


def test_valid_loan_request():
    p = loan_profile()
    assert p.needs_loan_analysis
    assert p.loan_triggers == {FinancialNeed.HOME_LOAN}


def test_loan_term_must_be_a_listed_choice():
    with pytest.raises(ValidationError, match="Kỳ hạn vay"):
        loan_profile(financial_needs=[FinancialNeed.HOME_LOAN], loan_term_months=17)


def test_all_loan_term_choices_accepted():
    for term in LOAN_TERM_CHOICES:
        p = loan_profile(financial_needs=[FinancialNeed.HOME_LOAN], loan_term_months=term)
        assert p.loan_term_months == term


# ----------------------------------------------------- cảnh báo chất lượng

def test_clean_profile_has_no_flags():
    p = profile(average_monthly_income=Decimal("30000000"),
                average_monthly_expense=Decimal("20000000"))
    assert p.data_quality_flags() == []


def test_expense_exceeding_income_is_flagged_not_rejected():
    """Chi > thu là hợp lệ (hộ đang khó khăn) nhưng phải nổi cờ."""
    p = profile(average_monthly_income=Decimal("10000000"),
                average_monthly_expense=Decimal("15000000"))
    assert DataQualityFlag.EXPENSE_EXCEEDS_INCOME in p.data_quality_flags()


def test_savings_rate_above_60_percent_is_flagged():
    p = profile(average_monthly_income=Decimal("30000000"),
                average_monthly_expense=Decimal("5000000"))
    assert DataQualityFlag.SAVINGS_RATE_TOO_HIGH in p.data_quality_flags()


def test_missing_expense_is_flagged():
    p = profile(average_monthly_expense=None)
    assert DataQualityFlag.MISSING_EXPENSE in p.data_quality_flags()


def test_loan_exceeding_asset_price_is_flagged():
    p = loan_profile(financial_needs=[FinancialNeed.HOME_LOAN],
                     asset_price=Decimal("800000000"),
                     loan_amount=Decimal("900000000"),
                     loan_term_months=60)
    assert DataQualityFlag.LOAN_EXCEEDS_ASSET_PRICE in p.data_quality_flags()


# ---------------------------------------------------------- trường bổ sung

def test_new_fields_accepted():
    p = profile(occupation=OccupationType.OFFICE_STAFF,
                employment_years=Decimal("8.5"))
    assert OCCUPATION_TO_HOME_CREDIT[p.occupation] == "Core staff"


def test_occupation_and_employment_belong_to_the_loan_block():
    """Chốt 11/08/2026: hai trường này chỉ hỏi khi người dùng có nhu cầu vay.

    Người chỉ xem sức khỏe tài chính không phải nhập thêm ô nào — đó là lý do
    chúng nằm trong khối vay chứ không ở phần nhân thân.
    """
    # Không có nhu cầu vay → bỏ trống vẫn hợp lệ.
    assert profile().occupation is None

    # Có nhu cầu vay mà thiếu → chặn.
    with pytest.raises(ValidationError, match="Nghề nghiệp"):
        profile(financial_needs=[FinancialNeed.HOME_LOAN],
                asset_price=Decimal("2000000000"),
                loan_amount=Decimal("1400000000"),
                loan_term_months=240,
                employment_years=Decimal("10"))

    with pytest.raises(ValidationError, match="Số năm đi làm"):
        profile(financial_needs=[FinancialNeed.HOME_LOAN],
                asset_price=Decimal("2000000000"),
                loan_amount=Decimal("1400000000"),
                loan_term_months=240,
                occupation=OccupationType.OFFICE_STAFF)


def test_json_round_trip():
    """Payload đi qua HTTP phải khôi phục nguyên vẹn."""
    p = loan_profile(assets=[AssetType.REAL_ESTATE, AssetType.VEHICLE],
                     financial_needs=[FinancialNeed.HOME_LOAN],
                     asset_price=Decimal("1000000000"),
                     loan_amount=Decimal("600000000"),
                     loan_term_months=120)
    assert HouseholdProfile.model_validate_json(p.model_dump_json()) == p
