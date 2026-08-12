"""F01 task 12 — kiểm tra feature tỉ lệ.

Test trọng tâm là `test_features_are_currency_invariant`: cùng một hộ gia
đình, mô tả bằng hai đơn vị tiền tệ lệch nhau 340 lần, phải cho ra feature
TRÙNG KHÍT. Đó là toàn bộ lý do mục 2.1 tồn tại — nếu test này đỏ thì model
train trên Home Credit không dùng được cho người dùng Việt Nam.
"""
from __future__ import annotations

from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from hfml.data import loader
from hfml.data.features.builder import (
    ALL_FORM_FEATURES,
    ALL_HOME_CREDIT_FEATURES,
    FORM_ONLY_FEATURES,
    HOME_CREDIT_ONLY_FEATURES,
    RATIO_FEATURES,
    SHARED_FEATURES,
    Availability,
    build_from_home_credit,
    build_from_profile,
    feature_catalog,
    median_income_per_capita,
    safe_divide,
)
from hfml.data.preprocessing.cleaner import normalize_missing
from hfml.data.schema import FinancialNeed, HouseholdProfile, OccupationType


def profile(scale: Decimal = Decimal(1), **overrides) -> HouseholdProfile:
    """Hồ sơ mẫu. `scale` nhân mọi số tiền — dùng để thử bất biến đơn vị."""
    base = dict(
        representative_name="Nguyễn Văn A",
        birth_year=1986,
        household_size=4,
        children_count=2,
        has_dependents=True,
        occupation=OccupationType.OFFICE_STAFF,
        employment_years=Decimal("10"),
        average_monthly_income=Decimal("30000000") * scale,
        average_monthly_expense=Decimal("20000000") * scale,
        has_debt=True,
        total_current_debt=Decimal("120000000") * scale,
        monthly_debt_payment=Decimal("6000000") * scale,
        has_savings=True,
        savings_amount=Decimal("60000000") * scale,
        financial_needs=[FinancialNeed.HOME_LOAN],
        asset_price=Decimal("2000000000") * scale,
        loan_amount=Decimal("1400000000") * scale,
        loan_term_months=240,
    )
    base.update(overrides)
    return HouseholdProfile(**base)


# ================== TEST TRỌNG TÂM: bất biến đơn vị tiền tệ ==================

def test_features_are_currency_invariant():
    """Cùng một hộ, hai đơn vị tiền tệ lệch 340 lần → feature trùng khít.

    Đây là điều kiện để model train trên Home Credit áp được cho VN.
    """
    vnd = build_from_profile(profile())
    # Cùng hộ đó nhưng quy về đơn vị của Home Credit.
    other_currency = build_from_profile(profile(scale=Decimal(1) / Decimal(340)))

    pd.testing.assert_frame_equal(vnd, other_currency, atol=1e-9)


def test_every_registered_feature_is_currency_invariant():
    """Tử số là tiền thì mẫu số cũng phải là tiền, nếu không domain gap quay lại."""
    for f in RATIO_FEATURES:
        assert f.currency_invariant, f"{f.name}: {f.dimension}"


def test_income_per_capita_is_not_raw_money():
    """PLAN §2.1 ghi `income_per_capita` (tiền ÷ người = vẫn là tiền)."""
    names = {f.name for f in RATIO_FEATURES}
    assert "income_per_capita" not in names
    assert "income_per_capita_ratio" in names


def test_ltv_and_markup_are_not_the_same_feature():
    """Cái bẫy chính của task 12: hai đại lượng khác nghĩa mà suýt trùng tên.

    Home Credit `AMT_CREDIT/AMT_GOODS_PRICE` luôn ≥ 1,0 (đã cộng phí vào
    khoản vay) — đó là tỉ lệ đội giá. Form `loan_amount/asset_price` < 1,0 —
    đó mới là LTV. Gộp chúng lại là đưa model ra ngoài phân phối.
    """
    by_name = {f.name: f for f in RATIO_FEATURES}
    assert by_name["ltv"].availability is Availability.FORM_ONLY
    assert by_name["credit_goods_markup"].availability is Availability.HOME_CREDIT_ONLY
    assert by_name["ltv"].home_credit is None
    assert by_name["credit_goods_markup"].form is None


def test_features_with_semantic_mismatch_carry_a_caveat():
    """Chỗ nào hai nguồn lệch nghĩa phải ghi lại — đó là giới hạn của model."""
    by_name = {f.name: f for f in RATIO_FEATURES}
    for name in ("dti", "credit_goods_markup", "ltv"):
        assert len(by_name[name].caveat) > 40, name


# ------------------------------------------------------------ safe_divide

def test_safe_divide_returns_nan_not_inf():
    """`inf` chảy xuống dưới sẽ làm scaler nổ và imputer không bắt được."""
    out = safe_divide(pd.Series([1.0, 2.0, 3.0]), pd.Series([0.0, -1.0, np.nan]))
    assert out.isna().all()


def test_safe_divide_normal_case():
    out = safe_divide(pd.Series([10.0, 20.0]), pd.Series([2.0, 5.0]))
    assert out.tolist() == [5.0, 4.0]


# ------------------------------------------------------- sinh từ hồ sơ form

def test_profile_features_have_fixed_columns_and_order():
    """Thiếu hoặc lệch thứ tự cột là lỗi im lặng lúc inference."""
    out = build_from_profile(profile())
    assert list(out.columns) == list(ALL_FORM_FEATURES)
    assert len(out) == 1


def test_empty_optional_fields_give_nan_not_crash():
    """Người dùng bỏ trống thì ra NaN, vẫn đủ cột."""
    minimal = HouseholdProfile(
        representative_name="B", household_size=1, children_count=0,
        has_dependents=False, average_monthly_income=Decimal("10000000"),
        has_debt=False, has_savings=False,
    )
    out = build_from_profile(minimal)
    assert list(out.columns) == list(ALL_FORM_FEATURES)
    assert np.isnan(out["savings_months"].iloc[0]) or out["savings_months"].iloc[0] == 0
    assert np.isnan(out["ltv"].iloc[0])


def test_known_ratios_are_computed_correctly():
    out = build_from_profile(profile()).iloc[0]
    assert out["dti"] == pytest.approx(6 / 30)                    # 6tr / 30tr
    assert out["ltv"] == pytest.approx(1400 / 2000)
    assert out["credit_income_ratio"] == pytest.approx(1400 / (30 * 12))
    assert out["children_ratio"] == pytest.approx(2 / 4)
    assert out["savings_months"] == pytest.approx(60 / 20)
    assert out["debt_income_ratio"] == pytest.approx(120 / (30 * 12))
    assert out["savings_rate"] == pytest.approx((30 - 20) / 30)
    assert out["expense_income_ratio"] == pytest.approx(20 / 30)


def test_no_debt_means_zero_not_unknown():
    """Người không nợ có DTI = 0, khác hẳn 'không biết DTI'."""
    out = build_from_profile(profile(has_debt=False, total_current_debt=None,
                                     monthly_debt_payment=None)).iloc[0]
    assert out["dti"] == 0.0
    assert out["debt_income_ratio"] == 0.0


def test_no_savings_means_zero_months():
    out = build_from_profile(profile(has_savings=False, savings_amount=None)).iloc[0]
    assert out["savings_months"] == 0.0


def test_age_and_employment_ratio():
    out = build_from_profile(profile(birth_year=1986,
                                     employment_years=Decimal("10"))).iloc[0]
    from datetime import date
    age = date.today().year - 1986
    assert out["age_years"] == pytest.approx(age)
    assert out["employment_ratio"] == pytest.approx(10 / age)


def test_income_per_capita_ratio_needs_reference():
    """Không có mức tham chiếu thì để NaN, không bịa ra một con số."""
    without = build_from_profile(profile()).iloc[0]["income_per_capita_ratio"]
    assert np.isnan(without)

    with_ref = build_from_profile(profile(), reference_income_per_capita=100_000_000)
    assert with_ref.iloc[0]["income_per_capita_ratio"] == pytest.approx(
        (30_000_000 * 12 / 4) / 100_000_000)


# ------------------------------------------------------- danh mục feature

def test_three_availability_groups_partition_cleanly():
    groups = [set(SHARED_FEATURES), set(FORM_ONLY_FEATURES),
              set(HOME_CREDIT_ONLY_FEATURES)]
    for i, a in enumerate(groups):
        for b in groups[i + 1:]:
            assert not a & b
    assert set(ALL_FORM_FEATURES) == groups[0] | groups[1]
    assert set(ALL_HOME_CREDIT_FEATURES) == groups[0] | groups[2]


def test_formula_presence_matches_availability():
    """Khai `FORM_ONLY` thì phải thật sự không có công thức Home Credit."""
    for f in RATIO_FEATURES:
        if f.availability is Availability.FORM_ONLY:
            assert f.home_credit is None and f.form, f.name
        elif f.availability is Availability.HOME_CREDIT_ONLY:
            assert f.form is None and f.home_credit, f.name
        else:
            assert f.home_credit and f.form, f.name


def test_catalog_covers_every_feature():
    cat = feature_catalog()
    assert list(cat["name"]) == [f.name for f in RATIO_FEATURES]
    assert cat["description"].str.len().min() > 3
    assert cat["currency_invariant"].all()


# ------------------------------------------------------ trên dataset thật

needs_dataset = pytest.mark.skipif(
    not loader.resolve(loader.PRIMARY_FILE).exists(),
    reason="chưa tải dataset Home Credit")

HC_COLUMNS = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
              "CNT_CHILDREN", "CNT_FAM_MEMBERS", "DAYS_BIRTH", "DAYS_EMPLOYED"]


@needs_dataset
def test_home_credit_feature_sets():
    df = normalize_missing(loader.load_application_train(columns=HC_COLUMNS, nrows=5000))

    full = build_from_home_credit(df, reference_income_per_capita=73_575.0)
    assert list(full.columns) == list(ALL_HOME_CREDIT_FEATURES)

    reduced = build_from_home_credit(df, reference_income_per_capita=73_575.0,
                                     feature_set="reduced")
    assert list(reduced.columns) == list(SHARED_FEATURES)
    # Bộ rút gọn là tập con thật sự của bộ full — bảng so sánh §7.2 mới có nghĩa.
    assert set(reduced.columns) < set(full.columns)


@needs_dataset
def test_reduced_set_is_exactly_what_the_form_can_supply():
    """Điều kiện để model 'rút gọn' deploy được: form sinh đủ mọi cột của nó."""
    from_form = set(build_from_profile(profile()).columns)
    assert set(SHARED_FEATURES) <= from_form


@needs_dataset
def test_markup_is_always_at_least_one():
    """Bằng chứng cho caveat: đại lượng này không phải LTV."""
    df = normalize_missing(loader.load_application_train(columns=HC_COLUMNS))
    markup = build_from_home_credit(df)["credit_goods_markup"].dropna()
    assert markup.quantile(0.01) >= 1.0
    assert markup.median() == pytest.approx(1.119, abs=0.01)


@needs_dataset
def test_home_credit_ratios_are_plausible():
    """Feature sinh ra phải nằm trong dải hợp lý, không phải rác."""
    df = normalize_missing(loader.load_application_train(columns=HC_COLUMNS))
    out = build_from_home_credit(df)

    assert out["dti"].median() == pytest.approx(0.163, abs=0.01)
    assert out["credit_income_ratio"].median() == pytest.approx(3.27, abs=0.05)
    assert 20 < out["age_years"].median() < 60
    # Sentinel đã thành NaN ở task 8 nên không còn ai "đi làm 1000 năm".
    assert out["employment_years"].max() < 60
    assert (out["employment_ratio"].dropna() <= 1).all()


@needs_dataset
def test_no_infinities_leak_through():
    df = normalize_missing(loader.load_application_train(columns=HC_COLUMNS))
    out = build_from_home_credit(df)
    assert not np.isinf(out.to_numpy(dtype=float)).any()


@needs_dataset
def test_vietnamese_profile_lands_inside_home_credit_distribution():
    """Kiểm chứng cuối cùng của mục 2.1: hồ sơ VN phải rơi vào vùng phân phối
    mà model đã học, chứ không nằm ngoài rìa."""
    df = normalize_missing(loader.load_application_train(columns=HC_COLUMNS))
    hc = build_from_home_credit(df)
    vn = build_from_profile(profile()).iloc[0]

    for name in SHARED_FEATURES:
        if np.isnan(vn[name]) or hc[name].isna().all():
            continue
        low, high = hc[name].quantile([0.01, 0.99])
        assert low <= vn[name] <= high, (
            f"{name}: hồ sơ VN = {vn[name]:.3f} nằm ngoài dải "
            f"[{low:.3f}, {high:.3f}] của Home Credit")


@needs_dataset
def test_median_income_per_capita_is_positive():
    df = loader.load_application_train(columns=["AMT_INCOME_TOTAL", "CNT_FAM_MEMBERS"])
    assert median_income_per_capita(df) > 0
