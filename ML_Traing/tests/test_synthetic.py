"""F03 task 3, 4 — kiểm tra dân số hộ synthetic của ML01.

Hai nhóm test ở đây khác hẳn nhau về mục đích:

    Cổng kiểm chứng (PLAN.md §6.2)  dân số có DÙNG ĐƯỢC để train không —
        mỗi lớp ≥ 10%, ranh giới đủ dày.
    Ràng buộc schema              hộ sinh ra có phải hộ HỢP LỆ không —
        số con < số nhân khẩu, có nợ thì phải có khoản trả.

Nhóm thứ hai quan trọng hơn vẻ ngoài: dân số vi phạm schema nghĩa là model
được train trên những hộ mà `HouseholdProfile` sẽ từ chối lúc inference.
"""
from __future__ import annotations

import pandas as pd
import pytest

from hfml.data.schema import ASSET_COLUMNS, AssetType, asset_column
from hfml.data.synthetic import (
    ASSET_BASE_RATES,
    HOUSEHOLD_SIZE_WEIGHTS,
    PopulationParams,
    generate_households,
    population_summary,
)
from hfml.ml.ml01_recommendation.labeler import (
    class_distribution,
    distance_to_boundary,
    label_frame,
)


@pytest.fixture(scope="module")
def population() -> pd.DataFrame:
    return generate_households()


# ------------------------------------------ CỔNG KIỂM CHỨNG (PLAN §6.2)

def test_gate_every_class_reaches_ten_percent(population):
    """Cổng 1 — lớp nào dưới 10% thì bảng per-class không đọc được.

    Không qua thì chỉnh THAM SỐ SINH DÂN SỐ (`PopulationParams`), tuyệt đối
    không chỉnh ngưỡng `g(·)` cho vừa dữ liệu: ngưỡng đã chốt và có dẫn
    nguồn ở PLAN.md §6.1b, sửa nó là đảo ngược quan hệ nhân quả.
    """
    table = class_distribution(label_frame(population))
    thin = table.loc[table["share"] < 0.10, "label"].tolist()
    assert not thin, f"lớp dưới 10%: {thin}\n{table.to_string(index=False)}"


def test_gate_decision_boundary_is_not_too_clean(population):
    """Cổng 2 — ranh giới phải có bề dày, nếu không mọi thuật toán đạt 100%.

    PLAN.md §6.2 đặt mốc 8% hồ sơ nằm trong dải ±10% quanh một ngưỡng.
    """
    near = float((distance_to_boundary(population) <= 0.10).mean())
    assert near >= 0.08, f"chỉ {near:.1%} hồ sơ sát biên, ranh giới quá sạch"


def test_population_covers_households_that_outspend_income(population):
    """Không có hộ nào chi ≥ thu thì nhóm EMERGENCY không thể tồn tại.

    Đây đúng là chỗ 300 hộ thật trong DB không dùng được: chúng được seed
    rất lành, 0 hộ chi vượt thu.
    """
    ratio = population["average_monthly_expense"] / population["average_monthly_income"]
    assert float((ratio >= 1.0).mean()) > 0.02


# ---------------------------------------------------- RÀNG BUỘC SCHEMA

def test_children_count_stays_below_household_size(population):
    """`HouseholdProfile` từ chối hộ có số con ≥ số nhân khẩu."""
    assert (population["children_count"] < population["household_size"]).all()


def test_debt_flag_agrees_with_debt_amounts(population):
    """Có nợ thì phải có cả dư nợ và khoản trả — schema bắt buộc khai đủ."""
    with_debt = population[population["has_debt"]]
    assert (with_debt["monthly_debt_payment"] > 0).all()
    assert (with_debt["total_current_debt"] > 0).all()

    without = population[~population["has_debt"]]
    assert (without["monthly_debt_payment"] == 0).all()
    assert (without["total_current_debt"] == 0).all()


def test_savings_flag_agrees_with_savings_amount(population):
    """`has_savings` phải tương đương `savings_amount > 0`, cả hai chiều.

    Làm tròn về mốc 100k có thể kéo một khoản tiết kiệm bé tí về 0 — lúc đó
    cờ và số tiền mâu thuẫn nhau.
    """
    assert (population["has_savings"] == (population["savings_amount"] > 0)).all()


def test_money_columns_are_non_negative(population):
    for col in ("average_monthly_income", "average_monthly_expense",
                "savings_amount", "total_current_debt", "monthly_debt_payment"):
        assert (population[col] >= 0).all(), col


def test_income_is_never_zero(population):
    """Thu nhập 0 làm mọi tỉ lệ của `g(·)` vô nghĩa."""
    assert (population["average_monthly_income"] > 0).all()


def test_age_stays_inside_configured_range(population):
    p = PopulationParams()
    assert population["age"].between(p.age_min, p.age_max).all()


def test_household_size_uses_measured_distribution(population):
    assert set(population["household_size"].unique()) <= set(HOUSEHOLD_SIZE_WEIGHTS)


def test_no_missing_values_anywhere(population):
    """`None` không được lọt vào dân số train — xem `ZERO_WHEN_ABSENT`."""
    assert int(population.isna().sum().sum()) == 0


# ------------------------------------------------------------- TÀI SẢN

def test_asset_columns_are_boolean(population):
    for col in ASSET_COLUMNS:
        assert population[col].dtype == bool, col


def test_asset_ownership_tracks_measured_db_rates(population):
    """Tỉ lệ sở hữu bám tỉ lệ đo từ 450 dòng `tblassets`.

    `asset_health_effect` đối xứng quanh `health = 0,5` nên tỉ lệ trung bình
    được bảo toàn — lệch nhiều nghĩa là hiệu ứng đã cắt biên (clip), lúc đó
    con số "lấy từ DB" không còn đúng nữa.
    """
    for asset, base in ASSET_BASE_RATES.items():
        actual = float(population[asset_column(asset)].mean())
        assert actual == pytest.approx(base, abs=0.02), asset.value


def test_asset_columns_follow_schema_naming():
    assert ASSET_COLUMNS == tuple(f"has_asset_{a.value}" for a in AssetType)


# ------------------------------------------------------- TÁI LẬP & THAM SỐ

def test_generation_is_reproducible():
    """Cùng seed cho cùng dân số — điều kiện của F06 task 6."""
    small = PopulationParams(n=500)
    assert generate_households(small, seed=42).equals(generate_households(small, seed=42))


def test_different_seed_gives_different_population():
    small = PopulationParams(n=500)
    assert not generate_households(small, seed=42).equals(
        generate_households(small, seed=7))


def test_row_count_follows_params():
    assert len(generate_households(PopulationParams(n=123))) == 123


def test_params_actually_drive_generation():
    """Tham số phải có tác dụng thật — nếu không, cách xử lý khi lớp < 10%
    mà PLAN.md §6.2 chỉ định sẽ không làm được gì cả."""
    rich = PopulationParams(n=2_000, income_log_mu=18.0)
    poor = PopulationParams(n=2_000, income_log_mu=16.0)
    assert (generate_households(rich)["average_monthly_income"].median()
            > generate_households(poor)["average_monthly_income"].median())

    no_debt = generate_households(PopulationParams(n=2_000, debt_prob=0.0))
    assert not no_debt["has_debt"].any()


# -------------------------------------------------------------- BÁO CÁO

def test_population_summary_compares_against_real_db(population):
    """Bảng đối chiếu với 300 hộ thật — phần giải trình trong báo cáo."""
    table = population_summary(population)
    assert list(table.columns) == ["chỉ số", "synthetic", "DB thật (300 hộ)"]
    assert not table.isna().any().any()
    assert len(table) >= 6
