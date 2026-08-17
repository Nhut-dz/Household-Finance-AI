"""Test cho ML02 task 1 — khám phá Home Credit (`hfml.ml.ml02_credit_risk`).

Không test nào chạm vào dataset thật: 307.511 dòng × 122 cột làm bộ test mất
vài phút và phụ thuộc một file 1,4 GB không có trong git. Mọi test dựng dữ
liệu nhỏ có tính chất ĐÃ BIẾT rồi kiểm phép đo có tìm ra đúng tính chất đó
không.

Mỗi test ứng với một cách phép đo có thể hỏng âm thầm — tức vẫn trả về một
con số trông hợp lý trong khi nó sai.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hfml.ml.ml02_credit_risk.explore import (
    MASS_POINT_SHARE,
    MISSING_BIN,
    RARE_BIN,
    aggregate_bureau,
    build_ratio_features,
    distribution_table,
    form_coverage,
    information_value,
    iv_band,
    merge_bureau,
    rank_by_information_value,
    unreachable_columns,
    woe_table,
    FORM_FIELDS,
)

SEED = 42


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(SEED)


# --------------------------------------------------------------------------
# WoE / IV
# --------------------------------------------------------------------------
def test_iv_is_zero_for_a_feature_independent_of_the_label(rng):
    """Cột không liên quan gì tới nhãn phải cho IV gần 0.

    Đây là phép thử cơ bản nhất: nếu một cột nhiễu thuần tuý vẫn ra IV cao thì
    mọi thứ tính từ nó đều vô nghĩa, và bảng xếp hạng chỉ là xếp hạng nhiễu.
    """
    n = 20_000
    noise = pd.Series(rng.normal(size=n))
    y = pd.Series(rng.binomial(1, 0.08, size=n))

    assert information_value(noise, y) < 0.02


def test_iv_grows_with_the_strength_of_the_relationship(rng):
    """Cột liên quan chặt hơn phải cho IV cao hơn — thứ tự mới là thứ dùng được."""
    n = 20_000
    x = pd.Series(rng.normal(size=n))

    # Cùng một biến x, chỉ khác độ mạnh của hệ số.
    weak = pd.Series(rng.binomial(1, 1 / (1 + np.exp(-(-2.5 + 0.3 * x)))))
    strong = pd.Series(rng.binomial(1, 1 / (1 + np.exp(-(-2.5 + 1.5 * x)))))

    assert information_value(x, strong) > information_value(x, weak)


def test_missing_values_get_their_own_bin_instead_of_being_dropped():
    """NaN phải thành một khoảng riêng, không được biến mất.

    Ở Home Credit việc thiếu dữ liệu tự nó dự báo được vỡ nợ (PLAN.md §4.3b).
    Bỏ NaN khỏi bảng WoE là vứt đúng phần thông tin đó, mà bảng vẫn trông bình
    thường nên không ai phát hiện.
    """
    x = pd.Series([1.0, 2.0, 3.0, 4.0] * 25 + [np.nan] * 100)
    y = pd.Series([0, 0, 0, 1] * 25 + [1] * 100)

    table = woe_table(x, y, bins=4)

    assert MISSING_BIN in set(table["bin"])
    assert table.loc[table["bin"] == MISSING_BIN, "n"].iloc[0] == 100
    # Tổng số hồ sơ trong bảng phải bằng số dòng đầu vào — không mất dòng nào.
    assert table["n"].sum() == len(x)


def test_missing_bin_captures_its_higher_default_rate():
    """Khoảng NaN phải mang đúng tỉ lệ vỡ nợ của nhóm thiếu dữ liệu."""
    x = pd.Series([1.0] * 200 + [np.nan] * 100)
    y = pd.Series([0] * 200 + [1] * 50 + [0] * 50)

    table = woe_table(x, y)
    missing_row = table[table["bin"] == MISSING_BIN].iloc[0]

    assert missing_row["bad_rate"] == pytest.approx(0.5)
    assert missing_row["lift"] > 1.0


def test_zero_inflated_column_does_not_collapse_into_one_bin():
    """Cột dồn khối phải tách được, không được sập về một khoảng.

    Đây là lỗi ĐÃ SẬP khi chạy lần đầu trên dữ liệu thật:
    `BUREAU_TOTAL_OVERDUE` có 98,92% giá trị bằng 0 nên mọi mốc phân vị của
    `pd.qcut` trùng nhau, `duplicates="drop"` gộp hết còn một khoảng, và IV ra
    đúng 0,0000 — trông y hệt "cột không có tín hiệu" trong khi thật ra phép
    đo không chạy. Test này canh đúng chỗ đó.
    """
    n = 10_000
    n_nonzero = int(n * 0.01)
    x = pd.Series([0.0] * (n - n_nonzero) + list(np.linspace(1, 500, n_nonzero)))
    # Nhóm khác 0 vỡ nợ nhiều hơn hẳn — tín hiệu phải bắt được.
    y = pd.Series([0] * (n - n_nonzero) + [1] * n_nonzero)

    table = woe_table(x, y)

    assert len(table) > 1, "cột dồn khối bị sập về một khoảng"
    assert f"={0:g}" in set(table["bin"]), "giá trị khối chưa được tách riêng"
    assert information_value(x, y) > 0.02


def test_thin_tail_stays_measurable_after_the_mass_point_is_split_off():
    """Phần đuôi mỏng phải gộp thành ít khoảng, không bị chia vụn rồi mất hút.

    Lỗi thứ hai đã gặp trên dữ liệu thật, ngay sau khi sửa lỗi thứ nhất:
    `BUREAU_TOTAL_OVERDUE` tách khối 0 xong chỉ còn **1,08%** dân số; chia
    tiếp thành 10 thập phân vị thì mỗi khoảng ~0,1%, đều nhỏ hơn `min_share`
    nên bị loại khỏi phép tính lift — cột có tín hiệu thật lại báo lift 0,99,
    tức "an toàn hơn trung bình". Số khoảng phải co theo phần còn lại.
    """
    n = 10_000
    n_nonzero = int(n * 0.012)
    x = pd.Series([0.0] * (n - n_nonzero) + list(np.linspace(1, 500, n_nonzero)))
    y = pd.Series([0] * (n - n_nonzero) + [1] * n_nonzero)

    df = pd.DataFrame({"SK_ID_CURR": np.arange(n), "TARGET": y, "overdue": x})
    row = rank_by_information_value(df).iloc[0]

    assert row["worst_bin"] != f"={0:g}", "đuôi mỏng bị loại, lift chỉ còn nhóm 0"
    assert row["max_lift"] > 1.0


def test_mass_point_threshold_is_respected():
    """Giá trị chiếm dưới ngưỡng thì KHÔNG tách riêng, để tránh vụn bảng."""
    n = 10_000
    minor = int(n * (MASS_POINT_SHARE / 2))     # dưới ngưỡng
    x = pd.Series([7.0] * minor + list(np.linspace(0, 1, n - minor)))
    y = pd.Series(np.random.default_rng(SEED).binomial(1, 0.1, size=n))

    bins = set(woe_table(x, y)["bin"])

    assert f"={7:g}" not in bins


def test_rare_categories_are_grouped_together():
    """Hạng mục quá nhỏ phải gộp vào `__RARE__`, không đứng riêng.

    Một hạng mục 5 hồ sơ có thể cho WoE rất lớn hoàn toàn do ngẫu nhiên; để
    nó đứng riêng là đẩy một cột vô dụng lên đầu bảng xếp hạng.
    """
    x = pd.Series(["A"] * 500 + ["B"] * 495 + ["C"] * 5)
    y = pd.Series([0] * 500 + [0] * 495 + [1] * 5)

    bins = set(woe_table(x, y)["bin"])

    assert "C" not in bins
    assert RARE_BIN in bins


def test_woe_stays_finite_when_a_bin_has_no_defaults():
    """Khoảng không có hồ sơ vỡ nợ nào vẫn phải cho số hữu hạn, không `inf`.

    Không hiệu chỉnh thì `log(0)` cho `-inf`, IV thành `inf`, và cột đó chiếm
    đỉnh bảng xếp hạng vĩnh viễn.
    """
    x = pd.Series(["A"] * 500 + ["B"] * 500)
    y = pd.Series([0] * 500 + [1] * 500)

    table = woe_table(x, y)

    assert np.isfinite(table["woe"]).all()
    assert np.isfinite(information_value(x, y))


def test_single_class_label_raises_instead_of_returning_a_number():
    """Nhãn một lớp thì WoE không định nghĩa được — phải báo lỗi, không đoán."""
    x = pd.Series([1.0, 2.0] * 50)
    y = pd.Series([0] * 100)

    with pytest.raises(ValueError, match="một lớp"):
        woe_table(x, y)


def test_iv_band_boundaries():
    """Thang diễn giải phải xếp đúng ở chính các mốc, không lệch một bậc."""
    assert iv_band(0.60) == "đáng ngờ (nghi rò rỉ nhãn)"
    assert iv_band(0.50) == "đáng ngờ (nghi rò rỉ nhãn)"
    assert iv_band(0.4999) == "mạnh"
    assert iv_band(0.30) == "mạnh"
    assert iv_band(0.10) == "trung bình"
    assert iv_band(0.02) == "yếu"
    assert iv_band(0.0) == "gần như vô dụng"


# --------------------------------------------------------------------------
# Bảng xếp hạng
# --------------------------------------------------------------------------
def _ranking_frame(rng) -> pd.DataFrame:
    n = 5_000
    x = rng.normal(size=n)
    return pd.DataFrame({
        "SK_ID_CURR": np.arange(n),
        "TARGET": rng.binomial(1, 1 / (1 + np.exp(-(-2.5 + 1.5 * x)))),
        "strong": x,
        "noise": rng.normal(size=n),
        "constant": np.ones(n),
    })


def test_ranking_excludes_id_and_target_but_keeps_everything_else(rng):
    """Khoá hồ sơ và nhãn không bao giờ là feature; cột còn lại phải đủ mặt."""
    ranking = rank_by_information_value(_ranking_frame(rng))

    assert set(ranking["column"]) == {"strong", "noise", "constant"}


def test_ranking_is_sorted_and_puts_the_real_signal_on_top(rng):
    ranking = rank_by_information_value(_ranking_frame(rng))

    assert ranking.iloc[0]["column"] == "strong"
    assert ranking["iv"].is_monotonic_decreasing


def test_constant_column_is_reported_not_silently_dropped(rng):
    """Cột hằng số vẫn phải có mặt kèm lý do — biến mất là mất dấu vết."""
    ranking = rank_by_information_value(_ranking_frame(rng))
    row = ranking[ranking["column"] == "constant"].iloc[0]

    assert row["iv"] == pytest.approx(0.0, abs=1e-9)
    assert "hằng số" in row["note"]


def test_max_lift_ignores_the_rare_bucket():
    """`max_lift` không được lấy từ `__RARE__` — túi đó không phải một nhóm người.

    Lỗi đã gặp trên dữ liệu thật: `FLAG_DOCUMENT_2` bật ở 13/307.511 hồ sơ,
    gộp vào `__RARE__` rồi leo lên đầu bảng lift với 3,81 thuần tuý do ngẫu
    nhiên.
    """
    n = 2_000
    rare = 8
    df = pd.DataFrame({
        "SK_ID_CURR": np.arange(n),
        # Nhóm hiếm vỡ nợ 100%, nhóm còn lại 10%.
        "TARGET": [1] * rare + [1] * 200 + [0] * (n - rare - 200),
        "flag": ["rare"] * rare + ["common"] * (n - rare),
    })

    row = rank_by_information_value(df).iloc[0]

    assert row["worst_bin"] != RARE_BIN
    assert row["max_lift"] < 2.0


# --------------------------------------------------------------------------
# bureau.csv
# --------------------------------------------------------------------------
@pytest.fixture
def bureau() -> pd.DataFrame:
    """Ba khách hàng: 1 sạch · 2 đang quá hạn · 3 không có mặt trong file.

    Phải có đủ `BUREAU_COLUMNS` — từ 15/08/2026 phép gộp do
    `features.aggregate_bureau` (task 3) sở hữu và nó cần thêm dư nợ lẫn hạn
    mức để dựng ba tỉ lệ của nhóm lịch sử tín dụng.
    """
    return pd.DataFrame({
        "SK_ID_CURR": [1, 1, 2],
        "SK_ID_BUREAU": [10, 11, 12],
        "CREDIT_ACTIVE": ["Closed", "Active", "Active"],
        "CREDIT_DAY_OVERDUE": [0, 0, 45],
        "AMT_CREDIT_SUM": [91_323.0, 50_000.0, 20_000.0],
        "AMT_CREDIT_SUM_DEBT": [0.0, 30_000.0, 18_000.0],
        "AMT_CREDIT_SUM_OVERDUE": [0.0, 0.0, 5_000.0],
        "DAYS_CREDIT": [-730, -365, -1_095],
    })


def test_bureau_aggregate_matches_the_four_form_questions(bureau):
    """Bốn ô mục C của form phải khớp ĐÚNG định nghĩa tổng hợp từ bureau."""
    agg = aggregate_bureau(bureau)

    assert agg.loc[1, "BUREAU_LOAN_COUNT"] == 2
    assert agg.loc[1, "BUREAU_OVERDUE_LOAN_COUNT"] == 0
    assert agg.loc[1, "BUREAU_HAS_OVERDUE"] == 0
    assert agg.loc[1, "BUREAU_TOTAL_OVERDUE"] == 0.0
    assert agg.loc[1, "BUREAU_ACTIVE_LOAN_COUNT"] == 1

    assert agg.loc[2, "BUREAU_OVERDUE_LOAN_COUNT"] == 1
    assert agg.loc[2, "BUREAU_HAS_OVERDUE"] == 1
    assert agg.loc[2, "BUREAU_TOTAL_OVERDUE"] == 5_000.0
    # Khoản cũ nhất cách đây 1.095 ngày = 3 năm.
    assert agg.loc[2, "BUREAU_HISTORY_YEARS"] == pytest.approx(3.0, abs=0.02)


def test_customer_without_bureau_record_gets_zero_not_median(bureau):
    """Không có bản ghi = CHƯA TỪNG VAY, phải điền 0 chứ không phải NaN.

    Để NaN rồi impute trung vị sẽ gán cho người chưa từng vay một lịch sử tín
    dụng trung bình mà họ không hề có — model học trên một hồ sơ không tồn tại.
    """
    app = pd.DataFrame({"SK_ID_CURR": [1, 2, 3], "TARGET": [0, 1, 0]})

    merged = merge_bureau(app, aggregate_bureau(bureau)).set_index("SK_ID_CURR")

    assert merged.loc[3, "BUREAU_LOAN_COUNT"] == 0
    assert merged.loc[3, "BUREAU_HAS_OVERDUE"] == 0
    assert merged.loc[3, "BUREAU_TOTAL_OVERDUE"] == 0.0
    assert merged.loc[3, "BUREAU_NO_RECORD"] == 1


def test_history_years_stays_missing_for_customers_without_record(bureau):
    """"Số năm có lịch sử tín dụng" của người chưa từng vay KHÔNG phải 0 năm.

    Nó không tồn tại. Điền 0 là khẳng định họ vừa mở quan hệ tín dụng hôm nay.
    """
    app = pd.DataFrame({"SK_ID_CURR": [1, 2, 3], "TARGET": [0, 1, 0]})

    merged = merge_bureau(app, aggregate_bureau(bureau)).set_index("SK_ID_CURR")

    assert pd.isna(merged.loc[3, "BUREAU_HISTORY_YEARS"])


def test_merge_keeps_every_application_row(bureau):
    """Nối bureau không được làm mất hồ sơ nào — left join, không phải inner."""
    app = pd.DataFrame({"SK_ID_CURR": [1, 2, 3, 4, 5], "TARGET": [0, 1, 0, 0, 1]})

    assert len(merge_bureau(app, aggregate_bureau(bureau))) == 5


# --------------------------------------------------------------------------
# Ánh xạ form ↔ Home Credit
# --------------------------------------------------------------------------
def test_every_form_field_is_declared_once():
    """Không trường nào khai hai lần — khai trùng thì bảng phủ sóng đếm sai."""
    names = [f.field for f in FORM_FIELDS]

    assert len(names) == len(set(names))


def test_form_field_source_is_one_of_the_known_screens():
    assert {f.source for f in FORM_FIELDS} <= {"household", "loan", "derived"}


def test_coverage_joins_measured_iv_onto_mapped_fields():
    """Trường ánh xạ được phải lấy đúng IV của cột tương ứng."""
    ranking = pd.DataFrame({
        "column": ["CODE_GENDER", "EXT_SOURCE_1"],
        "iv": [0.0386, 0.1508],
        "band": ["yếu", "trung bình"],
    })

    coverage = form_coverage(ranking)
    gender = coverage[coverage["field"] == "gender"].iloc[0]

    assert gender["home_credit"] == "CODE_GENDER"
    assert gender["iv"] == pytest.approx(0.0386)


def test_unmapped_fields_have_no_iv_instead_of_zero():
    """Trường Home Credit không có cột tương ứng phải để trống, không phải 0.

    Ghi 0 là khẳng định "cột này đo rồi, không có tín hiệu" — sai hoàn toàn so
    với "không có cột nào để đo".
    """
    ranking = pd.DataFrame({"column": ["CODE_GENDER"], "iv": [0.04], "band": ["yếu"]})

    coverage = form_coverage(ranking)
    savings = coverage[coverage["field"] == "savings_amount"].iloc[0]

    assert savings["home_credit"] == "—"
    assert pd.isna(savings["iv"])


def test_unreachable_lists_only_columns_the_form_cannot_supply():
    ranking = pd.DataFrame({
        "column": ["EXT_SOURCE_1", "CODE_GENDER", "ORGANIZATION_TYPE"],
        "iv": [0.15, 0.04, 0.06],
        "band": ["trung bình", "yếu", "yếu"],
    })

    unreachable = set(unreachable_columns(ranking)["column"])

    assert "CODE_GENDER" not in unreachable       # form lấy được
    assert {"EXT_SOURCE_1", "ORGANIZATION_TYPE"} <= unreachable


# --------------------------------------------------------------------------
# Feature tỉ lệ
# --------------------------------------------------------------------------
def test_employment_sentinel_is_excluded_from_the_ratio():
    """Sentinel 365243 phải bị loại trước khi tính `employment_ratio`.

    Để nguyên thì tỉ lệ ra khoảng -19,4 (1000 năm đi làm / 50 năm tuổi) và cả
    bảng phân phối thành vô nghĩa — nhưng nó vẫn là một con số, nên bảng trông
    vẫn bình thường.
    """
    app = pd.DataFrame({
        "AMT_ANNUITY": [1_000.0, 1_000.0],
        "AMT_INCOME_TOTAL": [10_000.0, 10_000.0],
        "AMT_CREDIT": [20_000.0, 20_000.0],
        "AMT_GOODS_PRICE": [18_000.0, 18_000.0],
        "CNT_CHILDREN": [1, 1],
        "CNT_FAM_MEMBERS": [3.0, 3.0],
        "DAYS_BIRTH": [-14_600, -14_600],
        "DAYS_EMPLOYED": [-3_650, 365_243],       # dòng 2 là sentinel
    })

    ratios = build_ratio_features(app)

    assert ratios["employment_ratio"].iloc[0] == pytest.approx(0.25)
    assert pd.isna(ratios["employment_ratio"].iloc[1])


def test_zero_income_becomes_missing_instead_of_infinity():
    """Thu nhập 0 phải cho NaN, không được cho `inf` chảy xuống bảng phân phối."""
    app = pd.DataFrame({
        "AMT_ANNUITY": [1_000.0],
        "AMT_INCOME_TOTAL": [0.0],
        "AMT_CREDIT": [20_000.0],
        "AMT_GOODS_PRICE": [18_000.0],
        "CNT_CHILDREN": [0],
        "CNT_FAM_MEMBERS": [2.0],
        "DAYS_BIRTH": [-14_600],
        "DAYS_EMPLOYED": [-3_650],
    })

    ratios = build_ratio_features(app)

    assert pd.isna(ratios["dti"].iloc[0])
    assert pd.isna(ratios["credit_income_ratio"].iloc[0])


def test_distribution_table_reports_every_feature_with_its_formula():
    frame = pd.DataFrame({"dti": [0.1, 0.2, 0.3, 0.4]})

    table = distribution_table(frame)

    assert list(table["feature"]) == ["dti"]
    assert table.iloc[0]["formula"] == "AMT_ANNUITY / AMT_INCOME_TOTAL"
    assert table.iloc[0]["p50"] == pytest.approx(0.25)
