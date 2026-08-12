"""F03 task 2, 4 — kiểm tra hàm sinh nhãn `g(·)` của ML01.

Test quan trọng nhất file này là `test_x_never_contains_label_drivers`. Rò rỉ
nhãn không làm gì sập cả: pipeline vẫn chạy, model vẫn train, metric còn ĐẸP
HƠN. Nó chỉ lộ ra khi có người hỏi "sao accuracy 99,8%?" — mà lúc đó báo cáo
đã in rồi. Đây là rủi ro circular labeling ở PLAN.md §14, và nó phải bị chặn
bằng test chứ không bằng trí nhớ.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hfml.data.schema import ASSET_COLUMNS, HouseholdProfile
from hfml.data.synthetic import generate_households
from hfml.ml.ml01_recommendation.labeler import (
    DEFAULT_THRESHOLDS,
    EXCLUDED_FROM_X,
    FORBIDDEN_IN_X,
    ORDERED_GROUPS,
    RAW_FEATURES,
    ZERO_WHEN_ABSENT,
    LabelThresholds,
    RecommendationGroup,
    add_label_noise,
    class_distribution,
    compute_indicators,
    distance_to_boundary,
    label_frame,
)


@pytest.fixture(scope="module")
def population() -> pd.DataFrame:
    """Dân số mặc định, sinh một lần dùng chung — 20.000 hộ không rẻ."""
    return generate_households()


def _household(**overrides) -> pd.DataFrame:
    """Một hộ ở nhóm GROWTH, cho phép ghi đè từng trường để dựng ca kiểm thử."""
    base = {
        "average_monthly_income": 20_000_000.0,
        "average_monthly_expense": 10_000_000.0,   # savings_rate = 0,50
        "savings_amount": 100_000_000.0,           # savings_months = 10
        "total_current_debt": 0.0,
        "monthly_debt_payment": 0.0,               # dti = 0
    }
    base.update(overrides)
    return pd.DataFrame([base])


# ------------------------------------------------------- CHỐNG RÒ RỈ NHÃN

def test_x_never_contains_label_drivers():
    """`X` không được chứa biến mà `g(·)` đặt ngưỡng lên (PLAN.md §6.1c).

    Đưa `savings_months` / `dti` / `savings_rate` vào `X` thì một cây sâu 3
    tầng học thuộc nguyên `g(·)`, mọi thuật toán đạt ~100%, và bảng so sánh
    4 thuật toán mất sạch ý nghĩa.
    """
    leaked = set(RAW_FEATURES) & set(FORBIDDEN_IN_X)
    assert not leaked, f"rò rỉ nhãn: {sorted(leaked)} vừa là feature vừa là biến của g(·)"


def test_indicator_columns_stay_out_of_feature_frame(population):
    """`compute_indicators` trả khung RIÊNG, không nối vào dữ liệu hộ."""
    indicators = compute_indicators(population)
    assert set(indicators.columns) == {"savings_months", "dti", "savings_rate"}
    assert not set(indicators.columns) & set(population.columns)


def test_generated_population_matches_x_exactly(population):
    """Cột sinh ra và `RAW_FEATURES` phải khớp HAI CHIỀU.

    Thiếu cột thì train nổ ngay — vô hại. Thừa cột mới nguy: nó lặng lẽ đi
    vào `X` nếu chỗ nào đó dùng `df.columns` thay vì `RAW_FEATURES`.
    """
    assert set(population.columns) == set(RAW_FEATURES)


def test_hidden_health_variable_is_not_exported(population):
    """`health` là công cụ sinh, lộ ra ngoài là rò rỉ trực tiếp cấu trúc nhãn."""
    assert "health" not in population.columns


# --------------------------------------------------- FEATURE SET & LÝ DO

def test_excluded_fields_are_real_form_fields():
    """Mọi trường trong `EXCLUDED_FROM_X` phải có thật trong schema.

    Chặn hai thứ: gõ sai tên, và mục đã cũ còn sót lại sau khi schema đổi —
    cả hai đều biến bảng giải trình thành thông tin sai trong báo cáo.
    """
    fields = set(HouseholdProfile.model_fields)
    unknown = set(EXCLUDED_FROM_X) - fields
    assert not unknown, f"không có trong HouseholdProfile: {sorted(unknown)}"


def test_excluded_fields_carry_a_reason():
    """Loại một trường thì phải nói được vì sao — hội đồng sẽ hỏi.

    Chấp nhận hai dạng: lý do viết đủ, hoặc trỏ sang một trường khác trong
    chính bảng này (`xem \\`occupation\\``) — nhóm KHỐI VAY dùng chung một lý
    do, chép lại bốn lần chỉ tạo thêm bốn chỗ để quên cập nhật.
    """
    for field, reason in EXCLUDED_FROM_X.items():
        if len(reason) > 40:
            continue
        pointed = [k for k in EXCLUDED_FROM_X if k != field and f"`{k}`" in reason]
        assert pointed, f"{field}: lý do quá sơ sài và không trỏ sang trường nào"
        assert len(EXCLUDED_FROM_X[pointed[0]]) > 40, (
            f"{field} trỏ sang {pointed[0]}, mà chỗ đó cũng không có lý do")


def test_loan_block_fields_are_excluded():
    """Trường của KHỐI VAY không được vào `X` của ML01.

    Chúng chỉ hiện khi người dùng chọn `home_loan`, mà ML01 chấm sức khỏe
    tài chính cho MỌI hồ sơ — train trên dân số ai cũng có nghề nghiệp rồi
    suy luận cho người bỏ trống là lệch phân phối train/inference.
    """
    for field in ("occupation", "employment_years", "asset_price",
                  "loan_amount", "loan_term_months"):
        assert field not in RAW_FEATURES
        assert field in EXCLUDED_FROM_X


def test_assets_enter_x_as_multi_hot():
    """`assets` là danh sách nhiều lựa chọn — phải trải thành 6 cột nhị phân."""
    assert set(ASSET_COLUMNS) <= set(RAW_FEATURES)
    assert "assets" not in RAW_FEATURES


def test_conditional_money_columns_are_zero_not_missing(population):
    """`None` ở ba cột này nghĩa là 0, không phải "chưa biết"."""
    assert set(ZERO_WHEN_ABSENT) <= set(RAW_FEATURES)
    assert not population[list(ZERO_WHEN_ABSENT)].isna().any().any()
    no_savings = population.loc[~population["has_savings"], "savings_amount"]
    assert (no_savings == 0).all()


# ----------------------------------------------------------- CHỈ SỐ g(·)

def test_indicators_match_definition():
    df = _household()
    ind = compute_indicators(df).iloc[0]
    assert ind["savings_months"] == pytest.approx(10.0)
    assert ind["dti"] == pytest.approx(0.0)
    assert ind["savings_rate"] == pytest.approx(0.50)


def test_zero_expense_gives_infinite_buffer():
    """Không chi tiêu thì đệm là vô hạn — `inf`, không phải `NaN`.

    Dùng `NaN` thì so sánh `< 1` trả về False một cách tình cờ đúng, nhưng
    `distance_to_boundary` sẽ hỏng theo kiểu khó truy.
    """
    ind = compute_indicators(_household(average_monthly_expense=0.0)).iloc[0]
    assert ind["savings_months"] == np.inf


def test_zero_income_does_not_raise():
    """Thu nhập 0 là dữ liệu bẩn, nhưng `g(·)` phải trả nhãn chứ không nổ."""
    labels = label_frame(_household(average_monthly_income=0.0))
    assert labels.iloc[0] in {g.value for g in ORDERED_GROUPS}


# ----------------------------------------------------- THANG MỨC ĐỘ g(·)

@pytest.mark.parametrize("overrides,expected", [
    # Chi vượt thu → savings_rate < 0
    ({"average_monthly_expense": 25_000_000.0}, RecommendationGroup.EMERGENCY),
    # Đệm dưới 1 tháng chi tiêu
    ({"savings_amount": 5_000_000.0}, RecommendationGroup.EMERGENCY),
    # dti = 0,50 ≥ 0,40
    ({"monthly_debt_payment": 10_000_000.0}, RecommendationGroup.DEBT_FOCUS),
    # Đệm 2 tháng: qua mốc 1 nhưng dưới mốc 3
    ({"savings_amount": 20_000_000.0}, RecommendationGroup.BUILD_BUFFER),
    # savings_rate = 0,05 < 0,10
    ({"average_monthly_expense": 19_000_000.0,
      "savings_amount": 200_000_000.0}, RecommendationGroup.BUILD_BUFFER),
    ({}, RecommendationGroup.GROWTH),
])
def test_each_threshold_routes_to_its_group(overrides, expected):
    assert label_frame(_household(**overrides)).iloc[0] == expected.value


def test_most_severe_group_wins_when_conditions_overlap():
    """Hộ thỏa nhiều điều kiện nhận nhãn NẶNG NHẤT — `g(·)` đơn trị nhờ vậy.

    Hộ dưới đây vừa không có đệm (EMERGENCY) vừa dti 0,50 (DEBT_FOCUS) vừa
    tiết kiệm âm. Không có quy tắc thứ tự thì nhãn phụ thuộc thứ tự viết
    điều kiện — tức là phụ thuộc may rủi.
    """
    df = _household(
        average_monthly_expense=22_000_000.0,
        savings_amount=0.0,
        monthly_debt_payment=10_000_000.0,
    )
    assert label_frame(df).iloc[0] == RecommendationGroup.EMERGENCY.value


def test_severity_order_is_emergency_first():
    assert ORDERED_GROUPS[0] is RecommendationGroup.EMERGENCY
    assert ORDERED_GROUPS[-1] is RecommendationGroup.GROWTH
    assert [g.severity for g in ORDERED_GROUPS] == [0, 1, 2, 3]


def test_thresholds_match_plan_section_6_1b():
    """Ngưỡng đã chốt và có dẫn nguồn — đổi chúng là đổi định nghĩa bài toán.

    PLAN.md §6.2 nói rõ: lớp nào dưới 10% thì chỉnh THAM SỐ SINH DÂN SỐ,
    không chỉnh ngưỡng cho vừa dữ liệu.
    """
    t = DEFAULT_THRESHOLDS
    assert (t.emergency_savings_months, t.debt_focus_dti,
            t.buffer_savings_months, t.buffer_savings_rate) == (1.0, 0.40, 3.0, 0.10)


def test_thresholds_are_frozen():
    with pytest.raises(Exception):
        DEFAULT_THRESHOLDS.debt_focus_dti = 0.99  # type: ignore[misc]


def test_custom_thresholds_are_honoured():
    """Ngưỡng truyền vào phải thực sự được dùng — nếu không, test trên vô nghĩa."""
    df = _household(monthly_debt_payment=5_000_000.0)      # dti = 0,25
    assert label_frame(df).iloc[0] == RecommendationGroup.GROWTH.value
    loose = LabelThresholds(debt_focus_dti=0.20)
    assert label_frame(df, loose).iloc[0] == RecommendationGroup.DEBT_FOCUS.value


def test_label_frame_is_total_and_index_preserving(population):
    labels = label_frame(population)
    assert len(labels) == len(population)
    assert labels.index.equals(population.index)
    assert not labels.isna().any()
    assert set(labels.unique()) <= {g.value for g in ORDERED_GROUPS}


# ------------------------------------------------------------ NHIỄU NHÃN

def test_all_boundary_terms_share_one_scale():
    """Năm số hạng của `distance_to_boundary` phải cùng đơn vị TƯƠNG ĐỐI.

    `add_label_noise` so cả năm với một `boundary_width` duy nhất, nên số
    hạng nào lệch đơn vị sẽ chi phối vùng nhiễu nhãn.

    Hộ dưới đây để dành 5% thu nhập — cách ngưỡng `savings_rate < 0,10` đúng
    một nửa, và KHÔNG hề sát ranh giới dòng tiền âm. Bản cũ để `sr_0` ở đơn
    vị tuyệt đối nên chấm nó 0,05 và xếp vào diện sát biên.
    """
    df = _household(average_monthly_expense=19_000_000.0,   # savings_rate = 0,05
                    savings_amount=100_000_000.0)           # savings_months ≈ 5,3
    assert float(distance_to_boundary(df).iloc[0]) == pytest.approx(0.5)


def test_zero_savings_rate_is_still_on_the_boundary():
    """Đổi thước không được làm mất ngưỡng: chi đúng bằng thu là sát biên."""
    df = _household(average_monthly_expense=20_000_000.0)   # savings_rate = 0
    assert float(distance_to_boundary(df).iloc[0]) == pytest.approx(0.0)


def test_no_single_term_dominates_the_boundary_band(population):
    """Không số hạng nào được một mình chiếm quá nửa vùng sát biên.

    Đây là dấu hiệu của lỗi lệch đơn vị: `sr_0` từng một mình quét 15,95%
    dân số trong khi bốn số hạng kia chỉ 2,6–5,9%.
    """
    near = distance_to_boundary(population) <= 0.10
    total = float(near.mean())
    thresholds = DEFAULT_THRESHOLDS
    ind = compute_indicators(population)
    per_term = {
        "sm_1": (ind["savings_months"] - thresholds.emergency_savings_months).abs()
                / thresholds.emergency_savings_months,
        "sm_3": (ind["savings_months"] - thresholds.buffer_savings_months).abs()
                / thresholds.buffer_savings_months,
        "dti": (ind["dti"] - thresholds.debt_focus_dti).abs() / thresholds.debt_focus_dti,
        "sr_0": ind["savings_rate"].abs() / thresholds.buffer_savings_rate,
        "sr_10": (ind["savings_rate"] - thresholds.buffer_savings_rate).abs()
                 / thresholds.buffer_savings_rate,
    }
    for name, gap in per_term.items():
        share = float((gap.replace(np.inf, np.nan) <= 0.10).mean())
        assert share <= total * 0.60, (
            f"{name} một mình chiếm {share:.1%} / {total:.1%} — lệch đơn vị?")


def test_noise_only_touches_near_boundary_rows(population):
    """Đảo một hộ nằm SÂU trong `GROWTH` là nhiễu vô nghĩa (PLAN.md §6.2)."""
    labels = label_frame(population)
    noisy = add_label_noise(labels, population, rate=0.03, boundary_width=0.10)
    flipped = labels.index[labels != noisy]
    assert len(flipped) > 0
    assert (distance_to_boundary(population).loc[flipped] <= 0.10).all()


def test_noise_only_moves_to_adjacent_severity(population):
    """`GROWTH` → `BUILD_BUFFER` hợp lý; `GROWTH` → `EMERGENCY` thì không."""
    labels = label_frame(population)
    noisy = add_label_noise(labels, population)
    flipped = labels.index[labels != noisy]
    for idx in flipped:
        before = RecommendationGroup(labels.loc[idx]).severity
        after = RecommendationGroup(noisy.loc[idx]).severity
        assert abs(before - after) == 1, f"nhảy {abs(before - after)} bậc mức độ"


def test_noise_rate_matches_plan(population):
    """3% của TOÀN dân số, đúng con số chốt ở PLAN.md §6.2."""
    labels = label_frame(population)
    noisy = add_label_noise(labels, population, rate=0.03)
    assert int((labels != noisy).sum()) == pytest.approx(len(labels) * 0.03, rel=0.02)


def test_noise_is_reproducible(population):
    """Cùng seed cho cùng kết quả — điều kiện của F06 task 6."""
    labels = label_frame(population)
    a = add_label_noise(labels, population, seed=42)
    b = add_label_noise(labels, population, seed=42)
    c = add_label_noise(labels, population, seed=7)
    assert a.equals(b)
    assert not a.equals(c)


def test_zero_rate_leaves_labels_untouched(population):
    labels = label_frame(population)
    assert add_label_noise(labels, population, rate=0.0).equals(labels)


# -------------------------------------------------------- PHÂN BỐ 4 LỚP

def test_class_distribution_covers_all_four_groups(population):
    table = class_distribution(label_frame(population))
    assert list(table["label"]) == [g.value for g in ORDERED_GROUPS]
    assert table["n"].sum() == len(population)
    assert table["share"].sum() == pytest.approx(1.0)


def test_class_distribution_handles_empty_input():
    table = class_distribution(pd.Series([], dtype=object))
    assert (table["share"] == 0.0).all()
