"""F01 task 9 — kiểm tra duplicate và dữ liệu bất hợp lệ.

Điều quan trọng nhất được kiểm ở đây là **bất đối xứng train/inference**:
lúc train được bỏ dòng, lúc inference thì không — người dùng đang chờ kết
quả, trả về "hồ sơ bị loại" là hệ thống hỏng chứ không phải dữ liệu hỏng.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hfml.data import loader
from hfml.data.preprocessing.cleaner import FLAG_SUFFIX
from hfml.data.preprocessing.validator import (
    INVALID_RULES,
    OutlierClipper,
    clean_for_training,
    drop_duplicates,
    drop_invalid_rows,
    duplicate_id_mask,
    duplicate_row_mask,
    invalid_mask,
    validation_report,
)


def frame(**overrides) -> pd.DataFrame:
    data = {
        "SK_ID_CURR": [1, 2, 3, 4],
        "TARGET": [0, 1, 0, 0],
        "AMT_INCOME_TOTAL": [100_000.0, 200_000.0, 150_000.0, 180_000.0],
        "AMT_CREDIT": [500_000.0, 600_000.0, 450_000.0, 700_000.0],
        "DAYS_BIRTH": [-12000.0, -15000.0, -9000.0, -20000.0],
        "DAYS_EMPLOYED": [-2000.0, -3000.0, -1000.0, -5000.0],
        "CNT_CHILDREN": [1, 0, 2, 3],
        "CNT_FAM_MEMBERS": [3.0, 2.0, 4.0, 5.0],
    }
    data.update(overrides)
    return pd.DataFrame(data)


# ------------------------------------------------------------- duplicate

def test_clean_frame_has_no_duplicates():
    df = frame()
    assert not duplicate_id_mask(df).any()
    assert not duplicate_row_mask(df).any()


def test_duplicate_id_keeps_first():
    df = frame(SK_ID_CURR=[1, 1, 2, 3])
    assert duplicate_id_mask(df).tolist() == [False, True, False, False]


def test_duplicate_row_detected_even_with_different_id():
    """Khác ID nhưng trùng toàn bộ đặc trưng vẫn là bản sao."""
    df = frame()
    dup = pd.concat([df, df.tail(1).assign(SK_ID_CURR=[99])], ignore_index=True)
    assert duplicate_row_mask(dup).sum() == 1


def test_drop_duplicates_resets_index():
    df = frame(SK_ID_CURR=[1, 1, 2, 3])
    out = drop_duplicates(df)
    assert len(out) == 3
    assert out.index.tolist() == [0, 1, 2]


def test_drop_duplicates_ignores_full_row_by_default():
    """Trên bộ feature rút gọn, hai người khác nhau dễ trùng vài cột.

    Mặc định chỉ khử theo ID, nếu không sẽ xóa nhầm dữ liệu train hợp lệ.
    """
    df = pd.DataFrame({"SK_ID_CURR": [1, 2], "AMT_INCOME_TOTAL": [100.0, 100.0]})
    assert len(drop_duplicates(df)) == 2
    assert len(drop_duplicates(df, full_row=True)) == 1


# ------------------------------------------------------------ quy tắc

def codes_violated(df) -> set[str]:
    rep = validation_report(df)
    return set(rep.loc[rep["n_violations"] > 0, "code"])


def test_clean_frame_violates_nothing():
    assert codes_violated(frame()) == set()


def test_nonpositive_income_detected():
    assert "nonpositive_income" in codes_violated(frame(AMT_INCOME_TOTAL=[0.0, 1.0, 2.0, 3.0]))


def test_negative_amount_detected():
    assert "negative_amount" in codes_violated(
        frame(AMT_CREDIT=[-1.0, 600_000.0, 450_000.0, 700_000.0]))


def test_positive_days_detected():
    assert "days_positive" in codes_violated(
        frame(DAYS_EMPLOYED=[5.0, -3000.0, -1000.0, -5000.0]))


def test_employed_before_birth_detected():
    df = frame(DAYS_EMPLOYED=[-99_000.0, -3000.0, -1000.0, -5000.0])
    assert "employed_before_birth" in codes_violated(df)


def test_children_exceed_family_detected():
    assert "children_exceed_family" in codes_violated(
        frame(CNT_CHILDREN=[3, 0, 2, 3], CNT_FAM_MEMBERS=[3.0, 2.0, 4.0, 5.0]))


def test_negative_count_detected():
    assert "negative_count" in codes_violated(frame(CNT_CHILDREN=[-1, 0, 2, 3]))


def test_flag_columns_are_not_mistaken_for_data():
    """Cờ `_MISSING` cũng bắt đầu bằng DAYS_/AMT_/CNT_ nhưng giá trị 1 là hợp lệ."""
    df = frame()
    df["DAYS_EMPLOYED" + FLAG_SUFFIX] = np.array([1, 1, 0, 0], dtype="int8")
    df["AMT_INCOME" + FLAG_SUFFIX] = np.array([1, 0, 0, 0], dtype="int8")
    assert codes_violated(df) == set()


def test_rule_on_absent_column_is_not_a_violation():
    """Bộ feature rút gọn thiếu cột — 'không áp dụng được' khác 'vi phạm'."""
    reduced = frame()[["SK_ID_CURR", "TARGET", "AMT_INCOME_TOTAL"]]
    assert codes_violated(reduced) == set()


def test_nan_is_not_a_violation():
    """Thiếu dữ liệu là việc của task 8, không phải task 9."""
    assert codes_violated(frame(CNT_FAM_MEMBERS=[np.nan] * 4)) == set()


def test_report_covers_every_rule():
    rep = validation_report(frame())
    assert list(rep["code"]) == [r.code for r in INVALID_RULES]
    assert all(rep["description"].str.len() > 10)


def test_drop_invalid_removes_only_violating_rows():
    df = frame(AMT_INCOME_TOTAL=[0.0, 200_000.0, 150_000.0, 180_000.0])
    out = drop_invalid_rows(df)
    assert len(out) == 3
    assert 1 not in out["SK_ID_CURR"].tolist()


def test_invalid_mask_combines_rules():
    df = frame(AMT_INCOME_TOTAL=[0.0, 200_000.0, 150_000.0, 180_000.0],
               CNT_CHILDREN=[1, 0, 2, 9])
    assert invalid_mask(df).tolist() == [True, False, False, True]


# ------------------------------------------------------------ ngoại lai

def test_clipper_caps_extreme_value():
    train = pd.DataFrame({"x": list(range(1000))})
    clipper = OutlierClipper(lower_quantile=0.01, upper_quantile=0.99).fit(train)
    out = clipper.transform(pd.DataFrame({"x": [-500, 500, 50_000]}))
    low, high = clipper.bounds_["x"]
    assert out["x"].tolist() == [low, 500, high]


def test_clipper_learns_bounds_from_train_only():
    """Biên phải đến từ train. Học trên toàn bộ dữ liệu là rò rỉ."""
    train = pd.DataFrame({"x": [1.0] * 100})
    test = pd.DataFrame({"x": [1_000_000.0]})
    clipper = OutlierClipper().fit(train)
    assert clipper.transform(test)["x"].iloc[0] == 1.0


def test_clipper_skips_flags_id_and_target():
    df = pd.DataFrame({
        "SK_ID_CURR": [1, 2, 3],
        "TARGET": [0, 1, 0],
        "DAYS_EMPLOYED" + FLAG_SUFFIX: np.array([0, 1, 0], dtype="int8"),
        "x": [1.0, 2.0, 3.0],
    })
    clipper = OutlierClipper().fit(df)
    assert set(clipper.bounds_) == {"x"}


def test_clipper_reports_which_rows_were_clipped():
    """Inference phải gắn cờ, không được im lặng sửa số của người dùng."""
    train = pd.DataFrame({"x": list(range(1000))})
    clipper = OutlierClipper(lower_quantile=0.01, upper_quantile=0.99).fit(train)
    incoming = pd.DataFrame({"x": [500.0, 99_999.0]})
    assert clipper.clipped_mask(incoming).tolist() == [False, True]


def test_clipper_rejects_bad_quantiles():
    with pytest.raises(ValueError, match="Phân vị không hợp lệ"):
        OutlierClipper(lower_quantile=0.9, upper_quantile=0.1).fit(pd.DataFrame({"x": [1.0]}))


def test_clipper_ignores_text_columns():
    df = pd.DataFrame({"x": [1.0, 2.0], "name": ["a", "b"]})
    clipper = OutlierClipper().fit(df)
    assert set(clipper.bounds_) == {"x"}
    assert clipper.transform(df)["name"].tolist() == ["a", "b"]


def test_clipper_composes_in_pipeline():
    from sklearn.pipeline import Pipeline

    from hfml.data.preprocessing.cleaner import MissingNormalizer

    pipe = Pipeline([("missing", MissingNormalizer()), ("clip", OutlierClipper())])
    out = pipe.fit_transform(frame())
    assert len(out) == len(frame())


# ------------------------------------------------------------ điểm vào

def test_clean_for_training_removes_both_kinds():
    df = frame(SK_ID_CURR=[1, 1, 2, 3], AMT_INCOME_TOTAL=[100_000.0, 100_000.0, 0.0, 180_000.0])
    out = clean_for_training(df)
    assert len(out) == 2                      # bỏ 1 trùng + 1 thu nhập = 0


def test_clean_for_training_keeps_clean_frame_intact():
    df = frame()
    pd.testing.assert_frame_equal(clean_for_training(df), df)


def test_no_clean_for_inference_function_exists():
    """Cố ý không có — lúc inference không được bỏ dòng nào."""
    from hfml.data.preprocessing import validator

    assert not hasattr(validator, "clean_for_inference")


# ------------------------------------------------------ trên dataset thật

@pytest.mark.skipif(not loader.resolve(loader.PRIMARY_FILE).exists(),
                    reason="chưa tải dataset Home Credit")
def test_home_credit_has_no_duplicates():
    """Khẳng định "0 dòng trùng" trong docstring chỉ đúng trên ĐỦ 122 cột."""
    df = loader.load_application_train()
    assert not duplicate_id_mask(df).any()
    assert not duplicate_row_mask(df).any()


@pytest.mark.skipif(not loader.resolve(loader.PRIMARY_FILE).exists(),
                    reason="chưa tải dataset Home Credit")
def test_home_credit_violates_only_the_known_sentinel_rule():
    df = loader.load_application_train(
        columns=["AMT_INCOME_TOTAL", "DAYS_BIRTH", "DAYS_EMPLOYED",
                 "CNT_CHILDREN", "CNT_FAM_MEMBERS"])
    rep = validation_report(df)
    violated = set(rep.loc[rep["n_violations"] > 0, "code"])
    # Chỉ `days_positive` được phép vi phạm: đó là sentinel, task 8 xử lý.
    assert violated <= {"days_positive"}


@pytest.mark.skipif(not loader.resolve(loader.PRIMARY_FILE).exists(),
                    reason="chưa tải dataset Home Credit")
def test_sentinel_stops_violating_after_task8_normalisation():
    """Chạy task 8 trước thì task 9 không còn thấy vi phạm nào."""
    from hfml.data.preprocessing.cleaner import normalize_missing

    df = loader.load_application_train(
        columns=["AMT_INCOME_TOTAL", "DAYS_BIRTH", "DAYS_EMPLOYED",
                 "CNT_CHILDREN", "CNT_FAM_MEMBERS"])
    rep = validation_report(normalize_missing(df))
    assert rep["n_violations"].sum() == 0
