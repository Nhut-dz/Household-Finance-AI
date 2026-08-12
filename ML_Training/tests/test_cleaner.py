"""F01 task 8 — kiểm tra xử lý missing values.

Trọng tâm hai điều, và cả hai đều là lỗi im lặng nếu sai:

1.  `MissingNormalizer` KHÔNG được học gì từ dữ liệu. Sai điều này là rò rỉ
    dữ liệu mà test metric vẫn đẹp.
2.  Cờ `_MISSING` phải sinh TRƯỚC khi sentinel bị thay bằng NaN, nếu không
    thì mất phân biệt giữa "365243" và missing thật.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hfml.data.preprocessing import cleaner
from hfml.data.preprocessing.cleaner import (
    FLAG_SUFFIX,
    MISSING_FLAGS,
    REJECTED_FLAGS,
    HighMissingDropper,
    MissingNormalizer,
)


def frame() -> pd.DataFrame:
    return pd.DataFrame({
        "SK_ID_CURR": [1, 2, 3, 4],
        "TARGET": [0, 1, 0, 0],
        "DAYS_EMPLOYED": [-1000.0, 365243.0, -500.0, np.nan],
        "OCCUPATION_TYPE": ["Laborers", None, "Drivers", "Managers"],
        "ORGANIZATION_TYPE": ["Business", "XNA", "School", "Unknown"],
        "EXT_SOURCE_1": [0.5, np.nan, np.nan, 0.7],
        "EXT_SOURCE_3": [0.1, 0.2, np.nan, 0.4],
        "TOTALAREA_MODE": [0.1, np.nan, np.nan, np.nan],
        "AMT_REQ_CREDIT_BUREAU_HOUR": [0.0, np.nan, 1.0, np.nan],
    })


# ----------------------------------------------------------- sentinel

def test_sentinel_becomes_nan():
    out = cleaner.replace_sentinels(frame())
    assert out["DAYS_EMPLOYED"].isna().sum() == 2      # 1 sẵn có + 1 sentinel
    assert not (out["DAYS_EMPLOYED"] == 365243).any()


def test_replace_sentinels_does_not_mutate_input():
    df = frame()
    cleaner.replace_sentinels(df)
    assert (df["DAYS_EMPLOYED"] == 365243).any(), "hàm đã sửa DataFrame gốc"


# -------------------------------------------------------- chuỗi giả

def test_placeholder_strings_become_nan():
    out = cleaner.replace_placeholders(frame())
    assert out["ORGANIZATION_TYPE"].isna().sum() == 2   # XNA + Unknown
    assert set(out["ORGANIZATION_TYPE"].dropna()) == {"Business", "School"}


def test_placeholder_does_not_touch_numeric():
    out = cleaner.replace_placeholders(frame())
    assert out["EXT_SOURCE_1"].isna().sum() == 2        # y như đầu vào


# --------------------------------------------------------------- cờ

def test_flag_catches_sentinel_and_real_nan():
    out = cleaner.add_missing_flags(frame())
    # dòng 1 = sentinel, dòng 3 = NaN thật → cả hai đều là "thiếu"
    assert out["DAYS_EMPLOYED" + FLAG_SUFFIX].tolist() == [0, 1, 0, 1]


def test_flags_are_int8():
    out = cleaner.add_missing_flags(frame())
    for flag in MISSING_FLAGS:
        if flag.source in frame().columns:
            assert out[flag.name].dtype == np.int8, flag.name


def test_normalize_order_preserves_sentinel_information():
    """Cờ phải sinh trước khi thay sentinel — đây là bug dễ mắc nhất."""
    out = cleaner.normalize_missing(frame())
    assert out["DAYS_EMPLOYED" + FLAG_SUFFIX].tolist() == [0, 1, 0, 1]
    assert out["DAYS_EMPLOYED"].isna().sum() == 2


def test_flag_skipped_when_source_column_absent():
    """Bộ feature rút gọn không có EXT_SOURCE_* — không được vì thế mà lỗi."""
    reduced = frame()[["SK_ID_CURR", "TARGET", "DAYS_EMPLOYED", "OCCUPATION_TYPE"]]
    out = cleaner.normalize_missing(reduced)
    assert "DAYS_EMPLOYED" + FLAG_SUFFIX in out.columns
    assert "EXT_SOURCE_1" + FLAG_SUFFIX not in out.columns


def test_no_flag_can_be_disabled():
    out = cleaner.normalize_missing(frame(), add_flags=False)
    assert not [c for c in out.columns if c.endswith(FLAG_SUFFIX)]


# ------------------------------------------------- bằng chứng cho từng cờ

def test_every_flag_documents_its_evidence():
    """Mỗi cờ phải có số đo biện minh — không cờ nào 'thêm cho chắc'."""
    for flag in MISSING_FLAGS:
        assert 0 < flag.missing_rate < 1, flag.name
        assert abs(flag.lift - 1.0) > 0.10, f"{flag.name} lift quá gần 1, không đáng giữ"
        assert flag.name.endswith(FLAG_SUFFIX)


def test_flag_names_unique():
    names = [f.name for f in MISSING_FLAGS]
    assert len(names) == len(set(names))


def test_rejected_flags_are_documented():
    """Cột bị loại phải có lý do, để trả lời 'sao không làm cho cột này?'."""
    assert "EXT_SOURCE_2" in REJECTED_FLAGS
    assert all(reason for reason in REJECTED_FLAGS.values())
    assert not set(REJECTED_FLAGS) & {f.source for f in MISSING_FLAGS}


# ----------------------------------------------------- transformer sklearn

def test_normalizer_is_stateless():
    """fit() không được học gì — nếu học thì chạy trước split là rò rỉ."""
    tr = MissingNormalizer()
    before = set(vars(tr))
    tr.fit(frame())
    learned = set(vars(tr)) - before
    assert learned <= {"feature_names_in_"}, f"đã học thêm: {learned}"


def test_normalizer_fit_on_subset_gives_same_result():
    """Bằng chứng thực nghiệm cho tính không trạng thái."""
    df = frame()
    full = MissingNormalizer().fit(df).transform(df)
    partial = MissingNormalizer().fit(df.head(2)).transform(df)
    pd.testing.assert_frame_equal(full, partial)


def test_normalizer_feature_names_out():
    tr = MissingNormalizer().fit(frame())
    names = list(tr.get_feature_names_out())
    assert "DAYS_EMPLOYED" + FLAG_SUFFIX in names
    assert names[:2] == ["SK_ID_CURR", "TARGET"]


def test_dropper_learns_only_from_fit_data():
    """Ngưỡng phải tính trên train, không phải trên toàn bộ dữ liệu."""
    train = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [np.nan] * 4})
    test = pd.DataFrame({"a": [np.nan] * 4, "b": [1.0, 2.0, 3.0, 4.0]})

    dropper = HighMissingDropper(threshold=0.6).fit(train)
    assert dropper.columns_to_drop_ == ["b"]
    # `a` thiếu 100% ở test nhưng vẫn giữ, vì quyết định đến từ train.
    assert list(dropper.transform(test).columns) == ["a"]


def test_dropper_never_drops_flags():
    df = pd.DataFrame({
        "x": [np.nan] * 4,
        "DAYS_EMPLOYED" + FLAG_SUFFIX: np.array([1, 1, 1, 1], dtype="int8"),
    })
    dropper = HighMissingDropper(threshold=0.5).fit(df)
    assert dropper.columns_to_drop_ == ["x"]


def test_dropper_feature_names_out():
    train = pd.DataFrame({"a": [1.0, 2.0], "b": [np.nan, np.nan]})
    dropper = HighMissingDropper(threshold=0.6).fit(train)
    assert list(dropper.get_feature_names_out()) == ["a"]


def test_pipeline_composition():
    """Hai bước phải ghép được vào sklearn Pipeline (điều kiện cho task 14)."""
    from sklearn.pipeline import Pipeline

    pipe = Pipeline([
        ("missing", MissingNormalizer()),
        ("drop", HighMissingDropper(threshold=0.60)),
    ])
    out = pipe.fit_transform(frame())
    assert "DAYS_EMPLOYED" + FLAG_SUFFIX in out.columns
    # TOTALAREA_MODE thiếu 75% > 60% → bị bỏ, nhưng cờ của nó còn lại.
    assert "TOTALAREA_MODE" not in out.columns
    assert "BUILDING_INFO" + FLAG_SUFFIX in out.columns


# ------------------------------------------------------------- tóm tắt

def test_missing_summary_sorted():
    summary = cleaner.missing_summary(cleaner.normalize_missing(frame()))
    assert summary["missing_rate"].is_monotonic_decreasing
    assert set(summary.columns) == {"column", "missing", "missing_rate"}


@pytest.mark.skipif(
    not __import__("hfml.data.loader", fromlist=["loader"]).resolve("application_train").exists(),
    reason="chưa tải dataset Home Credit",
)
def test_measured_rates_match_real_data():
    """Số đo ghi trong MISSING_FLAGS phải khớp dataset thật, không phải bịa."""
    from hfml.data import loader

    sources = [f.source for f in MISSING_FLAGS]
    df = loader.load_application_train(columns=sources)
    flagged = cleaner.add_missing_flags(df)

    for flag in MISSING_FLAGS:
        actual = float(flagged[flag.name].mean())
        assert actual == pytest.approx(flag.missing_rate, abs=0.005), (
            f"{flag.name}: ghi {flag.missing_rate:.2%} nhưng đo được {actual:.2%}")
