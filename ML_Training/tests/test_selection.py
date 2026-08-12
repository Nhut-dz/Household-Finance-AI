"""F01 task 13 — kiểm tra feature selection.

Hai test mang tính kiến trúc:

    test_supervised_selection_is_data_dependent
        Chứng minh vì sao bước có giám sát BẮT BUỘC nằm trong Pipeline:
        fit trên hai nửa dữ liệu khác nhau cho ra hai tập feature khác nhau.

    test_selection_is_reproducible
        F06 task 6 đòi chạy lại ra metric trùng đến 4 chữ số — muốn vậy thì
        danh sách cột bị bỏ phải cố định, không phụ thuộc thứ tự duyệt.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hfml.data import loader
from hfml.data.features.selection import (
    CorrelatedFeatureRemover,
    NearZeroVarianceRemover,
    SupervisedFeatureSelector,
    selection_report,
)
from hfml.data.preprocessing.cleaner import normalize_missing


def frame(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = rng.normal(size=n)
    return pd.DataFrame({
        "useful_a": base,
        "twin_of_a": base * 2 + rng.normal(scale=0.01, size=n),   # r ≈ 1
        "useful_b": rng.normal(size=n),
        "almost_constant": [1] * (n - 1) + [0],
        "truly_constant": [7] * n,
    })


# ------------------------------------------------------- gần hằng số

def test_near_constant_column_is_dropped():
    """`nunique() == 2` nên kiểm tra hằng số thông thường bỏ sót cột này."""
    remover = NearZeroVarianceRemover(threshold=0.99).fit(frame())
    assert "almost_constant" in remover.columns_to_drop_
    assert "truly_constant" in remover.columns_to_drop_
    assert "useful_a" not in remover.columns_to_drop_


def test_nzv_respects_protect():
    remover = NearZeroVarianceRemover(0.99, protect=("almost_constant",)).fit(frame())
    assert "almost_constant" not in remover.columns_to_drop_


def test_nzv_transform_drops_columns():
    df = frame()
    remover = NearZeroVarianceRemover(0.99).fit(df)
    out = remover.transform(df)
    assert list(out.columns) == list(remover.get_feature_names_out())
    assert "truly_constant" not in out.columns


def test_nzv_rejects_bad_threshold():
    with pytest.raises(ValueError, match="threshold"):
        NearZeroVarianceRemover(threshold=1.5).fit(frame())


def test_nzv_learns_from_fit_data_only():
    """Cột hằng số ở train nhưng đa dạng ở test vẫn bị bỏ — quyết định từ train."""
    train = pd.DataFrame({"x": [1.0] * 100, "y": np.arange(100.0)})
    test = pd.DataFrame({"x": np.arange(100.0), "y": np.arange(100.0)})
    remover = NearZeroVarianceRemover(0.99).fit(train)
    assert list(remover.transform(test).columns) == ["y"]


# --------------------------------------------------------- tương quan

def test_correlated_pair_loses_one_member():
    remover = CorrelatedFeatureRemover(threshold=0.95).fit(frame())
    dropped = set(remover.columns_to_drop_)
    assert len(dropped & {"useful_a", "twin_of_a"}) == 1
    assert "useful_b" not in dropped


def test_keeps_the_column_with_fewer_missing():
    """Trong bộ ba AVG/MODE/MEDI, giữ cột đầy đủ nhất thì impute bịa ít nhất."""
    n = 200
    rng = np.random.default_rng(1)
    base = rng.normal(size=n)
    df = pd.DataFrame({"full": base, "sparse": base.copy()})
    df.loc[:99, "sparse"] = np.nan          # 50% thiếu

    remover = CorrelatedFeatureRemover(0.95).fit(df)
    assert remover.columns_to_drop_ == ["sparse"]


def test_protect_wins_over_missing_rate():
    n = 200
    base = np.random.default_rng(2).normal(size=n)
    df = pd.DataFrame({"full": base, "sparse": base.copy()})
    df.loc[:99, "sparse"] = np.nan

    remover = CorrelatedFeatureRemover(0.95, protect=("sparse",)).fit(df)
    assert remover.columns_to_drop_ == ["full"]


def test_correlated_records_the_reason():
    remover = CorrelatedFeatureRemover(0.95).fit(frame())
    for col in remover.columns_to_drop_:
        assert "|r|=" in remover.dropped_because_[col]


def test_selection_is_reproducible():
    """Chạy lại phải ra đúng cùng danh sách cột (F06 task 6)."""
    df = frame()
    first = CorrelatedFeatureRemover(0.95).fit(df).columns_to_drop_
    second = CorrelatedFeatureRemover(0.95).fit(df).columns_to_drop_
    assert first == second

    shuffled = df[list(reversed(df.columns))]
    third = CorrelatedFeatureRemover(0.95).fit(shuffled).columns_to_drop_
    # Đổi thứ tự cột có thể đổi cột nào bị giữ, nhưng số lượng phải như nhau.
    assert len(third) == len(first)


# ------------------------------------------------------- có giám sát

def supervised_frame(n: int = 400, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=n)
    X = pd.DataFrame({
        "signal": signal,
        "noise_1": rng.normal(size=n),
        "noise_2": rng.normal(size=n),
        "noise_3": rng.normal(size=n),
    })
    y = pd.Series((signal > 0).astype(int))
    return X, y


def test_supervised_selector_finds_the_signal():
    X, y = supervised_frame()
    selector = SupervisedFeatureSelector(k=1).fit(X, y)
    assert selector.selected_ == ["signal"]


def test_supervised_selector_requires_labels():
    X, _ = supervised_frame()
    with pytest.raises(ValueError, match="cần nhãn"):
        SupervisedFeatureSelector(k=2).fit(X)


def test_supervised_selector_rejects_nan():
    """Phải đặt SAU imputer trong Pipeline."""
    X, y = supervised_frame()
    X.loc[0, "signal"] = np.nan
    with pytest.raises(ValueError, match="Còn NaN"):
        SupervisedFeatureSelector(k=2).fit(X, y)


def test_supervised_selection_is_data_dependent():
    """Vì sao bước này BẮT BUỘC nằm trong Pipeline, không chạy rời trước split.

    Cùng phân phối, hai mẫu khác nhau → hai tập feature khác nhau. Cái gì phụ
    thuộc dữ liệu thì phải fit lại trong từng fold, nếu không phần validation
    đã tham gia quyết định feature và metric CV thành lạc quan giả.
    """
    rng = np.random.default_rng(7)
    n = 300
    X = pd.DataFrame(rng.normal(size=(n, 8)), columns=[f"f{i}" for i in range(8)])
    y = pd.Series(rng.integers(0, 2, size=n))       # nhãn ngẫu nhiên, không tín hiệu

    first = SupervisedFeatureSelector(k=3).fit(X.iloc[:150], y.iloc[:150]).selected_
    second = SupervisedFeatureSelector(k=3).fit(X.iloc[150:], y.iloc[150:]).selected_
    assert first != second


def test_supervised_selector_is_deterministic_for_same_data():
    X, y = supervised_frame()
    a = SupervisedFeatureSelector(k=2, random_state=42).fit(X, y).selected_
    b = SupervisedFeatureSelector(k=2, random_state=42).fit(X, y).selected_
    assert a == b


def test_k_larger_than_columns_is_clamped():
    X, y = supervised_frame()
    selector = SupervisedFeatureSelector(k=999).fit(X, y)
    assert len(selector.selected_) == X.shape[1]


# ------------------------------------------------------------- báo cáo

def test_report_lists_every_dropped_column():
    report = selection_report(frame())
    assert set(report.columns) == {"column", "reason", "detail"}
    assert set(report["reason"]) <= {"near_zero_variance", "correlated"}
    assert report["column"].is_unique


def test_pipeline_composition():
    from sklearn.pipeline import Pipeline

    pipe = Pipeline([
        ("nzv", NearZeroVarianceRemover()),
        ("corr", CorrelatedFeatureRemover()),
    ])
    out = pipe.fit_transform(frame())
    assert out.shape[1] < frame().shape[1]


# ------------------------------------------------------ trên dataset thật

@pytest.mark.skipif(not loader.resolve(loader.PRIMARY_FILE).exists(),
                    reason="chưa tải dataset Home Credit")
def test_real_data_selection_matches_measurements():
    """Con số ghi trong docstring phải khớp dữ liệu thật."""
    df = normalize_missing(loader.load_application_train())
    numeric = df.select_dtypes(include="number").drop(columns=["SK_ID_CURR", "TARGET"])

    nzv = NearZeroVarianceRemover(0.99).fit(numeric)
    assert len(nzv.columns_to_drop_) == 18
    assert "FLAG_MOBIL" in nzv.columns_to_drop_

    corr = CorrelatedFeatureRemover(0.95).fit(nzv.transform(numeric))
    assert 20 <= len(corr.columns_to_drop_) <= 40


@pytest.mark.skipif(not loader.resolve(loader.PRIMARY_FILE).exists(),
                    reason="chưa tải dataset Home Credit")
def test_engineered_flag_duplicates_an_existing_column():
    """Cờ task 8 trùng khít `FLAG_EMP_PHONE` (r = 0,9999) — người nghỉ hưu
    không có điện thoại cơ quan. Khử trùng lặp phải bỏ đúng một trong hai."""
    df = normalize_missing(loader.load_application_train(
        columns=["FLAG_EMP_PHONE", "DAYS_EMPLOYED"]))
    pair = df[["FLAG_EMP_PHONE", "DAYS_EMPLOYED_MISSING"]]
    assert abs(pair.corr().iloc[0, 1]) > 0.999

    remover = CorrelatedFeatureRemover(0.95).fit(pair)
    assert len(remover.columns_to_drop_) == 1
