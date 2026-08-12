"""F01 task 14 — kiểm tra Pipeline tiền xử lý đóng gói.

Đây là mục PLAN.md §13 ghi "không được cắt", nên bộ test này phải chứng minh
được ba điều, mỗi điều tương ứng một cách hệ thống có thể hỏng âm thầm:

    1. `fit` chỉ dùng thống kê của TRAIN     → không rò rỉ dữ liệu
    2. `joblib` round-trip giữ nguyên hành vi → inference khớp training
    3. Thứ tự feature ổn định và tra được     → xác suất không vô nghĩa
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from hfml.data import loader
from hfml.data.preprocessing.cleaner import FLAG_SUFFIX
from hfml.data.preprocessing.pipeline import (
    build_preprocessing_pipeline,
    feature_names,
    fit_preprocessing,
)


def frame(n: int = 300, seed: int = 0) -> pd.DataFrame:
    """Khung dữ liệu giống application_train thu nhỏ, có đủ các loại bẩn."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "SK_ID_CURR": np.arange(n),
        "AMT_INCOME_TOTAL": rng.lognormal(11.9, 0.5, n),
        "AMT_CREDIT": rng.lognormal(13.0, 0.6, n),
        "DAYS_BIRTH": -rng.integers(7500, 25000, n).astype(float),
        "DAYS_EMPLOYED": -rng.integers(100, 15000, n).astype(float),
        "CNT_CHILDREN": rng.integers(0, 4, n),
        "OCCUPATION_TYPE": rng.choice(["Laborers", "Managers", "Drivers"], n),
        "ORGANIZATION_TYPE": rng.choice(["Business", "School", "XNA"], n),
        "CONSTANT_FLAG": np.ones(n),
        "MOSTLY_EMPTY": np.where(rng.random(n) < 0.9, np.nan, 1.0),
    })
    # Sentinel của Home Credit ở 18% số dòng, giống dữ liệu thật.
    df.loc[df.index[: int(n * 0.18)], "DAYS_EMPLOYED"] = 365243.0
    return df


def labels(n: int = 300, seed: int = 1) -> pd.Series:
    return pd.Series(np.random.default_rng(seed).integers(0, 2, n))


# ============ 1. Không rò rỉ: chỉ dùng thống kê của TRAIN ============

def test_transform_uses_train_statistics_only():
    """Giá trị cực đoan chỉ có ở test không được thay đổi phép biến đổi."""
    train = frame(200, seed=0)
    test = frame(50, seed=9)
    test.loc[test.index[0], "AMT_INCOME_TOTAL"] = 1e12   # ngoại lai chỉ ở test

    pipe = build_preprocessing_pipeline()
    pipe, _ = fit_preprocessing(pipe, train)

    out = pipe.transform(test)
    # Bị kẹp về biên học từ train, không kéo theo cả cột.
    assert out["AMT_INCOME_TOTAL"].max() <= train["AMT_INCOME_TOTAL"].max()


def test_fitting_on_all_data_differs_from_fitting_on_train():
    """Bằng chứng vì sao phải fit trên train: hai cách cho kết quả khác nhau.

    Nếu hai cách cho kết quả y hệt thì việc 'fit trên train' chỉ là hình thức.
    Khác nhau nghĩa là tập test THẬT SỰ có ảnh hưởng nếu để nó lọt vào fit.
    """
    train = frame(200, seed=0)
    test = frame(200, seed=5)
    test["AMT_INCOME_TOTAL"] *= 50            # test lệch phân phối hẳn

    only_train = build_preprocessing_pipeline().fit(train).transform(test)
    leaked = build_preprocessing_pipeline().fit(pd.concat([train, test])).transform(test)

    assert not np.allclose(only_train["AMT_INCOME_TOTAL"],
                           leaked["AMT_INCOME_TOTAL"])


def test_imputation_uses_train_median():
    train = frame(200, seed=0)
    test = frame(20, seed=3)
    test["AMT_CREDIT"] = np.nan

    pipe = build_preprocessing_pipeline().fit(train)
    out = pipe.transform(test)
    # Mọi dòng test nhận cùng một giá trị — trung vị của train.
    assert out["AMT_CREDIT"].nunique() == 1
    assert not out["AMT_CREDIT"].isna().any()


# ============ 2. joblib round-trip ============

def test_joblib_roundtrip_preserves_behaviour(tmp_path):
    """Model được dump cùng pipeline; load lại phải cho ra ĐÚNG cùng kết quả."""
    import joblib

    train, test = frame(200, seed=0), frame(50, seed=4)
    pipe = build_preprocessing_pipeline().fit(train)
    before = pipe.transform(test)

    path = tmp_path / "preprocessing.joblib"
    joblib.dump(pipe, path)
    after = joblib.load(path).transform(test)

    pd.testing.assert_frame_equal(before, after)


def test_column_selectors_are_picklable():
    """Hàm chọn cột phải ở cấp module — lambda thì joblib không dump được."""
    import pickle

    pipe = build_preprocessing_pipeline().fit(frame())
    assert pickle.loads(pickle.dumps(pipe)) is not None


# ============ 3. Thứ tự feature ============

def test_feature_names_match_output_columns():
    pipe = build_preprocessing_pipeline().fit(frame())
    out = pipe.transform(frame(50, seed=2))
    assert feature_names(pipe) == list(out.columns)


def test_feature_order_is_stable_across_transforms():
    """Thứ tự cột sai là lỗi im lặng: model vẫn chạy, xác suất vô nghĩa."""
    pipe = build_preprocessing_pipeline().fit(frame())
    first = list(pipe.transform(frame(30, seed=6)).columns)
    second = list(pipe.transform(frame(30, seed=7)).columns)
    assert first == second == feature_names(pipe)


def test_id_and_target_never_reach_the_features():
    train = frame()
    train["TARGET"] = labels()
    pipe = build_preprocessing_pipeline().fit(train)
    assert "SK_ID_CURR" not in feature_names(pipe)
    assert "TARGET" not in feature_names(pipe)


def test_refitting_same_data_gives_same_feature_order():
    """F06 task 6: chạy lại phải tái lập được."""
    train = frame()
    a = feature_names(build_preprocessing_pipeline().fit(train))
    b = feature_names(build_preprocessing_pipeline().fit(train))
    assert a == b


# ============ Hành vi từng bước đã ghép đúng ============

def test_sentinel_and_flag_survive_the_pipeline():
    pipe = build_preprocessing_pipeline().fit(frame())
    names = feature_names(pipe)
    assert any(n.endswith(FLAG_SUFFIX) for n in names), \
        "cờ _MISSING bị mất trên đường đi"


def test_near_constant_column_is_removed():
    pipe = build_preprocessing_pipeline().fit(frame())
    assert "CONSTANT_FLAG" not in feature_names(pipe)


def test_high_missing_column_is_removed():
    pipe = build_preprocessing_pipeline(high_missing_threshold=0.6).fit(frame())
    assert "MOSTLY_EMPTY" not in feature_names(pipe)


def test_output_has_no_missing_values():
    """Cây của sklearn không nhận NaN — pipeline phải trả ra bảng sạch."""
    pipe = build_preprocessing_pipeline().fit(frame())
    out = pipe.transform(frame(50, seed=8))
    assert not out.isna().to_numpy().any()


def test_output_is_entirely_numeric():
    pipe = build_preprocessing_pipeline().fit(frame())
    out = pipe.transform(frame(50, seed=8))
    assert all(pd.api.types.is_numeric_dtype(out[c]) for c in out.columns)


def test_unknown_category_at_inference_does_not_crash():
    """F06 task 1 — người dùng gửi giá trị chưa từng có trong train."""
    pipe = build_preprocessing_pipeline().fit(frame())
    incoming = frame(5, seed=11)
    incoming["OCCUPATION_TYPE"] = "Phi hành gia"
    out = pipe.transform(incoming)
    assert len(out) == 5
    assert not out.isna().to_numpy().any()


def test_single_row_inference():
    """Lúc inference chỉ có MỘT hồ sơ — không được vỡ vì thiếu thống kê."""
    pipe = build_preprocessing_pipeline().fit(frame())
    out = pipe.transform(frame(1, seed=12))
    assert out.shape == (1, len(feature_names(pipe)))


# ============ Tùy chọn cấu hình ============

def test_supervised_selection_stays_inside_the_pipeline():
    """Bước nhìn nhãn phải là một step, để mỗi CV fold fit lại."""
    train, y = frame(), labels()
    pipe = build_preprocessing_pipeline(select_k=4).fit(train, y)
    assert "select" in dict(pipe.steps)
    assert len(feature_names(pipe)) == 4


def test_scaling_can_be_enabled():
    pipe = build_preprocessing_pipeline(scaling="robust").fit(frame())
    assert not pipe.transform(frame(20, seed=13)).isna().to_numpy().any()


def test_onehot_widens_the_output():
    ordinal = build_preprocessing_pipeline(encoding="ordinal").fit(frame())
    onehot = build_preprocessing_pipeline(encoding="onehot").fit(frame())
    assert len(feature_names(onehot)) > len(feature_names(ordinal))


def test_decorrelation_can_be_skipped():
    pipe = build_preprocessing_pipeline(correlation_threshold=None).fit(frame())
    assert "decorrelate" not in dict(pipe.steps)


def test_protect_keeps_a_column_alive():
    protected = build_preprocessing_pipeline(protect=("CONSTANT_FLAG",)).fit(frame())
    assert "CONSTANT_FLAG" in feature_names(protected)


def test_pipeline_is_a_single_sklearn_object():
    """PLAN §4.4: toàn bộ impute → encode → scale nằm trong MỘT Pipeline."""
    pipe = build_preprocessing_pipeline()
    assert isinstance(pipe, Pipeline)
    assert [name for name, _ in pipe.steps][:4] == [
        "missing", "drop_high_missing", "clip", "encode"]


# ============ Trên dữ liệu thật ============

@pytest.mark.skipif(not loader.resolve(loader.PRIMARY_FILE).exists(),
                    reason="chưa tải dataset Home Credit")
def test_end_to_end_on_home_credit():
    from sklearn.model_selection import train_test_split

    df = loader.load_application_train(nrows=20_000)
    y = df["TARGET"]
    X = df.drop(columns=["TARGET"])
    X_train, X_test, y_train, _ = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    pipe = build_preprocessing_pipeline()
    pipe, train_out = fit_preprocessing(pipe, X_train, y_train)
    test_out = pipe.transform(X_test)

    assert list(train_out.columns) == list(test_out.columns) == feature_names(pipe)
    assert not test_out.isna().to_numpy().any()
    # 121 cột đầu vào rút xuống một bộ gọn hơn hẳn nhưng vẫn đáng kể.
    assert 30 < train_out.shape[1] < X_train.shape[1]


@pytest.mark.skipif(not loader.resolve(loader.PRIMARY_FILE).exists(),
                    reason="chưa tải dataset Home Credit")
def test_redundant_flags_are_dropped_without_losing_information():
    """Home Credit đã có sẵn cột mang đúng thông tin của hai cờ này.

    Bỏ chúng không mất gì; nhưng `protect=` giữ lại được khi cần tên feature
    đọc hiểu được cho SHAP → tầng llm.
    """
    from hfml.data.preprocessing.pipeline import INTERPRETABLE_FLAGS

    df = loader.load_application_train(nrows=30_000)
    X = df.drop(columns=["TARGET"])

    plain = build_preprocessing_pipeline().fit(X, df["TARGET"])
    dropped = dict(plain.steps)["decorrelate"].dropped_because_
    assert "DAYS_EMPLOYED_MISSING" in dropped
    assert "FLAG_EMP_PHONE" in dropped["DAYS_EMPLOYED_MISSING"]

    protected = build_preprocessing_pipeline(protect=INTERPRETABLE_FLAGS).fit(
        X, df["TARGET"])
    assert set(INTERPRETABLE_FLAGS) <= set(feature_names(protected))
    # Giữ cờ thì cột gốc trùng thông tin bị bỏ thay — số feature không đổi.
    assert len(feature_names(protected)) == len(feature_names(plain))


@pytest.mark.skipif(not loader.resolve(loader.PRIMARY_FILE).exists(),
                    reason="chưa tải dataset Home Credit")
def test_trained_model_survives_dump_and_load(tmp_path):
    """Kịch bản thật của F03/F04 task 15: dump pipeline + model chung một file."""
    import joblib
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier

    df = loader.load_application_train(nrows=10_000)
    y = df["TARGET"]
    X = df.drop(columns=["TARGET"])
    X_train, X_test, y_train, _ = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    bundle = Pipeline([
        ("prep", build_preprocessing_pipeline()),
        ("model", DecisionTreeClassifier(max_depth=4, random_state=42)),
    ]).fit(X_train, y_train)

    before = bundle.predict_proba(X_test)
    path = tmp_path / "ml02.joblib"
    joblib.dump(bundle, path)
    after = joblib.load(path).predict_proba(X_test)

    np.testing.assert_array_equal(before, after)
