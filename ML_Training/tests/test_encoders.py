"""F01 task 10 + 11 — kiểm tra encoding và scaling.

Hai test quan trọng nhất ở đây không kiểm "chạy được", mà kiểm hai khẳng
định mà báo cáo sẽ phải bảo vệ trước hội đồng:

    test_unknown_category_does_not_crash
        Giá trị lạ lúc inference phải ra mã riêng, không ném lỗi (F06 task 1).

    test_scaling_does_not_change_tree_predictions
        Bằng chứng thực nghiệm cho quyết định KHÔNG scale ở task 11.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier

from hfml.data.preprocessing.cleaner import FLAG_SUFFIX
from hfml.data.preprocessing.encoders import (
    MISSING_CODE,
    UNKNOWN_CODE,
    build_categorical_encoder,
    build_categorical_transformer,
    build_numeric_transformer,
    build_scaler,
    classify_columns,
)


def frame() -> pd.DataFrame:
    return pd.DataFrame({
        "SK_ID_CURR": [1, 2, 3, 4],
        "TARGET": [0, 1, 0, 1],
        "AMT_INCOME_TOTAL": [100_000.0, 200_000.0, np.nan, 180_000.0],
        "DAYS_BIRTH": [-12000.0, -15000.0, -9000.0, -20000.0],
        "OCCUPATION_TYPE": ["Laborers", "Managers", None, "Drivers"],
        "CODE_GENDER": ["F", "M", "F", "M"],
        "DAYS_EMPLOYED" + FLAG_SUFFIX: np.array([0, 1, 0, 1], dtype="int8"),
    })


# ------------------------------------------------------- phân loại cột

def test_classify_splits_four_groups():
    g = classify_columns(frame())
    assert g.numeric == ["AMT_INCOME_TOTAL", "DAYS_BIRTH"]
    assert g.categorical == ["OCCUPATION_TYPE", "CODE_GENDER"]
    assert g.flags == ["DAYS_EMPLOYED" + FLAG_SUFFIX]
    assert g.excluded == ["SK_ID_CURR", "TARGET"]


def test_flags_are_not_treated_as_numeric():
    """Cờ đã là 0/1 — encode lại vô nghĩa, scale thì mất ý nghĩa nhị phân."""
    g = classify_columns(frame())
    assert not set(g.flags) & set(g.numeric)


def test_id_and_target_never_become_features():
    g = classify_columns(frame())
    assert "SK_ID_CURR" not in g.features
    assert "TARGET" not in g.features


def test_exclude_is_configurable():
    g = classify_columns(frame(), exclude=("SK_ID_CURR",))
    assert "TARGET" in g.numeric


# ------------------------------------------------- task 10: encoding

def test_ordinal_encodes_without_widening():
    enc = build_categorical_encoder("ordinal")
    cats = frame()[["OCCUPATION_TYPE", "CODE_GENDER"]]
    out = enc.fit_transform(cats)
    assert out.shape == cats.shape           # 2 cột vào, 2 cột ra


def test_missing_gets_its_own_code():
    """Không điền mode — gán cho người ta một nghề họ không làm."""
    enc = build_categorical_encoder("ordinal")
    out = enc.fit_transform(frame()[["OCCUPATION_TYPE"]])
    assert out[2, 0] == MISSING_CODE


def test_unknown_category_does_not_crash():
    """F06 task 1: người dùng gửi nghề lạ, encoder phải trả mã riêng."""
    enc = build_categorical_encoder("ordinal")
    enc.fit(frame()[["OCCUPATION_TYPE"]])
    out = enc.transform(pd.DataFrame({"OCCUPATION_TYPE": ["Phi hành gia"]}))
    assert out[0, 0] == UNKNOWN_CODE


def test_unknown_and_missing_have_different_codes():
    """Cây phải phân biệt được 'chưa từng thấy' với 'không khai'."""
    assert UNKNOWN_CODE != MISSING_CODE


def test_onehot_widens_and_handles_unknown():
    tr = build_categorical_transformer("onehot")
    cats = frame()[["OCCUPATION_TYPE", "CODE_GENDER"]]
    out = tr.fit_transform(cats)
    assert out.shape[1] > cats.shape[1]
    # Giá trị lạ vẫn không được ném lỗi.
    tr.transform(pd.DataFrame({"OCCUPATION_TYPE": ["Phi hành gia"], "CODE_GENDER": ["X"]}))


def test_invalid_strategy_rejected():
    with pytest.raises(ValueError, match="strategy không hợp lệ"):
        build_categorical_encoder("target_encoding")


def test_encoder_learns_categories_from_train_only():
    """Hạng mục phải đến từ train. Fit trên toàn bộ dữ liệu là rò rỉ."""
    enc = build_categorical_encoder("ordinal")
    enc.fit(pd.DataFrame({"c": ["a", "b"]}))
    out = enc.transform(pd.DataFrame({"c": ["a", "b", "z"]}))
    assert out[2, 0] == UNKNOWN_CODE


# -------------------------------------------------- task 11: scaling

def test_default_is_no_scaling():
    assert build_scaler("none") is None


@pytest.mark.parametrize("kind", ["standard", "minmax", "robust"])
def test_scalers_available_when_needed(kind):
    assert build_scaler(kind) is not None


def test_invalid_scaler_rejected():
    with pytest.raises(ValueError, match="kind không hợp lệ"):
        build_scaler("zscore")


def test_numeric_transformer_imputes_with_median():
    """Trung vị chứ không trung bình — ngoại lai kéo trung bình đi rất xa."""
    tr = build_numeric_transformer(scaling="none")
    train = pd.DataFrame({"x": [1.0, 2.0, 3.0, 1_000_000.0]})
    tr.fit(train)
    out = tr.transform(pd.DataFrame({"x": [np.nan]}))
    assert out[0, 0] == 2.5                  # trung vị, không phải 250.001,5


def test_numeric_transformer_can_scale():
    tr = build_numeric_transformer(scaling="standard")
    out = tr.fit_transform(pd.DataFrame({"x": [1.0, 2.0, 3.0]}))
    assert abs(float(np.mean(out))) < 1e-9


def test_scaling_does_not_change_tree_predictions():
    """Bằng chứng cho quyết định không scale ở task 11.

    Cây bất biến với biến đổi đơn điệu: ngưỡng `x ≤ 147.150` và
    `x_scaled ≤ 0,31` cho ra ĐÚNG một phép phân hoạch.
    """
    rng = np.random.default_rng(42)
    n = 400
    X = pd.DataFrame({
        # Hai cột lệch nhau 10^9 lần, giống AMT_INCOME_TOTAL vs REGION_POP.
        "income": rng.lognormal(12, 1, n),
        "region_pop": rng.uniform(0.0003, 0.07, n),
    })
    y = ((X["income"] > X["income"].median()) ^ (X["region_pop"] > 0.03)).astype(int)

    plain = build_numeric_transformer(scaling="none")
    scaled = build_numeric_transformer(scaling="standard")

    pred_plain = DecisionTreeClassifier(random_state=42).fit(
        plain.fit_transform(X), y).predict(plain.transform(X))
    pred_scaled = DecisionTreeClassifier(random_state=42).fit(
        scaled.fit_transform(X), y).predict(scaled.transform(X))

    assert np.array_equal(pred_plain, pred_scaled)


def test_robust_scaler_resists_outliers():
    """Nếu bật scaling thì dùng robust — standard bị ngoại lai kéo lệch."""
    x = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 117_000_000.0]})
    standard = build_scaler("standard").fit_transform(x)
    robust = build_scaler("robust").fit_transform(x)
    # Với standard, 4 giá trị bình thường bị nén sát nhau về ~0.
    assert float(np.ptp(standard[:4])) < 0.01
    assert float(np.ptp(robust[:4])) > 1.0


# ------------------------------------------------------ ghép nhóm

def test_column_transformer_composition():
    """Ba nhánh phải ghép được — đây là bộ khung của task 14."""
    from hfml.data.preprocessing.encoders import PassthroughFlags

    df = frame()
    g = classify_columns(df)
    ct = ColumnTransformer([
        ("num", build_numeric_transformer(), g.numeric),
        ("cat", build_categorical_transformer(), g.categorical),
        ("flag", PassthroughFlags(), g.flags),
    ])
    out = ct.fit_transform(df)
    assert out.shape == (4, len(g.features))

    names = list(ct.get_feature_names_out())
    assert len(names) == len(g.features)


def test_passthrough_flags_keeps_values():
    from hfml.data.preprocessing.encoders import PassthroughFlags

    flags = frame()[["DAYS_EMPLOYED" + FLAG_SUFFIX]]
    out = PassthroughFlags().fit_transform(flags)
    pd.testing.assert_frame_equal(out, flags)
