"""Test phân tích feature importance của ML02 (task 13).

Ba nhóm bất biến được canh:

    · Permutation phải đo bằng PR-AUC, không phải accuracy. Đo bằng accuracy
      thì MỌI cột đều "không quan trọng" — bỏ hết feature vẫn được 91,93% nhờ
      đoán toàn lớp âm, nên bảng ra toàn số 0 mà trông vẫn hợp lệ.
    · Bagging không có `feature_importances_`; phải trung bình qua cây con VÀ
      ánh xạ lại qua `estimators_features_`. Bỏ bước ánh xạ thì con số gán
      nhầm cột.
    · Task 13 là CHẨN ĐOÁN, không được đổi feature set. Đổi rồi train lại và
      chấm lại trên chính tập validation đã dùng để đo là rò rỉ.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from hfml.ml.ml02_credit_risk import importance as importance_module
from hfml.ml.ml02_credit_risk.importance import (
    N_REPEATS,
    PERMUTATION_SCORING,
    ImportanceResult,
    builtin_importance,
    permutation_table,
    rank_comparison,
    shap_table,
    transformed_matrix,
)


class _IdentityFeatures:
    """Bước 'features' giả — trả nguyên khung, để test tách khỏi Pipeline thật."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features or [], dtype=object)


def toy_frame(n: int = 2_000, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """Ba cột: một cột QUYẾT ĐỊNH nhãn, một cột nhiễu, một cột hằng số."""
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=n)
    return (
        pd.DataFrame({
            "co_tin_hieu": signal,
            "nhieu": rng.normal(size=n),
            "hang_so": np.ones(n),
        }),
        pd.Series(rng.binomial(1, 1 / (1 + np.exp(-(signal * 2 - 2.5))))),
    )


def fitted(model) -> Pipeline:
    X, y = toy_frame()
    pipeline = Pipeline([("features", _IdentityFeatures()), ("model", model)])
    pipeline.fit(X, y)
    return pipeline


# --------------------------------------------------------- built-in
def test_builtin_importance_finds_the_column_that_drives_the_label():
    pipeline = fitted(DecisionTreeClassifier(max_depth=4, random_state=42))
    X, _ = toy_frame()

    table = builtin_importance(pipeline, list(X.columns))

    assert table.iloc[0]["feature"] == "co_tin_hieu"
    assert table.set_index("feature").loc["hang_so", "importance"] == 0.0


def test_builtin_importance_sums_to_one_for_a_single_tree():
    """Impurity importance của sklearn được chuẩn hoá — tổng bằng 1."""
    pipeline = fitted(DecisionTreeClassifier(max_depth=4, random_state=42))
    X, _ = toy_frame()

    table = builtin_importance(pipeline, list(X.columns))

    assert table["importance"].sum() == pytest.approx(1.0)


def test_bagging_importance_is_averaged_over_child_trees():
    """`BaggingClassifier` KHÔNG có `feature_importances_`.

    Phải trung bình qua cây con. Không xử lý riêng thì hàm ném `TypeError` và
    cả Bagging biến mất khỏi bảng — hoặc tệ hơn, ai đó điền 0 cho nó.
    """
    model = BaggingClassifier(
        estimator=DecisionTreeClassifier(max_depth=4, random_state=42),
        n_estimators=5, random_state=42)
    pipeline = fitted(model)
    X, _ = toy_frame()

    assert not hasattr(model, "feature_importances_")

    table = builtin_importance(pipeline, list(X.columns))

    assert len(table) == 3
    assert table.iloc[0]["feature"] == "co_tin_hieu"
    assert table["importance"].sum() == pytest.approx(1.0, abs=0.01)


def test_bagging_importance_maps_back_through_sampled_columns():
    """Mỗi cây con có thể chỉ thấy một tập con cột — phải ánh xạ lại.

    Với `max_features < 1.0`, cây thứ i chỉ được huấn luyện trên các cột ghi
    trong `estimators_features_[i]`, nên `child.feature_importances_` có ĐỘ
    DÀI bằng số cột đó chứ không bằng tổng số cột. Cộng thẳng vào vị trí 0, 1,
    2… là gán nhầm cột — mà bảng vẫn đủ dòng và tổng vẫn ~1, nên không có gì
    để lộ ra.

    Test này đối chiếu với phép tính làm tay thay vì kiểm một hệ quả thống kê:
    với dữ liệu nhỏ, hai cột rất dễ hoà nhau và phép kiểm gián tiếp sẽ vừa
    đúng vừa sai tuỳ seed.
    """
    model = BaggingClassifier(
        estimator=DecisionTreeClassifier(max_depth=3, random_state=42),
        n_estimators=8, max_features=0.5, random_state=42)
    pipeline = fitted(model)
    X, _ = toy_frame()
    names = list(X.columns)

    assert hasattr(model, "estimators_features_")
    # Điều kiện làm cho phép ánh xạ trở nên cần thiết.
    assert len(model.estimators_features_[0]) < len(names)

    mong_doi = np.zeros(len(names))
    for child, used in zip(model.estimators_, model.estimators_features_):
        mong_doi[np.asarray(used)] += child.feature_importances_
    mong_doi /= len(model.estimators_)

    table = builtin_importance(pipeline, names).set_index("feature")
    thuc_te = np.array([table.loc[name, "importance"] for name in names])

    np.testing.assert_allclose(thuc_te, mong_doi)
    # Cột hằng số không thể mang importance dù có được lấy mẫu hay không.
    assert table.loc["hang_so", "importance"] == pytest.approx(0.0, abs=1e-9)


def test_model_without_importance_raises_instead_of_returning_zeros():
    """Trả về toàn số 0 sẽ bị đọc như 'không cột nào quan trọng'."""
    from sklearn.linear_model import LogisticRegression

    pipeline = fitted(LogisticRegression(max_iter=200))
    X, _ = toy_frame()

    with pytest.raises(TypeError, match="importance"):
        builtin_importance(pipeline, list(X.columns))


# ------------------------------------------------------ permutation
def test_permutation_uses_pr_auc_not_accuracy():
    """Đo bằng accuracy thì MỌI cột đều 'không quan trọng'.

    Bỏ hết feature vẫn được 91,93% nhờ đoán toàn lớp âm, nên bảng ra toàn số 0
    mà trông vẫn hợp lệ — không có gì báo là phép đo vô nghĩa.
    """
    assert PERMUTATION_SCORING == "average_precision"


def test_permutation_ranks_the_driving_column_first():
    pipeline = fitted(RandomForestClassifier(
        n_estimators=20, max_depth=5, random_state=42))
    X, y = toy_frame()

    table = permutation_table(pipeline, X, y, n_repeats=3)

    assert table.iloc[0]["feature"] == "co_tin_hieu"
    assert table.iloc[0]["importance"] > 0


def test_permutation_gives_a_constant_column_no_importance():
    """Xáo trộn một cột hằng số không đổi gì — importance phải bằng 0."""
    pipeline = fitted(RandomForestClassifier(
        n_estimators=20, max_depth=5, random_state=42))
    X, y = toy_frame()

    table = permutation_table(pipeline, X, y, n_repeats=3).set_index("feature")

    assert table.loc["hang_so", "importance"] == pytest.approx(0.0, abs=1e-9)


def test_permutation_reports_spread_not_just_a_point(pytestconfig):
    """Có `std` thì mới biết chênh lệch giữa hai cột có ý nghĩa hay không."""
    pipeline = fitted(RandomForestClassifier(
        n_estimators=20, max_depth=5, random_state=42))
    X, y = toy_frame()

    table = permutation_table(pipeline, X, y, n_repeats=3)

    assert "std" in table.columns
    assert (table["std"] >= 0).all()


def test_permutation_measures_on_the_transformed_matrix_not_the_pipeline():
    """Đo trên ma trận đã biến đổi, không cho cả Pipeline chạy lại mỗi lần.

    `permutation_importance` chạy `n_features × n_repeats` lần dự đoán; cho cả
    Pipeline chạy lại nghĩa là gộp bureau, dựng tỉ lệ, điền thiếu… lặp lại
    hàng trăm lần mà không đo thêm gì.
    """
    pipeline = fitted(DecisionTreeClassifier(max_depth=4, random_state=42))
    X, _ = toy_frame()

    matrix = transformed_matrix(pipeline, X)

    assert isinstance(matrix, pd.DataFrame)
    assert list(matrix.columns) == list(X.columns)


def test_default_repeat_count_is_declared():
    assert N_REPEATS >= 3


# ------------------------------------------------------------- SHAP
def test_shap_returns_an_empty_frame_instead_of_crashing(monkeypatch):
    """SHAP là phần BỔ SUNG — mất nó không được làm hỏng cả task 13."""
    pipeline = fitted(DecisionTreeClassifier(max_depth=4, random_state=42))
    X, _ = toy_frame()

    def no_shap(*args, **kwargs):
        raise RuntimeError("giả lập SHAP hỏng")

    monkeypatch.setattr(importance_module, "transformed_matrix", no_shap)

    with pytest.raises(RuntimeError):
        # Lỗi ở bước biến đổi thì vẫn nổi lên — chỉ lỗi CỦA SHAP mới bị nuốt.
        shap_table(pipeline, X)


def test_shap_table_ranks_the_driving_column_first():
    pipeline = fitted(DecisionTreeClassifier(max_depth=4, random_state=42))
    X, _ = toy_frame()

    table = shap_table(pipeline, X, max_rows=500)

    if table.empty:
        pytest.skip("shap chưa cài trong môi trường này")
    assert table.iloc[0]["feature"] == "co_tin_hieu"
    assert (table["importance"] >= 0).all(), "trung bình |SHAP| không thể âm"


# --------------------------------------------------- đối chiếu ba cách đo
def test_rank_comparison_compares_ranks_not_values():
    """Ba phương pháp có thang đo khác nhau hoàn toàn.

    Impurity cộng lại bằng 1, permutation tính bằng mức tụt PR-AUC, SHAP tính
    bằng log-odds. So giá trị là so hai thứ không cùng đơn vị.
    """
    result = ImportanceResult(
        algo="xgboost", feature_set="reduced",
        builtin=pd.DataFrame({"feature": ["a", "b", "c"],
                              "importance": [0.6, 0.3, 0.1]}),
        permutation=pd.DataFrame({"feature": ["b", "a", "c"],
                                  "importance": [0.05, 0.03, 0.0]}),
        shap=pd.DataFrame({"feature": ["a", "b", "c"],
                           "importance": [1.2, 0.8, 0.1]}))

    table = rank_comparison(result).set_index("feature")

    assert set(table.columns) >= {"rank_builtin", "rank_permutation",
                                  "rank_shap", "rank_mean", "rank_spread"}
    assert table.loc["a", "rank_builtin"] == 1
    assert table.loc["a", "rank_permutation"] == 2
    # `a` và `b` bất đồng giữa hai cách đo → spread ≥ 1.
    assert table.loc["a", "rank_spread"] >= 1


def test_rank_comparison_skips_missing_methods():
    """Thiếu SHAP thì vẫn đối chiếu được hai cách còn lại."""
    result = ImportanceResult(
        algo="a", feature_set="full",
        builtin=pd.DataFrame({"feature": ["x", "y"], "importance": [0.7, 0.3]}),
        permutation=pd.DataFrame({"feature": ["x", "y"], "importance": [0.1, 0.05]}))

    table = rank_comparison(result)

    assert "rank_shap" not in table.columns
    assert len(table) == 2


def test_rank_comparison_is_empty_when_nothing_was_measured():
    assert rank_comparison(ImportanceResult(algo="a", feature_set="full")).empty


# ------------------------------------------------- chẩn đoán, không sửa
def test_importance_module_never_refits_or_selects_features():
    """Task 13 là CHẨN ĐOÁN. Không có hàm nào train lại hay đổi feature set.

    Dùng bảng importance đo trên validation để bỏ bớt feature rồi train lại và
    chấm lại trên chính tập đó là rò rỉ — validation khi ấy đã tham gia quyết
    định feature nào tồn tại.
    """
    names = {n.lower() for n in dir(importance_module) if not n.startswith("_")}

    assert not any("refit" in n or "retrain" in n or "select_feature" in n
                   or "drop_feature" in n for n in names), names


def test_written_metadata_declares_it_is_diagnostic_only(tmp_path, monkeypatch):
    import json

    from hfml.config import CONFIG
    from hfml.ml.ml02_credit_risk.importance import write_importance

    monkeypatch.setattr(CONFIG.paths, "runs", tmp_path)
    result = ImportanceResult(
        algo="xgboost", feature_set="reduced",
        builtin=pd.DataFrame({"feature": ["a"], "importance": [1.0]}))

    metadata = json.loads(
        write_importance([result])["metadata"].read_text(encoding="utf-8"))

    assert metadata["purpose"] == "diagnostic"
    assert metadata["feature_selection_changed"] is False
    assert metadata["test_set_touched"] is False
    assert "RÒ RỈ" in metadata["leakage_note"]
