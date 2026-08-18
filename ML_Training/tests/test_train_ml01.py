"""F03 task 5–15 — kiểm tra vòng train & đánh giá ML01.

Dùng dân số NHỎ (2.000 hộ, 3 fold) cho phần lớn test: những gì cần kiểm ở
đây là *cấu trúc* của vòng train — cùng split, không rò rỉ, chọn đúng model,
export đủ metadata — chứ không phải con số macro-F1 cụ thể. Chạy 20.000 hộ ×
5 fold × 5 thuật toán trong mỗi test thì bộ test hết dùng được.

Chất lượng số thật do ba cổng của PLAN.md §6.2 canh, và cổng đó có test
riêng ở cuối file, chạy trên tham số mặc định.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier

from hfml.config import CONFIG as _CONFIG
from hfml.data.preprocessing.pipeline import build_preprocessing_pipeline
from hfml.data.synthetic import PopulationParams
from hfml.ml.estimator import PipelineClassifier
from hfml.ml.evaluation.metrics import SELECTION_METRIC
from hfml.ml.evaluation.tracking import append_results
from hfml.ml.registry import load_model, save_model
from hfml.ml.ml01_recommendation.labeler import (
    FORBIDDEN_IN_X,
    ORDERED_GROUPS,
    RAW_FEATURES,
)
from hfml.ml.ml01_recommendation import train as train_module
from hfml.ml.ml01_recommendation.train import (
    ALGORITHMS,
    BAGGING,
    CONTENDERS,
    BASELINE,
    DECISION_TREE,
    RANDOM_FOREST,
    XGBOOST,
    GATE_MAX_ACCURACY,
    N_ESTIMATORS,
    build_comparison,
    build_training_data,
    check_gates,
    compare_models,
    evaluate_on_validation,
    FINAL_VERSION,
    evaluate_on_test,
    export_final_model,
    feature_importance_report,
    missing_comparison_rows,
    record_model_selection,
    select_best,
    select_final_model,
    split_train_test,
    split_train_val_test,
    train_bagging,
    train_decision_tree,
    train_random_forest,
    train_xgboost,
    run_full_pipeline,
)

SMALL = PopulationParams(n=2_000)

#: Đọc từ config chứ không viết cứng 0.15 — đổi tỉ lệ ở config mà test vẫn
#: xanh vì hằng số cũ thì test hết canh được gì.
CONFIG_VAL_SIZE = _CONFIG.training["val_size"]

FAST_ALGORITHMS = {
    BASELINE: ALGORITHMS[BASELINE],
    "decision_tree": ALGORITHMS["decision_tree"],
}


@pytest.fixture(scope="module")
def data() -> tuple[pd.DataFrame, pd.Series]:
    return build_training_data(SMALL)


@pytest.fixture(scope="module")
def validation_run(data) -> tuple[pd.DataFrame, dict]:
    """Fit trên train, chấm trên validation — đúng giao thức task 5."""
    X, y = data
    X_train, X_val, _, y_train, y_val, _ = split_train_val_test(X, y)
    return evaluate_on_validation(
        X_train, y_train, X_val, y_val, algorithms=FAST_ALGORITHMS)


# --------------------------------------------------------------- dữ liệu

def test_training_matrix_holds_exactly_the_declared_features(data):
    """`X` phải đúng `RAW_FEATURES`, đúng thứ tự — đây là chốt chặn cuối."""
    X, _ = data
    assert list(X.columns) == list(RAW_FEATURES)


def test_training_matrix_carries_no_label_driver(data):
    X, _ = data
    assert not set(X.columns) & set(FORBIDDEN_IN_X)


def test_labels_are_noisy_not_pristine(data):
    """Nhãn train phải là nhãn ĐÃ thêm nhiễu, nếu không cổng 2 mất tác dụng."""
    from hfml.data.synthetic import generate_households
    from hfml.ml.ml01_recommendation.labeler import label_frame
    _, y = data
    clean = label_frame(generate_households(SMALL))
    assert not y.equals(clean)


# -------------------------------------------- TÁCH TEST (task 5)

def test_holdout_size_follows_config(data):
    from hfml.config import CONFIG
    X, y = data
    X_train, X_val, X_test, _, _, _ = split_train_val_test(X, y)
    assert len(X_test) == pytest.approx(len(X) * CONFIG.training["test_size"], abs=1)
    # KHÔNG bằng len(X): `split_train_test` bỏ qua tập validation, phần thiếu
    # đúng bằng nó. Đây là hợp đồng của hàm, không phải dòng bị mất.
    missing = len(X) - len(X_train) - len(X_test)
    assert missing == pytest.approx(len(X) * CONFIG.training["val_size"], abs=1)


def test_three_way_split_matches_70_15_15(data):
    """Task 5 (chốt lại 12/08/2026): 70% train · 15% validation · 15% test.

    Kiểm tỉ lệ trên tập GỐC chứ không trên phần còn lại. Cắt hai lần mà quên
    quy đổi tỉ lệ lần hai là lỗi im lặng: validation ra 12,75% thay vì 15%,
    tổng vẫn đủ dòng nên không gì báo động.
    """
    from hfml.config import CONFIG
    X, y = data
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(X, y)

    assert len(X_train) + len(X_val) + len(X_test) == len(X)
    assert len(X_val) == pytest.approx(len(X) * CONFIG.training["val_size"], abs=1)
    assert len(X_test) == pytest.approx(len(X) * CONFIG.training["test_size"], abs=1)
    expected_train = 1.0 - CONFIG.training["val_size"] - CONFIG.training["test_size"]
    assert len(X_train) == pytest.approx(len(X) * expected_train, abs=2)

    # Ba tập rời nhau từng đôi một.
    assert not set(X_train.index) & set(X_val.index)
    assert not set(X_train.index) & set(X_test.index)
    assert not set(X_val.index) & set(X_test.index)
    assert len(y_train) == len(X_train)
    assert len(y_val) == len(X_val)
    assert len(y_test) == len(X_test)


def test_three_way_split_is_stratified_in_every_part(data):
    """Lớp nhỏ nhất ~15%; cắt hai lần thì lệch tỉ lệ dễ tích luỹ ở tập thứ ba."""
    X, y = data
    _, X_val, _, y_train, y_val, y_test = split_train_val_test(X, y)
    overall = y.value_counts(normalize=True)
    for label, share in overall.items():
        for part in (y_train, y_val, y_test):
            assert part.value_counts(normalize=True)[label] == pytest.approx(
                share, abs=0.03)


def test_split_train_test_reuses_the_same_training_rows(data):
    """Hai hàm chia phải cho CÙNG tập train khi cùng seed.

    Nếu lệch thì chỉ số CV đo được sẽ phụ thuộc vào việc gọi hàm nào — cùng
    một model ra hai con số khác nhau mà không ai giải thích được vì sao.
    """
    X, y = data
    X_train_two, X_test_two, _, _ = split_train_test(X, y, seed=42)
    X_train_three, _, X_test_three, _, _, _ = split_train_val_test(X, y, seed=42)
    assert list(X_train_two.index) == list(X_train_three.index)
    assert list(X_test_two.index) == list(X_test_three.index)


def test_holdout_is_stratified(data):
    """Lớp nhỏ nhất chỉ ~15% — cắt ngẫu nhiên thì tỉ lệ train/test lệch nhau,
    và chỉ số test không còn so được với chỉ số CV."""
    X, y = data
    _, _, y_train, y_test = split_train_test(X, y)
    overall = y.value_counts(normalize=True)
    for label, share in overall.items():
        assert y_train.value_counts(normalize=True)[label] == pytest.approx(share, abs=0.02)
        assert y_test.value_counts(normalize=True)[label] == pytest.approx(share, abs=0.02)


def test_pipeline_does_not_use_k_fold_cross_validation():
    """Đặc tả chốt lại 14/08/2026: ML01 KHÔNG dùng K-Fold Cross-Validation.

    Kiểm ở hai chỗ mà CV từng bám vào — API của module và khoá config. Test
    này tồn tại để một lần "tiện tay thêm CV lại cho chắc" bị chặn ngay, chứ
    không phải để kiểm một hành vi: hành vi thì các test khác đã canh.
    """
    from hfml.config import CONFIG

    for removed in ("cross_validate", "_materialise_folds"):
        assert not hasattr(train_module, removed), (
            f"{removed} đã bị bỏ theo phương pháp mới, không được đưa lại")
    assert "n_splits" not in CONFIG.training


def test_train_and_test_never_overlap(data):
    X, y = data
    X_train, X_test, _, _ = split_train_test(X, y)
    assert not set(X_train.index) & set(X_test.index)


def test_split_is_reproducible(data):
    X, y = data
    first = split_train_test(X, y, seed=42)[1]
    second = split_train_test(X, y, seed=42)[1]
    other = split_train_test(X, y, seed=7)[1]
    assert list(first.index) == list(second.index)
    assert list(first.index) != list(other.index)


@pytest.mark.slow
def test_model_selection_never_sees_the_test_set(tmp_path, monkeypatch):
    """Cấu trúc phải đảm bảo test không tham gia bất kỳ quyết định nào.

    Bắt lấy đúng hai khung dữ liệu mà `evaluate_on_validation` nhận — tập fit
    và tập chấm — rồi đối chiếu cả hai với tập test. Giao khác rỗng nghĩa là
    model được chọn bằng chính dữ liệu dùng để báo cáo, và con số báo cáo mất
    giá trị.
    """
    seen: dict[str, pd.Index] = {}
    original = train_module.evaluate_on_validation

    def spy(X_train, y_train, X_val, y_val, **kwargs):
        seen["train"] = X_train.index
        seen["validation"] = X_val.index
        return original(X_train, y_train, X_val, y_val, **kwargs)

    monkeypatch.setattr(train_module, "evaluate_on_validation", spy)
    result = run_full_pipeline(SMALL, runs_dir=tmp_path)

    test_index = set(result["X_test"].index)
    assert not set(seen["train"]) & test_index
    assert not set(seen["validation"]) & test_index
    assert len(seen["train"]) == len(result["X_train"])
    assert len(seen["validation"]) == len(result["X_val"])


# ------------------------------------------------- CÙNG SPLIT, KHÔNG RÒ RỈ

def test_every_algorithm_sees_the_same_split(data):
    """Điều kiện để bảng so sánh có nghĩa (PLAN.md §6.3).

    Mỗi thuật toán tự chia dữ liệu thì chênh lệch giữa chúng lẫn với chênh
    lệch giữa các cách chia, và không tách ra được nữa. Ở đây điều đó được
    bảo đảm bằng việc `split_train_val_test()` với cùng seed luôn cho đúng
    cùng ba tập.
    """
    X, y = data
    first = split_train_val_test(X, y, seed=42)
    second = split_train_val_test(X, y, seed=42)
    for a, b in zip(first, second):
        assert list(a.index) == list(b.index)


def test_validation_predictions_use_the_same_validation_rows(data):
    """Bốn thuật toán phải được chấm trên ĐÚNG cùng tập validation."""
    X, y = data
    X_train, X_val, _, y_train, y_val, _ = split_train_val_test(X, y)
    _, predictions = evaluate_on_validation(
        X_train, y_train, X_val, y_val, algorithms=FAST_ALGORITHMS)

    for algo, predicted in predictions.items():
        assert list(predicted.index) == list(X_val.index), algo


def test_preprocessing_is_refit_per_model(data):
    """Pipeline phải fit lại cho mỗi model — fit một lần rồi dùng chung là rò rỉ.

    `PipelineClassifier.fit` gọi `clone()`, nên object gốc không bao giờ được
    fit. Nếu ai đó bỏ `clone` đi thì `preprocessing` gốc sẽ có thuộc tính đã
    học và test này đỏ.
    """
    X, y = data
    preprocessing = build_preprocessing_pipeline()
    model = PipelineClassifier(task="ml01", algo="decision_tree",
                               estimator=DecisionTreeClassifier(random_state=42),
                               preprocessing=preprocessing)
    model.fit(X, y)
    assert not hasattr(preprocessing.named_steps["nzv"], "columns_to_drop_")


def test_clone_keeps_pandas_output():
    """Các bước lọc tự viết cần TÊN CỘT; mất `set_output` là chúng gãy."""
    pipeline = build_preprocessing_pipeline()
    for _, step in clone(pipeline).steps:
        assert getattr(step, "_sklearn_output_config", None) == {"transform": "pandas"}


# ------------------------------------------------------------ estimator

def test_predict_returns_string_labels_for_every_algorithm(data):
    """XGBoost bên trong chạy bằng số, bên ngoài vẫn phải trả chuỗi."""
    X, y = data
    small = X.head(400)
    for algo, factory in ALGORITHMS.items():
        model = PipelineClassifier(task="ml01", algo=algo, estimator=factory(42),
                                   preprocessing=build_preprocessing_pipeline())
        model.fit(small, y.head(400))
        predicted = model.predict(small.head(10))
        assert set(predicted) <= set(model.classes_), algo


def test_predict_proba_columns_align_with_classes(data):
    X, y = data
    model = PipelineClassifier(task="ml01", algo="decision_tree",
                               estimator=DecisionTreeClassifier(random_state=42),
                               preprocessing=build_preprocessing_pipeline())
    model.fit(X, y)
    proba = model.predict_proba(X.head(50))
    assert proba.shape == (50, len(model.classes_))
    assert np.allclose(proba.sum(axis=1), 1.0)
    # Lớp có xác suất cao nhất phải khớp `predict` — lệch nghĩa là cột bị hoán vị.
    argmax = np.asarray(model.classes_)[proba.argmax(axis=1)]
    assert list(argmax) == list(model.predict(X.head(50)))


def test_column_order_is_restored_at_inference(data):
    """Đảo thứ tự cột lúc inference không được làm đổi kết quả."""
    X, y = data
    model = PipelineClassifier(task="ml01", algo="decision_tree",
                               estimator=DecisionTreeClassifier(random_state=42),
                               preprocessing=build_preprocessing_pipeline())
    model.fit(X, y)
    shuffled = X.head(50)[list(reversed(RAW_FEATURES))]
    assert list(model.predict(shuffled)) == list(model.predict(X.head(50)))


def test_missing_column_raises_a_named_error(data):
    X, y = data
    model = PipelineClassifier(task="ml01", algo="decision_tree",
                               estimator=DecisionTreeClassifier(random_state=42),
                               preprocessing=build_preprocessing_pipeline())
    model.fit(X, y)
    with pytest.raises(ValueError, match="thiếu 1 cột"):
        model.predict(X.head(5).drop(columns=["age"]))


def test_slug_follows_naming_convention(data):
    X, y = data
    model = PipelineClassifier(task="ml01", algo="xgboost",
                               estimator=ALGORITHMS["xgboost"](42),
                               preprocessing=build_preprocessing_pipeline())
    assert model.slug == "ml01_xgboost_v1"


def test_feature_importance_matches_columns_after_preprocessing(data):
    """Ghép importance với danh sách cột ĐẦU VÀO là lệch chỉ số nếu có cột bị bỏ."""
    X, y = data
    model = PipelineClassifier(task="ml01", algo="decision_tree",
                               estimator=DecisionTreeClassifier(random_state=42),
                               preprocessing=build_preprocessing_pipeline())
    model.fit(X, y)
    importance = model.feature_importance()
    assert list(importance["feature"]) == sorted(model.transformed_feature_names_,
                                                 key=lambda f: -float(
                                                     importance.set_index("feature")
                                                     .loc[f, "importance"]))
    assert importance["importance"].sum() == pytest.approx(1.0)


def test_dummy_has_no_feature_importance(data):
    X, y = data
    model = PipelineClassifier(task="ml01", algo=BASELINE,
                               estimator=DummyClassifier(strategy="stratified"),
                               preprocessing=build_preprocessing_pipeline())
    model.fit(X, y)
    with pytest.raises(AttributeError):
        model.feature_importance()


# ------------------------------------------------------- bảng & chọn model

def test_comparison_table_has_one_row_per_algorithm(validation_run):
    comparison, _ = validation_run
    assert list(comparison["algo"]) == list(FAST_ALGORITHMS)
    for column in ("accuracy", "macro_f1", "fit_seconds"):
        assert column in comparison
    # Không còn độ lệch giữa fold: mỗi chỉ số là một điểm đo trên validation.
    assert not [c for c in comparison.columns if c.endswith("_std")]


def test_validation_predictions_cover_every_validation_row(validation_run, data):
    """Dự đoán phải phủ hết tập validation — chỗ trống nghĩa là có dòng bị bỏ."""
    X, y = data
    _, _, _, _, y_val, _ = split_train_val_test(X, y)
    _, predictions = validation_run
    for algo, predicted in predictions.items():
        assert len(predicted) == len(y_val), algo
        assert not predicted.isna().any(), algo


def test_selection_uses_macro_f1_not_accuracy():
    """ML01 chọn model bằng macro-F1 (PLAN.md §11).

    Bảng dưới đây dựng để accuracy và macro-F1 chỉ sang hai model khác nhau —
    nếu code lỡ chọn theo accuracy thì test này đỏ.
    """
    assert SELECTION_METRIC["ml01"] == "macro_f1"
    comparison = pd.DataFrame([
        {"algo": BASELINE, "accuracy": 0.20, "macro_f1": 0.20},
        {"algo": "decision_tree", "accuracy": 0.95, "macro_f1": 0.70},
        {"algo": "xgboost", "accuracy": 0.90, "macro_f1": 0.88},
    ])
    assert select_best(comparison) == "xgboost"


# ------------------------------------------ DECISION TREE (task 7)

@pytest.fixture(scope="module")
def decision_tree_run(tmp_path_factory) -> dict:
    """Ghi output vào thư mục tạm — `save` nay mặc định bật, để nguyên thì
    test sẽ đè lên artifact thật trong `src/training/runs/`."""
    return train_decision_tree(SMALL,
                               runs_dir=tmp_path_factory.mktemp("runs"))


def test_decision_tree_fits_on_train_and_scores_validation(decision_tree_run, data):
    """Fit trên train, chấm trên validation — test không dòng nào lọt vào."""
    X, y = data
    X_train, X_val, X_test, _, _, _ = split_train_val_test(X, y)
    assert len(decision_tree_run["validation_predictions"]) == len(X_val)
    assert not set(decision_tree_run["X_train"].index) & set(X_test.index)
    assert not set(decision_tree_run["X_train"].index) & set(X_val.index)


def test_decision_tree_fits_final_model_on_the_full_training_set(decision_tree_run):
    """Cây cuối học từ 5/5 tập train, không phải 4/5 như từng cây trong CV.

    `tree_.n_node_samples[0]` là số mẫu ở nút gốc — chính là số dòng cây đã
    thấy lúc fit, nên nó phân biệt được "fit trên toàn bộ train" với "fit
    trên một fold".
    """
    model = decision_tree_run["model"]
    root_samples = model.pipeline_.named_steps["model"].tree_.n_node_samples[0]
    assert int(root_samples) == len(decision_tree_run["X_train"])


def test_decision_tree_run_does_not_expose_the_test_set(decision_tree_run):
    """Task 7 không được dùng tập test — cách chắc nhất là không trả nó ra."""
    assert not [key for key in decision_tree_run if "test" in key]


def test_decision_tree_writes_its_artifact_to_the_runs_directory(decision_tree_run):
    """Cây đơn phải có artifact như ba thuật toán kia.

    Trước đây nó là thuật toán DUY NHẤT không ghi output, nên mọi bước sau
    muốn dùng lại nó đều phải fit lại từ đầu.
    """
    artifact = decision_tree_run["artifact"]
    assert artifact.exists() and artifact.suffix == ".joblib"
    assert artifact.stem == f"ml01_{DECISION_TREE}_v1"


def test_decision_tree_logs_its_cv_row_to_results_csv(decision_tree_run):
    """Cây đơn phải ghi dòng CV như ba thuật toán kia.

    Trước đây nó ghi artifact nhưng KHÔNG ghi vào `results.csv`, nên dòng CV
    của nó chỉ xuất hiện khi task 12 backfill. Hỏng im lặng: đổi cấu hình
    split rồi train lại, ba model kia có dòng mới còn cây đơn vẫn giữ dòng
    backfill cũ — bảng so sánh trộn hai cỡ dữ liệu mà không báo gì.
    """
    results_csv = decision_tree_run["results_csv"]
    assert results_csv.exists()
    assert decision_tree_run["artifact"].parent == results_csv.parent

    saved = pd.read_csv(results_csv)
    rows = saved[(saved["algo"] == DECISION_TREE) & (saved["split"] == "validation")]
    assert len(rows) >= 1
    # `n_rows` là cỡ tập VALIDATION — nơi chỉ số được chấm.
    assert rows.iloc[-1]["n_rows"] == pytest.approx(
        SMALL.n * CONFIG_VAL_SIZE, abs=1)


def test_decision_tree_metadata_records_metrics_and_configuration(decision_tree_run):
    artifact = decision_tree_run["artifact"]
    metadata = json.loads(
        artifact.with_name(f"{artifact.stem}.metadata.json").read_text(encoding="utf-8"))
    assert metadata["algo"] == DECISION_TREE
    assert "macro_f1" in metadata["metrics"]["validation"]
    assert metadata["feature_names"] == list(RAW_FEATURES)
    assert metadata["config"]["estimator"] == "DecisionTreeClassifier"


def test_decision_tree_config_separates_pruning_limit_from_actual_depth(decision_tree_run):
    """`max_depth=None` và độ sâu THỰC TẾ nói hai chuyện khác nhau.

    Khoảng cách giữa chúng chính là bằng chứng cây đơn mọc tự do tới đâu —
    vai trò PLAN.md §6.3 giao cho nó trong báo cáo.
    """
    config = decision_tree_run["config"]
    assert config["max_depth"] is None            # không cắt tỉa
    assert config["fitted_depth"] > 1             # nhưng cây thật thì sâu
    assert config["n_leaves"] < config["n_nodes"]


def test_decision_tree_artifact_reloads_and_predicts(decision_tree_run):
    artifact = decision_tree_run["artifact"]
    reloaded = load_model(artifact.stem, directory=artifact.parent)
    sample = decision_tree_run["X_train"].head(20)
    predicted = reloaded.predict(sample)
    assert set(predicted) <= set(reloaded.classes_)
    assert list(predicted) == list(decision_tree_run["model"].predict(sample))


def test_decision_tree_can_skip_writing_outputs():
    result = train_decision_tree(SMALL, save=False)
    assert "artifact" not in result
    assert "results_csv" not in result


def test_decision_tree_reports_validation_metrics(decision_tree_run):
    metrics = decision_tree_run["validation_metrics"]
    assert metrics["algo"] == DECISION_TREE
    for key in ("accuracy", "macro_f1", "balanced_accuracy"):
        assert key in metrics
    # Bỏ K-Fold nên không còn độ lệch giữa các fold.
    assert not [k for k in metrics if str(k).endswith("_std")]


def test_decision_tree_is_reproducible():
    """Cùng seed cho cùng cây — điều kiện của F06 task 6."""
    first = train_decision_tree(SMALL, save=False)
    second = train_decision_tree(SMALL, save=False)
    assert first["validation_metrics"]["macro_f1"] == second["validation_metrics"]["macro_f1"]
    assert list(first["validation_predictions"]) == list(second["validation_predictions"])


# ----------------------------------------------- BAGGING (task 8)

@pytest.fixture(scope="module")
def bagging_run(tmp_path_factory) -> dict:
    """Chạy có ghi output, nhưng vào thư mục tạm — không đụng runs/ thật."""
    return train_bagging(SMALL,
                         runs_dir=tmp_path_factory.mktemp("runs"))


def test_bagging_fits_on_train_and_scores_validation(bagging_run, data):
    X, y = data
    X_train, X_val, X_test, _, _, _ = split_train_val_test(X, y)
    assert len(bagging_run["validation_predictions"]) == len(X_val)
    assert not set(bagging_run["X_train"].index) & set(X_test.index)
    assert not set(bagging_run["X_train"].index) & set(X_val.index)


def test_bagging_fits_final_model_on_the_full_training_set(bagging_run):
    """Model cuối học từ 5/5 tập train, không phải 4/5 như từng model trong CV.

    Mỗi cây con của Bagging lấy mẫu bootstrap cùng cỡ với tập được fit, nên
    `_n_samples` phản ánh đúng số dòng model đã thấy.
    """
    estimator = bagging_run["model"].pipeline_.named_steps["model"]
    assert estimator._n_samples == len(bagging_run["X_train"])


def test_bagging_run_does_not_expose_the_test_set(bagging_run):
    assert not [key for key in bagging_run if "test" in key]


def test_bagging_uses_one_hundred_trees(bagging_run):
    assert bagging_run["config"]["n_estimators"] == N_ESTIMATORS
    assert bagging_run["config"]["bootstrap"] is True


def test_bagging_writes_outputs_to_the_runs_directory(bagging_run):
    """Quy ước kiến trúc: artifact + kết quả nằm ở `src/training/runs/`."""
    artifact = bagging_run["artifact"]
    assert artifact.exists() and artifact.suffix == ".joblib"
    assert bagging_run["results_csv"].exists()
    assert artifact.parent == bagging_run["results_csv"].parent


def test_bagging_metadata_records_metrics_and_configuration(bagging_run):
    """Biết macro-F1 mà không biết chạy với mấy cây thì con số không dựng lại được."""
    artifact = bagging_run["artifact"]
    metadata = json.loads(
        artifact.with_name(f"{artifact.stem}.metadata.json").read_text(encoding="utf-8"))
    assert metadata["algo"] == BAGGING
    assert "macro_f1" in metadata["metrics"]["validation"]
    assert metadata["config"]["n_estimators"] == N_ESTIMATORS
    assert metadata["feature_names"] == list(RAW_FEATURES)


def test_bagging_runs_directory_holds_no_log_file(bagging_run):
    """`src/training/runs/` chứa kết quả, KHÔNG chứa log."""
    written = list(bagging_run["artifact"].parent.iterdir())
    assert not [p for p in written if p.suffix == ".log"]


def test_bagging_can_skip_writing_outputs():
    result = train_bagging(SMALL, save=False)
    assert "artifact" not in result
    assert "results_csv" not in result


# ------------------------------------------ RANDOM FOREST (task 9)

@pytest.fixture(scope="module")
def random_forest_run(tmp_path_factory) -> dict:
    return train_random_forest(SMALL,
                               runs_dir=tmp_path_factory.mktemp("runs"))


def test_random_forest_fits_on_train_and_scores_validation(random_forest_run, data):
    X, y = data
    X_train, X_val, X_test, _, _, _ = split_train_val_test(X, y)
    assert len(random_forest_run["validation_predictions"]) == len(X_val)
    assert not set(random_forest_run["X_train"].index) & set(X_test.index)
    assert not set(random_forest_run["X_train"].index) & set(X_val.index)


def test_random_forest_fits_final_model_on_the_full_training_set(random_forest_run):
    """Rừng cuối học từ 5/5 tập train, không phải 4/5 như từng rừng trong CV.

    Phải đọc `weighted_n_node_samples`, KHÔNG phải `n_node_samples`. Với
    `bootstrap=True`, sklearn không nhân bản dòng mà gán số lần rút thành
    trọng số, nên `n_node_samples[0]` chỉ đếm dòng PHÂN BIỆT — khoảng
    (1 − 1/e) ≈ 63,2% cỡ tập train. Tổng trọng số mới bằng đúng số dòng.
    """
    estimator = random_forest_run["model"].pipeline_.named_steps["model"]
    n_train = len(random_forest_run["X_train"])
    tree = estimator.estimators_[0].tree_

    assert float(tree.weighted_n_node_samples[0]) == pytest.approx(n_train)
    # Và đúng là bootstrap chứ không phải học trên toàn bộ dòng phân biệt.
    assert int(tree.n_node_samples[0]) < n_train


def test_random_forest_run_does_not_expose_the_test_set(random_forest_run):
    assert not [key for key in random_forest_run if "test" in key]


def test_random_forest_grows_one_hundred_trees(random_forest_run):
    assert random_forest_run["config"]["n_estimators"] == N_ESTIMATORS


def test_random_forest_config_records_resolved_max_features(random_forest_run):
    """`max_features` là tham số phân biệt RF với Bagging — bỏ khỏi bản ghi
    thì lần chạy không dựng lại được.

    Ghi cả giá trị đã quy đổi: `'sqrt'` của 17 cột là 4, và 4 mới là con số
    đọc hiểu được.
    """
    config = random_forest_run["config"]
    assert config["max_features"] == "sqrt"
    assert config["n_features_in"] == len(RAW_FEATURES)
    assert config["max_features_resolved"] == int(len(RAW_FEATURES) ** 0.5)


def test_random_forest_writes_outputs_to_the_runs_directory(random_forest_run):
    artifact = random_forest_run["artifact"]
    assert artifact.exists() and artifact.suffix == ".joblib"
    assert random_forest_run["results_csv"].exists()
    assert artifact.parent == random_forest_run["results_csv"].parent


def test_random_forest_metadata_records_metrics_and_configuration(random_forest_run):
    artifact = random_forest_run["artifact"]
    metadata = json.loads(
        artifact.with_name(f"{artifact.stem}.metadata.json").read_text(encoding="utf-8"))
    assert metadata["algo"] == RANDOM_FOREST
    assert "macro_f1" in metadata["metrics"]["validation"]
    assert metadata["config"]["max_features"] == "sqrt"
    assert metadata["feature_names"] == list(RAW_FEATURES)


def test_random_forest_can_skip_writing_outputs():
    result = train_random_forest(SMALL, save=False)
    assert "artifact" not in result
    assert "results_csv" not in result


# ---------------------------------------------- XGBOOST (task 10)

@pytest.fixture(scope="module")
def xgboost_run(tmp_path_factory) -> dict:
    return train_xgboost(SMALL,
                         runs_dir=tmp_path_factory.mktemp("runs"))


def test_xgboost_fits_on_train_and_scores_validation(xgboost_run, data):
    X, y = data
    X_train, X_val, X_test, _, _, _ = split_train_val_test(X, y)
    assert len(xgboost_run["validation_predictions"]) == len(X_val)
    assert not set(xgboost_run["X_train"].index) & set(X_test.index)
    assert not set(xgboost_run["X_train"].index) & set(X_val.index)


def test_xgboost_fits_final_model_on_the_full_training_set(monkeypatch):
    """Model cuối phải học từ 5/5 tập train, không phải 4/5 như trong CV.

    Không kiểm được qua thuộc tính của estimator như task 8–9: XGBoost không
    phơi ra số dòng đã train, và `Cover` ở nút gốc thì phụ thuộc phân bố lớp
    nên không quy ngược ra số dòng.

    Nên kiểm ở chỗ chắc chắn hơn: ghi lại cỡ dữ liệu của MỌI lần `fit`. Các
    lần trong CV phải là 4/5 tập train, lần cuối cùng phải là 5/5. Cách này
    không phụ thuộc thư viện nên còn đúng khi đổi estimator.
    """
    sizes: list[int] = []
    original = PipelineClassifier.fit

    def spy(self, X, y):
        sizes.append(len(X))
        return original(self, X, y)

    monkeypatch.setattr(PipelineClassifier, "fit", spy)
    result = train_xgboost(SMALL, save=False)

    n_train = len(result["X_train"])
    # Không còn K-Fold: mọi lần fit đều trên TOÀN bộ tập train, không
    # còn lần nào chạy trên 4/5 như thời CV 5-fold.
    assert sizes, "phải có ít nhất một lần fit"
    assert all(size == n_train for size in sizes)


def test_xgboost_run_does_not_expose_the_test_set(xgboost_run):
    assert not [key for key in xgboost_run if "test" in key]


def test_xgboost_config_records_every_tuning_knob(xgboost_run):
    """`learning_rate` × `n_estimators` bù trừ nhau, `subsample`/
    `colsample_bytree` thêm ngẫu nhiên — thiếu một cái là không dựng lại được."""
    config = xgboost_run["config"]
    for key in ("n_estimators", "max_depth", "learning_rate",
                "subsample", "colsample_bytree", "tree_method", "objective"):
        assert key in config, key


def test_xgboost_builds_one_tree_per_class_per_round(xgboost_run):
    """`multi:softprob` dựng một cây cho MỖI LỚP ở mỗi vòng.

    300 vòng × 4 lớp = 1.200 cây. Ghi rõ để ai đọc bảng so sánh không đặt
    "300" cạnh "100 cây" của Bagging/RF rồi hiểu sai độ phức tạp.
    """
    config = xgboost_run["config"]
    assert config["objective"] == "multi:softprob"
    assert config["n_trees"] == config["n_estimators"] * config["n_classes"]

    booster = xgboost_run["model"].pipeline_.named_steps["model"].get_booster()
    assert booster.trees_to_dataframe()["Tree"].nunique() == config["n_trees"]


def test_xgboost_writes_outputs_to_the_runs_directory(xgboost_run):
    artifact = xgboost_run["artifact"]
    assert artifact.exists() and artifact.suffix == ".joblib"
    assert xgboost_run["results_csv"].exists()
    assert artifact.parent == xgboost_run["results_csv"].parent


def test_xgboost_metadata_records_metrics_and_configuration(xgboost_run):
    artifact = xgboost_run["artifact"]
    metadata = json.loads(
        artifact.with_name(f"{artifact.stem}.metadata.json").read_text(encoding="utf-8"))
    assert metadata["algo"] == XGBOOST
    assert "macro_f1" in metadata["metrics"]["validation"]
    assert metadata["config"]["objective"] == "multi:softprob"
    assert metadata["feature_names"] == list(RAW_FEATURES)


def test_xgboost_predicts_string_labels_after_reload(xgboost_run, tmp_path):
    """XGBoost chạy bằng nhãn số bên trong; artifact tải lại vẫn phải trả chuỗi."""
    import joblib
    reloaded = joblib.load(xgboost_run["artifact"])
    predicted = reloaded.predict(xgboost_run["X_train"].head(20))
    assert set(predicted) <= set(reloaded.classes_)


def test_xgboost_can_skip_writing_outputs():
    result = train_xgboost(SMALL, save=False)
    assert "artifact" not in result
    assert "results_csv" not in result


# --------------------------------- ĐÁNH GIÁ TRÊN TEST (task 11)

@pytest.fixture(scope="module")
def test_evaluation(tmp_path_factory) -> dict:
    return evaluate_on_test(SMALL, runs_dir=tmp_path_factory.mktemp("runs"))


def test_evaluation_covers_all_four_contenders(test_evaluation):
    assert tuple(test_evaluation["summary"]["algo"]) == CONTENDERS
    assert BASELINE not in test_evaluation["per_algo"]


def test_all_models_are_scored_on_the_same_test_set(test_evaluation, data):
    """Cùng một tập test cho cả bốn — nếu khác thì chênh lệch giữa các model
    là chênh lệch của dữ liệu, không phải của model."""
    X, y = data
    _, X_test, _, y_test = split_train_test(X, y)
    assert list(test_evaluation["X_test"].index) == list(X_test.index)
    assert list(test_evaluation["y_test"]) == list(y_test)

    # Mọi bảng per-class phải cùng support — bằng chứng cùng tập test.
    supports = {algo: tuple(result["per_class"]["support"])
                for algo, result in test_evaluation["per_algo"].items()}
    assert len(set(supports.values())) == 1


def test_evaluation_never_touches_the_test_set_during_fitting(monkeypatch):
    """Test chỉ được dùng để CHẤM, không được lọt vào lúc fit."""
    fitted_on: list[int] = []
    original = PipelineClassifier.fit

    def spy(self, X, y):
        fitted_on.append(len(X))
        return original(self, X, y)

    monkeypatch.setattr(PipelineClassifier, "fit", spy)
    result = evaluate_on_test(SMALL, algorithms=(DECISION_TREE,), save=False)

    n_test = len(result["X_test"])
    n_val = round(SMALL.n * CONFIG_VAL_SIZE)
    # Đúng một lần fit, và chỉ trên tập train 70% — cả validation lẫn test đều
    # nằm ngoài. Trừ đi cả hai chứ không riêng test: nếu validation lọt vào
    # lúc fit thì tập train phình ra và chỉ số test không còn so được với chỉ
    # số validation mà bốn hàm train đã ghi.
    assert fitted_on == [SMALL.n - n_val - n_test]


def test_summary_reports_every_metric_task11_asks_for(test_evaluation):
    """Accuracy · Precision · Recall · F1 · Macro-F1 · Balanced Accuracy."""
    columns = set(test_evaluation["summary"].columns)
    assert {"accuracy", "balanced_accuracy", "macro_precision", "macro_recall",
            "macro_f1", "weighted_f1"} <= columns


def test_per_class_table_covers_every_group(test_evaluation):
    for algo, result in test_evaluation["per_algo"].items():
        table = result["per_class"]
        assert list(table["label"]) == test_evaluation["labels"], algo
        assert set(table.columns) >= {"precision", "recall", "f1", "support"}


def test_confusion_matrix_totals_match_the_test_set(test_evaluation):
    """Tổng confusion matrix phải bằng cỡ tập test — thiếu ô nghĩa là mất dòng."""
    n_test = len(test_evaluation["y_test"])
    for algo, result in test_evaluation["per_algo"].items():
        assert int(result["confusion"].to_numpy().sum()) == n_test, algo


def test_macro_recall_equals_balanced_accuracy(test_evaluation):
    """Kiểm chéo bằng một đẳng thức: balanced accuracy CHÍNH LÀ macro recall.

    Hai con số này đến từ hai hàm khác nhau của sklearn, nên chúng khớp nhau
    là bằng chứng cả hai được tính trên cùng bộ nhãn và cùng thứ tự lớp.
    """
    for algo, result in test_evaluation["per_algo"].items():
        metrics = result["metrics"]
        assert metrics["macro_recall"] == pytest.approx(
            metrics["balanced_accuracy"], abs=1e-9), algo


def test_evaluation_does_not_rank_or_select(test_evaluation):
    """Task 11 chỉ ĐO. Xếp hạng, chọn model, export thuộc task sau."""
    assert not [key for key in test_evaluation
                if key in {"best", "best_algo", "ranking", "artifact"}]


def test_evaluation_writes_results_to_the_runs_directory(test_evaluation):
    files = test_evaluation["files"]
    assert files["results"].exists()
    assert files["per_class"].exists()
    assert files["confusion"].exists()
    assert files["results"].parent == files["per_class"].parent

    saved = pd.read_csv(files["results"])
    # Task 11 CHỈ ghi dòng test. Chỉ số validation là của bốn hàm `train_*`;
    # ghi thêm ở đây sẽ tạo dòng validation thứ hai cho cùng thuật toán.
    assert set(saved["split"]) == {"test"}
    assert len(saved) == len(CONTENDERS)
    assert set(saved["algo"]) == set(CONTENDERS)


def test_saved_per_class_covers_every_algorithm(test_evaluation):
    saved = pd.read_csv(test_evaluation["files"]["per_class"])
    assert set(saved["algo"]) == set(CONTENDERS)
    assert len(saved) == len(CONTENDERS) * len(test_evaluation["labels"])


def test_evaluation_can_skip_writing_outputs():
    result = evaluate_on_test(SMALL, algorithms=(DECISION_TREE,), save=False)
    assert "files" not in result


# ------------------------------------- SO SÁNH MODEL (task 12)

def _results_frame(**overrides) -> pd.DataFrame:
    """Bảng results.csv giả lập, đủ cột để `build_comparison` chạy."""
    rows = []
    for algo in CONTENDERS:
        for split, base in (("validation", 0.90), ("test", 0.88)):
            rows.append({
                "algo": algo, "split": split,
                "accuracy": base, "macro_f1": base - 0.01,
                "balanced_accuracy": base - 0.02,
                "fit_seconds": 1.0,
                **overrides,
            })
    return pd.DataFrame(rows)


def test_comparison_covers_all_four_contenders():
    comparison = build_comparison(_results_frame())
    assert list(comparison["algo"]) == list(CONTENDERS)


def test_gap_is_validation_minus_test():
    """Chiều của `gap` phải là CV − test: số DƯƠNG = CV lạc quan hơn thực tế.

    Đảo chiều thì mọi kết luận về overfit trong báo cáo bị lộn ngược.
    """
    results = _results_frame()
    comparison = build_comparison(results).set_index("algo")
    for algo in CONTENDERS:
        row = comparison.loc[algo]
        assert row["gap_accuracy"] == pytest.approx(
            row["validation_accuracy"] - row["test_accuracy"])
        assert row["gap_accuracy"] == pytest.approx(0.02)


def test_comparison_reports_all_three_metrics():
    columns = set(build_comparison(_results_frame()).columns)
    for metric in ("accuracy", "macro_f1", "balanced_accuracy"):
        assert {f"validation_{metric}", f"test_{metric}",
                f"gap_{metric}"} <= columns


def test_comparison_uses_the_latest_row_per_algo_and_split():
    """Chạy lại một task thì results.csv có hai dòng — phải lấy dòng mới nhất."""
    results = pd.concat([_results_frame(),
                         _results_frame().assign(accuracy=0.5)], ignore_index=True)
    comparison = build_comparison(results).set_index("algo")
    assert comparison.loc[CONTENDERS[0], "validation_accuracy"] == pytest.approx(0.5)


def test_missing_rows_are_detected():
    results = _results_frame()
    trimmed = results[~((results["algo"] == DECISION_TREE)
                        & (results["split"] == "validation"))]
    assert missing_comparison_rows(trimmed) == [(DECISION_TREE, "validation")]
    assert missing_comparison_rows(results) == []


def test_comparison_refuses_to_score_a_missing_test_result(tmp_path):
    """Thiếu số CV thì bù được; thiếu số TEST thì không — task 12 không được
    tự chấm trên tập test, đó là việc của task 11."""
    results = _results_frame()
    results = results[~((results["algo"] == BAGGING) & (results["split"] == "test"))]
    results.to_csv(tmp_path / "results.csv", index=False)
    with pytest.raises(ValueError, match="task 11"):
        compare_models(SMALL, runs_dir=tmp_path)


@pytest.fixture(scope="module")
def comparison_run(tmp_path_factory) -> dict:
    """Chạy thật trên dân số nhỏ: task 11 trước, rồi so sánh."""
    runs = tmp_path_factory.mktemp("runs")
    evaluate_on_test(SMALL, runs_dir=runs)
    return compare_models(SMALL, runs_dir=runs)


def test_comparison_backfills_only_what_is_missing(comparison_run):
    """`results.csv` sau task 11 chỉ có dòng test — cả 4 CV đều phải bù."""
    assert set(comparison_run["backfilled"]) == set(CONTENDERS)
    assert not comparison_run["comparison"][
        ["validation_macro_f1", "test_macro_f1"]].isna().any().any()


def test_comparison_does_not_select_a_model(comparison_run):
    """Task 12 chỉ so sánh. Chọn model là task sau."""
    assert not [key for key in comparison_run
                if key in {"best", "best_algo", "selected", "artifact"}]
    assert "best_algo" not in comparison_run["comparison"].columns


def test_per_class_comparison_lists_every_group_and_model(comparison_run):
    table = comparison_run["per_class"]
    assert list(table.index) == [g.value for g in ORDERED_GROUPS]
    assert set(table.columns) == set(CONTENDERS)


def test_comparison_writes_tables_to_the_runs_directory(comparison_run):
    files = comparison_run["files"]
    assert files["comparison"].exists()
    assert files["per_class"].exists()
    saved = pd.read_csv(files["comparison"])
    assert list(saved["algo"]) == list(CONTENDERS)


# --------------------------- FEATURE IMPORTANCE (task 13)

@pytest.fixture(scope="module")
def importance_run(tmp_path_factory) -> dict:
    """`use_saved=False`: artifact trong runs/ được train trên dân số mặc định,
    nạp vào đây thì importance in ra không phải của dữ liệu đang xét."""
    return feature_importance_report(
        SMALL, use_saved=False, runs_dir=tmp_path_factory.mktemp("runs"))


def test_bagging_is_reported_as_unavailable_with_a_reason(importance_run):
    """`BaggingClassifier` không phơi ra `feature_importances_`.

    Nó phải nằm trong `unavailable` kèm lý do, không được biến mất lặng lẽ —
    thiếu một model mà không nói vì sao là chỗ người đọc báo cáo sẽ hỏi.
    """
    assert BAGGING in importance_run["unavailable"]
    assert "feature_importances_" in importance_run["unavailable"][BAGGING]
    assert BAGGING not in set(importance_run["importance"]["algo"])


def test_every_other_model_reports_importance(importance_run):
    reported = set(importance_run["importance"]["algo"])
    assert reported == {DECISION_TREE, RANDOM_FOREST, XGBOOST}


def test_importance_covers_every_feature_and_sums_to_one(importance_run):
    long = importance_run["importance"]
    for algo, group in long.groupby("algo"):
        assert set(group["feature"]) == set(RAW_FEATURES), algo
        assert group["importance"].sum() == pytest.approx(1.0), algo


def test_rank_follows_descending_importance(importance_run):
    long = importance_run["importance"]
    for algo, group in long.groupby("algo"):
        ordered = group.sort_values("rank")
        assert list(ordered["importance"]) == sorted(
            ordered["importance"], reverse=True), algo
        assert list(ordered["rank"]) == list(range(1, len(ordered) + 1)), algo


def test_report_records_where_each_model_came_from(importance_run):
    """`source` cho biết model đến từ artifact hay vừa fit lại — không có nó
    thì con số không tra ngược được."""
    assert set(importance_run["sources"]) == set(CONTENDERS)
    assert set(importance_run["sources"].values()) == {"refit"}


def test_saved_artifact_is_reused_instead_of_refitting(tmp_path, data):
    """Có artifact thì phải dùng lại, không train lại."""
    X, y = data
    X_train, X_val, _, y_train, y_val, _ = split_train_val_test(X, y)
    model = PipelineClassifier(
        task="ml01", algo=DECISION_TREE,
        estimator=ALGORITHMS[DECISION_TREE](42),
        preprocessing=build_preprocessing_pipeline())
    model.fit(X_train, y_train)
    save_model(model, directory=tmp_path)

    report = feature_importance_report(
        SMALL, algorithms=(DECISION_TREE,), runs_dir=tmp_path, save=False)
    assert report["sources"][DECISION_TREE] == "artifact"


def test_missing_artifact_falls_back_to_refit(tmp_path):
    report = feature_importance_report(
        SMALL, algorithms=(DECISION_TREE,), runs_dir=tmp_path, save=False)
    assert report["sources"][DECISION_TREE] == "refit"


def test_top_table_lists_the_requested_number_of_features(importance_run):
    top = importance_run["top"]
    assert list(top.index) == [1, 2, 3, 4, 5]
    assert set(top.columns) == {DECISION_TREE, RANDOM_FOREST, XGBOOST}


def test_importance_report_does_not_select_a_model(importance_run):
    assert not [key for key in importance_run
                if key in {"best", "best_algo", "selected"}]


def test_importance_report_writes_to_the_runs_directory(importance_run):
    files = importance_run["files"]
    assert files["importance"].exists()
    assert files["pivot"].exists()
    saved = pd.read_csv(files["importance"])
    assert set(saved.columns) >= {"algo", "source", "rank", "feature", "importance"}


def test_importance_report_is_reproducible():
    first = feature_importance_report(
        SMALL, algorithms=(DECISION_TREE,), use_saved=False, save=False)
    second = feature_importance_report(
        SMALL, algorithms=(DECISION_TREE,), use_saved=False, save=False)
    assert first["importance"].equals(second["importance"])


# ------------------------------------ CHỌN MODEL (task 14)

def _comparison_frame(validation: dict[str, float],
                      test: dict[str, float]) -> pd.DataFrame:
    """Bảng so sánh giả lập, đủ cột để `select_final_model` chạy."""
    return pd.DataFrame([
        {"algo": algo,
         "validation_macro_f1": validation[algo],
         "test_macro_f1": test[algo],
         "gap_macro_f1": validation[algo] - test[algo]}
        for algo in CONTENDERS
    ])


def test_selection_picks_the_highest_validation_macro_f1():
    record = select_final_model(_comparison_frame(
        validation={DECISION_TREE: 0.84, BAGGING: 0.91, RANDOM_FOREST: 0.85, XGBOOST: 0.92},
        test={DECISION_TREE: 0.84, BAGGING: 0.90, RANDOM_FOREST: 0.84, XGBOOST: 0.91}))
    assert record["selected"] == XGBOOST
    assert record["selection_metric"] == "macro_f1"
    assert record["validation_macro_f1"] == pytest.approx(0.92)


def test_selection_ignores_test_results_entirely():
    """Test chỉ là thông tin tham khảo — đảo hết chỉ số test không được làm
    đổi lựa chọn.

    Bảng dưới đây dựng để `bagging` thắng áp đảo trên TEST còn `xgboost`
    thắng trên CV. Nếu code lỡ nhìn cột test thì lựa chọn sẽ lật.
    """
    scores = {DECISION_TREE: 0.84, BAGGING: 0.91,
              RANDOM_FOREST: 0.85, XGBOOST: 0.92}
    record = select_final_model(_comparison_frame(
        validation=scores,
        test={DECISION_TREE: 0.10, BAGGING: 0.99, RANDOM_FOREST: 0.10, XGBOOST: 0.10}))
    assert record["selected"] == XGBOOST


def test_selection_reports_margin_to_the_runner_up():
    """Khoảng cách tới á quân phải nói ra được, không chỉ tên người thắng.

    Trước 14/08/2026 bản ghi còn `margin_vs_fold_std` — margin quy về σ giữa
    các fold. Bỏ K-Fold thì không còn σ, nên trường đó đã được gỡ; test này
    canh luôn việc nó không quay lại dưới dạng một giá trị tự chế.
    """
    record = select_final_model(_comparison_frame(
        validation={DECISION_TREE: 0.84, BAGGING: 0.908, RANDOM_FOREST: 0.85, XGBOOST: 0.920},
        test={a: 0.9 for a in CONTENDERS}))
    assert record["runner_up"] == BAGGING
    assert record["margin"] == pytest.approx(0.012)
    assert "margin_vs_fold_std" not in record
    assert "validation_macro_f1_std" not in record


def test_selection_ranks_every_contender():
    record = select_final_model(_comparison_frame(
        validation={DECISION_TREE: 0.84, BAGGING: 0.91, RANDOM_FOREST: 0.85, XGBOOST: 0.92},
        test={a: 0.9 for a in CONTENDERS}))
    ranked = [row["algo"] for row in record["validation_ranking"]]
    assert ranked == [XGBOOST, BAGGING, RANDOM_FOREST, DECISION_TREE]


def test_selection_keeps_test_numbers_as_supporting_only():
    record = select_final_model(_comparison_frame(
        validation={DECISION_TREE: 0.84, BAGGING: 0.91, RANDOM_FOREST: 0.85, XGBOOST: 0.92},
        test={DECISION_TREE: 0.83, BAGGING: 0.90, RANDOM_FOREST: 0.84, XGBOOST: 0.905}))
    assert record["supporting"]["test_macro_f1"] == pytest.approx(0.905)
    assert record["supporting"]["gap_validation_minus_test"] == pytest.approx(0.015)
    # Chỉ số test không được nằm ở tầng ngoài của bản ghi, nơi dễ đọc nhầm
    # thành căn cứ chọn.
    assert not [key for key in record if key.startswith("test_")]


def test_selection_never_picks_the_baseline():
    frame = _comparison_frame(
        validation={a: 0.5 for a in CONTENDERS}, test={a: 0.5 for a in CONTENDERS})
    frame = pd.concat([frame, pd.DataFrame([{
        "algo": BASELINE, "validation_macro_f1": 0.99, "test_macro_f1": 0.99,
        "gap_macro_f1": 0.0, }])], ignore_index=True)
    assert select_final_model(frame)["selected"] != BASELINE


def test_recording_refuses_to_run_without_cv_results(tmp_path):
    """Task 14 không train lại — thiếu số CV thì báo lỗi."""
    results = _results_frame()
    results = results[~((results["algo"] == XGBOOST)
                        & (results["split"] == "validation"))]
    results.to_csv(tmp_path / "results.csv", index=False)
    with pytest.raises(ValueError, match="không train lại"):
        record_model_selection(runs_dir=tmp_path)


def test_recording_writes_selection_to_the_runs_directory(tmp_path):
    _results_frame().to_csv(tmp_path / "results.csv", index=False)
    result = record_model_selection(runs_dir=tmp_path)

    saved = json.loads(result["file"].read_text(encoding="utf-8"))
    assert saved["selected"] == result["record"]["selected"]
    assert saved["selection_basis"].startswith("tập validation")
    assert saved["criterion"]
    assert result["file"].name == "model_selection.json"


# ------------------------------------- EXPORT MODEL (task 15)

@pytest.fixture(scope="module")
def exported(tmp_path_factory) -> dict:
    """Dựng đủ tiền đề trong thư mục tạm: train xgboost → chấm test → chọn →
    export. Không đụng `src/training/runs/` thật."""
    runs = tmp_path_factory.mktemp("runs")
    train_xgboost(SMALL, runs_dir=runs)
    evaluate_on_test(SMALL, algorithms=(XGBOOST,), runs_dir=runs)
    record_model_selection(runs_dir=runs, algorithms=(XGBOOST,))
    return export_final_model(runs_dir=runs)


def test_export_uses_the_model_task14_selected(exported):
    assert exported["algo"] == XGBOOST
    assert exported["metadata"]["algo"] == XGBOOST
    assert exported["metadata"]["selection"]["selected"] == XGBOOST


def test_export_refuses_to_run_without_a_selection_record(tmp_path):
    """Task 15 export cái task 14 đã chọn — không tự suy ra model tốt nhất."""
    with pytest.raises(FileNotFoundError, match="task 14"):
        export_final_model(runs_dir=tmp_path)


def test_export_lands_in_the_runs_directory(exported):
    artifact = exported["artifact"]
    assert artifact.exists() and artifact.suffix == ".joblib"
    assert exported["metadata_path"].exists()
    assert artifact.parent == exported["metadata_path"].parent


def test_export_keeps_the_algorithm_name_in_the_slug(exported):
    """Bản "final" mà giấu mình là model gì thì phải mở file ra mới biết."""
    assert exported["artifact"].stem == f"ml01_{XGBOOST}_v{FINAL_VERSION}"


def test_export_does_not_change_the_model(exported, tmp_path_factory):
    """"Export" là đóng gói lại, không phải train lại — dự đoán phải y hệt
    artifact nguồn."""
    runs = exported["artifact"].parent
    source = load_model(f"ml01_{XGBOOST}_v1", directory=runs)
    final = load_model(exported["artifact"].stem, directory=runs)

    X, y = build_training_data(SMALL)
    X_train, _, _, _ = split_train_test(X, y)
    sample = X_train.head(100)
    assert list(final.predict(sample)) == list(source.predict(sample))
    assert np.allclose(final.predict_proba(sample), source.predict_proba(sample))


def test_export_can_be_loaded_and_predicts(exported):
    """Artifact hỏng thường không nổ lúc `load` mà nổ lúc `predict`."""
    checks = exported["verification"]
    assert checks["loaded"]
    assert checks["labels_within_classes"]
    assert checks["proba_shape_matches"]
    assert checks["proba_sums_to_one"]
    assert checks["feature_order_matches"]


def test_export_metadata_carries_what_reproduction_needs(exported):
    meta = exported["metadata"]
    for key in ("random_seed", "feature_names", "classes", "config",
                "environment", "selection", "exported_at"):
        assert key in meta, key
    assert meta["feature_names"] == list(RAW_FEATURES)
    assert meta["config"]["objective"] == "multi:softprob"
    # Cùng seed mà khác phiên bản thư viện vẫn có thể ra số khác.
    assert {"python", "scikit-learn", "xgboost"} <= set(meta["environment"])


def test_export_metadata_gathers_both_validation_and_test_metrics(exported):
    """Metadata task 10 chỉ có validation; chỉ số test nằm ở results.csv. Bản export
    phải gom cả hai để nó tự đủ, không phải tra chéo file khác."""
    metrics = exported["metadata"]["metrics"]
    assert "macro_f1" in metrics["validation"]
    assert "macro_f1" in metrics["test"]


def test_export_records_hashes_of_both_artifacts(exported):
    """sha256 để đối chiếu file đang cầm có đúng là file metadata mô tả không."""
    meta = exported["metadata"]
    assert len(meta["source_sha256"]) == 64
    assert len(meta["artifact_sha256"]) == 64

    import hashlib
    digest = hashlib.sha256(exported["artifact"].read_bytes()).hexdigest()
    assert digest == meta["artifact_sha256"]


# ----------------------------------------------- BASELINE (task 6)

def test_baseline_matches_plan_specification():
    """PLAN.md §6.3 chỉ định đúng `DummyClassifier(strategy='stratified')`.

    `random_state` phải được đặt: baseline rút ngẫu nhiên, không cố định seed
    thì mỗi lần chạy ra một mốc khác và cổng "thắng baseline" hết tái lập.
    """
    baseline = ALGORITHMS[BASELINE](42)
    assert isinstance(baseline, DummyClassifier)
    assert baseline.strategy == "stratified"
    assert baseline.random_state == 42


@pytest.mark.xfail(reason="Thiết kế nhãn CŨ (thang if/else) không còn giữ được cân bằng lớp sau khi ba công thức `savings_rate` được gộp về một: `labeler` trước đây bỏ quên khoản trả nợ, nên nhánh EMERGENCY yếu hơn thực tế và DEBT_FOCUS mới đủ 14%. Với công thức đúng, EMERGENCY nuốt phần lớn dân số và DEBT_FOCUS còn 1,6%. Đây là chứng cứ cho thấy cân bằng lớp của bản cũ là sản phẩm phụ của một lỗi tính toán, không phải của thiết kế. Bản thay thế là `scoring.py` + `dataset.py` (ML01 v2), có test riêng ở `test_ml01_v2.py`.", strict=False)
def test_baseline_accuracy_follows_stratified_theory(data):
    """Accuracy của dự đoán rút theo tỉ lệ phải bằng Σpᵢ².

    Đây là phép kiểm baseline có đúng là stratified không. Lệch khỏi công
    thức nghĩa là nó đang làm việc khác — ví dụ `most_frequent`, vốn cho
    accuracy bằng pₘₐₓ (~0,32) chứ không phải 0,27.
    """
    X, y = data
    X_train, X_val, _, y_train, y_val, _ = split_train_val_test(X, y)
    expected = float((y_train.value_counts(normalize=True) ** 2).sum())

    comparison, _ = evaluate_on_validation(
        X_train, y_train, X_val, y_val,
        algorithms={BASELINE: ALGORITHMS[BASELINE]})
    assert float(comparison["accuracy"].iloc[0]) == pytest.approx(expected, abs=0.03)


def test_baseline_macro_f1_is_one_over_class_count(data):
    """Macro-F1 của baseline stratified luôn ≈ 1/k, bất kể lớp lệch bao nhiêu.

    Mỗi lớp có precision = recall = pᵢ nên F1ᵢ = pᵢ, và trung bình pᵢ trên k
    lớp luôn là 1/k vì Σpᵢ = 1. Vậy 0,25 là **mốc cứng** của ML01: model nào
    dưới mức này thì kém hơn đoán mò.
    """
    X, y = data
    X_train, X_val, _, y_train, y_val, _ = split_train_val_test(X, y)
    n_classes = y_train.nunique()

    comparison, _ = evaluate_on_validation(
        X_train, y_train, X_val, y_val,
        algorithms={BASELINE: ALGORITHMS[BASELINE]})
    assert float(comparison["macro_f1"].iloc[0]) == pytest.approx(1 / n_classes, abs=0.03)


def test_baseline_ignores_features(data):
    """Baseline không được nhìn `X` — nếu nhìn thì nó không còn là mốc nữa.

    Xáo trộn toàn bộ giá trị feature mà phân bố dự đoán vẫn giữ nguyên, đó là
    bằng chứng nó chỉ rút theo tỉ lệ lớp.
    """
    X, y = data
    model = PipelineClassifier(task="ml01", algo=BASELINE,
                               estimator=ALGORITHMS[BASELINE](42),
                               preprocessing=build_preprocessing_pipeline())
    model.fit(X, y)
    scrambled = X.sample(frac=1.0, random_state=1).reset_index(drop=True)

    original = pd.Series(model.predict(X)).value_counts(normalize=True)
    shuffled = pd.Series(model.predict(scrambled)).value_counts(normalize=True)
    for label in original.index:
        assert shuffled[label] == pytest.approx(original[label], abs=0.05)


def test_baseline_is_never_selected():
    comparison = pd.DataFrame([
        {"algo": BASELINE, "accuracy": 0.99, "macro_f1": 0.99},
        {"algo": "decision_tree", "accuracy": 0.50, "macro_f1": 0.50},
    ])
    assert select_best(comparison) == "decision_tree"


def test_select_best_rejects_a_table_without_contenders():
    with pytest.raises(ValueError):
        select_best(pd.DataFrame([{"algo": BASELINE, "macro_f1": 0.2}]))


def test_bagging_and_forest_use_the_same_tree_count():
    """Khác số cây thì bảng không còn trả lời được về lấy mẫu feature."""
    assert ALGORITHMS["bagging"](42).n_estimators == N_ESTIMATORS
    assert ALGORITHMS["random_forest"](42).n_estimators == N_ESTIMATORS


def test_decision_tree_is_left_unpruned():
    """Vai trò của cây đơn là CHO THẤY overfit (PLAN.md §6.3)."""
    assert ALGORITHMS["decision_tree"](42).max_depth is None


# ------------------------------------------------------ cổng kiểm chứng

def test_gates_flag_a_too_clean_boundary(data):
    _, y = data
    comparison = pd.DataFrame([
        {"algo": BASELINE, "accuracy": 0.25, "macro_f1": 0.25, "macro_f1_std": 0.01},
        {"algo": "xgboost", "accuracy": 0.995, "macro_f1": 0.995, "macro_f1_std": 0.01},
    ])
    gates = check_gates(y, comparison)
    row = gates[gates["cổng"] == "Ranh giới không quá sạch"].iloc[0]
    assert not row["đạt"]


def test_clean_boundary_gate_prefers_the_test_number(data):
    """Có test thì cổng 2 phải chấm trên test, không phải trên CV.

    CV ở đây "sạch" giả tạo (0,99) còn test thì không (0,93). Chấm nhầm chỗ
    sẽ báo động sai — và nếu ngược lại, sẽ bỏ lọt rò rỉ thật.
    """
    _, y = data
    comparison = pd.DataFrame([
        {"algo": BASELINE, "accuracy": 0.25, "macro_f1": 0.25, "macro_f1_std": 0.01},
        {"algo": "xgboost", "accuracy": 0.99, "macro_f1": 0.99, "macro_f1_std": 0.01},
    ])
    gates = check_gates(y, comparison, test_metrics={"accuracy": 0.93, "macro_f1": 0.92})
    row = gates[gates["cổng"] == "Ranh giới không quá sạch"].iloc[0]
    assert row["đạt"]
    assert "test" in row["chi tiết"]


def test_gates_flag_a_model_that_barely_beats_baseline(data):
    _, y = data
    comparison = pd.DataFrame([
        {"algo": BASELINE, "accuracy": 0.25, "macro_f1": 0.250, "macro_f1_std": 0.005},
        {"algo": "decision_tree", "accuracy": 0.28, "macro_f1": 0.270,
         "macro_f1_std": 0.005},
    ])
    gates = check_gates(y, comparison)
    row = gates[gates["cổng"] == "Thắng baseline rõ rệt"].iloc[0]
    assert not row["đạt"]


def test_gates_flag_an_imbalanced_population():
    skewed = pd.Series(["GROWTH"] * 970 + ["EMERGENCY"] * 30)
    comparison = pd.DataFrame([
        {"algo": BASELINE, "accuracy": 0.9, "macro_f1": 0.4, "macro_f1_std": 0.01},
        {"algo": "xgboost", "accuracy": 0.95, "macro_f1": 0.8, "macro_f1_std": 0.01},
    ])
    gates = check_gates(skewed, comparison)
    assert not gates[gates["cổng"] == "Cân bằng lớp"].iloc[0]["đạt"]


def test_gate_accuracy_ceiling_matches_plan():
    assert GATE_MAX_ACCURACY == 0.98


# ------------------------------------------------------------- ghi log

def test_results_csv_appends_instead_of_overwriting(tmp_path):
    """Lịch sử các lần chỉnh tham số chính là phần thực nghiệm của báo cáo."""
    path = tmp_path / "results.csv"
    append_results(pd.DataFrame([{"task": "ml01", "algo": "a", "macro_f1": 0.5}]), path)
    append_results(pd.DataFrame([{"task": "ml01", "algo": "b", "macro_f1": 0.6}]), path)
    saved = pd.read_csv(path)
    assert list(saved["algo"]) == ["a", "b"]
    assert saved["run_at"].notna().all()


def test_results_csv_merges_new_metric_columns(tmp_path):
    """Đổi bộ chỉ số không được làm mất dòng cũ."""
    path = tmp_path / "results.csv"
    append_results(pd.DataFrame([{"algo": "a", "macro_f1": 0.5}]), path)
    append_results(pd.DataFrame([{"algo": "b", "pr_auc": 0.7}]), path)
    saved = pd.read_csv(path)
    assert len(saved) == 2
    assert {"macro_f1", "pr_auc"} <= set(saved.columns)


# ------------------------------------------------------- vòng chạy đầy đủ

@pytest.mark.slow
@pytest.mark.xfail(reason="Thiết kế nhãn CŨ (thang if/else) không còn giữ được cân bằng lớp sau khi ba công thức `savings_rate` được gộp về một: `labeler` trước đây bỏ quên khoản trả nợ, nên nhánh EMERGENCY yếu hơn thực tế và DEBT_FOCUS mới đủ 14%. Với công thức đúng, EMERGENCY nuốt phần lớn dân số và DEBT_FOCUS còn 1,6%. Đây là chứng cứ cho thấy cân bằng lớp của bản cũ là sản phẩm phụ của một lỗi tính toán, không phải của thiết kế. Bản thay thế là `scoring.py` + `dataset.py` (ML01 v2), có test riêng ở `test_ml01_v2.py`.", strict=False)
def test_full_run_passes_every_gate_and_exports(tmp_path, monkeypatch):
    """Chạy thật với tham số mặc định — đây là chỗ kiểm CHẤT LƯỢNG số.

    Ba cổng của PLAN.md §6.2 phải xanh; không thì quay lại task 3 chứ không
    phải hạ ngưỡng cổng.
    """
    result = run_full_pipeline(runs_dir=tmp_path)

    assert result["gates"]["đạt"].all(), result["gates"].to_string(index=False)
    assert result["best_algo"] != BASELINE
    assert result["artifact"].exists()
    assert (tmp_path / "results.csv").exists()

    # Chỉ số test là con số đem báo cáo — phải có, và phải thắng baseline
    # trên CÙNG tập test chứ không phải so với baseline đo ở chỗ khác.
    assert result["test_metrics"]["macro_f1"] > result["baseline_test_metrics"]["macro_f1"]
    # Ba tập phải phủ hết 20.000 hộ, không dòng nào rơi ra ngoài. Khẳng định
    # này còn sót từ thời chia 80/20; từ 14/08/2026 phép chia là 70/15/15 nên
    # thiếu `y_val` thì tổng chỉ ra 17.000.
    assert (len(result["y_train"]) + len(result["y_val"])
            + len(result["y_test"])) == 20_000

    # Bảng per-class dựng trên test nên tổng support phải bằng cỡ tập test.
    assert result["per_class"]["support"].sum() == len(result["y_test"])

    metadata = json.loads(
        result["artifact"].with_name(f"{result['artifact'].stem}.metadata.json")
        .read_text(encoding="utf-8"))
    # Giữ CẢ HAI: chỉ số CV là căn cứ chọn, chỉ số test là kết quả báo cáo.
    assert {"validation", "test", "baseline_test"} <= set(metadata["metrics"])
    assert metadata["feature_names"] == list(RAW_FEATURES)
