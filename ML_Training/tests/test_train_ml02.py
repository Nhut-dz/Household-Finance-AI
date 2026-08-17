"""Test phần khung train của ML02 và Decision Tree (task 7).

Không test nào chạm dataset thật — mỗi test dựng dữ liệu nhỏ có cấu trúc đã
biết. Trọng tâm là những thứ nếu hỏng thì **bảng so sánh ở task 12 so nhầm mà
không có gì để lộ ra**:

    · Pipeline không được fit lại trên train  → rò rỉ, chỉ số đẹp giả
    · Tập test bị chạm ở task 7–12            → mất tính độc lập của đánh giá
    · Tỉ số phạt không đến từ task 4          → bốn thuật toán học trên hai
                                                mức phạt khác nhau
"""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

from hfml.config import CONFIG
from hfml.ml.ml02_credit_risk import train as train_module
from hfml.ml.ml02_credit_risk.clean import ID_COLUMN, TARGET_COLUMN
from hfml.ml.ml02_credit_risk.imbalance import scale_pos_weight_from
from hfml.ml.ml02_credit_risk.features import (
    aggregate_bureau,
    split_features_and_target,
)
from hfml.ml.ml02_credit_risk.train import (
    ALGORITHMS,
    BAGGING,
    DECISION_TREE,
    FEATURE_SETS,
    MIN_LEAF_SHARE,
    N_ESTIMATORS,
    RANDOM_FOREST,
    RF_MAX_FEATURES,
    XGBOOST,
    XGBOOST_PARAMS,
    TrainedModel,
    TrainingData,
    decision_tree_params,
    fit_and_evaluate,
    results_frame,
    save_run,
    train_bagging,
    train_decision_tree,
    train_random_forest,
    train_xgboost,
)


def application(n: int = 3_000, seed: int = 0) -> pd.DataFrame:
    """Hồ sơ đã làm sạch, thu nhỏ — nhãn có liên hệ THẬT với feature.

    Nhãn ngẫu nhiên thuần thì mọi model đều cho PR-AUC ≈ tỉ lệ nền và không
    test nào phân biệt được "train chạy đúng" với "train chẳng học gì".
    """
    rng = np.random.default_rng(seed)
    income = rng.uniform(100_000, 300_000, size=n)
    credit = rng.uniform(200_000, 900_000, size=n)
    # Vay càng nhiều so với thu nhập càng dễ vỡ nợ.
    risk = 1 / (1 + np.exp(-(credit / income - 3.0)))
    return pd.DataFrame({
        ID_COLUMN: np.arange(1, n + 1),
        TARGET_COLUMN: rng.binomial(1, risk * 0.25),
        "AMT_INCOME_TOTAL": income,
        "AMT_CREDIT": credit,
        "AMT_ANNUITY": rng.uniform(10_000, 40_000, size=n),
        "AMT_GOODS_PRICE": rng.uniform(180_000, 850_000, size=n),
        "CNT_CHILDREN": rng.integers(0, 3, size=n),
        "CNT_FAM_MEMBERS": rng.integers(3, 6, size=n).astype(float),
        "DAYS_BIRTH": -rng.integers(8_000, 22_000, size=n),
        "DAYS_EMPLOYED": -rng.integers(100, 7_000, size=n),
        "CODE_GENDER": rng.choice(["M", "F"], size=n),
        "NAME_EDUCATION_TYPE": rng.choice(
            ["Higher education", "Secondary / secondary special"], size=n),
    })


def bureau_for(df: pd.DataFrame) -> pd.DataFrame:
    ids = np.repeat(df[ID_COLUMN].to_numpy()[: int(len(df) * 0.85)], 2)
    rng = np.random.default_rng(7)
    return pd.DataFrame({
        ID_COLUMN: ids,
        "SK_ID_BUREAU": np.arange(len(ids)),
        "CREDIT_ACTIVE": rng.choice(["Active", "Closed"], size=len(ids)),
        "CREDIT_DAY_OVERDUE": rng.choice([0, 0, 0, 30], size=len(ids)),
        "AMT_CREDIT_SUM": rng.uniform(10_000, 200_000, size=len(ids)),
        "AMT_CREDIT_SUM_DEBT": rng.uniform(0, 100_000, size=len(ids)),
        "AMT_CREDIT_SUM_OVERDUE": rng.choice([0.0, 0.0, 0.0, 5_000.0], size=len(ids)),
        "DAYS_CREDIT": -rng.integers(100, 3_000, size=len(ids)),
    })


@pytest.fixture
def data() -> TrainingData:
    df = application(3_000)
    train, validation = df.iloc[:2_200], df.iloc[2_200:]
    X_train, y_train = split_features_and_target(train)
    X_val, y_val = split_features_and_target(validation)
    return TrainingData(X_train, y_train, X_val, y_val,
                        aggregate_bureau(bureau_for(df)))


# --------------------------------------------------------------- siêu tham số
def test_min_leaf_scales_with_the_training_set():
    """Ràng buộc lá suy từ CỠ dữ liệu, không phải một con số dò được.

    215.257 dòng × 0,1% ≈ 215 hồ sơ mỗi lá; ở tỉ lệ dương 8,07% thì lá đó chứa
    ~17 ca vỡ nợ — vừa đủ để xác suất của lá có nghĩa.
    """
    assert decision_tree_params(215_257)["min_samples_leaf"] == 215
    assert decision_tree_params(20_000)["min_samples_leaf"] == 20
    assert decision_tree_params(215_257)["min_samples_leaf"] == int(
        215_257 * MIN_LEAF_SHARE)


def test_min_leaf_never_drops_below_one():
    """Tập rất nhỏ vẫn phải cho tham số hợp lệ, không phải 0."""
    assert decision_tree_params(100)["min_samples_leaf"] >= 1


def test_decision_tree_uses_the_project_seed():
    assert decision_tree_params(1_000)["random_state"] == CONFIG.random_seed


# ------------------------------------------------------- cân bằng lớp task 4
def test_decision_tree_gets_balanced_class_weight_from_task_four(data):
    """Tỉ số phạt phải đến từ `imbalance_params`, không hardcode ở task 7.

    Một chỗ duy nhất quyết định thì bốn thuật toán không thể lệch nhau. Hardcode
    ở mỗi task train là cách chắc chắn nhất để task 12 so nhầm.
    """
    model = train_decision_tree(data, feature_sets=("reduced",))[0]
    estimator = model.pipeline.named_steps["model"]

    assert isinstance(estimator, DecisionTreeClassifier)
    assert estimator.class_weight == "balanced"


# ------------------------------------------------------------ chống rò rỉ
def test_training_data_has_no_test_attribute():
    """Không có thuộc tính test thì không có cách nào lỡ tay chạm vào nó.

    Tập test khoá tới task 14. Ràng buộc đó được cài bằng CẤU TRÚC dữ liệu chứ
    không phải bằng lời dặn trong tài liệu.
    """
    fields = set(TrainingData.__dataclass_fields__)

    assert not any("test" in f for f in fields), fields
    assert {"X_train", "y_train", "X_validation", "y_validation"} <= fields


def test_fit_and_evaluate_never_receives_a_test_set():
    """Muốn chạm tập test phải sửa chữ ký hàm — tức phải cố ý."""
    params = set(inspect.signature(fit_and_evaluate).parameters)

    assert not any("test" in p for p in params), params


def test_pipeline_is_refit_inside_each_training_run(data):
    """Pipeline phải học trung vị/phân vị từ CHÍNH tập train của lần train đó.

    Nạp một Pipeline fit sẵn ở nơi khác thì thống kê của nó đến từ một tập
    khác — và nếu tập đó chứa dòng nay thuộc validation thì đó là rò rỉ.
    """
    model = train_decision_tree(data, feature_sets=("reduced",))[0]
    builder = (model.pipeline
               .named_steps["features"]
               .named_steps["features"])

    # `fit` đã chạy → transformer mang thống kê học được.
    assert builder.reference_income_per_capita_ is not None
    assert hasattr(builder, "engineered_names_")


def test_two_runs_on_the_same_data_give_the_same_result(data):
    """Cùng seed, cùng dữ liệu → cùng chỉ số. Không tái lập thì không so được."""
    a = train_decision_tree(data, feature_sets=("reduced",))[0]
    b = train_decision_tree(data, feature_sets=("reduced",))[0]

    assert a.pr_auc == pytest.approx(b.pr_auc)
    assert a.metrics_validation["roc_auc"] == pytest.approx(
        b.metrics_validation["roc_auc"])


# ------------------------------------------------------------------ kết quả
def test_model_learns_something_above_the_random_floor(data):
    """Nhãn có liên hệ thật với feature nên PR-AUC phải vượt tỉ lệ nền.

    Không vượt nghĩa là vòng train đang chạy nhưng chẳng học được gì — mà
    `fit()` vẫn xong và `predict_proba` vẫn trả số đều đặn.
    """
    model = train_decision_tree(data, feature_sets=("reduced",))[0]

    assert model.pr_auc > model.metrics_validation["base_rate"]
    assert model.metrics_validation["pr_auc_lift"] > 1.0


def test_both_feature_sets_are_trained(data):
    models = train_decision_tree(data)

    assert [m.feature_set for m in models] == list(FEATURE_SETS)
    assert all(m.algo == DECISION_TREE for m in models)


def test_full_set_has_more_features_than_reduced(data):
    """Bộ FULL giữ cả cột Home Credit mà form không thu được."""
    reduced, full = train_decision_tree(data)

    assert full.n_features > reduced.n_features


def test_overfit_gap_is_measured_not_assumed(data):
    """Khoảng cách train − validation là thứ duy nhất cho thấy học thuộc.

    Với cây đơn thì đó chính là vai trò của nó trong báo cáo (§6.3), nên con
    số này phải được ĐO chứ không phải một nhận xét định tính.
    """
    model = train_decision_tree(data, feature_sets=("reduced",))[0]

    assert model.metrics_train, "thiếu chỉ số trên train"
    assert model.overfit_gap == pytest.approx(
        model.metrics_train["pr_auc"] - model.metrics_validation["pr_auc"])


def test_result_row_is_marked_as_validation_and_holdout(data):
    """Dòng trong `results.csv` phải nói rõ đo trên tập nào và không dùng CV."""
    row = train_decision_tree(data, feature_sets=("reduced",))[0].row()

    assert row["split"] == "validation"
    assert row["n_splits"] == 1          # holdout, KHÔNG K-Fold
    assert row["task"] == "ml02"
    assert row["random_seed"] == CONFIG.random_seed


def test_results_frame_has_one_row_per_model(data):
    table = results_frame(train_decision_tree(data))

    assert len(table) == len(FEATURE_SETS)
    assert {"pr_auc", "recall_positive", "overfit_gap"} <= set(table.columns)


# ------------------------------------------------------------------- lưu
def test_saved_artifact_is_marked_intermediate_not_an_export(tmp_path, monkeypatch, data):
    """Artifact của task 7 KHÔNG phải export — export là task 15.

    Ghi rõ trong metadata để không ai nạp nhầm file này vào service và tưởng
    đã có đủ feature list, label mapping và cấu hình cho inference.
    """
    import json

    monkeypatch.setattr(CONFIG.paths, "runs", tmp_path)
    model = train_decision_tree(data, feature_sets=("reduced",))[0]

    written = save_run(model)
    metadata = json.loads(written["metadata"].read_text(encoding="utf-8"))

    assert metadata["artifact_kind"] == "intermediate"
    assert metadata["evaluated_on"] == "validation"
    assert metadata["test_set_touched"] is False


def test_saved_pipeline_reloads_and_predicts(tmp_path, monkeypatch, data):
    """Artifact phải nạp lại và cho ĐÚNG xác suất cũ.

    Không kiểm điều này thì task 11–14 có thể đang đọc một file hỏng mà vẫn
    ra số, chỉ là số khác.
    """
    import joblib

    monkeypatch.setattr(CONFIG.paths, "runs", tmp_path)
    model = train_decision_tree(data, feature_sets=("reduced",))[0]
    truoc = model.pipeline.predict_proba(data.X_validation)[:, 1]

    path = save_run(model)["pipeline"]
    sau = joblib.load(path).predict_proba(data.X_validation)[:, 1]

    np.testing.assert_allclose(truoc, sau)


def test_slug_identifies_algorithm_and_feature_set():
    model = TrainedModel(algo="decision_tree", feature_set="reduced",
                         pipeline=None)

    assert model.slug == "ml02_decision_tree_reduced"


# ------------------------------------------------------ task 8 · Bagging
def test_bagging_puts_class_weight_on_the_child_tree_not_itself(data):
    """`BaggingClassifier` KHÔNG có tham số `class_weight` (task 4).

    Đặt nhầm lên ngoài sẽ `TypeError`; tệ hơn là nếu bị nuốt trong `**kwargs`
    thì model train MẤT CÂN BẰNG trong khi bảng cấu hình vẫn ghi là đã cân
    bằng — và bảng so sánh task 12 so một model có trọng số với ba model
    không có.
    """
    model = train_bagging(data, feature_sets=("reduced",))[0]
    estimator = model.pipeline.named_steps["model"]

    assert isinstance(estimator, BaggingClassifier)
    assert not hasattr(estimator, "class_weight")
    assert estimator.estimator.class_weight == "balanced"


def test_bagging_child_tree_matches_task_seven_exactly(data):
    """Cây con phải dùng ĐÚNG siêu tham số của Decision Tree ở task 7.

    Đó là điều kiện để đọc được bảng so sánh: Bagging chỉ khác ở chỗ lấy 50 mẫu
    bootstrap rồi trung bình, nên chênh lệch PR-AUC **chính là** phần do giảm
    phương sai. Cây con có `min_samples_leaf` khác thì con số đó lẫn cả phần
    "cây được điều tiết khác đi".
    """
    model = train_bagging(data, feature_sets=("reduced",))[0]
    child = model.pipeline.named_steps["model"].estimator
    expected = decision_tree_params(model.n_train)

    assert child.min_samples_leaf == expected["min_samples_leaf"]
    assert child.max_depth == expected["max_depth"]


def test_bagging_keeps_all_features_at_every_split(data):
    """`max_features=1.0` là ranh giới với Random Forest ở task 9.

    RF khác Bagging đúng ở chỗ lấy mẫu feature tại mỗi lát cắt. Bagging cũng
    lấy mẫu feature thì hai thuật toán trùng nhau và §6.3 mất chỗ đối chiếu
    hai cơ chế giảm phương sai.
    """
    model = train_bagging(data, feature_sets=("reduced",))[0]

    assert model.pipeline.named_steps["model"].max_features == 1.0


def test_bagging_and_random_forest_share_one_tree_count():
    """Số cây phải dùng chung, nếu không chênh lệch lẫn cả phần 'nhiều cây hơn'."""
    assert N_ESTIMATORS == 50
    assert train_module.N_ESTIMATORS is N_ESTIMATORS


def test_bagging_actually_builds_the_requested_number_of_trees(data):
    model = train_bagging(data, feature_sets=("reduced",))[0]

    assert len(model.pipeline.named_steps["model"].estimators_) == N_ESTIMATORS


def test_bagging_reduces_the_overfit_gap_of_a_single_tree(data):
    """Trung bình 50 cây bootstrap phải bớt học thuộc hơn một cây đơn.

    Đây là lý do Bagging có mặt trong bảng bốn thuật toán (§6.3). Không giảm
    thì hoặc cấu hình sai, hoặc các cây con giống hệt nhau — mà cả hai đều
    không lộ ra ở chỉ số validation.
    """
    cay_don = train_decision_tree(data, feature_sets=("reduced",))[0]
    bagging = train_bagging(data, feature_sets=("reduced",))[0]

    assert bagging.overfit_gap < cay_don.overfit_gap


def test_bagging_is_reproducible(data):
    """Như Random Forest: `n_jobs=-1` làm thứ tự cộng dồn không cố định.

    Xem `test_random_forest_is_reproducible` để biết vì sao dung sai 1e-9.
    """
    a = train_bagging(data, feature_sets=("reduced",))[0]
    b = train_bagging(data, feature_sets=("reduced",))[0]

    assert a.pr_auc == pytest.approx(b.pr_auc, abs=1e-9)


def test_bagging_rows_are_labelled_with_its_own_name(data):
    models = train_bagging(data)

    assert all(m.algo == BAGGING for m in models)
    assert models[0].slug == "ml02_bagging_reduced"


# ------------------------------------------------ task 9 · Random Forest
def test_random_forest_subsamples_features_at_each_split(data):
    """`max_features='sqrt'` là THỨ DUY NHẤT phân biệt RF với Bagging.

    Bagging xét đủ cột ở mỗi lát cắt (`max_features=1.0`); RF chỉ xét √p. Nếu
    hai bên trùng tham số này thì hai thuật toán là một, và chênh lệch trong
    bảng so sánh không đo cơ chế nào cả.
    """
    rf = train_random_forest(data, feature_sets=("reduced",))[0]
    bagging = train_bagging(data, feature_sets=("reduced",))[0]

    assert rf.pipeline.named_steps["model"].max_features == RF_MAX_FEATURES
    assert bagging.pipeline.named_steps["model"].max_features == 1.0
    assert rf.pipeline.named_steps["model"].max_features != \
        bagging.pipeline.named_steps["model"].max_features


def test_random_forest_matches_bagging_on_everything_else(data):
    """Mọi tham số khác phải giống hệt task 8, nếu không chênh lệch lẫn nhiều thứ."""
    rf = train_random_forest(data, feature_sets=("reduced",))[0]
    forest = rf.pipeline.named_steps["model"]
    expected = decision_tree_params(rf.n_train)

    assert isinstance(forest, RandomForestClassifier)
    assert forest.n_estimators == N_ESTIMATORS
    assert forest.min_samples_leaf == expected["min_samples_leaf"]
    assert forest.max_depth == expected["max_depth"]
    assert forest.random_state == CONFIG.random_seed


def test_random_forest_uses_balanced_not_balanced_subsample(data):
    """`'balanced_subsample'` tính lại trọng số trên từng mẫu bootstrap.

    Khi đó tỉ số phạt dao động quanh 11,39 thay vì đúng bằng nó, và Random
    Forest không còn nhận cùng mức phạt với ba thuật toán kia — task 12 sẽ so
    trên hai sân khác nhau mà không có gì trong bảng để lộ ra.
    """
    model = train_random_forest(data, feature_sets=("reduced",))[0]

    assert model.pipeline.named_steps["model"].class_weight == "balanced"


def test_random_forest_builds_the_shared_number_of_trees(data):
    model = train_random_forest(data, feature_sets=("reduced",))[0]

    assert len(model.pipeline.named_steps["model"].estimators_) == N_ESTIMATORS


def test_random_forest_is_reproducible(data):
    """Cùng seed → cùng model, trong giới hạn của số dấu phẩy động.

    KHÔNG đòi trùng khít từng bit, và đây là lý do có đo: với `n_jobs=-1`,
    `predict_proba` cộng dồn kết quả các cây theo thứ tự do bộ lập lịch quyết
    định, mà phép cộng số thực không kết hợp. Đo được lệch tối đa **1,1e-16**
    giữa hai lần chạy — dưới ngưỡng biểu diễn của `float64`.

    Vì sao vẫn phải để dung sai chứ không so `==`: PR-AUC là chỉ số **xếp
    hạng**, nên một chênh lệch 1e-16 đủ để đảo thứ tự hai hồ sơ có xác suất
    sát nhau và làm chỉ số nhích một chút. Test này từng trượt đúng một lần,
    khi máy đang chạy song song một tác vụ nặng khác.

    Ngưỡng 1e-9 vẫn chặt hơn nhiều so với yêu cầu tái lập của F06 task 6
    (metric trùng tới 4 chữ số thập phân).
    """
    a = train_random_forest(data, feature_sets=("reduced",))[0]
    b = train_random_forest(data, feature_sets=("reduced",))[0]

    assert a.pr_auc == pytest.approx(b.pr_auc, abs=1e-9)


def test_random_forest_rows_are_labelled_with_its_own_name(data):
    models = train_random_forest(data)

    assert all(m.algo == RANDOM_FOREST for m in models)
    assert models[0].slug == "ml02_random_forest_reduced"


def test_random_forest_beats_the_random_floor(data):
    model = train_random_forest(data, feature_sets=("reduced",))[0]

    assert model.pr_auc > model.metrics_validation["base_rate"]


# ----------------------------------------------------- task 10 · XGBoost
def test_xgboost_uses_scale_pos_weight_not_class_weight(data):
    """XGBoost KHÔNG có `class_weight` — cân bằng lớp qua `scale_pos_weight`.

    Con số phải đến từ task 4 và được tính trên RIÊNG tập train. Hardcode
    11,3872 (số đo trên toàn bộ dataset) là để thống kê của validation/test
    góp phần định hình hàm mất mát.
    """
    model = train_xgboost(data, feature_sets=("reduced",))[0]
    estimator = model.pipeline.named_steps["model"]

    assert isinstance(estimator, XGBClassifier)
    assert estimator.get_params().get("class_weight") in (None, "deprecated")
    assert estimator.scale_pos_weight == pytest.approx(
        scale_pos_weight_from(data.y_train))


def test_xgboost_penalty_matches_the_other_three_algorithms(data):
    """Bốn thuật toán phải nhận CÙNG tỉ số phạt dù dùng hai cơ chế khác tên.

    Task 4 đã kiểm bằng số: `class_weight='balanced'` cho tỉ số trọng số trùng
    khít `scale_pos_weight` tới sáu chữ số. Test này canh lại ở tầng train —
    lệch nhau thì task 12 so trên hai sân khác nhau mà bảng không lộ ra.
    """
    xgb = train_xgboost(data, feature_sets=("reduced",))[0]
    weights = compute_class_weight(
        "balanced", classes=np.array([0, 1]), y=data.y_train)

    assert xgb.pipeline.named_steps["model"].scale_pos_weight == pytest.approx(
        weights[1] / weights[0], rel=1e-9)


def test_xgboost_optimises_the_metric_it_is_judged_by(data):
    """`eval_metric='aucpr'` khớp chỉ số CHỌN MODEL của ML02.

    Để mặc định `logloss` thì thứ XGBoost tối ưu bên trong lệch khỏi thứ đem
    đi so ở task 12.
    """
    model = train_xgboost(data, feature_sets=("reduced",))[0]

    assert model.pipeline.named_steps["model"].eval_metric == "aucpr"


def test_xgboost_does_not_use_early_stopping(data):
    """Early stopping cần dừng theo validation — chính tập dùng để báo cáo.

    Dừng theo nó rồi lại chấm trên nó là chọn tham số trên tập đánh giá, và
    con số báo cáo thành lạc quan hơn thực tế.
    """
    model = train_xgboost(data, feature_sets=("reduced",))[0]
    params = model.pipeline.named_steps["model"].get_params()

    assert params.get("early_stopping_rounds") is None
    assert "early_stopping_rounds" not in XGBOOST_PARAMS


def test_xgboost_config_is_fixed_not_tuned():
    """Cấu hình đặt trước, không dò trên validation.

    `learning_rate=0.1` + `n_estimators=200` là bản thay cho mặc định 0.3 + 100:
    bước nhỏ hơn, nhiều bước hơn — cách khắc phục sách vở cho boosting học thuộc.
    """
    assert XGBOOST_PARAMS["learning_rate"] == 0.1
    assert XGBOOST_PARAMS["n_estimators"] == 200
    assert XGBOOST_PARAMS["max_depth"] == 6


def test_xgboost_is_reproducible(data):
    a = train_xgboost(data, feature_sets=("reduced",))[0]
    b = train_xgboost(data, feature_sets=("reduced",))[0]

    assert a.pr_auc == pytest.approx(b.pr_auc)


def test_xgboost_rows_are_labelled_with_its_own_name(data):
    models = train_xgboost(data)

    assert all(m.algo == XGBOOST for m in models)
    assert models[0].slug == "ml02_xgboost_reduced"


def test_xgboost_beats_the_random_floor(data):
    model = train_xgboost(data, feature_sets=("reduced",))[0]

    assert model.pr_auc > model.metrics_validation["base_rate"]


# ------------------------------------------------- bốn thuật toán đã đủ
def test_all_four_algorithms_are_implemented():
    """Task 7–10 đã xong. So sánh và chọn model là task 12, 14."""
    names = {n for n in dir(train_module) if n.startswith("train_")}

    assert {"train_decision_tree", "train_bagging",
            "train_random_forest", "train_xgboost"} <= names
    assert ALGORITHMS == (DECISION_TREE, BAGGING, RANDOM_FOREST, XGBOOST)


def test_training_module_does_not_select_a_best_model():
    """Chọn model là task 14 — tầng train không được kết luận thay.

    Có một hàm `select_best` ở đây thì task 12–14 dễ gọi nó và bỏ qua phần
    đánh giá độc lập, mà đó mới là chỗ ràng buộc "không dùng test để chọn".
    """
    names = {n.lower() for n in dir(train_module)}

    assert not any("select" in n or "best" in n for n in names), names
