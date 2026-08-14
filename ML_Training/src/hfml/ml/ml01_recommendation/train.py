"""Train & đánh giá ML01 — 4 thuật toán + baseline (F03 task 5–15, PLAN.md §6.3).

Ba tập, ba vai trò khác nhau (task 5)
-------------------------------------
    train  80%   CV 5-fold trên đây để SO SÁNH 4 thuật toán và CHỌN model
    test   20%   khoá lại, chấm ĐÚNG MỘT LẦN cho model đã chọn

Vì sao không thể gộp: nếu chọn model bằng chính con số đem đi báo cáo thì
con số đó lạc quan có hệ thống — nó là điểm cao nhất trong 4 lần thử, và
"cao nhất trong nhiều lần thử" luôn lệch lên. Ở đây độ lệch nhỏ (4 ứng viên,
khoảng cách rõ), nhưng cấu trúc phải đúng thì mới trả lời được "tập test
đâu?".

`StratifiedKFold` bên trong tập train đóng vai VALIDATION — không cần cắt
thêm một tập validation cố định, vì mỗi fold đã luân phiên làm việc đó, và
với 4 lớp thì CV cho ước lượng ổn định hơn một lần cắt duy nhất.

Model export ra artifact được fit trên tập TRAIN, không phải toàn bộ dữ liệu.
Đổi lại 20% dữ liệu, ta có được điều quan trọng hơn: chỉ số trong
`metadata.json` mô tả **đúng cái model nằm trong file đó**. Fit lại trên 100%
rồi gắn chỉ số đo từ model khác là một sự lệch nhỏ nhưng không giải thích nổi.

Điều kiện để bảng so sánh có nghĩa
----------------------------------
Bốn thuật toán phải chạy trên **cùng split, cùng feature, cùng seed**. Ở đây
điều đó không phải lời hứa mà là cấu trúc code: `_materialise_folds()` sinh
danh sách fold MỘT LẦN rồi mọi thuật toán nhận đúng danh sách ấy. Không chỗ
nào gọi `StratifiedKFold` lần thứ hai.

Nếu mỗi thuật toán tự chia dữ liệu thì chênh lệch macro-F1 giữa chúng lẫn với
chênh lệch giữa các cách chia — và không cách nào tách hai thứ đó ra nữa.

Vì sao Bagging và RandomForest cùng 100 cây
-------------------------------------------
Hai thuật toán này chỉ khác nhau ở chỗ RF lấy mẫu ngẫu nhiên FEATURE tại mỗi
lần chẻ nhánh. Để số cây khác nhau thì bảng so sánh không còn trả lời được
"lấy mẫu feature có giúp gì không" — chênh lệch có thể chỉ do số cây.

`DecisionTreeClassifier` thì cố ý để KHÔNG giới hạn độ sâu: vai trò của nó
trong báo cáo (PLAN.md §6.3) là cho thấy cây đơn overfit thế nào, cắt tỉa nó
là bỏ mất điều cần cho thấy.

Chỉ số chọn model là Macro-F1, không phải accuracy
--------------------------------------------------
Bốn lớp không cân bằng (15%–32%). Accuracy vẫn được tính và vào bảng, nhưng
model nào được chọn thì do macro-F1 quyết định (PLAN.md §11).
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from typing import Callable, Final

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from hfml.config import CONFIG
from hfml.data.preprocessing.pipeline import build_preprocessing_pipeline
from hfml.data.synthetic import PopulationParams, generate_households
from hfml.logger import get_logger, log_run_context
from hfml.ml.estimator import PipelineClassifier
from hfml.ml.evaluation.metrics import (
    SELECTION_METRIC,
    aggregate_folds,
    classification_metrics,
    confusion_table,
    per_class_table,
)
from hfml.ml.evaluation.tracking import append_results
from hfml.ml.ml01_recommendation.labeler import (
    ORDERED_GROUPS,
    RAW_FEATURES,
    add_label_noise,
    class_distribution,
    label_frame,
)
from hfml.ml.registry import load_model, save_model

log = get_logger(__name__)

TASK: Final[str] = "ml01"

#: Số cây dùng chung cho Bagging và RandomForest — xem docstring đầu file.
N_ESTIMATORS: Final[int] = 100

#: Tên baseline trong bảng so sánh. Không phải model dự tuyển, nên
#: `select_best()` loại nó ra.
BASELINE: Final[str] = "baseline_dummy"

#: Tên thuật toán của task 7 trong `ALGORITHMS` và trong bảng kết quả.
DECISION_TREE: Final[str] = "decision_tree"

#: Tên thuật toán của task 8.
BAGGING: Final[str] = "bagging"

#: Tên thuật toán của task 9.
RANDOM_FOREST: Final[str] = "random_forest"

#: Tên thuật toán của task 10.
XGBOOST: Final[str] = "xgboost"


def _decision_tree(seed: int) -> BaseEstimator:
    # Không giới hạn độ sâu — cố ý, xem docstring đầu file.
    return DecisionTreeClassifier(random_state=seed)


def _bagging(seed: int) -> BaseEstimator:
    return BaggingClassifier(
        estimator=DecisionTreeClassifier(random_state=seed),
        n_estimators=N_ESTIMATORS, random_state=seed, n_jobs=-1)


def _random_forest(seed: int) -> BaseEstimator:
    """RandomForest MẶC ĐỊNH — `max_features='sqrt'`, cố ý không chỉnh.

    Ở bài toán này RF **thua Bagging** (macro-F1 0,853 so với 0,910) và gần
    như ngang cây đơn (0,851) — ngược với kỳ vọng thông thường. Đo thử ngày
    11/08/2026 cho thấy nguyên nhân nằm đúng ở lấy mẫu feature:

        max_features='sqrt'  (~4/17 cột)   macro-F1  0,8529
        max_features=0.5     (~8/17 cột)             0,8964
        max_features=None    (17/17 cột)             0,9106  ≈ Bagging 0,9097

    Ranh giới của `g(·)` là các TỈ LỆ (`tiết kiệm ÷ chi tiêu`,
    `trả nợ ÷ thu nhập`). Muốn xấp xỉ một tỉ lệ bằng lát cắt song song trục
    thì cây phải chẻ luân phiên trên tử số và mẫu số. Cho mỗi lát cắt chỉ
    thấy 4 trong 17 cột thì phần lớn lần chẻ không có sẵn cả hai, nên lấy
    mẫu feature ở đây **phá** nhiều hơn giúp.

    Vì vậy giữ nguyên `sqrt`: đặt `max_features=None` sẽ biến RF thành đúng
    Bagging, và cột RF trong bảng so sánh mất hết nội dung. Chênh lệch này
    chính là phần trả lời cho "lấy mẫu ngẫu nhiên feature giúp được gì" —
    câu trả lời ở đây là *không*, và nói được vì sao thì giá trị hơn một
    bảng bốn dòng đều đẹp.
    """
    return RandomForestClassifier(
        n_estimators=N_ESTIMATORS, random_state=seed, n_jobs=-1)


def _xgboost(seed: int) -> BaseEstimator:
    return XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.1,
        subsample=0.9, colsample_bytree=0.9,
        tree_method="hist", eval_metric="mlogloss",
        random_state=seed, n_jobs=-1)


def _baseline(seed: int) -> BaseEstimator:
    """Baseline bắt buộc của PLAN.md §6.3 — `DummyClassifier(strategy='stratified')`.

    Nó **không nhìn `X`**: mỗi dự đoán là một lần rút ngẫu nhiên theo đúng tỉ
    lệ 4 lớp của tập train. Đó chính là điều làm nó thành mốc so sánh — mọi
    thứ một model thật đạt được VƯỢT trên mức này là phần do feature đóng góp,
    không phải do cấu trúc lớp.

    Hai giá trị kỳ vọng, tính được bằng tay và dùng để kiểm baseline có chạy
    đúng bản chất không (đo trên train 16.000 dòng, seed 42):

        accuracy  =  Σ pᵢ²   =  0,2721      đo được 0,2698
        macro-F1  =  1 / k   =  0,2500      đo được 0,2483

    Macro-F1 ra đúng `1/k` bất kể lớp cân bằng hay không: với dự đoán rút theo
    tỉ lệ, mỗi lớp có precision = recall = pᵢ nên F1ᵢ = pᵢ, và trung bình của
    pᵢ trên k lớp luôn là 1/k vì Σpᵢ = 1. Vì vậy **0,25 là mốc cứng** cho ML01:
    model nào có macro-F1 dưới 0,25 thì kém hơn đoán mò.

    Accuracy thì ngược lại, phụ thuộc phân bố (Σpᵢ² = 0,272 ở đây, sẽ cao hơn
    nếu lớp lệch hơn) — thêm một lý do nữa để chọn model bằng macro-F1.
    """
    return DummyClassifier(strategy="stratified", random_state=seed)


#: Bốn thuật toán của PLAN.md §6.3 cộng baseline. Thứ tự = thứ tự trong bảng.
ALGORITHMS: Final[dict[str, Callable[[int], BaseEstimator]]] = {
    BASELINE: _baseline,
    "decision_tree": _decision_tree,
    "bagging": _bagging,
    "random_forest": _random_forest,
    "xgboost": _xgboost,
}


# --------------------------------------------------------------- dữ liệu

def build_training_data(
    params: PopulationParams | None = None,
    seed: int = 42,
    noise_rate: float = 0.03,
    boundary_width: float = 0.10,
) -> tuple[pd.DataFrame, pd.Series]:
    """Sinh dân số, gán nhãn `g(·)`, thêm nhiễu. Trả `(X, y)`.

    Biến mục tiêu `y` là **nhóm định hướng tài chính** (Financial Recommendation
    Group) — `y.name == "recommendation_group"`, nhận một trong bốn giá trị
    EMERGENCY · DEBT_FOCUS · BUILD_BUFFER · GROWTH. Không phải điểm hay mức độ
    sức khỏe tài chính; xem docstring đầu `labeler.py`.

    `X` lấy đúng `RAW_FEATURES` chứ không phải `df` — đó là chỗ chặn rò rỉ
    nhãn cuối cùng trước khi dữ liệu vào model (PLAN.md §6.1c).
    """
    df = generate_households(params, seed=seed)
    y = add_label_noise(label_frame(df), df, rate=noise_rate,
                        boundary_width=boundary_width, seed=seed)
    return df[list(RAW_FEATURES)], y


def split_train_val_test(
    X: pd.DataFrame,
    y: pd.Series,
    val_size: float | None = None,
    test_size: float | None = None,
    seed: int | None = None,
):
    """Chia ba tập của task 5 (chốt lại 12/08/2026): 70% / 15% / 15%.

        train       70%   `StratifiedKFold` 5-fold chạy trên đây — vẫn là
                          căn cứ CHỌN MODEL, và là chỗ duy nhất sinh ra σ
                          giữa các fold mà task 14 dùng để đo cách biệt
        validation  15%   chấm một lần để đối chiếu với CV; KHÔNG tham gia
                          chọn model
        test        15%   khoá lại, CHỈ dùng cho đánh giá cuối

    CV 5-fold được giữ nguyên chứ không bị validation thay thế. Hai thứ trả
    lời hai câu khác nhau: CV cho biết chỉ số dao động bao nhiêu giữa các lát
    cắt (nên mới có σ), validation cho một con số trên tập chưa từng fit. Bỏ
    CV thì tiêu chí "hơn á quân bao nhiêu lần σ" của task 14 mất căn cứ.

    Cắt hai lần chứ không một lần: lần đầu tách `test`, lần sau tách
    `validation` ra khỏi phần còn lại. Tỉ lệ lần hai phải quy đổi theo phần
    còn lại (`0.15 / 0.85`) chứ không phải 0.15 — lấy thẳng 0.15 của phần còn
    lại thì validation chỉ còn 12,75% tổng.

    `stratify` ở cả hai lần là bắt buộc: lớp nhỏ nhất chiếm ~15%, cắt ngẫu
    nhiên thì tỉ lệ 4 lớp giữa ba tập lệch nhau và các chỉ số hết so được.
    """
    val_size = CONFIG.training["val_size"] if val_size is None else val_size
    test_size = CONFIG.training["test_size"] if test_size is None else test_size
    seed = CONFIG.random_seed if seed is None else seed

    if val_size + test_size >= 1.0:
        raise ValueError(
            f"val_size ({val_size}) + test_size ({test_size}) phải nhỏ hơn 1.0 — "
            "không còn dòng nào cho tập train.")

    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed)

    # Quy đổi sang tỉ lệ CỦA PHẦN CÒN LẠI để validation đúng bằng `val_size`
    # của tập gốc.
    val_of_rest = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest, test_size=val_of_rest, stratify=y_rest, random_state=seed)

    total = len(X)
    log.info("Tách dữ liệu: train %d (%.0f%%) · validation %d (%.0f%%) · test %d (%.0f%%)",
             len(X_train), len(X_train) / total * 100,
             len(X_val), len(X_val) / total * 100,
             len(X_test), len(X_test) / total * 100)
    return X_train, X_val, X_test, y_train, y_val, y_test


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float | None = None,
    seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Chỉ tập train (70%) và tập test (15%) — bỏ qua validation.

    Dùng cho những chỗ chỉ cần hai vế đó: CV chạy trên train, đánh giá cuối
    chạy trên test. Tập validation KHÔNG bị trộn vào train — nó bị bỏ đi khỏi
    giá trị trả về, nên `len(train) + len(test)` nhỏ hơn `len(X)` đúng bằng
    phần validation. Cần cả ba thì gọi `split_train_val_test()`.

    Cắt giống hệt `split_train_val_test()` với cùng seed, nên tập train ở đây
    và tập train ở đó là một — chỉ số CV không phụ thuộc vào việc gọi hàm nào.

    Tập test không được tham gia CV, chọn model, hay tinh chỉnh siêu tham số.
    `test_model_selection_never_sees_the_test_set` canh ràng buộc này.
    """
    X_train, _, X_test, y_train, _, y_test = split_train_val_test(
        X, y, test_size=test_size, seed=seed)
    return X_train, X_test, y_train, y_test


def _materialise_folds(X: pd.DataFrame, y: pd.Series, n_splits: int, seed: int):
    """Sinh fold MỘT LẦN để mọi thuật toán dùng chung — xem docstring đầu file."""
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(X, y))


# ------------------------------------------------------------ đánh giá

def cross_validate(
    X: pd.DataFrame,
    y: pd.Series,
    algorithms: dict[str, Callable[[int], BaseEstimator]] | None = None,
    n_splits: int | None = None,
    seed: int | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """Chạy CV cho từng thuật toán trên cùng bộ fold.

    Trả về `(bảng so sánh, dự đoán out-of-fold)`. Dự đoán OOF cho phép dựng
    confusion matrix và bảng per-class trên TOÀN bộ dữ liệu mà mỗi dòng vẫn
    được dự đoán bởi model chưa từng thấy nó — trung thực hơn hẳn việc fit
    lại rồi chấm trên chính tập train.
    """
    algorithms = algorithms or ALGORITHMS
    seed = CONFIG.random_seed if seed is None else seed
    n_splits = CONFIG.training["n_splits"] if n_splits is None else n_splits

    labels = [g.value for g in ORDERED_GROUPS]
    folds = _materialise_folds(X, y, n_splits, seed)
    preprocessing = build_preprocessing_pipeline()

    rows: list[dict] = []
    oof: dict[str, pd.Series] = {}

    for algo, factory in algorithms.items():
        fold_metrics: list[dict[str, float]] = []
        predictions = pd.Series(index=X.index, dtype=object)
        started = time.perf_counter()

        for train_idx, valid_idx in folds:
            model = PipelineClassifier(
                task=TASK, algo=algo,
                estimator=factory(seed), preprocessing=preprocessing)
            model.fit(X.iloc[train_idx], y.iloc[train_idx])

            predicted = model.predict(X.iloc[valid_idx])
            predictions.iloc[valid_idx] = predicted
            fold_metrics.append(classification_metrics(
                y.iloc[valid_idx], predicted, labels=labels))

        rows.append({"algo": algo, **aggregate_folds(fold_metrics),
                     "fit_seconds": round(time.perf_counter() - started, 2)})
        oof[algo] = predictions
        log.info("%-16s macro-F1 %.4f · accuracy %.4f (%.1fs)",
                 algo, rows[-1]["macro_f1"], rows[-1]["accuracy"],
                 rows[-1]["fit_seconds"])

    return pd.DataFrame(rows), oof


# ------------------------------------------- task 7 — Decision Tree

def train_decision_tree(
    params: PopulationParams | None = None,
    seed: int | None = None,
    n_splits: int | None = None,
    test_size: float | None = None,
    save: bool = True,
    runs_dir=None,
) -> dict:
    """Task 7 — train `DecisionTreeClassifier` theo giao thức chốt ở task 5.

    Trình tự:

        1. tách 80/20 bằng `split_train_test()` (task 5)
        2. CV 5-fold **chỉ trên tập train**, chỉ chạy Decision Tree
        3. fit cây cuối cùng trên **toàn bộ** tập train
        4. ghi artifact + metadata vào `src/training/runs/`

    Tập test cố ý **không nằm trong giá trị trả về**. Task 7 không được dùng
    nó, và cách chắc chắn nhất để không dùng nhầm là không đưa nó ra khỏi
    hàm — tách lại bằng cùng `seed` thì luôn ra đúng tập cũ, nên không mất gì.

    Bước 3 không thừa so với bước 2: CV cho *ước lượng* chỉ số bằng 5 cây,
    mỗi cây học từ 4/5 tập train. Cây đem dùng phải học từ cả 5/5, và đó là
    một cây khác với cả năm cây kia.

    Bước 4 được bổ sung sau: task 7 vốn được giao trước khi chốt quy ước ghi
    output, nên ba thuật toán sau có artifact còn cây đơn thì không. Việc
    train và chỉ số của nó **không đổi** — chỉ thêm phần lưu lại kết quả.

    Cấu hình lần chạy ghi kèm cả `max_depth=None` lẫn độ sâu THỰC TẾ của cây
    đã fit: hai con số này nói hai chuyện khác nhau, và chính khoảng cách
    giữa chúng là bằng chứng cây đơn mọc tự do tới đâu (PLAN.md §6.3).
    """
    seed = CONFIG.random_seed if seed is None else seed
    n_splits = CONFIG.training["n_splits"] if n_splits is None else n_splits
    test_size = CONFIG.training["test_size"] if test_size is None else test_size
    runs_dir = CONFIG.paths.runs if runs_dir is None else runs_dir
    log_run_context(log)

    X, y = build_training_data(params, seed=seed)
    X_train, _, y_train, _ = split_train_test(X, y, test_size, seed)

    comparison, oof = cross_validate(
        X_train, y_train,
        algorithms={DECISION_TREE: ALGORITHMS[DECISION_TREE]},
        n_splits=n_splits, seed=seed)

    model = PipelineClassifier(
        task=TASK, algo=DECISION_TREE,
        estimator=ALGORITHMS[DECISION_TREE](seed),
        preprocessing=build_preprocessing_pipeline())
    model.fit(X_train, y_train)

    estimator = model.pipeline_.named_steps["model"]
    tree = estimator.tree_
    run_config = {
        "estimator": "DecisionTreeClassifier",
        "criterion": str(estimator.criterion),
        "max_depth": estimator.max_depth,          # None = không cắt tỉa
        "fitted_depth": int(estimator.get_depth()),
        "n_nodes": int(tree.node_count),
        "n_leaves": int(estimator.get_n_leaves()),
        "min_samples_split": int(estimator.min_samples_split),
        "min_samples_leaf": int(estimator.min_samples_leaf),
        "n_splits": int(n_splits),
        "test_size": float(test_size),
        "n_train_rows": int(len(X_train)),
    }
    log.info("Cây cuối: %d dòng train · %d nút · sâu %d",
             int(tree.n_node_samples[0]), run_config["n_nodes"],
             run_config["fitted_depth"])

    result = {
        "X_train": X_train,
        "y_train": y_train,
        "cv_metrics": comparison.iloc[0].to_dict(),
        "config": run_config,
        "oof": oof[DECISION_TREE],
        "model": model,
    }

    if save:
        cv_metrics = comparison.iloc[0].to_dict()
        cv_metrics.pop("algo", None)
        result["artifact"] = save_model(
            model, metrics={"cv": cv_metrics},
            directory=runs_dir, extra={"config": run_config})

        # Task 7 từng bỏ qua bước này, nên dòng CV của decision_tree chỉ xuất
        # hiện khi task 12 backfill. Hậu quả im lặng: đổi cấu hình split rồi
        # train lại, ba model kia có dòng mới còn decision_tree vẫn giữ dòng
        # backfill CŨ — bảng so sánh trộn hai cỡ dữ liệu mà không báo gì.
        logged = comparison.copy()
        logged["split"] = "cv_train"
        logged.insert(0, "task", TASK)
        logged.insert(2, "feature_set", "default")
        logged.insert(3, "random_seed", seed)
        logged.insert(4, "n_rows", len(X_train))
        logged.insert(5, "n_splits", n_splits)
        logged["note"] = "task7 decision_tree"
        result["results_csv"] = append_results(logged, runs_dir / "results.csv")

    return result


# ----------------------------------------------- task 8 — Bagging

def train_bagging(
    params: PopulationParams | None = None,
    seed: int | None = None,
    n_splits: int | None = None,
    test_size: float | None = None,
    save: bool = True,
    runs_dir=None,
) -> dict:
    """Task 8 — train `BaggingClassifier` theo giao thức chốt ở task 5.

    Cùng trình tự với task 7: tách 80/20 → CV 5-fold **chỉ trên train** →
    fit model cuối trên **toàn bộ** train. Tập test không được trả ra khỏi
    hàm, nên task này không thể lỡ dùng nó.

    Khác task 7 ở chỗ có ghi output: theo quy ước kiến trúc, metric, cấu hình
    và model artifact của một lần train nằm ở `src/training/runs/`
    (`CONFIG.paths.runs`). File log KHÔNG thuộc đó — log vẫn đi `logs/`.

    Cấu hình lần chạy được ghi vào `metadata.json` chứ không chỉ ghi metric:
    biết Bagging đạt macro-F1 bao nhiêu mà không biết nó chạy với mấy cây thì
    con số đó không dựng lại được.
    """
    seed = CONFIG.random_seed if seed is None else seed
    n_splits = CONFIG.training["n_splits"] if n_splits is None else n_splits
    test_size = CONFIG.training["test_size"] if test_size is None else test_size
    runs_dir = CONFIG.paths.runs if runs_dir is None else runs_dir
    log_run_context(log)

    X, y = build_training_data(params, seed=seed)
    X_train, _, y_train, _ = split_train_test(X, y, test_size, seed)

    comparison, oof = cross_validate(
        X_train, y_train,
        algorithms={BAGGING: ALGORITHMS[BAGGING]},
        n_splits=n_splits, seed=seed)

    model = PipelineClassifier(
        task=TASK, algo=BAGGING,
        estimator=ALGORITHMS[BAGGING](seed),
        preprocessing=build_preprocessing_pipeline())
    model.fit(X_train, y_train)

    estimator = model.pipeline_.named_steps["model"]
    run_config = {
        "estimator": "BaggingClassifier",
        "base_estimator": "DecisionTreeClassifier",
        "n_estimators": int(estimator.n_estimators),
        "bootstrap": bool(estimator.bootstrap),
        "n_splits": int(n_splits),
        "test_size": float(test_size),
        "n_train_rows": int(len(X_train)),
    }
    cv_metrics = comparison.iloc[0].to_dict()
    cv_metrics.pop("algo", None)

    result = {
        "X_train": X_train,
        "y_train": y_train,
        "cv_metrics": comparison.iloc[0].to_dict(),
        "oof": oof[BAGGING],
        "model": model,
        "config": run_config,
    }

    if save:
        result["artifact"] = save_model(
            model, metrics={"cv": cv_metrics},
            directory=runs_dir, extra={"config": run_config})

        logged = comparison.copy()
        logged["split"] = "cv_train"
        logged.insert(0, "task", TASK)
        logged.insert(2, "feature_set", "default")
        logged.insert(3, "random_seed", seed)
        logged.insert(4, "n_rows", len(X_train))
        logged.insert(5, "n_splits", n_splits)
        logged["note"] = "task8 bagging"
        result["results_csv"] = append_results(logged, runs_dir / "results.csv")

    return result


# ------------------------------------------ task 9 — Random Forest

def train_random_forest(
    params: PopulationParams | None = None,
    seed: int | None = None,
    n_splits: int | None = None,
    test_size: float | None = None,
    save: bool = True,
    runs_dir=None,
) -> dict:
    """Task 9 — train `RandomForestClassifier` theo giao thức chốt ở task 5.

    Cùng trình tự task 7 và 8: tách 80/20 → CV 5-fold **chỉ trên train** →
    fit model cuối trên **toàn bộ** train. Tập test không trả ra khỏi hàm.

    Cấu hình ghi lại thêm `max_features` — với RandomForest đó là tham số
    phân biệt nó với Bagging, nên bỏ nó khỏi bản ghi thì lần chạy này không
    dựng lại được. Ghi cả giá trị đã quy đổi (`'sqrt'` → số cột thực tế mỗi
    lát cắt được nhìn), vì đó mới là con số đọc hiểu được.
    """
    seed = CONFIG.random_seed if seed is None else seed
    n_splits = CONFIG.training["n_splits"] if n_splits is None else n_splits
    test_size = CONFIG.training["test_size"] if test_size is None else test_size
    runs_dir = CONFIG.paths.runs if runs_dir is None else runs_dir
    log_run_context(log)

    X, y = build_training_data(params, seed=seed)
    X_train, _, y_train, _ = split_train_test(X, y, test_size, seed)

    comparison, oof = cross_validate(
        X_train, y_train,
        algorithms={RANDOM_FOREST: ALGORITHMS[RANDOM_FOREST]},
        n_splits=n_splits, seed=seed)

    model = PipelineClassifier(
        task=TASK, algo=RANDOM_FOREST,
        estimator=ALGORITHMS[RANDOM_FOREST](seed),
        preprocessing=build_preprocessing_pipeline())
    model.fit(X_train, y_train)

    estimator = model.pipeline_.named_steps["model"]
    run_config = {
        "estimator": "RandomForestClassifier",
        "n_estimators": int(estimator.n_estimators),
        "max_features": str(estimator.max_features),
        # sklearn 1.9 bỏ `RandomForestClassifier.max_features_`; giá trị đã
        # quy đổi chỉ còn trên từng cây con. Đọc từ đó thay vì tự tính lại —
        # tự tính là chép luật quy đổi của sklearn ra chỗ thứ hai, rồi lệch.
        "max_features_resolved": int(estimator.estimators_[0].max_features_),
        "n_features_in": int(estimator.n_features_in_),
        "bootstrap": bool(estimator.bootstrap),
        "n_splits": int(n_splits),
        "test_size": float(test_size),
        "n_train_rows": int(len(X_train)),
    }
    cv_metrics = comparison.iloc[0].to_dict()
    cv_metrics.pop("algo", None)

    log.info("Rừng cuối: %d cây · mỗi lát cắt nhìn %d/%d cột",
             run_config["n_estimators"], run_config["max_features_resolved"],
             run_config["n_features_in"])

    result = {
        "X_train": X_train,
        "y_train": y_train,
        "cv_metrics": comparison.iloc[0].to_dict(),
        "oof": oof[RANDOM_FOREST],
        "model": model,
        "config": run_config,
    }

    if save:
        result["artifact"] = save_model(
            model, metrics={"cv": cv_metrics},
            directory=runs_dir, extra={"config": run_config})

        logged = comparison.copy()
        logged["split"] = "cv_train"
        logged.insert(0, "task", TASK)
        logged.insert(2, "feature_set", "default")
        logged.insert(3, "random_seed", seed)
        logged.insert(4, "n_rows", len(X_train))
        logged.insert(5, "n_splits", n_splits)
        logged["note"] = "task9 random_forest"
        result["results_csv"] = append_results(logged, runs_dir / "results.csv")

    return result


# ----------------------------------------------- task 10 — XGBoost

def train_xgboost(
    params: PopulationParams | None = None,
    seed: int | None = None,
    n_splits: int | None = None,
    test_size: float | None = None,
    save: bool = True,
    runs_dir=None,
) -> dict:
    """Task 10 — train `XGBClassifier` theo giao thức chốt ở task 5.

    Cùng trình tự task 7–9: tách 80/20 → CV 5-fold **chỉ trên train** → fit
    model cuối trên **toàn bộ** train. Tập test không trả ra khỏi hàm.

    Cấu hình ghi lại đầy đủ hơn ba thuật toán trước vì XGBoost có nhiều siêu
    tham số cùng ảnh hưởng kết quả (`learning_rate` × `n_estimators` bù trừ
    nhau, `subsample`/`colsample_bytree` thêm ngẫu nhiên). Thiếu một cái là
    lần chạy không dựng lại được.

    Ghi cả `n_trees`: với `multi:softprob`, XGBoost dựng **một cây cho mỗi
    lớp ở mỗi vòng**, nên 300 vòng × 4 lớp = 1.200 cây. Ai đọc bảng so sánh
    mà thấy "300" cạnh "100 cây" của Bagging/RF sẽ hiểu sai độ phức tạp.
    """
    seed = CONFIG.random_seed if seed is None else seed
    n_splits = CONFIG.training["n_splits"] if n_splits is None else n_splits
    test_size = CONFIG.training["test_size"] if test_size is None else test_size
    runs_dir = CONFIG.paths.runs if runs_dir is None else runs_dir
    log_run_context(log)

    X, y = build_training_data(params, seed=seed)
    X_train, _, y_train, _ = split_train_test(X, y, test_size, seed)

    comparison, oof = cross_validate(
        X_train, y_train,
        algorithms={XGBOOST: ALGORITHMS[XGBOOST]},
        n_splits=n_splits, seed=seed)

    model = PipelineClassifier(
        task=TASK, algo=XGBOOST,
        estimator=ALGORITHMS[XGBOOST](seed),
        preprocessing=build_preprocessing_pipeline())
    model.fit(X_train, y_train)

    estimator = model.pipeline_.named_steps["model"]
    booster = estimator.get_booster()
    rounds = int(booster.num_boosted_rounds())
    run_config = {
        "estimator": "XGBClassifier",
        "objective": str(estimator.get_params()["objective"]),
        "n_estimators": int(estimator.n_estimators),
        "n_classes": int(estimator.n_classes_),
        "n_trees": rounds * int(estimator.n_classes_),
        "max_depth": int(estimator.max_depth),
        "learning_rate": float(estimator.learning_rate),
        "subsample": float(estimator.subsample),
        "colsample_bytree": float(estimator.colsample_bytree),
        "tree_method": str(estimator.tree_method),
        "eval_metric": str(estimator.eval_metric),
        "n_splits": int(n_splits),
        "test_size": float(test_size),
        "n_train_rows": int(len(X_train)),
    }
    cv_metrics = comparison.iloc[0].to_dict()
    cv_metrics.pop("algo", None)

    log.info("XGBoost cuối: %d vòng × %d lớp = %d cây · sâu %d",
             rounds, run_config["n_classes"], run_config["n_trees"],
             run_config["max_depth"])

    result = {
        "X_train": X_train,
        "y_train": y_train,
        "cv_metrics": comparison.iloc[0].to_dict(),
        "oof": oof[XGBOOST],
        "model": model,
        "config": run_config,
    }

    if save:
        result["artifact"] = save_model(
            model, metrics={"cv": cv_metrics},
            directory=runs_dir, extra={"config": run_config})

        logged = comparison.copy()
        logged["split"] = "cv_train"
        logged.insert(0, "task", TASK)
        logged.insert(2, "feature_set", "default")
        logged.insert(3, "random_seed", seed)
        logged.insert(4, "n_rows", len(X_train))
        logged.insert(5, "n_splits", n_splits)
        logged["note"] = "task10 xgboost"
        result["results_csv"] = append_results(logged, runs_dir / "results.csv")

    return result


# -------------------------------------- task 11 — đánh giá trên test

#: Bốn thuật toán dự tuyển, KHÔNG gồm baseline. Thứ tự cố định để bảng kết
#: quả giữa các lần chạy xếp giống nhau.
CONTENDERS: Final[tuple[str, ...]] = (DECISION_TREE, BAGGING, RANDOM_FOREST, XGBOOST)


def evaluate_on_test(
    params: PopulationParams | None = None,
    seed: int | None = None,
    test_size: float | None = None,
    algorithms: tuple[str, ...] = CONTENDERS,
    save: bool = True,
    runs_dir=None,
) -> dict:
    """Task 11 — chấm 4 thuật toán trên **cùng một** tập validation và test.

    Mỗi model được fit trên tập train (70%) rồi dự đoán hai tập giữ riêng:
    validation (15%) và test (15%). Cả bốn thuật toán dùng đúng cùng hai tập
    đó — chúng đến từ `split_train_val_test()` với cùng `seed` — nên chênh
    lệch giữa các model là chênh lệch của model, không phải của dữ liệu.

    Chấm validation ở đây thuần tuý để ĐỐI CHIẾU. Nó không tham gia chọn
    model: task 14 chỉ đọc chỉ số CV. Có ba con số cho cùng một model (CV,
    validation, test) thì lệch bất thường giữa chúng lộ ra ngay, thay vì phải
    tin vào một phép đo duy nhất.

    Hàm này **chỉ đo**. Nó không xếp hạng, không chọn model, không export —
    những việc đó thuộc task sau. Vì vậy giá trị trả về là một `dict` khoá
    theo tên thuật toán, không có trường nào kiểu `best`.

    Kết quả test tuyệt đối không được quay ngược lại ảnh hưởng training hay
    chọn model. Ở đây điều đó được bảo đảm về cấu trúc: hàm nhận danh sách
    thuật toán từ ngoài vào và trả kết quả ra, không có nhánh nào đọc chỉ số
    rồi đổi cách fit.
    """
    seed = CONFIG.random_seed if seed is None else seed
    test_size = CONFIG.training["test_size"] if test_size is None else test_size
    runs_dir = CONFIG.paths.runs if runs_dir is None else runs_dir
    log_run_context(log)

    X, y = build_training_data(params, seed=seed)
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(
        X, y, test_size=test_size, seed=seed)
    labels = [g.value for g in ORDERED_GROUPS]
    log.info("Đánh giá %d thuật toán trên validation %d dòng và test %d dòng",
             len(algorithms), len(X_val), len(X_test))

    per_algo: dict[str, dict] = {}
    validation_metrics: dict[str, dict] = {}
    for algo in algorithms:
        model = PipelineClassifier(
            task=TASK, algo=algo,
            estimator=ALGORITHMS[algo](seed),
            preprocessing=build_preprocessing_pipeline())
        model.fit(X_train, y_train)

        validation_metrics[algo] = classification_metrics(
            y_val, model.predict(X_val), labels=labels)

        predicted = model.predict(X_test)
        metrics = classification_metrics(y_test, predicted, labels=labels)
        per_algo[algo] = {
            "metrics": metrics,
            "per_class": per_class_table(y_test, predicted, labels),
            "confusion": confusion_table(y_test, predicted, labels),
        }
        log.info("%-16s validation macro-F1 %.4f · test macro-F1 %.4f",
                 algo, validation_metrics[algo]["macro_f1"], metrics["macro_f1"])

    summary = pd.DataFrame(
        [{"algo": algo, **per_algo[algo]["metrics"]} for algo in algorithms])
    validation_summary = pd.DataFrame(
        [{"algo": algo, **validation_metrics[algo]} for algo in algorithms])

    result = {
        "X_test": X_test,
        "y_test": y_test,
        "X_val": X_val,
        "y_val": y_val,
        "labels": labels,
        "per_algo": per_algo,
        "summary": summary,
        "validation_summary": validation_summary,
    }

    if save:
        result["files"] = _save_test_evaluation(
            per_algo, summary, algorithms, runs_dir, seed, len(X_test),
            validation_summary=validation_summary, n_val=len(X_val))
    return result


def _save_test_evaluation(per_algo, summary, algorithms, runs_dir, seed, n_test,
                          validation_summary=None, n_val=None):
    """Ghi kết quả đánh giá ra `src/training/runs/`.

    Bảng per-class và confusion của 4 model gộp vào MỘT file mỗi loại, phân
    biệt bằng cột `algo` — tám file rời cho cùng một lần chạy chỉ làm thư mục
    khó đọc, mà đọc được vẫn phải mở từng cái để ghép lại.

    Chỉ số validation đi chung `results.csv` với `split="validation"`, không
    ra file riêng: nó là cùng một loại số với `cv_train` và `test`, tách file
    thì so ba tập lại phải tự ghép.
    """
    runs_dir.mkdir(parents=True, exist_ok=True)

    def _tag(frame, split: str, n_rows: int, note: str):
        tagged = frame.copy()
        tagged["split"] = split
        tagged["n_rows"] = n_rows
        tagged["note"] = note
        return tagged

    rows = []
    if validation_summary is not None:
        rows.append(_tag(validation_summary, "validation", n_val,
                         "task11 validation"))
    rows.append(_tag(summary, "test", n_test, "task11 test evaluation"))

    # `append_results` sắp lại cột theo `LEADING_COLUMNS`, nên ở đây chỉ cần
    # có đủ cột chứ không cần chèn đúng vị trí.
    logged = pd.concat(rows, ignore_index=True)
    logged["task"] = TASK
    logged["feature_set"] = "default"
    logged["random_seed"] = seed
    files = {"results": append_results(logged, runs_dir / "results.csv")}

    per_class = pd.concat(
        [per_algo[a]["per_class"].assign(algo=a) for a in algorithms],
        ignore_index=True)
    per_class = per_class[["algo", "label", "precision", "recall", "f1", "support"]]
    files["per_class"] = runs_dir / "test_per_class.csv"
    per_class.to_csv(files["per_class"], index=False, encoding="utf-8")

    confusion = pd.concat(
        [per_algo[a]["confusion"].reset_index().assign(algo=a) for a in algorithms],
        ignore_index=True)
    files["confusion"] = runs_dir / "test_confusion.csv"
    confusion.to_csv(files["confusion"], index=False, encoding="utf-8")

    log.info("Ghi đánh giá test → %s", runs_dir)
    return files


# ------------------------------------------ task 12 — so sánh model

#: Ba chỉ số đưa vào bảng so sánh, kèm khoảng cách CV → test cho mỗi cái.
COMPARISON_METRICS: Final[tuple[str, ...]] = (
    "accuracy", "macro_f1", "balanced_accuracy")


def load_results(runs_dir=None) -> pd.DataFrame:
    """Đọc `results.csv` đã tích luỹ qua task 8–11.

    Một `(algo, split)` có thể có nhiều dòng nếu chạy lại nhiều lần —
    `build_comparison` lấy dòng MỚI NHẤT, vì `append_results` ghi nối theo
    thứ tự thời gian.
    """
    runs_dir = CONFIG.paths.runs if runs_dir is None else runs_dir
    path = runs_dir / "results.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Chưa có {path}. Cần chạy task 8–11 trước khi so sánh.")
    return pd.read_csv(path)


def missing_comparison_rows(
    results: pd.DataFrame,
    algorithms: tuple[str, ...] = CONTENDERS,
) -> list[tuple[str, str]]:
    """Liệt kê `(algo, split)` còn thiếu để dựng được bảng so sánh đầy đủ."""
    have = set(map(tuple, results[["algo", "split"]].dropna().to_numpy()))
    return [(algo, split) for algo in algorithms
            for split in ("cv_train", "test") if (algo, split) not in have]


def build_comparison(
    results: pd.DataFrame,
    algorithms: tuple[str, ...] = CONTENDERS,
    metrics: tuple[str, ...] = COMPARISON_METRICS,
) -> pd.DataFrame:
    """Ghép chỉ số CV và test cạnh nhau, kèm khoảng cách giữa hai bên.

    `gap = CV − test`, cố ý theo chiều đó: số DƯƠNG nghĩa là CV lạc quan hơn
    thực tế, và đó là chiều đáng lo. Số âm nghĩa là model làm tốt hơn trên
    dữ liệu chưa từng thấy — hiếm, và thường chỉ là dao động mẫu.

    Khoảng cách này đọc cùng với `*_std` giữa các fold mới có nghĩa: gap nhỏ
    hơn một độ lệch chuẩn thì không phân biệt được với nhiễu.
    """
    latest = (results.dropna(subset=["split"])
              .groupby(["algo", "split"], as_index=False).last()
              .set_index(["algo", "split"]))

    rows = []
    for algo in algorithms:
        row: dict = {"algo": algo}
        for metric in metrics:
            cv = latest.loc[(algo, "cv_train"), metric] if (algo, "cv_train") in latest.index else np.nan
            test = latest.loc[(algo, "test"), metric] if (algo, "test") in latest.index else np.nan
            row[f"cv_{metric}"] = float(cv)
            row[f"test_{metric}"] = float(test)
            row[f"gap_{metric}"] = float(cv) - float(test)
        key = (algo, "cv_train")
        row["cv_macro_f1_std"] = (float(latest.loc[key, "macro_f1_std"])
                                  if key in latest.index else np.nan)
        row["fit_seconds"] = (float(latest.loc[key, "fit_seconds"])
                              if key in latest.index else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def build_per_class_comparison(runs_dir=None) -> pd.DataFrame:
    """Bảng F1 từng lớp × từng model, đọc từ `test_per_class.csv` (task 11).

    Chỉ số gộp giấu mất chỗ model mạnh/yếu: hai model cùng macro-F1 vẫn có
    thể hỏng ở hai lớp khác nhau, mà với bài toán này thì nhầm `EMERGENCY`
    không giống nhầm `GROWTH`.
    """
    runs_dir = CONFIG.paths.runs if runs_dir is None else runs_dir
    path = runs_dir / "test_per_class.csv"
    if not path.exists():
        raise FileNotFoundError(f"Chưa có {path}. Cần chạy task 11 trước.")

    per_class = pd.read_csv(path)
    order = [g.value for g in ORDERED_GROUPS]
    table = per_class.pivot_table(index="label", columns="algo", values="f1")
    return table.reindex(order)


def _backfill_cv_row(
    algo: str,
    params: PopulationParams | None,
    seed: int,
    n_splits: int,
    test_size: float,
    runs_dir,
) -> pd.DataFrame:
    """Chạy CV cho MỘT thuật toán còn thiếu số, rồi ghi bổ sung vào results.csv.

    Chỉ chạy đúng thuật toán thiếu, không đụng ba cái đã có — task 12 không
    được train lại những gì đã có số.
    """
    log.info("Thiếu CV của %s, chạy bổ sung", algo)
    X, y = build_training_data(params, seed=seed)
    X_train, _, y_train, _ = split_train_test(X, y, test_size, seed)
    comparison, _ = cross_validate(
        X_train, y_train, algorithms={algo: ALGORITHMS[algo]},
        n_splits=n_splits, seed=seed)

    logged = comparison.copy()
    logged["split"] = "cv_train"
    logged.insert(0, "task", TASK)
    logged.insert(2, "feature_set", "default")
    logged.insert(3, "random_seed", seed)
    logged.insert(4, "n_rows", len(X_train))
    logged.insert(5, "n_splits", n_splits)
    logged["note"] = f"task12 backfill cv ({algo})"
    append_results(logged, runs_dir / "results.csv")
    return logged


def compare_models(
    params: PopulationParams | None = None,
    seed: int | None = None,
    n_splits: int | None = None,
    test_size: float | None = None,
    algorithms: tuple[str, ...] = CONTENDERS,
    runs_dir=None,
    backfill: bool = True,
    save: bool = True,
) -> dict:
    """Task 12 — so sánh 4 thuật toán từ số ĐÃ CÓ của task 8–11.

    Không train lại thứ gì đã có số. Ngoại lệ duy nhất là `backfill`: nếu một
    thuật toán thiếu hẳn dòng CV thì không dựng được cột `gap` cho nó, và lúc
    đó chạy CV cho riêng nó là bắt buộc chứ không phải tiện tay.

    Hàm này **không chọn model**. Nó xếp bảng và tính khoảng cách; việc quyết
    định lấy model nào thuộc task sau.
    """
    seed = CONFIG.random_seed if seed is None else seed
    n_splits = CONFIG.training["n_splits"] if n_splits is None else n_splits
    test_size = CONFIG.training["test_size"] if test_size is None else test_size
    runs_dir = CONFIG.paths.runs if runs_dir is None else runs_dir

    results = load_results(runs_dir)
    missing = missing_comparison_rows(results, algorithms)

    backfilled: list[str] = []
    if missing and backfill:
        for algo, split in missing:
            if split != "cv_train":
                raise ValueError(
                    f"Thiếu kết quả test của {algo} — chạy lại task 11, "
                    "task 12 không tự chấm trên tập test.")
            _backfill_cv_row(algo, params, seed, n_splits, test_size, runs_dir)
            backfilled.append(algo)
        results = load_results(runs_dir)
        missing = missing_comparison_rows(results, algorithms)

    if missing:
        raise ValueError(f"Thiếu kết quả để so sánh: {missing}")

    comparison = build_comparison(results, algorithms)
    per_class = build_per_class_comparison(runs_dir)

    for _, row in comparison.iterrows():
        log.info("%-16s CV %.4f → test %.4f (gap %+.4f) macro-F1",
                 row["algo"], row["cv_macro_f1"],
                 row["test_macro_f1"], row["gap_macro_f1"])

    result = {
        "comparison": comparison,
        "per_class": per_class,
        "backfilled": backfilled,
    }

    if save:
        files = {"comparison": runs_dir / "model_comparison.csv",
                 "per_class": runs_dir / "model_comparison_per_class.csv"}
        comparison.to_csv(files["comparison"], index=False, encoding="utf-8")
        per_class.to_csv(files["per_class"], encoding="utf-8")
        log.info("Ghi bảng so sánh → %s", runs_dir)
        result["files"] = files

    return result


# ------------------------------ task 13 — feature importance

def _model_for_importance(algo, X_train, y_train, seed, runs_dir, use_saved):
    """Lấy model đã train: ưu tiên artifact đã lưu, không có thì mới fit.

    Trả về `(model, nguồn)`. Chỉ đọc artifact khi `use_saved` — chạy trên dân
    số khác mặc định (ví dụ trong test) mà vẫn nạp artifact cũ thì importance
    in ra không phải của dữ liệu đang xét.
    """
    if use_saved:
        try:
            model = load_model(f"{TASK}_{algo}_v1", directory=runs_dir)
            log.info("%-16s dùng artifact đã lưu", algo)
            return model, "artifact"
        except FileNotFoundError:
            pass

    model = PipelineClassifier(
        task=TASK, algo=algo,
        estimator=ALGORITHMS[algo](seed),
        preprocessing=build_preprocessing_pipeline())
    model.fit(X_train, y_train)
    log.info("%-16s chưa có artifact, fit lại trên tập train", algo)
    return model, "refit"


def feature_importance_report(
    params: PopulationParams | None = None,
    seed: int | None = None,
    test_size: float | None = None,
    algorithms: tuple[str, ...] = CONTENDERS,
    runs_dir=None,
    use_saved: bool = True,
    top_n: int = 5,
    save: bool = True,
) -> dict:
    """Task 13 — độ quan trọng feature của từng model ML01.

    Không phải thuật toán nào cũng có. `BaggingClassifier` **không** phơi ra
    `feature_importances_` (dù từng cây con của nó thì có), nên nó được ghi
    vào `unavailable` kèm lý do thay vì biến mất khỏi bảng — thiếu một model
    mà không nói vì sao là chỗ người đọc báo cáo sẽ hỏi.

    Model lấy từ artifact đã lưu nếu có, chỉ fit lại khi thiếu. Cột `source`
    ghi rõ từng model đến từ đâu, để con số tra ngược được.

    Hàm này **không chọn model** — nó xếp bảng importance, thế thôi.
    """
    seed = CONFIG.random_seed if seed is None else seed
    test_size = CONFIG.training["test_size"] if test_size is None else test_size
    runs_dir = CONFIG.paths.runs if runs_dir is None else runs_dir

    X, y = build_training_data(params, seed=seed)
    X_train, _, y_train, _ = split_train_test(X, y, test_size, seed)

    frames: list[pd.DataFrame] = []
    unavailable: dict[str, str] = {}
    sources: dict[str, str] = {}

    for algo in algorithms:
        model, source = _model_for_importance(
            algo, X_train, y_train, seed, runs_dir, use_saved)
        sources[algo] = source
        try:
            importance = model.feature_importance()
        except AttributeError as exc:
            unavailable[algo] = str(exc)
            log.info("%-16s không có feature importance: %s", algo, exc)
            continue

        importance = importance.assign(
            algo=algo, source=source,
            rank=range(1, len(importance) + 1))
        frames.append(importance[["algo", "source", "rank", "feature", "importance"]])

    if not frames:
        raise ValueError("Không thuật toán nào cung cấp feature importance")

    long = pd.concat(frames, ignore_index=True)
    pivot = long.pivot_table(index="feature", columns="algo", values="importance")
    # Xếp theo mức quan trọng TRUNG BÌNH giữa các model, để feature nào cũng
    # được nhiều model coi trọng thì nổi lên đầu bảng.
    pivot = pivot.assign(mean=pivot.mean(axis=1)).sort_values(
        "mean", ascending=False)

    top = (long[long["rank"] <= top_n]
           .pivot(index="rank", columns="algo", values="feature"))

    for algo in long["algo"].unique():
        best = long[(long["algo"] == algo) & (long["rank"] <= 3)]
        log.info("%-16s top-3: %s", algo,
                 ", ".join(f"{r.feature} {r.importance:.3f}"
                           for r in best.itertuples()))

    result = {
        "importance": long,
        "pivot": pivot,
        "top": top,
        "unavailable": unavailable,
        "sources": sources,
    }

    if save:
        runs_dir.mkdir(parents=True, exist_ok=True)
        files = {"importance": runs_dir / "feature_importance.csv",
                 "pivot": runs_dir / "feature_importance_pivot.csv"}
        long.to_csv(files["importance"], index=False, encoding="utf-8")
        pivot.to_csv(files["pivot"], encoding="utf-8")
        log.info("Ghi feature importance → %s", runs_dir)
        result["files"] = files

    return result


# -------------------------------------- task 14 — chọn model tốt nhất

def select_final_model(comparison: pd.DataFrame, task: str = TASK) -> dict:
    """Task 14 — chọn model theo **CV macro-F1**, không nhìn tập test.

    Điều "không chọn theo test" ở đây là ràng buộc CẤU TRÚC, không phải lời
    hứa: khung dữ liệu đưa vào `select_best()` được dựng lại chỉ từ cột
    `cv_*`, nên hàm quyết định **không có** con số test nào để mà dùng. Đổi
    hết chỉ số test đi thì lựa chọn vẫn y nguyên.

    Bản ghi trả về kèm `margin` (khoảng cách tới model hạng nhì) và
    `margin_vs_fold_std`. Chọn thì vẫn cứ argmax theo đúng yêu cầu, nhưng
    khoảng cách nhỏ hơn dao động giữa các fold thì phải nói ra — "thắng
    0,012 với σ 0,006" và "thắng áp đảo" là hai kết luận khác nhau.

    Chỉ số test được ghi vào `supporting`, đúng vai trò tham khảo.
    """
    metric = SELECTION_METRIC[task]
    cv_column, test_column = f"cv_{metric}", f"test_{metric}"

    # Chỉ cột CV đi vào quyết định. Không đưa cột test vào khung này.
    cv_only = (comparison[["algo", cv_column]]
               .rename(columns={cv_column: metric}))
    selected = select_best(cv_only, task)

    ranking = comparison.sort_values(cv_column, ascending=False).reset_index(drop=True)
    best_row = ranking.iloc[0]
    runner_up = ranking.iloc[1] if len(ranking) > 1 else None
    fold_std = float(best_row.get("cv_macro_f1_std", float("nan")))
    margin = (float(best_row[cv_column]) - float(runner_up[cv_column])
              if runner_up is not None else float("nan"))

    record = {
        "selected": selected,
        "task": task,
        "selection_metric": metric,
        "selection_basis": "cross-validation trên tập train (80%)",
        "criterion": (f"macro-F1 cao nhất khi CV {int(comparison.shape[0])} thuật toán "
                      f"trên cùng fold, cùng feature, cùng seed; "
                      f"baseline bị loại khỏi danh sách dự tuyển"),
        "cv_macro_f1": float(best_row[cv_column]),
        "cv_macro_f1_std": fold_std,
        "runner_up": str(runner_up["algo"]) if runner_up is not None else None,
        "margin": margin,
        "margin_vs_fold_std": (margin / fold_std) if fold_std else float("nan"),
        "cv_ranking": [
            {"algo": str(row["algo"]), "cv_macro_f1": float(row[cv_column])}
            for _, row in ranking.iterrows()
        ],
        # Tham khảo. KHÔNG tham gia quyết định trên.
        "supporting": {
            "test_macro_f1": float(best_row[test_column]),
            "gap_cv_minus_test": float(best_row[f"gap_{metric}"]),
            "note": "chỉ số test chỉ để đối chiếu, không dùng để chọn model",
        },
    }
    return record


def record_model_selection(
    runs_dir=None,
    algorithms: tuple[str, ...] = CONTENDERS,
    task: str = TASK,
    save: bool = True,
) -> dict:
    """Chọn model từ kết quả ĐÃ CÓ rồi ghi bản ghi ra `src/training/runs/`.

    Không train lại gì cả: đọc thẳng `results.csv` mà task 8–12 đã tích luỹ.
    Thiếu số thì báo lỗi chứ không tự chạy bù — task 14 chỉ có việc chọn.
    """
    runs_dir = CONFIG.paths.runs if runs_dir is None else runs_dir
    results = load_results(runs_dir)

    missing = missing_comparison_rows(results, algorithms)
    if [row for row in missing if row[1] == "cv_train"]:
        raise ValueError(
            f"Thiếu kết quả CV để chọn model: {missing}. "
            "Chạy task 8–12 trước; task 14 không train lại.")

    comparison = build_comparison(results, algorithms)
    record = select_final_model(comparison, task)

    log.info("Model được chọn: %s (CV macro-F1 %.4f, hơn %s %+.4f = %.1f×σ)",
             record["selected"], record["cv_macro_f1"], record["runner_up"],
             record["margin"], record["margin_vs_fold_std"])

    result = {"record": record, "comparison": comparison}
    if save:
        runs_dir.mkdir(parents=True, exist_ok=True)
        path = runs_dir / "model_selection.json"
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2),
                        encoding="utf-8")
        log.info("Ghi bản ghi chọn model → %s", path)
        result["file"] = path
    return result


# ------------------------------------------- task 15 — export model

#: Nhãn phiên bản của bản export cuối. Slug thành `ml01_xgboost_vfinal`, giữ
#: nguyên tên thuật toán trong slug — bản "final" mà giấu mất mình là model
#: gì thì người nhận phải mở file ra mới biết.
FINAL_VERSION: Final[str] = "final"


def _sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _environment() -> dict:
    """Phiên bản thư viện — thiếu nó thì "tái lập được" chỉ là nói miệng.

    Cùng seed mà khác phiên bản sklearn/xgboost vẫn có thể ra số khác.
    """
    import platform

    import sklearn
    import xgboost
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
    }


def export_final_model(runs_dir=None, verify: bool = True) -> dict:
    """Task 15 — export model đã được task 14 chọn.

    **Không chọn lại**: thuật toán lấy từ `model_selection.json` mà task 14
    đã ghi. Nếu chưa có file đó thì dừng, chứ không tự suy ra model tốt nhất.

    **Không train lại**: nạp đúng artifact đã lưu ở task 10, chỉ đóng gói lại
    kèm metadata đầy đủ. Trọng số không đổi — `verify` chứng minh điều đó
    bằng cách nạp bản export lên và dự đoán thử.

    Bản export gom đủ thứ cần để nhận dạng và tái lập:

        - thuật toán, seed, thứ tự feature, danh sách lớp
        - siêu tham số và cấu hình lần train (từ metadata task 10)
        - chỉ số CV **và** test (test nằm ở `results.csv`, không nằm trong
          metadata task 10 — gom lại đây để bản export tự đủ)
        - bản ghi chọn model của task 14
        - phiên bản thư viện
        - sha256 của cả artifact nguồn lẫn artifact export
    """
    runs_dir = CONFIG.paths.runs if runs_dir is None else runs_dir

    selection_path = runs_dir / "model_selection.json"
    if not selection_path.exists():
        raise FileNotFoundError(
            f"Chưa có {selection_path}. Task 15 export model mà TASK 14 đã "
            "chọn — chạy task 14 trước, task 15 không tự chọn.")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    algo = selection["selected"]

    source_slug = f"{TASK}_{algo}_v1"
    source_path = runs_dir / f"{source_slug}.joblib"
    model = load_model(source_slug, directory=runs_dir)
    source_meta = json.loads(
        (runs_dir / f"{source_slug}.metadata.json").read_text(encoding="utf-8"))

    # Chỉ số test của đúng thuật toán này, lấy dòng mới nhất trong results.csv.
    results = load_results(runs_dir)
    test_rows = results[(results["algo"] == algo) & (results["split"] == "test")]
    test_metrics = (
        test_rows.iloc[-1][["accuracy", "balanced_accuracy", "macro_precision",
                            "macro_recall", "macro_f1", "weighted_f1"]]
        .astype(float).to_dict() if not test_rows.empty else {})

    extra = {
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_artifact": source_path.name,
        "source_sha256": _sha256(source_path),
        "config": source_meta.get("config", {}),
        "selection": selection,
        "environment": _environment(),
    }
    metrics = {"cv": source_meta.get("metrics", {}).get("cv", {}),
               "test": test_metrics}

    # Chỉ đổi NHÃN phiên bản để slug của bản export khác bản train; trọng số
    # và mọi tham số của model giữ nguyên.
    model.version = FINAL_VERSION
    artifact = save_model(model, metrics=metrics, directory=runs_dir, extra=extra)

    # sha256 của chính file vừa ghi phải vá vào sau, vì không thể biết trước
    # hash của một file chứa chính hash đó.
    meta_path = artifact.with_name(f"{artifact.stem}.metadata.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["artifact_sha256"] = _sha256(artifact)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                         encoding="utf-8")

    log.info("Export model cuối: %s (%s, %.1f MB)",
             artifact.name, algo, artifact.stat().st_size / 1e6)

    result = {"artifact": artifact, "metadata_path": meta_path,
              "metadata": meta, "algo": algo}
    if verify:
        result["verification"] = _verify_export(artifact, runs_dir, meta)
    return result


def _verify_export(artifact, runs_dir, meta) -> dict:
    """Nạp lại bản export từ đĩa và dự đoán thử.

    Kiểm trên tập TRAIN, không phải test: chỗ này chỉ cần chứng minh file tải
    lên chạy được, không cần thêm một lần chấm test nào nữa.

    Một artifact hỏng thường không nổ lúc `joblib.load` mà nổ lúc `predict` —
    nên phải chạy tới dự đoán mới gọi là đã kiểm.
    """
    reloaded = load_model(artifact.stem, directory=runs_dir)

    config = meta.get("config", {})
    X, y = build_training_data(seed=meta["random_seed"])
    X_train, _, _, _ = split_train_test(
        X, y, config.get("test_size"), meta["random_seed"])
    sample = X_train.head(50)

    predicted = reloaded.predict(sample)
    proba = reloaded.predict_proba(sample)

    checks = {
        "loaded": True,
        "n_predictions": int(len(predicted)),
        "labels_within_classes": bool(set(predicted) <= set(reloaded.classes_)),
        "proba_shape_matches": proba.shape == (len(sample), len(reloaded.classes_)),
        "proba_sums_to_one": bool(np.allclose(proba.sum(axis=1), 1.0)),
        "feature_order_matches": list(reloaded.feature_names_) == meta["feature_names"],
    }
    failed = [name for name, ok in checks.items() if ok is False]
    if failed:
        raise ValueError(f"Bản export không qua kiểm tra: {failed}")
    log.info("Kiểm tra bản export: nạp được, dự đoán %d dòng, %d lớp",
             checks["n_predictions"], len(reloaded.classes_))
    return checks


# --------------------------------------------- cổng kiểm chứng (§6.2)

#: Mỗi lớp phải chiếm ít nhất chừng này, nếu không bảng per-class vô nghĩa.
GATE_MIN_CLASS_SHARE: Final[float] = 0.10
#: Model tốt nhất vượt mốc này nghĩa là ranh giới quá sạch — nhiều khả năng
#: có rò rỉ nhãn, hoặc vùng biên/nhiễu chưa đủ.
GATE_MAX_ACCURACY: Final[float] = 0.98
#: "Thắng baseline rõ rệt" cần một định nghĩa số. Đòi CẢ HAI: hơn tối thiểu
#: 0,05 macro-F1 tuyệt đối, VÀ khoảng cách lớn hơn 2× độ lệch chuẩn giữa các
#: fold. Chỉ dùng điều kiện tuyệt đối thì một model dao động mạnh vẫn "thắng";
#: chỉ dùng std thì một model ổn định mà kém vẫn qua.
GATE_BASELINE_MARGIN: Final[float] = 0.05


def check_gates(
    y: pd.Series,
    comparison: pd.DataFrame,
    test_metrics: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Ba cổng kiểm chứng của PLAN.md §6.2. Không qua thì quay lại task 3.

    Trả về bảng `(cổng, đạt, chi tiết)` thay vì ném exception: khi một cổng
    hỏng, cái cần biết là hỏng ở đâu và lệch bao nhiêu, chứ không phải một
    traceback dừng ngay ở cổng đầu tiên.

    Mỗi cổng chấm trên tập hợp lý với nó:

        Cân bằng lớp     toàn bộ dân số — đó là tính chất của dữ liệu
        Ranh giới sạch   tập TEST nếu có, vì đó mới là con số thật
        Thắng baseline   CV, vì chỉ ở đó cả 4 thuật toán mới có số để so
    """
    contenders = comparison[comparison["algo"] != BASELINE]
    baseline = comparison[comparison["algo"] == BASELINE]
    rows = []

    distribution = class_distribution(y)
    thin = distribution[distribution["share"] < GATE_MIN_CLASS_SHARE]
    rows.append({
        "cổng": "Cân bằng lớp",
        "điều kiện": f"mọi lớp ≥ {GATE_MIN_CLASS_SHARE:.0%}",
        "đạt": thin.empty,
        "chi tiết": "nhỏ nhất %.1f%% (%s)" % (
            distribution["share"].min() * 100,
            distribution.loc[distribution["share"].idxmin(), "label"]),
    })

    if test_metrics is not None:
        best_accuracy = float(test_metrics["accuracy"])
        source = "test"
    else:
        best_accuracy = float(contenders["accuracy"].max())
        source = "CV"
    rows.append({
        "cổng": "Ranh giới không quá sạch",
        "điều kiện": f"accuracy tốt nhất ≤ {GATE_MAX_ACCURACY}",
        "đạt": best_accuracy <= GATE_MAX_ACCURACY,
        "chi tiết": f"{best_accuracy:.4f} (đo trên {source})",
    })

    if baseline.empty:
        rows.append({"cổng": "Thắng baseline", "điều kiện": "có baseline",
                     "đạt": False, "chi tiết": "thiếu baseline trong bảng"})
    else:
        base_score = float(baseline["macro_f1"].iloc[0])
        base_std = float(baseline.get("macro_f1_std", pd.Series([0.0])).iloc[0])
        losers = []
        for _, row in contenders.iterrows():
            margin = float(row["macro_f1"]) - base_score
            spread = 2 * (float(row.get("macro_f1_std", 0.0)) + base_std)
            if margin < max(GATE_BASELINE_MARGIN, spread):
                losers.append(f"{row['algo']} (+{margin:.4f})")
        rows.append({
            "cổng": "Thắng baseline rõ rệt",
            "điều kiện": f"macro-F1 hơn baseline ≥ {GATE_BASELINE_MARGIN} và > 2σ",
            "đạt": not losers,
            "chi tiết": f"baseline {base_score:.4f}; "
                        + (", ".join(losers) + " không đạt" if losers else "mọi model đạt"),
        })

    return pd.DataFrame(rows)


def select_best(comparison: pd.DataFrame, task: str = TASK) -> str:
    """Chọn thuật toán theo chỉ số của bài toán — ML01 là macro-F1.

    Baseline bị loại khỏi danh sách dự tuyển: nó tồn tại để bị vượt qua, và
    một dân số hỏng đến mức baseline thắng thì đó là tín hiệu quay lại task 3,
    không phải model để đem đi deploy.
    """
    metric = SELECTION_METRIC[task]
    contenders = comparison[comparison["algo"] != BASELINE]
    if contenders.empty:
        raise ValueError("Bảng so sánh không có thuật toán nào ngoài baseline")
    return str(contenders.loc[contenders[metric].idxmax(), "algo"])


# --------------------------------------------------------------- chạy

def run_full_pipeline(
    params: PopulationParams | None = None,
    seed: int | None = None,
    n_splits: int | None = None,
    test_size: float | None = None,
    export: bool = True,
    note: str = "",
    runs_dir=None,
) -> dict:
    """Chạy trọn F03 task 5–15 trong MỘT lần gọi.

    Trình tự cố ý không đảo được: tách test → CV trên train → chọn model →
    mới chấm test. Tập test không tham gia bất cứ quyết định nào phía trước.

    Đây là đường chạy gộp, có trước khi F03 được tách thành từng task riêng.
    Nó vẫn được giữ vì hai test cấu trúc dựa vào nó — nhất là phép kiểm tập
    test không lọt vào bước chọn model, vốn chỉ kiểm được khi cả chuỗi chạy
    liền một mạch.

    Đường chạy CHÍNH THỨC theo từng task là `train_decision_tree()` …
    `export_final_model()`, gọi qua `scripts/train_ml01.py`. Hàm này từng tên
    là `train_ml01`, trùng tên với script đó và đã gây nhầm — đổi tên để hai
    thứ không còn lẫn.
    """
    seed = CONFIG.random_seed if seed is None else seed
    n_splits = CONFIG.training["n_splits"] if n_splits is None else n_splits
    runs_dir = CONFIG.paths.runs if runs_dir is None else runs_dir
    log_run_context(log)

    X, y = build_training_data(params, seed=seed)
    labels = [g.value for g in ORDERED_GROUPS]
    log.info("Dữ liệu ML01: %d dòng × %d feature", len(X), X.shape[1])

    X_train, X_test, y_train, y_test = split_train_test(X, y, test_size, seed)

    # -- Chọn model: CV chỉ trên tập train, test chưa được đụng tới.
    comparison, oof = cross_validate(X_train, y_train, n_splits=n_splits, seed=seed)
    best = select_best(comparison)
    log.info("Model được chọn theo macro-F1 (CV trên train): %s", best)

    # -- Fit model đã chọn trên toàn bộ tập train, rồi chấm test MỘT LẦN.
    final = PipelineClassifier(
        task=TASK, algo=best,
        estimator=ALGORITHMS[best](seed),
        preprocessing=build_preprocessing_pipeline())
    final.fit(X_train, y_train)

    test_predictions = final.predict(X_test)
    test_metrics = classification_metrics(y_test, test_predictions, labels=labels)
    log.info("TEST — macro-F1 %.4f · accuracy %.4f",
             test_metrics["macro_f1"], test_metrics["accuracy"])

    # Baseline cũng chấm trên test để bảng báo cáo có mốc so sánh cùng tập.
    baseline_model = PipelineClassifier(
        task=TASK, algo=BASELINE,
        estimator=ALGORITHMS[BASELINE](seed),
        preprocessing=build_preprocessing_pipeline())
    baseline_model.fit(X_train, y_train)
    baseline_test = classification_metrics(
        y_test, baseline_model.predict(X_test), labels=labels)

    gates = check_gates(y, comparison, test_metrics=test_metrics)

    result = {
        "X_train": X_train, "y_train": y_train,
        "X_test": X_test, "y_test": y_test,
        "comparison": comparison,          # CV trên train — dùng để CHỌN
        "oof": oof,
        "test_metrics": test_metrics,      # test — con số ĐEM BÁO CÁO
        "baseline_test_metrics": baseline_test,
        "gates": gates,
        "best_algo": best,
        "model": final,
        "distribution": class_distribution(y),
        "importance": final.feature_importance(),
        # Bảng per-class và confusion dựng từ dự đoán trên TEST — model chưa
        # từng thấy dòng nào trong đó.
        "per_class": per_class_table(y_test, test_predictions, labels),
        "confusion": confusion_table(y_test, test_predictions, labels),
    }

    if export:
        # Metadata mang CẢ HAI: chỉ số CV (căn cứ chọn) và chỉ số test (kết
        # quả báo cáo). Chỉ lưu một trong hai thì sau này không ai dựng lại
        # được vì sao model này được chọn.
        cv_metrics = comparison.loc[comparison["algo"] == best].iloc[0].to_dict()
        cv_metrics.pop("algo", None)
        result["artifact"] = save_model(final, metrics={
            "cv": cv_metrics,
            "test": test_metrics,
            "baseline_test": baseline_test,
        }, directory=runs_dir)

        logged = comparison.copy()
        logged["split"] = "cv_train"
        logged = pd.concat([logged, pd.DataFrame([
            {"algo": best, "split": "test", **test_metrics},
            {"algo": BASELINE, "split": "test", **baseline_test},
        ])], ignore_index=True)
        logged.insert(0, "task", TASK)
        logged.insert(2, "feature_set", "default")
        logged.insert(3, "random_seed", seed)
        logged.insert(4, "n_rows", len(X))
        logged.insert(5, "n_splits", n_splits)
        logged["note"] = note or f"best={best}"
        result["results_csv"] = append_results(logged, runs_dir / "results.csv")

    return result
