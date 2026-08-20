"""ML01 — Huấn luyện trên tập đã redesign (F03 · redesign 17/08/2026).

Chiến lược đánh giá, chọn MỘT và giữ nguyên
---------------------------------------------
    80% train  →  5-fold Stratified CV  →  chọn model & siêu tham số
    20% test   →  chạm ĐÚNG MỘT LẦN ở cuối

Không dùng thêm tập validation riêng. Hai chiến lược song song mà không có lý
do rõ ràng chỉ làm loãng ý nghĩa của cả hai: CV đã cho ước lượng kèm độ lệch
trên nhiều lần chia, còn một tập validation cố định thì không.

Test set không tham gia bất kỳ quyết định nào — không chọn model, không chỉnh
siêu tham số, không chọn ngưỡng. Chạm vào nó nhiều lần thì con số cuối cùng
không còn là ước lượng cho dữ liệu chưa thấy nữa, nó chỉ là một chỉ số đã bị
tối ưu gián tiếp.

Vì sao macro-F1 chứ không phải accuracy
-----------------------------------------
Bốn lớp không cân bằng (11,5% – 34,5%). Accuracy cho lớp lớn trọng số cao hơn,
nên một model bỏ mặc DEBT_FOCUS vẫn có accuracy trông đẹp. Macro-F1 tính trung
bình theo LỚP nên mọi lớp có tiếng nói ngang nhau.

Mất cân bằng xử lý bằng `class_weight`, KHÔNG bằng cách sửa nhãn.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier

from hfml.config import CONFIG
from hfml.logger import get_logger
from hfml.ml.base import BaseClassifier
from hfml.ml.ml01_recommendation import features as feature_mod
from hfml.ml.ml01_recommendation.scoring import GROUPS

log = get_logger(__name__)

TEST_SIZE: Final[float] = 0.20
N_FOLDS: Final[int] = 5
SEED: Final[int] = CONFIG.random_seed

#: Số lá tối thiểu tính theo tỉ lệ dân số, không phải số tuyệt đối.
#: Đặt số tuyệt đối thì đổi cỡ tập là đổi luôn độ phức tạp của cây mà không ai
#: chủ ý làm vậy.
MIN_LEAF_SHARE: Final[float] = 0.002


def build_candidates(n_rows: int) -> dict[str, Any]:
    """Các mô hình dự tuyển. `baseline` chỉ để so, không nằm trong danh sách chọn.

    Siêu tham số đặt ở mức vừa phải và GIỐNG NHAU về tinh thần giữa các thuật
    toán: mục tiêu là so thuật toán, nên một model được tinh chỉnh kỹ hơn ba
    model kia sẽ làm phép so sánh mất ý nghĩa.
    """
    min_leaf = max(5, int(n_rows * MIN_LEAF_SHARE))
    common = {"random_state": SEED}

    candidates: dict[str, Any] = {
        "baseline": DummyClassifier(strategy="stratified", **common),
        "decision_tree": DecisionTreeClassifier(
            max_depth=8, min_samples_leaf=min_leaf,
            class_weight="balanced", **common),
        "bagging": BaggingClassifier(
            estimator=DecisionTreeClassifier(
                max_depth=10, min_samples_leaf=min_leaf,
                class_weight="balanced", random_state=SEED),
            n_estimators=100, n_jobs=-1, **common),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=min_leaf,
            max_features="sqrt", class_weight="balanced_subsample",
            n_jobs=-1, **common),
    }

    try:
        from xgboost import XGBClassifier

        candidates["xgboost"] = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.08,
            subsample=0.9, colsample_bytree=0.9,
            objective="multi:softprob", num_class=len(GROUPS),
            tree_method="hist", n_jobs=-1, random_state=SEED,
            eval_metric="mlogloss")
    except ImportError:
        log.warning("Chưa cài xgboost — bỏ qua thuật toán này.")

    return candidates


class Ml01Model(BaseClassifier):
    """Model ML01 đã fit, đóng gói cùng danh sách feature.

    Không có ngưỡng, không có `if/else` nào trong `predict`. Nhãn là
    `argmax(predict_proba)`, và đó là toàn bộ logic dự đoán.
    """

    task = "ml01"
    feature_set = "v2"

    def __init__(self, algo: str, estimator, version: str = "1") -> None:
        self.algo = algo
        self.version = version
        self.estimator = estimator
        self.feature_names_: list[str] = list(feature_mod.FEATURES)
        self.classes_: list[str] = list(GROUPS)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Ml01Model":
        self.estimator.fit(X[self.feature_names_], _encode(y))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.estimator.predict_proba(X[self.feature_names_]))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Nhãn = lớp có xác suất cao nhất. Không có ngưỡng cứng nào ở đây."""
        index = self.predict_proba(X).argmax(axis=1)
        return np.array([self.classes_[i] for i in index])

    def predict_one(self, X: pd.DataFrame) -> dict:
        """Kết quả cho MỘT hồ sơ: nhãn + xác suất cả bốn nhóm."""
        proba = self.predict_proba(X)[0]
        order = int(proba.argmax())
        ranked = np.sort(proba)[::-1]
        return {
            "predicted_group": self.classes_[order],
            "probabilities": {name: round(float(p), 4)
                              for name, p in zip(self.classes_, proba)},
            # Chênh lệch giữa nhóm nhất và nhóm nhì. Nhỏ nghĩa là hồ sơ nằm ở
            # vùng giao — cần nói ra thay vì trình bày như một kết luận chắc.
            "prediction_confidence": round(float(ranked[0]), 4),
            "margin": round(float(ranked[0] - ranked[1]), 4),
        }


def _encode(y: pd.Series) -> np.ndarray:
    """Nhãn chuỗi → chỉ số, theo đúng thứ tự `GROUPS`.

    XGBoost cần nhãn số. Mã hoá tập trung ở đây để thứ tự lớp của
    `predict_proba` khớp `classes_` ở mọi thuật toán — lệch thứ tự là kiểu lỗi
    im lặng tệ nhất: model vẫn chạy, xác suất vẫn cộng thành 1, chỉ có điều
    gắn sai tên.
    """
    lookup = {name: i for i, name in enumerate(GROUPS)}
    return y.map(lookup).to_numpy()


@dataclass
class Ml01Report:
    """Kết quả một lần chạy huấn luyện, đủ để viết báo cáo mà không chạy lại."""

    cv_scores: dict = field(default_factory=dict)
    test_scores: dict = field(default_factory=dict)
    selected: str = ""
    selection_reason: str = ""
    confusion: list = field(default_factory=list)
    report_text: str = ""
    importance: dict = field(default_factory=dict)
    n_train: int = 0
    n_test: int = 0

    def to_dict(self) -> dict:
        return {
            "selected": self.selected,
            "selection_reason": self.selection_reason,
            "selection_metric": "cv_macro_f1",
            "n_train": self.n_train,
            "n_test": self.n_test,
            "cv": self.cv_scores,
            "test": self.test_scores,
            "confusion_matrix": self.confusion,
            "classes": list(GROUPS),
            "feature_importance": self.importance,
        }


def evaluate(model: Ml01Model, X: pd.DataFrame, y: pd.Series) -> dict:
    """Bộ chỉ số đầy đủ. KHÔNG chỉ accuracy — xem docstring đầu file."""
    predicted = model.predict(X)
    return {
        "accuracy": float(accuracy_score(y, predicted)),
        "macro_f1": float(f1_score(y, predicted, average="macro")),
        "macro_precision": float(precision_score(y, predicted, average="macro",
                                                 zero_division=0)),
        "macro_recall": float(recall_score(y, predicted, average="macro",
                                           zero_division=0)),
        "weighted_f1": float(f1_score(y, predicted, average="weighted")),
    }


def run(X: pd.DataFrame, y: pd.Series) -> tuple[Ml01Model, Ml01Report]:
    """Chạy trọn: chia tập → CV chọn model → fit lại → chấm test một lần."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)

    report = Ml01Report(n_train=len(y_train), n_test=len(y_test))
    folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    candidates = build_candidates(len(y_train))

    log.info("Chọn model bằng %d-fold CV trên %d hồ sơ train",
             N_FOLDS, len(y_train))

    for name, estimator in candidates.items():
        model = Ml01Model(name, estimator)
        scores = cross_val_score(
            estimator, X_train[model.feature_names_], _encode(y_train),
            cv=folds, scoring="f1_macro", n_jobs=1)
        report.cv_scores[name] = {
            "macro_f1_mean": float(scores.mean()),
            "macro_f1_std": float(scores.std()),
            "folds": [float(s) for s in scores],
        }
        log.info("  %-15s CV macro-F1 = %.4f ± %.4f",
                 name, scores.mean(), scores.std())

    # Baseline bị loại khỏi danh sách dự tuyển — nó ở đó để so, không để chọn.
    ranked = sorted(((n, s["macro_f1_mean"]) for n, s in report.cv_scores.items()
                     if n != "baseline"), key=lambda item: item[1], reverse=True)
    best, best_score = ranked[0]
    runner_up, runner_score = ranked[1]

    report.selected = best
    report.selection_reason = (
        f"CV macro-F1 cao nhất trên {N_FOLDS}-fold ({best_score:.4f}), "
        f"hơn {runner_up} {best_score - runner_score:+.4f}. "
        f"Baseline stratified đạt "
        f"{report.cv_scores['baseline']['macro_f1_mean']:.4f}.")

    # Fit lại trên TOÀN BỘ train rồi mới chấm test — đúng một lần.
    final = Ml01Model(best, build_candidates(len(y_train))[best]).fit(X_train, y_train)
    report.test_scores = evaluate(final, X_test, y_test)

    predicted = final.predict(X_test)
    report.confusion = confusion_matrix(
        y_test, predicted, labels=list(GROUPS)).tolist()
    report.report_text = classification_report(
        y_test, predicted, labels=list(GROUPS), zero_division=0)

    if hasattr(final.estimator, "feature_importances_"):
        report.importance = {
            name: float(value) for name, value in sorted(
                zip(final.feature_names_, final.estimator.feature_importances_),
                key=lambda item: item[1], reverse=True)}

    log.info("Chọn %s · test macro-F1 = %.4f · accuracy = %.4f",
             best, report.test_scores["macro_f1"], report.test_scores["accuracy"])
    return final, report
