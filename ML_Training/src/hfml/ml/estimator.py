"""Hiện thực cụ thể của `BaseClassifier` — preprocessing + estimator trong MỘT object.

`base.py` chỉ khai báo contract. File này là hiện thực duy nhất mà cả ML01 và
ML02 dùng, nên "một model = một file `.joblib`" là đúng theo nghĩa đen: mở
artifact ra là có cả bộ biến đổi đã fit lẫn estimator, không phải dựng lại
chuỗi tiền xử lý bằng tay lúc inference (PLAN.md §4.4).

Vì sao phải tự mã hóa nhãn
--------------------------
XGBoost không nhận nhãn dạng chuỗi, ba thuật toán còn lại thì nhận. Nếu để
mỗi thuật toán tự xoay xở thì `predict()` của XGBoost trả `0..3` còn ba cái
kia trả `"EMERGENCY"`, và mọi chỗ dùng kết quả phải biết mình đang cầm model
nào. Mã hóa ở đây một lần: estimator bên trong luôn thấy số, còn bên ngoài
`predict()` luôn trả chuỗi.

Thứ tự lớp lấy theo `np.unique` (tăng dần) — đúng quy ước của sklearn, nên
cột của `predict_proba` khớp `classes_` mà không cần hoán vị.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.pipeline import Pipeline

from hfml.logger import get_logger
from hfml.ml.base import BaseClassifier

log = get_logger(__name__)


class PipelineClassifier(BaseClassifier):
    """Bọc một `Pipeline` tiền xử lý và một estimator của sklearn/xgboost."""

    def __init__(
        self,
        *,
        task: str,
        algo: str,
        estimator: BaseEstimator,
        preprocessing: Pipeline,
        feature_set: str = "default",
        version: str = "1",
    ):
        self.task = task
        self.algo = algo
        self.feature_set = feature_set
        self.version = version
        self.preprocessing = preprocessing
        self.estimator = estimator

    # ------------------------------------------------------------ fit

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PipelineClassifier":
        """Fit cả chuỗi. Gọi trên tập TRAIN của một fold, không phải toàn bộ."""
        labels = np.asarray(y, dtype=object).astype(str)
        self.feature_names_ = list(X.columns)
        self.classes_ = [str(c) for c in np.unique(labels)]

        # `clone` để gọi `fit` nhiều lần (mỗi fold một lần) không tích lũy
        # trạng thái của lần trước — im lặng và rất khó truy nếu quên.
        self.pipeline_ = Pipeline([
            ("prep", clone(self.preprocessing)),
            ("model", clone(self.estimator)),
        ])
        self.pipeline_.fit(X, np.searchsorted(self.classes_, labels))

        prep = self.pipeline_.named_steps["prep"]
        self.transformed_feature_names_ = list(prep.get_feature_names_out())
        return self

    # -------------------------------------------------------- predict

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Trả nhãn dạng CHUỖI, kể cả khi bên trong là XGBoost."""
        codes = self.pipeline_.predict(self._aligned(X))
        return np.asarray(self.classes_, dtype=object)[np.asarray(codes, dtype=int)]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Xác suất từng lớp, cột theo đúng thứ tự `classes_`."""
        return self.pipeline_.predict_proba(self._aligned(X))

    def _aligned(self, X: pd.DataFrame) -> pd.DataFrame:
        """Sắp lại cột đúng thứ tự lúc fit.

        Thứ tự cột sai là lỗi im lặng: model vẫn chạy, vẫn trả xác suất, chỉ
        có điều xác suất đó vô nghĩa (xem `registry.py`). Thiếu cột thì báo
        ngay chứ đừng để `KeyError` của pandas nói hộ — thông báo của nó
        không cho biết model nào đang cần cột gì.
        """
        missing = [c for c in self.feature_names_ if c not in X.columns]
        if missing:
            raise ValueError(
                f"{self.slug}: thiếu {len(missing)} cột so với lúc train: {missing}")
        return X[self.feature_names_]

    # ---------------------------------------------------- diễn giải

    def feature_importance(self) -> pd.DataFrame:
        """Độ quan trọng feature, giảm dần (F03 task 13, F04 task 13).

        Tên cột lấy SAU tiền xử lý: các bước lọc có thể đã bỏ bớt cột, nên
        ghép `feature_importances_` với danh sách đầu vào là lệch chỉ số.

        `BaggingClassifier` không phơi ra `feature_importances_` — nhưng từng
        cây con của nó thì có, và RandomForest cũng chỉ là trung bình của các
        cây con như vậy. Nên tính cùng cách: cộng importance của từng cây rồi
        chia cho số cây. Phải ánh xạ qua `estimators_features_` vì với
        `max_features < 1.0` mỗi cây chỉ thấy một tập con cột — bỏ bước này
        thì con số gán nhầm cột mà bảng vẫn trông bình thường.

        Model không có cả hai (DummyClassifier) → `AttributeError`, để tầng
        báo cáo ghi lý do thay vì để model biến mất lặng lẽ.
        """
        model = self.pipeline_.named_steps["model"]
        names = self.transformed_feature_names_
        importances = _builtin_importances(model, len(names))
        if importances is None:
            raise AttributeError(
                f"{self.algo} không có feature_importances_ "
                "và không có cây con nào để trung bình")
        return (pd.DataFrame({"feature": names, "importance": importances})
                .sort_values("importance", ascending=False, ignore_index=True))


def _builtin_importances(model, n_features: int) -> np.ndarray | None:
    """`feature_importances_` của model, hoặc trung bình các cây con nếu model
    là ensemble không tự phơi ra thuộc tính này (Bagging). `None` nếu không có.
    """
    values = getattr(model, "feature_importances_", None)
    if values is not None:
        return np.asarray(values, dtype=float)

    children = getattr(model, "estimators_", None)
    if not children or not all(hasattr(c, "feature_importances_") for c in children):
        return None

    total = np.zeros(n_features, dtype=float)
    columns = getattr(model, "estimators_features_",
                      [np.arange(n_features)] * len(children))
    for child, used in zip(children, columns):
        total[np.asarray(used)] += child.feature_importances_
    return total / len(children)
