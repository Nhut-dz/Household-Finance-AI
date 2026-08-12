"""Contract chung cho ML01 và ML02 — mọi model đều là bộ phân loại.

Hai bài toán của đồ án đều là classification, nên contract chỉ cần ba phương
thức. `predict_proba` là bắt buộc, không phải tùy chọn (PLAN.md §11): tầng
`hfml.pipeline` hạ cấp xuống kết luận rule khi `max(predict_proba)` dưới
ngưỡng tin cậy, và ML02 ra quyết định theo ngưỡng xác suất.

Một model = preprocessing Pipeline + estimator, đóng gói chung trong một
object. Đây là điều kiện để `joblib.dump` một file duy nhất và để inference
dùng lại ĐÚNG bộ biến đổi đã fit trên train (PLAN.md §4.4).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseClassifier(ABC):
    """Giao diện tối thiểu mà training, registry và inference dựa vào."""

    #: Tên bài toán — "ml01" hoặc "ml02".
    task: str = "base"
    #: Thuật toán — "decision_tree" | "bagging" | "random_forest" | "xgboost".
    algo: str = "base"
    #: Bộ feature — "default" (ML01) | "full" | "reduced" (ML02).
    feature_set: str = "default"
    version: str = "1"

    #: Thứ tự cột lúc fit. Inference PHẢI dựng lại đúng thứ tự này.
    feature_names_: list[str]
    #: Nhãn theo đúng thứ tự cột của predict_proba.
    classes_: list[str]

    @property
    def slug(self) -> str:
        """Định danh artifact, ví dụ `ml02_xgboost_reduced_v1`."""
        parts = [self.task, self.algo]
        if self.feature_set != "default":
            parts.append(self.feature_set)
        return "_".join(parts) + f"_v{self.version}"

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseClassifier":
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        ...

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Xác suất từng lớp, shape (n_samples, len(classes_))."""
        ...
