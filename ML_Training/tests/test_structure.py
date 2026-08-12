"""F01 task 1 — kiểm tra cấu trúc project đứng vững.

Test này không kiểm tra nghiệp vụ, nó kiểm tra bộ khung: đủ 5 tầng, import
được, contract của tầng ml đúng như PLAN.md §11 yêu cầu, và seed dùng chung
đúng bằng 42. Chạy được ngay từ Tuần 1, trước khi có bất kỳ model nào.
"""
from __future__ import annotations

import importlib

import pytest

# 5 tầng theo PLAN.md §3, cộng api là lớp vỏ giao tiếp.
LAYERS = [
    "hfml.data",
    "hfml.data.preprocessing",
    "hfml.data.features",
    "hfml.rules",
    "hfml.ml",
    "hfml.ml.ml01_recommendation",
    "hfml.ml.ml02_credit_risk",
    "hfml.ml.evaluation",
    "hfml.pipeline",
    "hfml.llm",
    "hfml.api",
]


@pytest.mark.parametrize("module_name", LAYERS)
def test_layer_imports_and_is_documented(module_name: str):
    """Mỗi tầng phải import được và có docstring mô tả trách nhiệm."""
    module = importlib.import_module(module_name)
    assert module.__doc__, f"{module_name} thiếu docstring mô tả trách nhiệm tầng"


def test_classifier_contract_requires_predict_proba():
    """PLAN.md §11: predict_proba là bắt buộc, không phải tùy chọn."""
    from hfml.ml.base import BaseClassifier

    required = {"fit", "predict", "predict_proba"}
    assert required <= BaseClassifier.__abstractmethods__


def test_artifact_slug_naming():
    """Quy ước tên artifact ở PLAN.md §6.3 và §7.4."""
    from hfml.ml.base import BaseClassifier

    class _Stub(BaseClassifier):
        def fit(self, X, y): return self
        def predict(self, X): ...
        def predict_proba(self, X): ...

    ml01 = _Stub()
    ml01.task, ml01.algo = "ml01", "random_forest"
    assert ml01.slug == "ml01_random_forest_v1"

    ml02 = _Stub()
    ml02.task, ml02.algo, ml02.feature_set = "ml02", "xgboost", "reduced"
    assert ml02.slug == "ml02_xgboost_reduced_v1"


def test_shared_seed_is_42():
    """Cùng seed cho mọi bước — điều kiện để F06 task 6 tái lập được."""
    from hfml.config import CONFIG

    assert CONFIG.random_seed == 42
    assert CONFIG.paths.runs.exists()
