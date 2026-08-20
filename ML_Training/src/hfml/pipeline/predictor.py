"""AI-01 task 4, 5 — Inference ML01 và ML02 (F05 · M05).

Nạp artifact đã export ở F03 task 15 và F04 task 15, chạy dự đoán, trả kết quả
có cấu trúc. **Không train lại, không đụng training pipeline.**

Nạp một lần, giữ lại
---------------------
`lru_cache` giữ model trong bộ nhớ giữa các request. Nạp lại mỗi lần tốn vài
trăm mili-giây cho việc đọc từ đĩa, và với XGBoost thì đó là phần lớn thời
gian trả lời.

Nạp LƯỜI, không nạp lúc import: thiếu một artifact chỉ làm hỏng đúng model đó,
chứ không làm chết cả service. `/health` vẫn phải nói ra được là thiếu gì.

Thiếu model KHÔNG phải lỗi lập trình
-------------------------------------
`ModelUnavailable` là một trạng thái vận hành bình thường — artifact chưa được
train, chưa được copy sang máy chủ, hoặc phiên bản không khớp. Tầng
orchestrator bắt nó và trả về kết quả thiếu phần ML, kèm lý do; nó KHÔNG được
làm hỏng phần rule vốn chạy độc lập.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np
import pandas as pd

from hfml.config import CONFIG
from hfml.inference.lifecycle import MANAGER, ModelUnavailable
from hfml.inference.settings import ML01, ML02, SETTINGS
from hfml.logger import get_logger
from hfml.ml.ml01_recommendation.labeler import LABELS_VI, RecommendationGroup
from hfml.ml.ml02_credit_risk.export import RISK_LABELS_VI
from hfml.pipeline import confidence as conf
from hfml.pipeline.adapters import to_ml01_frame, to_ml02_frame

log = get_logger(__name__)

__all__ = [
    "ModelUnavailable", "PredictionResult", "predict_ml01", "predict_ml02",
    "model_status", "get_ml01", "get_ml02",
    "MISSING_INPUT", "MODEL_UNAVAILABLE", "PREDICTION_ERROR",
    "INVALID_PROBABILITY",
]


#: Vì sao một model không có kết quả. Bốn mã, và chúng KHÔNG cùng loại —
#: Epic AI-02 phải nói khác nhau cho từng mã:
#:
#:     missing_input        người dùng chưa khai đủ → bảo họ khai tiếp
#:     model_unavailable    artifact chưa có trên máy chủ → vấn đề vận hành
#:     prediction_error     model ném lỗi → vấn đề kỹ thuật
#:     invalid_probability  model trả số vô lý → thà thiếu còn hơn báo cáo sai
#:
#: Gộp cả bốn thành "model không khả dụng" là nói với người dùng rằng hệ thống
#: hỏng, trong khi thứ họ cần nghe là "bạn chưa điền màn Thông tin khoản vay".
MISSING_INPUT: Final[str] = "missing_input"
MODEL_UNAVAILABLE: Final[str] = "model_unavailable"
PREDICTION_ERROR: Final[str] = "prediction_error"
INVALID_PROBABILITY: Final[str] = "invalid_probability"


@dataclass
class PredictionResult:
    """Kết quả một model, đủ để tầng `llm` diễn đạt mà không phải tính thêm."""

    model: str
    available: bool
    label: str | None = None
    label_vi: str | None = None
    probability: float | None = None
    probabilities: list[dict] = field(default_factory=list)
    confidence: dict = field(default_factory=dict)
    model_version: str = ""
    error: str | None = None
    #: Mã lý do khi `available` là False — xem hằng số ở đầu file.
    reason_code: str | None = None

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "available": self.available,
            "label": self.label,
            "label_vi": self.label_vi,
            "probability": self.probability,
            "probabilities": self.probabilities,
            "confidence": self.confidence,
            "model_version": self.model_version,
            "error": self.error,
            "reason_code": self.reason_code,
        }

    @classmethod
    def unavailable(cls, model: str, reason: str,
                    code: str = MODEL_UNAVAILABLE) -> "PredictionResult":
        return cls(model=model, available=False, error=reason, reason_code=code)


# --------------------------------------------------------------------------
# Nạp artifact — uỷ quyền cho `inference.lifecycle`
# --------------------------------------------------------------------------
# Việc nạp, giữ trong bộ nhớ, đổi model lúc đang chạy đều thuộc về
# `ModelManager` (Epic AI-03 task 4). Ở đây chỉ còn phần dự đoán.
#
# Trước AI-03, file này tự giữ `lru_cache` và tự khai slug bằng hằng số, còn
# `api/main.py` có bản sao thứ hai của cả hai thứ đó — và bản sao ấy đã trôi
# sang một slug KHÔNG tồn tại trên đĩa mà không ai phát hiện.
def get_ml01():
    """Model ML01 — 4 nhóm khuyến nghị tài chính."""
    return MANAGER.get(ML01).model


def get_ml02():
    """Model ML02 — rủi ro tín dụng, kèm ngưỡng đã chốt ở F04 task 14."""
    return MANAGER.get(ML02).model


def model_status() -> dict:
    """Trạng thái hai model — cho `/health`."""
    return MANAGER.status()


# --------------------------------------------------------------------------
# Task 4 — ML01
# --------------------------------------------------------------------------
def predict_ml01(profile, age: int | None) -> PredictionResult:
    """Phân loại hộ vào 4 nhóm khuyến nghị.

    `age is None` (hồ sơ thiếu năm sinh) → trả về `unavailable` kèm lý do, chứ
    KHÔNG điền tuổi mặc định. Điền bừa thì model vẫn trả về một nhóm trông hợp
    lý và không ai biết nó dựa trên tuổi bịa (§6.1c).
    """
    if age is None:
        return PredictionResult.unavailable(
            "ml01", "Hồ sơ chưa có năm sinh nên chưa xác định được tuổi — "
                    "model bắt buộc cần trường này.", MISSING_INPUT)

    try:
        model = get_ml01()
    except ModelUnavailable as exc:
        return PredictionResult.unavailable("ml01", str(exc))

    try:
        frame = to_ml01_frame(profile, age)
        label = str(model.predict(frame)[0])
        proba = np.asarray(model.predict_proba(frame))[0]
    except Exception as exc:  # noqa: BLE001 — biên ngoài, không để traceback lọt
        log.exception("ML01 lỗi khi dự đoán")
        return PredictionResult.unavailable(
            "ml01", f"Model lỗi khi dự đoán: {exc}", PREDICTION_ERROR)

    classes = [str(c) for c in model.classes_]
    try:
        check = conf.check_ml01(proba)
    except conf.InvalidProbability as exc:
        return PredictionResult.unavailable(
            "ml01", f"Xác suất không hợp lệ: {exc}", INVALID_PROBABILITY)

    return PredictionResult(
        model="ml01",
        available=True,
        label=label,
        label_vi=LABELS_VI[RecommendationGroup(label)],
        probability=float(proba[classes.index(label)]),
        probabilities=[
            {"label": name,
             "label_vi": LABELS_VI[RecommendationGroup(name)],
             "probability": float(value)}
            for name, value in zip(classes, proba)
        ],
        confidence=check.to_dict(),
        model_version=model.slug,
    )


# --------------------------------------------------------------------------
# Task 5 — ML02
# --------------------------------------------------------------------------
def predict_ml02(profile, loan) -> PredictionResult:
    """Ước lượng rủi ro khoản vay.

    `loan is None` → `unavailable`, không phải lỗi: phần lớn người dùng chỉ
    muốn xem sức khoẻ tài chính và không khai khoản vay. Đưa "xác suất vỡ nợ
    8%" cho người không định vay là một con số vô nghĩa với họ (§7 phạm vi).
    """
    if loan is None:
        return PredictionResult.unavailable(
            "ml02", "Chưa có thông tin khoản vay — cần màn 'Thông tin khoản vay'.",
            MISSING_INPUT)

    try:
        model = get_ml02()
    except ModelUnavailable as exc:
        return PredictionResult.unavailable("ml02", str(exc))

    # Ngưỡng lấy từ `MANAGER`, không đọc thẳng `model.threshold`: cấu hình có
    # thể ghi đè ngưỡng, và đọc thẳng artifact ở đây sẽ tạo ra hai ngưỡng khác
    # nhau trong cùng một lượt — một cái phân nhãn, một cái đem đi báo cáo.
    threshold = MANAGER.threshold_for(ML02)
    try:
        frame = to_ml02_frame(profile, loan)
        probability = float(model.risk_probability(frame)[0])
        label = (str(model.predict(frame)[0]) if threshold is None
                 else ("HIGH_RISK" if probability >= threshold else "LOW_RISK"))
    except Exception as exc:  # noqa: BLE001
        log.exception("ML02 lỗi khi dự đoán")
        return PredictionResult.unavailable(
            "ml02", f"Model lỗi khi dự đoán: {exc}", PREDICTION_ERROR)

    try:
        check = conf.check_ml02(probability, float(threshold))
    except conf.InvalidProbability as exc:
        return PredictionResult.unavailable(
            "ml02", f"Xác suất không hợp lệ: {exc}", INVALID_PROBABILITY)

    return PredictionResult(
        model="ml02",
        available=True,
        label=label,
        label_vi=RISK_LABELS_VI[label],
        probability=probability,
        probabilities=[
            {"label": name, "label_vi": RISK_LABELS_VI[name],
             "probability": (probability if name == "HIGH_RISK"
                             else 1.0 - probability)}
            for name in model.classes_
        ],
        confidence=check.to_dict(),
        model_version=model.slug,
    )
