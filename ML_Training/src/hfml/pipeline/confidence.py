"""AI-01 task 7 — Kiểm tra confidence / probability (F05 · M05).

Ba việc, và việc thứ ba mới là việc khó:

    1. VALIDATE   xác suất phải nằm trong [0, 1] và tổng bằng 1
    2. NGƯỠNG     áp ngưỡng đã có của dự án, không đặt ngưỡng mới ở đây
    3. DIỄN ĐẠT   không biến xác suất thành sự chắc chắn

Hai ngưỡng, hai loại hoàn toàn khác nhau — đừng lẫn
-----------------------------------------------------
    `CONFIDENCE_THRESHOLD` (0,60, ở `config.yaml`)
        Ngưỡng TIN CẬY của ML01. `max(predict_proba) < 0,60` nghĩa là hồ sơ
        nằm gần ranh giới giữa hai nhóm — model vẫn chọn một nhãn, nhưng đó
        là phỏng đoán mong manh. Khi đó §8.1 yêu cầu hạ cấp xuống kết luận
        của rule và đánh dấu `low_confidence`.

    Ngưỡng của ML02 (0,1303, nằm TRONG artifact)
        Ngưỡng QUYẾT ĐỊNH `LOW_RISK` / `HIGH_RISK`. Nó không đo độ tin cậy —
        nó là chỗ cắt nhị phân, chốt ở F04 task 14 sau khi hiệu chuẩn.

Lẫn hai thứ này là sai kép: xác suất 0,20 của ML02 KHÔNG phải "kém tin cậy",
nó là một ước lượng rõ ràng rằng hồ sơ vượt ngưỡng 0,1303. Ngược lại, đòi
ML02 phải đạt 0,60 mới coi là "chắc chắn" thì với tỉ lệ nền 8,07% sẽ không
hồ sơ nào đạt.

Không biến xác suất thành sự chắc chắn
----------------------------------------
`describe()` sinh cụm từ mà tầng `llm` phải dùng nguyên văn. Không có cụm nào
mang nghĩa khẳng định — "khả năng cao", không phải "sẽ vỡ nợ". Đây là ràng
buộc của §8.2 guardrail 4, và nó phải nằm ở tầng tính chứ không phải để LLM
tự chọn chữ.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from hfml.config import CONFIG
from hfml.logger import get_logger

log = get_logger(__name__)

#: Sai số cho phép khi kiểm tổng xác suất — số dấu phẩy động không cộng khít.
PROBABILITY_TOLERANCE: Final[float] = 1e-6

#: Cụm từ mô tả mức xác suất, theo mốc dưới. Không cụm nào khẳng định chắc chắn.
_DESCRIPTIONS: Final[tuple[tuple[float, str], ...]] = (
    (0.80, "khả năng rất cao"),
    (0.60, "khả năng cao"),
    (0.40, "khả năng trung bình"),
    (0.20, "khả năng thấp"),
    (0.00, "khả năng rất thấp"),
)


class InvalidProbability(ValueError):
    """Xác suất không hợp lệ — dừng lại, đừng để nó chảy xuống báo cáo."""


@dataclass(frozen=True)
class ConfidenceCheck:
    """Kết quả kiểm một bộ xác suất."""

    confidence: float
    low_confidence: bool
    threshold: float
    description: str

    def to_dict(self) -> dict:
        return {
            "confidence": self.confidence,
            "low_confidence": self.low_confidence,
            "threshold": self.threshold,
            "description": self.description,
        }


def validate_probabilities(values, *, name: str = "probabilities") -> np.ndarray:
    """Kiểm một vector xác suất: hữu hạn, trong [0, 1], tổng bằng 1.

    Ném lỗi thay vì tự sửa. Một vector xác suất sai nghĩa là có gì đó hỏng ở
    tầng dưới; "chuẩn hoá lại cho tổng bằng 1" ở đây sẽ che mất chỗ hỏng và
    vẫn cho ra một kết quả trông bình thường.
    """
    array = np.asarray(values, dtype=float).ravel()

    if array.size == 0:
        raise InvalidProbability(f"{name}: vector rỗng.")
    if not np.isfinite(array).all():
        raise InvalidProbability(f"{name}: chứa NaN hoặc vô cực.")
    if (array < 0).any() or (array > 1).any():
        raise InvalidProbability(
            f"{name}: có giá trị ngoài [0, 1] (min {array.min():.4f}, "
            f"max {array.max():.4f}).")

    total = float(array.sum())
    if abs(total - 1.0) > PROBABILITY_TOLERANCE:
        raise InvalidProbability(f"{name}: tổng bằng {total:.6f}, không phải 1.")

    return array


def validate_single(value: float, *, name: str = "probability") -> float:
    """Kiểm MỘT xác suất lẻ (ví dụ P(vỡ nợ) của ML02).

    Không kiểm tổng — đây là xác suất của một biến cố, không phải một phân
    phối trên nhiều lớp.
    """
    number = float(value)
    if not np.isfinite(number):
        raise InvalidProbability(f"{name}: không hữu hạn ({value!r}).")
    if not 0.0 <= number <= 1.0:
        raise InvalidProbability(f"{name}: {number:.6f} nằm ngoài [0, 1].")
    return number


def describe(probability: float) -> str:
    """Cụm từ diễn đạt mức xác suất — tầng `llm` dùng NGUYÊN VĂN.

    Không cụm nào mang nghĩa khẳng định. "Khả năng rất cao" ở 0,85 vẫn để ngỏ
    15% còn lại; "sẽ xảy ra" thì không, và đó là điều §8.2 guardrail 4 cấm.
    """
    for floor, phrase in _DESCRIPTIONS:
        if probability >= floor:
            return phrase
    return _DESCRIPTIONS[-1][1]


def check_ml01(probabilities, threshold: float | None = None) -> ConfidenceCheck:
    """Kiểm độ tin cậy của ML01 — bài toán 4 lớp.

    `confidence` là xác suất của nhãn được chọn. Dưới ngưỡng thì §8.1 yêu cầu
    **hạ cấp xuống kết luận của rule** và nói ra, chứ không im lặng trình bày
    một phỏng đoán mong manh như một kết luận chắc chắn.
    """
    threshold = CONFIG.confidence_threshold if threshold is None else threshold
    array = validate_probabilities(probabilities, name="ml01.probabilities")
    confidence = float(array.max())

    return ConfidenceCheck(
        confidence=confidence,
        low_confidence=confidence < threshold,
        threshold=float(threshold),
        description=describe(confidence),
    )


def check_ml02(risk_probability: float, decision_threshold: float) -> ConfidenceCheck:
    """Kiểm kết quả ML02 — bài toán nhị phân có ngưỡng quyết định riêng.

    ⚠️ `decision_threshold` là ngưỡng CẮT `LOW_RISK`/`HIGH_RISK` (0,1303), KHÔNG
    phải ngưỡng tin cậy. Vì vậy `low_confidence` ở đây **không** so xác suất
    với ngưỡng đó — làm vậy là lẫn hai loại ngưỡng.

    Độ tin cậy của một quyết định nhị phân là **khoảng cách tới ranh giới**:
    hồ sơ có xác suất 0,1305 với ngưỡng 0,1303 nằm sát mép, đổi một chút dữ
    liệu là đổi nhãn. Hồ sơ 0,45 thì không.
    """
    probability = validate_single(risk_probability, name="ml02.risk_probability")
    threshold = validate_single(decision_threshold, name="ml02.threshold")

    # Khoảng cách tới ranh giới, chuẩn hoá theo phía hẹp hơn để hai bên so được.
    span = threshold if probability < threshold else (1.0 - threshold)
    margin = abs(probability - threshold) / span if span > 0 else 1.0

    return ConfidenceCheck(
        confidence=float(min(margin, 1.0)),
        # Nằm trong 20% khoảng cách tới ranh giới = sát mép, phải nói ra.
        low_confidence=margin < 0.20,
        threshold=threshold,
        description=describe(probability),
    )
