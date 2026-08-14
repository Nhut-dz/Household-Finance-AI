"""Pydantic request/response của tầng api.

Hiện chỉ phục vụ `POST /predict` (ML01). Contract ở đây là **17 feature đã
chuẩn hoá**, đúng tên và đúng ý nghĩa mà model được train — không phải tên cột
DB của backend.

Vì sao đặt ranh giới ở đây chứ không nhận thẳng hồ sơ dạng DB
--------------------------------------------------------------
Backend Laravel dùng tên cột riêng (`monthly_income`, `current_savings`, …) và
bộ giá trị tài sản riêng. Nếu ML service nhận hồ sơ thô rồi tự đoán, thì mỗi
lần backend đổi cột là ML service hỏng âm thầm. Bắt backend gửi đúng 17 feature
khiến chỗ lệch lộ ra ngay ở tầng validate, kèm tên field cụ thể.

`age` là bắt buộc, không có mặc định
------------------------------------
Model được train với `age` trong khoảng 22–62. Backend chỉ có `birth_year` và
trường đó cho phép bỏ trống. Khi thiếu, backend phải báo lỗi cho người dùng
chứ ML service KHÔNG tự điền một tuổi mặc định: điền bừa thì model vẫn trả về
một nhãn trông hợp lý, và không ai biết nó dựa trên tuổi bịa.
"""
from __future__ import annotations

from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from hfml.ml.ml01_recommendation.labeler import LABELS_VI, ORDERED_GROUPS

#: Ngưỡng trên của tiền, chặn số vô lý lọt vào model (1.000 tỉ VNĐ).
MAX_MONEY: Final[float] = 1_000_000_000_000.0


class Ml01PredictRequest(BaseModel):
    """17 feature của ML01, đúng thứ tự khai trong `RAW_FEATURES`.

    `extra="forbid"`: gửi thừa trường lạ thì báo lỗi chứ không nuốt im. Trường
    lạ gần như luôn là dấu hiệu backend map sai tên.
    """

    model_config = ConfigDict(extra="forbid")

    # -- Dòng tiền, nợ, tiết kiệm
    average_monthly_income: float = Field(..., gt=0, le=MAX_MONEY,
                                          description="Thu nhập trung bình tháng (VNĐ)")
    average_monthly_expense: float = Field(..., ge=0, le=MAX_MONEY,
                                           description="Chi tiêu trung bình tháng (VNĐ)")
    savings_amount: float = Field(..., ge=0, le=MAX_MONEY,
                                  description="Số tiền tiết kiệm; 0 khi không có")
    total_current_debt: float = Field(..., ge=0, le=MAX_MONEY,
                                      description="Tổng dư nợ; 0 khi không nợ")
    monthly_debt_payment: float = Field(..., ge=0, le=MAX_MONEY,
                                        description="Trả nợ hàng tháng; 0 khi không nợ")

    # -- Nhân khẩu
    household_size: int = Field(..., ge=1, le=50, description="Số người trong nhà")
    children_count: int = Field(..., ge=0, le=49, description="Số con")
    age: int = Field(..., ge=18, le=100, description="Tuổi người đại diện")

    # -- Cờ tình trạng
    has_debt: bool
    has_savings: bool
    has_dependents: bool = Field(..., description="Có phụng dưỡng người già")

    # -- Tài sản sở hữu, multi-hot
    has_asset_cash: bool = False
    has_asset_vehicle: bool = False
    has_asset_real_estate: bool = False
    has_asset_insurance: bool = False
    has_asset_gold: bool = False
    has_asset_investment: bool = False


class Ml01Probability(BaseModel):
    """Một dòng xác suất, kèm nhãn tiếng Việt để FE khỏi tự dịch."""

    label: str
    label_vi: str
    probability: float


class Ml01ModelConfidence(BaseModel):
    """Số liệu KỸ THUẬT về độ tin cậy — không phải kết quả dự đoán.

    Gom vào một khối riêng là có chủ ý. Trước đây `probabilities` nằm ngang
    hàng với `label` ở tầng ngoài, và FE vẽ bốn thanh cùng cỡ với nhãn thắng —
    người xem đọc thành "model trả về bốn kết quả". ML01 là phân loại đơn nhãn:
    kết quả đúng một nhãn, bốn xác suất chỉ là bên trong phép chọn nhãn đó.
    """

    confidence: float = Field(..., description="Xác suất của nhãn được dự đoán")
    low_confidence: bool = Field(
        ...,
        description="True khi xác suất cao nhất dưới ngưỡng tin cậy của config",
    )
    #: Xác suất đủ 4 lớp, xếp theo thang cấp thiết 🔴 → 🟢 chứ không theo xác
    #: suất giảm dần — thứ tự cố định giúp FE vẽ không nhảy giữa hai lần gọi.
    probabilities: list[Ml01Probability]


class Ml01PredictResponse(BaseModel):
    """Kết quả ML01 — Financial Recommendation Group Classification.

    `prediction` là **output nghiệp vụ duy nhất**: một nhóm định hướng tài
    chính. Mọi thứ trong `model_confidence` là thông tin kỹ thuật kèm theo.
    """

    prediction: str = Field(
        ...,
        description="Nhóm định hướng tài chính được dự đoán — MỘT nhãn. "
                    "Một trong EMERGENCY · DEBT_FOCUS · BUILD_BUFFER · GROWTH",
    )
    prediction_vi: str = Field(
        ..., description="Nhãn tiếng Việt của `prediction`, để hiển thị")
    model_confidence: Ml01ModelConfidence = Field(
        ..., description="Số liệu kỹ thuật; không trình bày như kết quả dự đoán")
    model_version: str = Field(..., description="Slug artifact đã dùng")


def build_probabilities(classes: list[str], values) -> list[Ml01Probability]:
    """Ghép `classes_` với vector xác suất, xếp theo thang mức độ.

    `classes_` của model xếp theo alphabet (`BUILD_BUFFER` trước `EMERGENCY`),
    còn thang mức độ nghiệp vụ là 🔴 EMERGENCY → 🟢 GROWTH. Ghép theo tên chứ
    không theo vị trí — dựa vào vị trí là đúng cho tới lần đầu tiên thứ tự lớp
    đổi, và lúc đó sai lặng lẽ.
    """
    by_label = {label: float(value) for label, value in zip(classes, values)}
    return [
        Ml01Probability(
            label=group.value,
            label_vi=LABELS_VI[group],
            probability=by_label.get(group.value, 0.0),
        )
        for group in ORDERED_GROUPS
    ]
