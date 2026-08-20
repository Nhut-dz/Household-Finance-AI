"""ML01 — Feature engineering (F03 · redesign 17/08/2026).

Feature dựng từ HỒ SƠ NGƯỜI DÙNG KHAI, không phải từ hoàn cảnh thật. Khoảng
cách giữa hai thứ đó chính là sai số Bayes của bài toán — xem `dataset.py`.

Vì sao chuyển sang tỉ số thay vì số tuyệt đối
-----------------------------------------------
Bản cũ đưa thẳng `savings_amount`, `average_monthly_income`… vào `X`. Hậu quả
đo được: `savings_amount` + `has_savings` chiếm 41,5% tổng trọng số, áp đảo
thu nhập + chi tiêu (28,6%).

Nguyên nhân không phải model chọn sai mà là cây chỉ cắt được theo từng trục.
Ranh giới tài chính có ý nghĩa lại gần như luôn là tỉ số — "tiết kiệm đủ mấy
tháng chi tiêu", "trả nợ chiếm bao nhiêu phần thu nhập". Một tỉ số là đường
cong trong không gian các số tuyệt đối, nên cây phải xấp xỉ nó bằng rất nhiều
nhát cắt, và cuối cùng bám vào biến nào dễ cắt nhất.

Đưa sẵn tỉ số thì mỗi khái niệm tài chính là MỘT trục, và trọng số phản ánh
tầm quan trọng nghiệp vụ thay vì phản ánh hình học của cây.

Đây là derived feature hợp lệ, KHÔNG phải rò rỉ nhãn
------------------------------------------------------
Ranh giới phân biệt hai loại: một feature là rò rỉ khi nó mang thông tin về
nhãn mà tại thời điểm dự đoán KHÔNG thể biết. Ở đây mọi tỉ số đều tính từ
đúng những con số mà người dùng điền vào form — không có gì đến từ tương lai,
và không có gì đến từ hàm chấm điểm.

Điều khiến bản cũ có vấn đề không phải là các tỉ số, mà là nhãn cũ được sinh
bằng ba phép so sánh bậc thang trên chính chúng. Nhãn mới là tổ hợp trơn của
hai mươi tín hiệu, tính trên GIÁ TRỊ THẬT — nên biết tỉ số đã khai không cho
phép suy ngược ra nhãn.
"""
from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

from hfml.data.schema import ASSET_COLUMNS
from hfml.rules import indicators as rule_indicators

#: Trần số tháng đệm. Trên mức này khác biệt không còn ý nghĩa nghiệp vụ, mà
#: đuôi dài thì kéo lệch mọi phép chuẩn hoá phía sau.
MAX_EMERGENCY_MONTHS: Final[float] = 60.0

#: Feature cuối cùng đưa vào `X`, nhóm theo ý nghĩa.
FEATURES: Final[tuple[str, ...]] = (
    # -- Quy mô (log vì tiền lệch phải rất mạnh)
    "log_income",
    "log_expense",
    "log_income_per_capita",

    # -- Cấu trúc dòng tiền: mỗi khái niệm tài chính một trục
    "expense_ratio",          # chi / thu
    "savings_rate",           # (thu − chi − trả nợ) / thu — CÓ DẤU
    "net_cashflow_ratio",     # dư trên mỗi người, chuẩn hoá theo thu nhập

    # -- Gánh nặng nợ
    "dti",                    # trả nợ / thu
    "debt_years",             # dư nợ / thu nhập năm
    "payment_share",          # trả nợ / tổng dòng ra

    # -- Dự phòng
    "emergency_months",       # tiết kiệm / chi tiêu
    "savings_to_income",      # tiết kiệm / thu nhập tháng
    "debt_to_savings",        # dư nợ / tiết kiệm

    # -- Nhân khẩu
    "household_size",
    "children_count",
    "dependency_ratio",       # người phụ thuộc / tổng thành viên
    "age",
    "has_dependents",

    # -- Tài sản, gộp thành một cột
    "asset_count",
)

#: Cột bị loại khỏi `X`, kèm lý do. Ghi lại để lần sau không ai thêm lại.
DROPPED: Final[dict[str, str]] = {
    "has_savings":
        "Trùng thông tin: đúng bằng `savings_amount > 0`, mà số dư đã có mặt "
        "qua `savings_to_income` và `emergency_months`. Ở bản cũ cột này còn "
        "chiếm 9,9% trọng số — trọng số dành cho một bit không thêm gì.",
    "has_debt":
        "Trùng thông tin: đúng bằng `total_current_debt > 0`, đã có qua `dti` "
        "và `debt_years`.",
    "savings_amount":
        "Số tuyệt đối. Giữ nguyên thì lại thành biến thay thế áp đảo như bản "
        "cũ. Thay bằng hai tỉ số mang đúng ý nghĩa nghiệp vụ.",
    "total_current_debt":
        "Số tuyệt đối — thay bằng `debt_years` và `debt_to_savings`.",
    "monthly_debt_payment":
        "Số tuyệt đối — thay bằng `dti` và `payment_share`.",
    "6 cột has_asset_*":
        "Gộp thành `asset_count`. Sáu cột nhị phân rời rạc chia nhỏ trọng số "
        "mà không cột nào đủ tín hiệu để cây dùng.",
}


def _safe_log(series: pd.Series) -> pd.Series:
    """`log1p` trên phần không âm — tiền lệch phải rất mạnh."""
    return np.log1p(series.astype(float).clip(lower=0.0))


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Dựng `X` từ hồ sơ đã khai. Trả về khung đúng thứ tự `FEATURES`.

    Dùng CHUNG cho train và inference. Dựng riêng ở hai nơi là cách chắc chắn
    nhất để hai bên tính lệch nhau mà không ai phát hiện — model vẫn chạy, vẫn
    trả xác suất, chỉ có điều xác suất đó vô nghĩa.
    """
    ind = rule_indicators.compute_frame(df)

    income = df["average_monthly_income"].astype(float).clip(lower=1.0)
    expense = df["average_monthly_expense"].astype(float).fillna(0.0)
    savings = df["savings_amount"].astype(float).fillna(0.0)
    payment = df["monthly_debt_payment"].astype(float).fillna(0.0)
    total_debt = df["total_current_debt"].astype(float).fillna(0.0)
    size = df["household_size"].astype(float).clip(lower=1.0)
    children = df["children_count"].astype(float).fillna(0.0)
    dependents = df["has_dependents"].astype(bool)

    outflow = (expense + payment).clip(lower=1.0)
    months = ind["emergency_months"].replace(np.inf, MAX_EMERGENCY_MONTHS)

    out = pd.DataFrame(index=df.index)
    out["log_income"] = _safe_log(income)
    out["log_expense"] = _safe_log(expense)
    out["log_income_per_capita"] = _safe_log(income / size)

    out["expense_ratio"] = expense / income
    out["savings_rate"] = ind["savings_rate"]
    out["net_cashflow_ratio"] = ind["net_cashflow"] / size / income

    out["dti"] = ind["dti"]
    out["debt_years"] = total_debt / (income * 12.0)
    out["payment_share"] = payment / outflow

    out["emergency_months"] = months.clip(upper=MAX_EMERGENCY_MONTHS)
    out["savings_to_income"] = savings / income
    # Không nợ thì tỉ lệ này bằng 0, không phải vô hạn.
    out["debt_to_savings"] = np.where(
        savings > 0, total_debt / savings.clip(lower=1.0), 0.0)

    out["household_size"] = size
    out["children_count"] = children
    out["dependency_ratio"] = (children + dependents.astype(float)) / size
    out["age"] = df["age"].astype(float) if "age" in df.columns else np.nan
    out["has_dependents"] = dependents.astype(int)

    present = [c for c in ASSET_COLUMNS if c in df.columns]
    out["asset_count"] = (df[present].astype(bool).sum(axis=1).astype(float)
                          if present else 0.0)

    return out[list(FEATURES)]
