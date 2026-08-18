"""Chỉ số tài chính — ĐỊNH NGHĨA DUY NHẤT cho toàn hệ thống (F02 · §5).

Mọi tầng đọc chỉ số từ đây: tầng quy tắc (RB01–RB05), hàm gán nhãn `g(·)` của
ML01, và bất cứ chỗ nào cần cùng những con số đó. Không tầng nào được tự tính
lại.

Vì sao file này ra đời
------------------------
Trước đó `savings_rate` có BA công thức khác nhau trong cùng một hệ thống:

    labeler.compute_indicators   (thu − chi) / thu                  ← bỏ trả nợ
    rb01_cashflow                (thu − chi − trả nợ) / thu         ← có dấu
    rb02_health            max(0, thu − chi − trả nợ) / thu         ← kẹp về 0

Đo trên 4.000 hộ sinh bằng seed 42:

    · 72,0% số hộ có `savings_rate` khác nhau giữa labeler và RB02,
      lệch trung bình 0,1393 — cao nhất 0,7519
    · 39,4% số hộ có dòng tiền âm, và với TOÀN BỘ nhóm này RB01 báo số âm
      (ví dụ −0,2548) còn RB02 báo đúng 0,0

Hậu quả không phải chuyện làm tròn. Người dùng đọc "tỷ lệ tiết kiệm 37,1%" từ
tầng quy tắc, trong khi ML01 gán nhãn cho họ dựa trên 51,4% — hai tầng nói về
cùng một hộ bằng hai con số khác nhau, và không tầng nào sai theo cách nhìn
thấy được.

Nghiêm trọng nhất là phép kẹp của RB02: nó xoá sạch dấu âm ở ĐÚNG nhóm đang
gặp khó khăn — nhóm mà độ lớn của thâm hụt là thông tin quan trọng nhất.

Quy ước: GIỮ DẤU, không kẹp
-----------------------------
`net_cashflow` và `savings_rate` đều có dấu. Hộ thâm hụt 5 triệu/tháng khác
hẳn hộ hoà vốn, và gộp cả hai thành 0 là vứt đi đúng thứ cần biết.

Nơi nào cần con số không âm để hiển thị thì tự kẹp ở chỗ hiển thị, KHÔNG kẹp
ở đây — kẹp tại nguồn nghĩa là mọi tầng phía sau mất thông tin mà không biết.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

#: Tên thay thế mà backend/DB dùng cho cùng một trường.
#: Giữ ở đây để mọi rule đọc hồ sơ theo cùng một cách.
_ALIASES: dict[str, tuple[str, ...]] = {
    "income": ("average_monthly_income", "monthly_income"),
    "expense": ("average_monthly_expense", "monthly_living_cost",
                "monthly_expense"),
    "debt_payment": ("monthly_debt_payment",),
    "savings": ("savings_amount", "current_savings"),
    "total_debt": ("total_current_debt", "total_debt"),
}


def _read(profile: Any, key: str) -> float:
    """Đọc một trường tiền, chấp nhận mọi tên gọi đã biết. Thiếu thì là 0."""
    for name in _ALIASES[key]:
        if hasattr(profile, name):
            value = getattr(profile, name)
        elif isinstance(profile, Mapping):
            value = profile.get(name)
        else:
            continue
        if value is not None:
            return float(value)
    return 0.0


@dataclass(frozen=True)
class FinancialIndicators:
    """Bộ chỉ số chuẩn của một hộ. Bất biến — tính một lần, dùng ở mọi tầng."""

    income: float
    expense: float
    debt_payment: float
    savings: float

    #: Thu − chi − trả nợ. CÓ DẤU: âm nghĩa là thâm hụt.
    net_cashflow: float
    #: Thu − chi, chưa trừ trả nợ. Dùng để tách phần chi sinh hoạt.
    living_cashflow: float
    #: `net_cashflow / income`. CÓ DẤU — xem docstring đầu file.
    savings_rate: float
    #: `debt_payment / income`.
    dti: float
    #: Số tháng chi tiêu mà khoản tiết kiệm gánh được.
    #:
    #: Không có chi tiêu thì đệm là vô hạn — trả `inf` chứ không phải `NaN`,
    #: để phép so sánh "< 3" vẫn cho kết quả đúng thay vì lặng lẽ thành False.
    emergency_months: float

    @property
    def is_deficit(self) -> bool:
        return self.net_cashflow < 0


def compute(profile: Any) -> FinancialIndicators:
    """Chỉ số của một hộ. Nhận `HouseholdProfile`, dict, hay bản ghi có thuộc tính."""
    income = _read(profile, "income")
    expense = _read(profile, "expense")
    debt_payment = _read(profile, "debt_payment")
    savings = _read(profile, "savings")

    net_cashflow = income - expense - debt_payment

    return FinancialIndicators(
        income=income,
        expense=expense,
        debt_payment=debt_payment,
        savings=savings,
        net_cashflow=net_cashflow,
        living_cashflow=income - expense,
        savings_rate=(net_cashflow / income) if income > 0 else 0.0,
        dti=(debt_payment / income) if income > 0 else 0.0,
        emergency_months=(
            savings / expense if expense > 0
            else (float("inf") if savings > 0 else 0.0)),
    )


def compute_frame(df):
    """Bản vector hoá cho cả `DataFrame` — dùng khi train ML01.

    Cùng công thức với `compute()`, chỉ khác cách chạy. Có test khoá việc hai
    đường phải cho cùng kết quả; lệch nhau là quay lại đúng vấn đề file này
    sinh ra để giải quyết.
    """
    import numpy as np
    import pandas as pd

    def col(key: str):
        for name in _ALIASES[key]:
            if name in df.columns:
                return df[name].astype(float).fillna(0.0)
        return pd.Series(0.0, index=df.index)

    income = col("income")
    expense = col("expense")
    debt_payment = col("debt_payment")
    savings = col("savings")

    net_cashflow = income - expense - debt_payment
    safe_income = income.where(income > 0)
    safe_expense = expense.where(expense > 0)

    return pd.DataFrame({
        "net_cashflow": net_cashflow,
        "living_cashflow": income - expense,
        "savings_rate": (net_cashflow / safe_income).fillna(0.0),
        "dti": (debt_payment / safe_income).fillna(0.0),
        # `inf` khi không có chi tiêu — cùng lý do với `compute()`.
        "emergency_months": (savings / safe_expense).replace(np.nan, np.inf),
    }, index=df.index)
