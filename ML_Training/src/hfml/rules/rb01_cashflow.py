"""RB01 — Tính toán thu nhập, chi tiêu sinh hoạt, nghĩa vụ nợ và dòng tiền ròng khả dụng.

Hàm thuần: cùng đầu vào luôn cho cùng một kết quả cấu trúc chuẩn.
"""
from __future__ import annotations

from typing import Any, Mapping
from hfml.data.schema import HouseholdProfile
from hfml.rules.thresholds import RB01Thresholds, DEFAULT_THRESHOLDS


def evaluate_cashflow(
    profile: HouseholdProfile | Mapping[str, Any],
    thresholds: RB01Thresholds | None = None,
) -> dict[str, Any]:
    """Tính dòng tiền ròng thực tế = Thu nhập - Chi tiêu sinh hoạt - Tiền trả nợ."""
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS.rb01

    if isinstance(profile, HouseholdProfile):
        income = float(profile.average_monthly_income)
        expense = float(profile.average_monthly_expense) if profile.average_monthly_expense is not None else 0.0
        debt_payment = float(profile.monthly_debt_payment) if profile.monthly_debt_payment is not None else 0.0
    else:
        # Hỗ trợ cả Pydantic schema keys (average_monthly_income) và DB column names (monthly_income)
        income = float(profile.get("average_monthly_income") or profile.get("monthly_income") or 0.0)
        expense = float(profile.get("average_monthly_expense") or profile.get("monthly_living_cost") or 0.0)
        debt_payment = float(profile.get("monthly_debt_payment") or 0.0)

    # 1. Dòng tiền ròng khả dụng thực tế sau khi trừ chi tiêu sinh hoạt và trả nợ
    net_cashflow = income - expense - debt_payment
    living_cashflow = income - expense
    savings_rate = (net_cashflow / income) if income > 0 else 0.0

    # 2. Xếp loại trạng thái dòng tiền
    if net_cashflow > thresholds.zero_tolerance:
        status = "POSITIVE"
        message_key = "cashflow_positive"
    elif abs(net_cashflow) <= thresholds.zero_tolerance:
        status = "BALANCED"
        message_key = "cashflow_balanced"
    else:
        status = "DEFICIT"
        message_key = "cashflow_deficit"

    return {
        "code": "RB01",
        "status": status,
        "value": {
            "income": round(income, 2),
            "expense": round(expense, 2),
            "debt_payment": round(debt_payment, 2),
            "net_cashflow": round(net_cashflow, 2),
            "living_cashflow": round(living_cashflow, 2),
            "savings_rate": round(savings_rate, 4),
            "savings_rate_percent": round(savings_rate * 100, 2),
        },
        "threshold": {
            "zero_tolerance": thresholds.zero_tolerance,
        },
        "message_key": message_key,
        "details": {
            "summary_vi": (
                f"Thu nhập: {income:,.0f}đ, Chi tiêu: {expense:,.0f}đ, Trả nợ: {debt_payment:,.0f}đ. "
                f"Dòng tiền ròng thực tế: {net_cashflow:,.0f}đ ({status})."
            )
        },
    }
