"""RB04 — Phân bổ ngân sách theo Quy tắc 50/30/20 (Rule Kê đơn đề xuất từ thu nhập).

Hàm thuần: cùng đầu vào luôn cho cùng một kết quả cấu trúc chuẩn.
Tương thích 100% các cột DB budget_needs, budget_wants, budget_savings, allocation_rule trong tblcalculation_results.
"""
from __future__ import annotations

from typing import Any, Mapping
from hfml.data.schema import HouseholdProfile
from hfml.rules.thresholds import RB04Thresholds, DEFAULT_THRESHOLDS


def evaluate_503020(
    profile: HouseholdProfile | Mapping[str, Any],
    thresholds: RB04Thresholds | None = None,
) -> dict[str, Any]:
    """Tính mức phân bổ ngân sách đề xuất (50% Thiết yếu, 30% Cá nhân, 20% Tiết kiệm)."""
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS.rb04

    if isinstance(profile, HouseholdProfile):
        income = float(profile.average_monthly_income)
        expense = float(profile.average_monthly_expense) if profile.average_monthly_expense is not None else 0.0
        debt_payment = float(profile.monthly_debt_payment) if profile.monthly_debt_payment is not None else 0.0
    else:
        income = float(profile.get("average_monthly_income") or profile.get("monthly_income") or 0.0)
        expense = float(profile.get("average_monthly_expense") or profile.get("monthly_living_cost") or 0.0)
        debt_payment = float(profile.get("monthly_debt_payment") or 0.0)

    # 1. Tính mức phân bổ đề xuất từ thu nhập (mỗi phần khớp 100% cột CSDL tblcalculation_results)
    needs_ratio = thresholds.needs_ratio  # 0.50
    wants_ratio = thresholds.wants_ratio  # 0.30
    savings_ratio = thresholds.savings_ratio  # 0.20

    budget_needs = income * needs_ratio
    budget_wants = income * wants_ratio
    budget_savings = income * savings_ratio
    max_living_cost = income * (needs_ratio + wants_ratio)  # 80% thu nhập

    net_savings = max(0.0, income - expense - debt_payment)
    actual_savings_rate = (net_savings / income) if income > 0 else 0.0

    # 2. Đánh giá trạng thái thực tế so với chuẩn 50/30/20
    if expense > max_living_cost:
        status = "OVERBUDGET"
        message_key = "budget_over_living"
    elif actual_savings_rate < savings_ratio:
        status = "UNDER_SAVING"
        message_key = "budget_under_saving"
    else:
        status = "BALANCED"
        message_key = "budget_balanced"

    return {
        "code": "RB04",
        "status": status,
        "value": {
            "income": round(income, 2),
            "actual_living_cost": round(expense, 2),
            "actual_debt_payment": round(debt_payment, 2),
            "actual_net_savings": round(net_savings, 2),
            "actual_savings_rate": round(actual_savings_rate, 4),
            "budget_needs": round(budget_needs, 2),
            "budget_wants": round(budget_wants, 2),
            "budget_savings": round(budget_savings, 2),
            "max_recommended_living_cost": round(max_living_cost, 2),
            "allocation_rule": "50/30/20",
        },
        "threshold": {
            "needs_ratio": needs_ratio,
            "wants_ratio": wants_ratio,
            "savings_ratio": savings_ratio,
        },
        "message_key": message_key,
        "details": {
            "summary_vi": (
                f"Đề xuất 50/30/20 cho Thu nhập {income:,.0f}đ: "
                f"Thiết yếu 50% ({budget_needs:,.0f}đ), Cá nhân 30% ({budget_wants:,.0f}đ), "
                f"Tiết kiệm 20% ({budget_savings:,.0f}đ). Chi tiêu thực tế: {expense:,.0f}đ ({status})."
            )
        },
    }
