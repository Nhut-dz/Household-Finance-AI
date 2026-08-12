"""RB03 — Tính tiến độ và khả năng đạt mục tiêu tiết kiệm.

Hàm thuần: cùng đầu vào luôn cho cùng một kết quả cấu trúc chuẩn.
Tương thích cột CSDL tblcalculation_results.recommended_monthly_saving.
"""
from __future__ import annotations

from typing import Any, Mapping
from hfml.data.schema import HouseholdProfile
from hfml.rules.thresholds import RB03Thresholds, DEFAULT_THRESHOLDS


def evaluate_savings_goal(
    profile: HouseholdProfile | Mapping[str, Any],
    target_amount: float | None = None,
    target_months: int | None = None,
    thresholds: RB03Thresholds | None = None,
) -> dict[str, Any]:
    """Tính mức tiết kiệm hàng tháng cần thiết và đánh giá tính khả thi."""
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS.rb03

    if isinstance(profile, HouseholdProfile):
        income = float(profile.average_monthly_income)
        expense = float(profile.average_monthly_expense) if profile.average_monthly_expense is not None else 0.0
        debt_payment = float(profile.monthly_debt_payment) if profile.monthly_debt_payment is not None else 0.0
        current_savings = float(profile.savings_amount) if profile.savings_amount is not None else 0.0
    else:
        # Đồng bộ đầy đủ trường từ FE Form Pydantic và DB Laravel Backend
        income = float(profile.get("average_monthly_income") or profile.get("monthly_income") or 0.0)
        expense = float(profile.get("average_monthly_expense") or profile.get("monthly_living_cost") or 0.0)
        debt_payment = float(profile.get("monthly_debt_payment") or 0.0)
        current_savings = float(profile.get("savings_amount") or profile.get("current_savings") or 0.0)

    # Nếu không truyền mục tiêu riêng, dùng mặc định 12 tháng
    months = target_months if target_months and target_months > 0 else thresholds.default_timeline_months
    target = target_amount if target_amount is not None else 0.0

    net_surplus = max(0.0, income - expense - debt_payment)
    remaining_goal = max(0.0, target - current_savings)
    required_monthly = remaining_goal / months if months > 0 else remaining_goal

    max_usable_surplus = net_surplus * thresholds.max_surplus_allocation_ratio

    if remaining_goal == 0.0:
        status = "COMPLETED"
        message_key = "goal_completed"
    elif required_monthly <= max_usable_surplus:
        status = "FEASIBLE"
        message_key = "goal_feasible"
    elif required_monthly <= net_surplus:
        status = "STRETCHED"
        message_key = "goal_stretched"
    else:
        status = "INFEASIBLE"
        message_key = "goal_infeasible"

    return {
        "code": "RB03",
        "status": status,
        "value": {
            "target_amount": round(target, 2),
            "current_savings": round(current_savings, 2),
            "remaining_goal": round(remaining_goal, 2),
            "target_months": months,
            "required_monthly_savings": round(required_monthly, 2),
            "recommended_monthly_saving": round(required_monthly, 2),  # Tương thích tên cột DB tblcalculation_results
            "available_net_surplus": round(net_surplus, 2),
        },
        "threshold": {
            "max_surplus_allocation_ratio": thresholds.max_surplus_allocation_ratio,
        },
        "message_key": message_key,
        "details": {
            "summary_vi": (
                f"Mục tiêu: {target:,.0f}đ ({months} tháng). "
                f"Cần tiết kiệm: {required_monthly:,.0f}đ/tháng (Số dư thặng dư khả dụng: {net_surplus:,.0f}đ/tháng)."
            )
        },
    }
