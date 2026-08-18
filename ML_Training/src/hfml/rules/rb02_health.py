"""RB02 — Đánh giá sức khỏe tài chính tổng hợp (DTI, Quỹ dự phòng, Tỷ lệ tiết kiệm, Gánh nặng phụ thuộc).

Hàm thuần: cùng đầu vào luôn cho cùng một kết quả cấu trúc chuẩn.
Đáp ứng chuẩn CSDL tblcalculation_results (dti_ratio dạng phần trăm 0-100, dti_status 3 mức LOW/MEDIUM/HIGH).
"""
from __future__ import annotations

from typing import Any, Mapping
from hfml.data.schema import HouseholdProfile
from hfml.rules import indicators
from hfml.rules.thresholds import RB02Thresholds, DEFAULT_THRESHOLDS


def _read_total_debt(profile: Any) -> float:
    """Tổng dư nợ — chỉ RB02 dùng nên không đưa vào bộ chỉ số chung."""
    if isinstance(profile, HouseholdProfile):
        return float(profile.total_current_debt or 0.0)
    return float(profile.get("total_current_debt") or profile.get("total_debt") or 0.0)


def evaluate_financial_health(
    profile: HouseholdProfile | Mapping[str, Any],
    thresholds: RB02Thresholds | None = None,
) -> dict[str, Any]:
    """Đánh giá 4 mức sức khỏe tài chính (EXCELLENT, GOOD, WARNING, CRITICAL)."""
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS.rb02

    # Chỉ số tiền tệ lấy từ ĐỊNH NGHĨA DUY NHẤT (`hfml.rules.indicators`).
    #
    # `savings_rate` ở đây TỪNG được kẹp về 0 bằng `max(0, ...)`. Phép kẹp đó
    # xoá sạch dấu âm ở đúng 39,4% số hộ đang thâm hụt — nhóm mà độ lớn của
    # thâm hụt là thông tin quan trọng nhất — và làm RB02 báo 0,0 trong khi
    # RB01 báo −0,2548 cho cùng một hộ. Nay cả hai dùng chung một con số.
    ind = indicators.compute(profile)
    income, expense = ind.income, ind.expense
    debt_payment, savings = ind.debt_payment, ind.savings
    total_debt = _read_total_debt(profile)
    net_savings = ind.net_cashflow
    dti = ind.dti
    savings_rate = ind.savings_rate
    emergency_months = ind.emergency_months

    if isinstance(profile, HouseholdProfile):
        has_dependents = bool(profile.has_dependents)
        household_size = int(profile.household_size)
        children_count = int(profile.children_count)
        assets = [a.value if hasattr(a, "value") else str(a) for a in profile.assets]
    else:
        has_dependents = bool(profile.get("has_dependents") or profile.get("supports_elderly") or False)
        household_size = int(profile.get("household_size") or 1)
        children_count = int(profile.get("children_count") or 0)
        assets_raw = profile.get("assets") or []
        assets = [a.value if hasattr(a, "value") else str(a) for a in assets_raw]

    dti_percent = dti * 100.0  # Tương thích cột tblcalculation_results.dti_ratio (0-100)
    if dti < 0.20:
        dti_status = "LOW"
    elif dti < 0.40:
        dti_status = "MEDIUM"
    else:
        dti_status = "HIGH"

    income_per_capita = (income / household_size) if household_size > 0 else income

    # Ngưỡng đệm khẩn cấp khuyến nghị: Nâng từ 3 tháng lên 6 tháng nếu có phụng dưỡng người già
    min_recommended_emergency_months = 6.0 if has_dependents else thresholds.emergency_good_min

    # 2. Thu thập các lý do cảnh báo
    reasons: list[str] = []

    if dti > thresholds.dti_warning_max:
        reasons.append(f"Tỷ lệ DTI trả nợ cao ({dti_percent:.1f}% > {thresholds.dti_warning_max * 100:.0f}%)")
    if emergency_months < min_recommended_emergency_months:
        if has_dependents:
            reasons.append(f"Có phụng dưỡng người già: Quỹ dự phòng mỏng ({emergency_months:.1f} tháng < 6.0 tháng dự phòng y tế)")
        else:
            reasons.append(f"Quỹ dự phòng mỏng ({emergency_months:.1f} tháng < {thresholds.emergency_warning_less:.0f} tháng)")
    if savings_rate < thresholds.savings_rate_warning_less:
        reasons.append(f"Tỷ lệ tiết kiệm thực tế thấp ({savings_rate:.1%} < {thresholds.savings_rate_warning_less:.0%})")

    # 3. Phân cấp 4 mức sức khỏe tài chính (EXCELLENT, GOOD, WARNING, CRITICAL)
    if (income - expense - debt_payment) < 0 or dti > thresholds.dti_warning_max or (emergency_months < thresholds.emergency_critical_less and savings_rate < 0.10):
        status = "CRITICAL"
        message_key = "health_critical"
    elif len(reasons) > 0:
        status = "WARNING"
        message_key = "health_warning"
    elif dti <= thresholds.dti_excellent_max and emergency_months >= thresholds.emergency_excellent_min and savings_rate >= thresholds.savings_rate_excellent_min:
        status = "EXCELLENT"
        message_key = "health_excellent"
    else:
        status = "GOOD"
        message_key = "health_good"

    return {
        "code": "RB02",
        "status": status,
        "value": {
            "income": round(income, 2),
            "expense": round(expense, 2),
            "debt_payment": round(debt_payment, 2),
            "total_debt": round(total_debt, 2),
            "savings": round(savings, 2),
            "dti": round(dti, 4),
            "dti_percent": round(dti_percent, 2),
            "dti_status": dti_status,
            "emergency_months": round(emergency_months, 2),
            "savings_rate": round(savings_rate, 4),
            "savings_rate_percent": round(savings_rate * 100, 2),
            "income_per_capita": round(income_per_capita, 2),
            "has_dependents": has_dependents,
            "household_size": household_size,
            "children_count": children_count,
            "assets_count": len(assets),
            "min_recommended_emergency_months": min_recommended_emergency_months,
        },
        "threshold": {
            "dti_good_max": thresholds.dti_good_max,
            "dti_warning_max": thresholds.dti_warning_max,
            "emergency_good_min": min_recommended_emergency_months,
            "emergency_excellent_min": thresholds.emergency_excellent_min,
            "savings_rate_good_min": thresholds.savings_rate_good_min,
        },
        "message_key": message_key,
        "details": {
            "reasons": reasons,
            "summary_vi": f"Sức khỏe tài chính: {status} (DTI: {dti_percent:.1f}%, Đệm khẩn cấp: {emergency_months:.1f} tháng, Tỷ lệ tiết kiệm: {savings_rate * 100:.1f}%)."
        },
    }
