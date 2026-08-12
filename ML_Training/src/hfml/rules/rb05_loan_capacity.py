"""RB05 — Đánh giá khả năng gánh nợ và hạn mức vay tối đa (DTI, LTV & Tài sản thế chấp).

Hàm thuần: cùng đầu vào luôn cho cùng một kết quả cấu trúc chuẩn.
Tương thích 100% cột CSDL tblcalculation_results.safe_loan_limit.
"""
from __future__ import annotations

from typing import Any, Mapping
from hfml.data.schema import HouseholdProfile
from hfml.rules.thresholds import RB05Thresholds, DEFAULT_THRESHOLDS


def _pmt_principal(monthly_payment: float, annual_rate: float, months: int) -> float:
    """Tính dư nợ gốc vay tối đa từ số tiền trả hàng tháng (PMT inverse formula)."""
    if monthly_payment <= 0 or months <= 0:
        return 0.0
    r = annual_rate / 12.0
    if r == 0:
        return monthly_payment * months
    pv = monthly_payment * (1.0 - (1.0 + r) ** (-months)) / r
    return float(pv)


def evaluate_loan_capacity(
    profile: HouseholdProfile | Mapping[str, Any],
    requested_loan: float | None = None,
    asset_price: float | None = None,
    term_months: int | None = None,
    thresholds: RB05Thresholds | None = None,
) -> dict[str, Any]:
    """Đánh giá khả năng vay tối đa theo DTI, LTV và tài sản đang sở hữu."""
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS.rb05

    if isinstance(profile, HouseholdProfile):
        income = float(profile.average_monthly_income)
        existing_debt_payment = float(profile.monthly_debt_payment) if profile.monthly_debt_payment is not None else 0.0
        total_debt = float(profile.total_current_debt) if profile.total_current_debt is not None else 0.0
        req_loan = requested_loan if requested_loan is not None else (float(profile.loan_amount) if profile.loan_amount is not None else 0.0)
        a_price = asset_price if asset_price is not None else (float(profile.asset_price) if profile.asset_price is not None else 0.0)
        months = term_months if term_months is not None else (profile.loan_term_months or 240)
        assets = [a.value if hasattr(a, "value") else str(a) for a in profile.assets]
    else:
        # Đồng bộ đầy đủ trường từ FE Form Pydantic và DB Laravel Backend
        income = float(profile.get("average_monthly_income") or profile.get("monthly_income") or 0.0)
        existing_debt_payment = float(profile.get("monthly_debt_payment") or 0.0)
        total_debt = float(profile.get("total_current_debt") or profile.get("total_debt") or 0.0)
        req_loan = requested_loan if requested_loan is not None else float(profile.get("loan_amount") or 0.0)
        a_price = asset_price if asset_price is not None else float(profile.get("asset_price") or 0.0)
        months = term_months if term_months is not None else int(profile.get("loan_term_months") or 240)
        assets_raw = profile.get("assets") or []
        assets = [a.value if hasattr(a, "value") else str(a) for a in assets_raw]

    # Phân tích năng lực thế chấp tài sản sở hữu (Hỗ trợ mã FE 'house'/'land'/'car' và mã DB 'real_estate'/'vehicle')
    has_real_estate = any(a in assets for a in ["house", "land", "real_estate"])
    has_vehicle = any(a in assets for a in ["car", "vehicle"])
    collateral_quality = "HIGH" if has_real_estate else ("MEDIUM" if has_vehicle else ("LOW" if assets else "NONE"))

    current_dti = (existing_debt_payment / income) if income > 0 else 0.0
    max_total_monthly_debt = income * thresholds.max_dti
    max_add_monthly_payment = max(0.0, max_total_monthly_debt - existing_debt_payment)

    # 1. Hạn mức vay tối đa theo DTI (dựa trên công thức tài chính PMT inverse)
    max_loan_by_dti = _pmt_principal(
        monthly_payment=max_add_monthly_payment,
        annual_rate=thresholds.assumed_annual_interest_rate,
        months=months,
    )

    # 2. Hạn mức vay tối đa theo LTV (nếu có giá trị tài sản mua)
    if a_price > 0:
        max_loan_by_ltv = a_price * thresholds.max_ltv
        max_allowed_loan = min(max_loan_by_dti, max_loan_by_ltv)
    else:
        max_loan_by_ltv = None
        max_allowed_loan = max_loan_by_dti

    ltv_requested = (req_loan / a_price) if a_price > 0 else None

    # 3. Phân loại trạng thái
    if current_dti >= thresholds.max_dti or max_add_monthly_payment <= 0:
        status = "REJECTED"
        message_key = "loan_rejected_dti_full"
    elif req_loan > 0 and req_loan <= max_allowed_loan:
        status = "APPROVED"
        message_key = "loan_approved"
    elif req_loan > max_allowed_loan:
        status = "WARNING"
        message_key = "loan_exceeds_capacity"
    else:
        status = "ELIGIBLE"
        message_key = "loan_eligible"

    return {
        "code": "RB05",
        "status": status,
        "value": {
            "income": round(income, 2),
            "existing_debt_payment": round(existing_debt_payment, 2),
            "total_debt": round(total_debt, 2),
            "current_dti": round(current_dti, 4),
            "max_allowed_monthly_payment": round(max_add_monthly_payment, 2),
            "max_allowed_loan": round(max_allowed_loan, 2),
            "safe_loan_limit": round(max_allowed_loan, 2),  # Tương thích cột CSDL tblcalculation_results.safe_loan_limit
            "max_loan_by_dti": round(max_loan_by_dti, 2),
            "max_loan_by_ltv": round(max_loan_by_ltv, 2) if max_loan_by_ltv is not None else None,
            "requested_loan": round(req_loan, 2),
            "requested_ltv": round(ltv_requested, 4) if ltv_requested is not None else None,
            "term_months": months,
            "assets_owned": assets,
            "collateral_quality": collateral_quality,
        },
        "threshold": {
            "max_dti": thresholds.max_dti,
            "max_ltv": thresholds.max_ltv,
            "assumed_annual_interest_rate": thresholds.assumed_annual_interest_rate,
        },
        "message_key": message_key,
        "details": {
            "summary_vi": (
                f"Hạn mức vay an toàn tối đa: {max_allowed_loan:,.0f}đ (Kỳ hạn {months} tháng). "
                f"Năng lực thế chấp hiện có: {collateral_quality}."
            )
        },
    }
