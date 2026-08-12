"""RuleEngine — Trình điều phối khởi chạy cả 5 rule tài chính (RB01, RB02, RB03, RB04, RB05).

Trả về một kết quả tổng hợp có cấu trúc sẵn sàng cho tầng LLM / API, bao gồm cả raw_json khớp 100% chuẩn CSDL tblcalculation_results.
"""
from __future__ import annotations

from typing import Any, Mapping
from hfml.data.schema import HouseholdProfile, FinancialNeed
from hfml.logger import get_logger
from hfml.rules.rb01_cashflow import evaluate_cashflow
from hfml.rules.rb02_health import evaluate_financial_health
from hfml.rules.rb03_savings_goal import evaluate_savings_goal
from hfml.rules.rb04_503020 import evaluate_503020
from hfml.rules.rb05_loan_capacity import evaluate_loan_capacity
from hfml.rules.thresholds import RuleThresholds, DEFAULT_THRESHOLDS

log = get_logger("hfml.rules.engine")


class RuleEngine:
    """Trình điều phối các quy tắc phân tích tài chính hộ gia đình."""

    def __init__(self, thresholds: RuleThresholds | None = None):
        self.thresholds = thresholds or DEFAULT_THRESHOLDS

    def evaluate(
        self,
        profile: HouseholdProfile | Mapping[str, Any],
        target_amount: float | None = None,
        target_months: int | None = None,
        requested_loan: float | None = None,
        asset_price: float | None = None,
        loan_term_months: int | None = None,
        need: FinancialNeed | str | None = None,
    ) -> dict[str, Any]:
        """Khởi chạy cả 5 quy tắc và tổng hợp kết quả có cấu trúc chuẩn."""
        log.info("Chạy RuleEngine cho hồ sơ hộ gia đình...")

        rb01_res = evaluate_cashflow(profile, thresholds=self.thresholds.rb01)
        rb02_res = evaluate_financial_health(profile, thresholds=self.thresholds.rb02)
        rb03_res = evaluate_savings_goal(
            profile,
            target_amount=target_amount,
            target_months=target_months,
            thresholds=self.thresholds.rb03,
        )
        rb04_res = evaluate_503020(profile, thresholds=self.thresholds.rb04)
        rb05_res = evaluate_loan_capacity(
            profile,
            requested_loan=requested_loan,
            asset_price=asset_price,
            term_months=loan_term_months,
            thresholds=self.thresholds.rb05,
        )

        # 1. Tổng hợp trạng thái chung (overall_status) từ cả 5 quy tắc
        if rb02_res["status"] == "CRITICAL" or rb01_res["status"] == "DEFICIT" or rb05_res["status"] == "REJECTED":
            overall_status = "CRITICAL"
        elif rb02_res["status"] == "WARNING" or rb03_res["status"] == "INFEASIBLE" or rb04_res["status"] == "OVERBUDGET" or rb05_res["status"] == "WARNING":
            overall_status = "WARNING"
        elif rb02_res["status"] == "EXCELLENT" and rb01_res["status"] == "POSITIVE" and rb04_res["status"] == "BALANCED":
            overall_status = "EXCELLENT"
        else:
            overall_status = "STABLE"

        rules_list = [rb01_res, rb02_res, rb03_res, rb04_res, rb05_res]

        # 2. Đóng gói raw_json theo chuẩn CSDL tblcalculation_results (PLAN.md §5.1)
        rb01_val = rb01_res.get("value", {})
        rb02_val = rb02_res.get("value", {})
        rb03_val = rb03_res.get("value", {})
        rb04_val = rb04_res.get("value", {})
        rb05_val = rb05_res.get("value", {})

        raw_json = {
            "scope": ["saving", "home_loan", "budget_50_30_20", "investment"],
            "intent": str(need) if need else "general_advice",
            "calculation": {
                "dti_percent": rb02_val.get("dti_percent", 0.0),
                "dti_status": rb02_val.get("dti_status", "LOW"),
                "monthly_surplus": rb01_val.get("net_cashflow", 0.0),
                "safe_new_monthly_payment": rb05_val.get("max_allowed_monthly_payment", 0.0),
                "safe_loan_limit": rb05_val.get("max_allowed_loan", 0.0),
                "recommended_monthly_saving": rb03_val.get("recommended_monthly_saving", 0.0),
                "budget_needs": rb04_val.get("budget_needs", 0.0),
                "budget_wants": rb04_val.get("budget_wants", 0.0),
                "budget_savings": rb04_val.get("budget_savings", 0.0),
                "allocation_rule": "50/30/20",
            },
            "profile_summary": {
                "income": rb01_val.get("income", 0.0),
                "expense": rb01_val.get("expense", 0.0),
                "debt_payment": rb01_val.get("debt_payment", 0.0),
                "savings": rb02_val.get("savings", 0.0),
                "total_debt": rb02_val.get("total_debt", 0.0),
                "has_dependents": rb02_val.get("has_dependents", False),
            },
        }

        return {
            "overall_status": overall_status,
            "rules": {
                "RB01": rb01_res,
                "RB02": rb02_res,
                "RB03": rb03_res,
                "RB04": rb04_res,
                "RB05": rb05_res,
            },
            "rules_list": rules_list,
            "raw_json": raw_json,
            "summary_vi": (
                f"Đánh giá tổng quan: {overall_status}. "
                f"Dòng tiền: {rb01_res['status']}, Sức khỏe: {rb02_res['status']}, "
                f"Mục tiêu tiết kiệm: {rb03_res['status']}, Ngân sách 50/30/20: {rb04_res['status']}, "
                f"Khả năng vay: {rb05_res['status']}."
            ),
        }
