"""Tầng api — FastAPI endpoints.

    GET  /health    kiểm tra service sống, kèm trạng thái model ML01
    POST /advise    tư vấn dựa trên tầng rule (F02)
    POST /predict   phân loại 4 nhóm khuyến nghị bằng model ML01 (F03)
"""
import functools
import re
from typing import Any, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from hfml.api.schemas import (
    Ml01PredictRequest,
    Ml01PredictResponse,
    build_probabilities,
)
from hfml.config import CONFIG
from hfml.logger import get_logger
from hfml.ml.ml01_recommendation.labeler import LABELS_VI, RAW_FEATURES, RecommendationGroup
from hfml.ml.registry import load_model
from hfml.rules.engine import RuleEngine
from hfml.rules.rb05_loan_capacity import evaluate_loan_capacity

log = get_logger(__name__)

app = FastAPI(title="Household Finance ML API", version="0.3.0")
rule_engine = RuleEngine()

#: Artifact được task 14 chọn và task 15 export. Đọc từ `CONFIG.paths.runs`.
ML01_SLUG = "ml01_xgboost_vfinal"


@functools.lru_cache(maxsize=1)
def get_ml01_model():
    """Nạp model ML01 một lần rồi giữ lại.

    Nạp lại mỗi request thì mất ~200ms cho việc đọc 2,8 MB từ đĩa, và với
    XGBoost 1.200 cây thì đó là phần lớn thời gian trả lời.

    Cố ý KHÔNG nạp lúc import: import lỗi thì cả service chết, kể cả
    `/health` và `/advise` vốn không cần model. Nạp lười thì thiếu artifact
    chỉ làm hỏng đúng `/predict`, và `/health` vẫn nói ra được là hỏng.
    """
    model = load_model(ML01_SLUG)
    log.info("Đã nạp model ML01: %s (%d feature, %d lớp)",
             ML01_SLUG, len(model.feature_names_), len(model.classes_))
    return model


class AdviseRequest(BaseModel):
    question: str
    household: dict[str, Any]


class AdviseResponse(BaseModel):
    response_text: str
    model_used: str = "HFML-RuleEngine-Advisor"
    suggested_questions: Optional[List[str]] = None
    tokens_used: Optional[int] = None


def parse_amount_from_text(text: str) -> float | None:
    """Trích xuất số tiền VNĐ từ câu hỏi người dùng (ví dụ: '5 tỷ', '500 triệu', '5.5 tỷ')."""
    match_ty = re.search(r"(\d+(?:[\.,]\d+)?)\s*(tỷ|ty|tỉ|ti)", text, re.IGNORECASE)
    if match_ty:
        val = float(match_ty.group(1).replace(",", "."))
        return val * 1_000_000_000

    match_trieu = re.search(r"(\d+(?:[\.,]\d+)?)\s*(triệu|trieu|tr)", text, re.IGNORECASE)
    if match_trieu:
        val = float(match_trieu.group(1).replace(",", "."))
        return val * 1_000_000

    match_digits = re.search(r"\b(\d{7,12})\b", text)
    if match_digits:
        return float(match_digits.group(1))

    return None


def parse_term_months_from_text(text: str) -> int | None:
    """Trích xuất thời hạn vay (số năm hoặc số tháng) từ câu hỏi người dùng (ví dụ: '5 năm', '60 tháng')."""
    match_nam = re.search(r"(\d+)\s*(năm|nam|n)\b", text, re.IGNORECASE)
    if match_nam:
        years = int(match_nam.group(1))
        if 1 <= years <= 30:
            return years * 12

    match_thang = re.search(r"(\d+)\s*(tháng|thang|th)\b", text, re.IGNORECASE)
    if match_thang:
        months = int(match_thang.group(1))
        if 1 <= months <= 360:
            return months

    return None


@app.get("/health")
def health() -> dict:
    """Trạng thái service, kèm việc model ML01 có nạp được không.

    Báo cả trạng thái model chứ không chỉ "ok": service sống mà thiếu artifact
    thì `/predict` hỏng, và người vận hành cần biết điều đó trước khi FE gọi
    tới rồi mới phát hiện.
    """
    try:
        model = get_ml01_model()
        ml01 = {"loaded": True, "slug": ML01_SLUG,
                "n_features": len(model.feature_names_),
                "classes": list(model.classes_)}
    except FileNotFoundError as exc:
        ml01 = {"loaded": False, "slug": ML01_SLUG, "error": str(exc)}

    return {"status": "ok", "service": "Household Finance ML Service", "ml01": ml01}


@app.post("/predict", response_model=Ml01PredictResponse)
def predict(req: Ml01PredictRequest) -> Ml01PredictResponse:
    """ML01 — phân loại hộ vào 4 nhóm khuyến nghị (F03).

    Cột được dựng theo ĐÚNG thứ tự `RAW_FEATURES` rồi mới đưa vào model.
    `PipelineClassifier` cũng tự sắp lại cột, nhưng làm ở đây thì lệch tên
    field lộ ra ngay tại biên chứ không chui vào trong pipeline.
    """
    try:
        model = get_ml01_model()
    except FileNotFoundError as exc:
        # 503 chứ không 500: service vẫn chạy, chỉ là chưa có artifact — đây
        # là lỗi triển khai, không phải lỗi của request.
        raise HTTPException(
            status_code=503,
            detail=f"Chưa có artifact ML01 ({ML01_SLUG}). Chạy train + export trước. {exc}",
        ) from exc

    payload = req.model_dump()
    frame = pd.DataFrame([[payload[name] for name in RAW_FEATURES]],
                         columns=list(RAW_FEATURES))

    try:
        label = str(model.predict(frame)[0])
        proba = model.predict_proba(frame)[0]
    except Exception as exc:  # noqa: BLE001 — biên ngoài, không để traceback lọt ra
        log.exception("ML01 dự đoán lỗi")
        raise HTTPException(status_code=500, detail=f"Model lỗi khi dự đoán: {exc}") from exc

    probabilities = build_probabilities(list(model.classes_), proba)
    confidence = next(p.probability for p in probabilities if p.label == label)

    return Ml01PredictResponse(
        label=label,
        label_vi=LABELS_VI[RecommendationGroup(label)],
        confidence=confidence,
        probabilities=probabilities,
        # Dưới ngưỡng thì FE phải nói ra là kết quả chưa chắc chắn, chứ không
        # hiển thị như một kết luận (PLAN.md §8.1).
        low_confidence=confidence < CONFIG.confidence_threshold,
        model_version=ML01_SLUG,
    )


@app.post("/advise", response_model=AdviseResponse)
def advise(req: AdviseRequest) -> AdviseResponse:
    household = req.household
    res = rule_engine.evaluate(household)

    overall_status = res.get("overall_status", "STABLE")
    rules = res.get("rules", {})

    # Lấy thông tin tài chính đã nhập
    total_debt = float(household.get("total_debt") or household.get("total_current_debt") or 0.0)
    debt_payment = float(household.get("monthly_debt_payment") or 0.0)
    savings = float(household.get("current_savings") or household.get("savings_amount") or 0.0)
    income = float(household.get("monthly_income") or household.get("average_monthly_income") or 0.0)
    expense = float(household.get("monthly_living_cost") or household.get("average_monthly_expense") or 0.0)
    has_dependents = bool(household.get("supports_elderly") or household.get("has_dependents") or False)
    assets_raw = household.get("assets") or []
    assets = [a.value if hasattr(a, "value") else str(a) for a in assets_raw]

    rb01 = rules.get("RB01", {})
    rb01_val = rb01.get("value", {})
    net_cashflow = rb01_val.get("net_cashflow", 0.0)

    rb02 = rules.get("RB02", {})
    rb02_val = rb02.get("value", {})
    dti = rb02_val.get("dti", 0.0)
    emerg_months = rb02_val.get("emergency_months", 0.0)
    savings_rate = rb02_val.get("savings_rate", 0.0)
    min_emerg_target = rb02_val.get("min_recommended_emergency_months", 3.0)

    rb05 = rules.get("RB05", {})
    rb05_val = rb05.get("value", {})
    max_add_payment = rb05_val.get("max_allowed_monthly_payment", 0.0)
    collateral_quality = rb05_val.get("collateral_quality", "NONE")

    rep_name = household.get("representative_name") or "bạn"
    question = req.question.strip()
    q_lower = question.lower()

    # Mô tả nợ, tiết kiệm & tài sản thế chấp (Sử dụng nhãn tiếng Việt chuẩn từ schema.py)
    debt_desc = f"{total_debt:,.0f} VNĐ (Trả gốc lãi {debt_payment:,.0f} VNĐ/tháng)" if total_debt > 0 or debt_payment > 0 else "Không có nợ"
    savings_desc = f"{savings:,.0f} VNĐ" if savings > 0 else "0 VNĐ (Chưa có tích lũy)"

    asset_label_list = []
    for a in assets:
        if a in ["house", "land", "real_estate"]:
            asset_label_list.append("Bất động sản")
        elif a in ["car", "vehicle"]:
            asset_label_list.append("Phương tiện (Xe)")
        elif a == "cash":
            asset_label_list.append("Tiền mặt/Tiền gửi")
        elif a == "gold":
            asset_label_list.append("Vàng")
        elif a == "insurance":
            asset_label_list.append("Bảo hiểm")
        elif a == "investment":
            asset_label_list.append("Đầu tư")
        else:
            asset_label_list.append(str(a).upper())

    asset_desc = ", ".join(list(dict.fromkeys(asset_label_list))) if asset_label_list else "Chưa có tài sản"

    # Trích xuất giá trị tài sản và kỳ hạn vay từ câu hỏi người dùng nếu có
    parsed_price = parse_amount_from_text(question)
    parsed_term = parse_term_months_from_text(question)
    term_months = parsed_term or int(household.get("loan_term_months") or 240)
    term_years_str = f"{term_months // 12} năm" if term_months % 12 == 0 else f"{term_months} tháng"

    # Xử lý phản hồi chuyên biệt theo từng Intent
    if "mua" in q_lower or "vay" in q_lower or parsed_price is not None:
        asset_price = parsed_price or float(household.get("asset_price") or 0.0)
        loan_eval = evaluate_loan_capacity(household, asset_price=asset_price, term_months=term_months)
        loan_val = loan_eval.get("value", {})
        max_loan = loan_val.get("max_allowed_loan", 0.0)
        max_pmt = loan_val.get("max_allowed_monthly_payment", 0.0)
        max_ltv_loan = loan_val.get("max_loan_by_ltv", 0.0)

        debt_note = f" (Đã trừ đi khoản nợ đang trả {debt_payment:,.0f} VNĐ/tháng)" if debt_payment > 0 else ""

        if asset_price > 0:
            down_payment = max(0.0, asset_price - max_loan)
            advice_detail = (
                f"🏡 **Tư vấn vay mua tài sản ({asset_price:,.0f} VNĐ, kỳ hạn {term_years_str})**:\n"
                f"- Hạn mức vay an toàn tối đa dựa trên thu nhập (DTI ≤ 40%, {term_years_str}): **{max_loan:,.0f} VNĐ**{debt_note}.\n"
                f"- Số tiền trả gốc lãi tối đa có thể gánh thêm: **~{max_pmt:,.0f} VNĐ/tháng**.\n"
                f"- Hạn mức vay tối đa theo tài sản thế chấp (LTV 70%): **{max_ltv_loan:,.0f} VNĐ**.\n"
                f"- Năng lực thế chấp tài sản sở hữu hiện tại ({asset_desc}): **Mức {collateral_quality}**.\n"
                f"- 💡 **Khuyên dùng**: Bạn có thể vay an toàn tối đa **{max_loan:,.0f} VNĐ** trong thời hạn **{term_years_str}**. "
                f"Do đó bạn cần chuẩn bị sẵn vốn tự có (tiền trả trước) tối thiểu **{down_payment:,.0f} VNĐ** ({down_payment/asset_price:.1%}) trước khi quyết định mua."
            )
        else:
            advice_detail = (
                f"🏦 **Tư vấn hạn mức vay an toàn (kỳ hạn {term_years_str})**:\n"
                f"- Hạn mức vay an toàn tối đa ({term_years_str}): **{max_loan:,.0f} VNĐ**{debt_note}.\n"
                f"- Khả năng trả gốc lãi vay mới tối đa: **{max_pmt:,.0f} VNĐ/tháng**.\n"
                f"- Năng lực thế chấp từ tài sản sở hữu ({asset_desc}): **Mức {collateral_quality}**."
            )

    elif "tiết kiệm" in q_lower or "tích lũy" in q_lower:
        buf_min = expense * min_emerg_target
        months_min = (buf_min / net_cashflow) if net_cashflow > 0 else 0
        elderly_note = " (Đã nâng lên 6 tháng do có phụng dưỡng người già)" if has_dependents else ""

        advice_detail = (
            f"🐖 **Gói tư vấn tiết kiệm & Quỹ dự phòng**:\n"
            f"- Số tiền tiết kiệm hiện tại: **{savings_desc}**.\n"
            f"- Số dư thặng dư khả dụng hàng tháng: **{net_cashflow:,.0f} VNĐ/tháng** (Đạt tỷ lệ tiết kiệm {savings_rate:.1%}).\n"
            f"- Mục tiêu quỹ dự phòng an toàn khuyến nghị ({min_emerg_target:.0f} tháng chi tiêu = {buf_min:,.0f} VNĐ){elderly_note}: Cần tích lũy khoảng **{months_min:.1f} tháng**."
        )

    elif "đầu tư" in q_lower:
        advice_detail = (
            f"📈 **Gói tư vấn phân bổ đầu tư**:\n"
            f"- Với thặng dư dòng tiền **{net_cashflow:,.0f} VNĐ/tháng** (Tỷ lệ tiết kiệm {savings_rate:.1%}):\n"
            f"- Trích **30% ({net_cashflow*0.3:,.0f} VNĐ)** cho tiền gửi tiết kiệm thanh khoản cao dự phòng.\n"
            f"- Trích **70% ({net_cashflow*0.7:,.0f} VNĐ)** đầu tư vào tài sản sinh lời an toàn (Trái phiếu / Chứng chỉ quỹ)."
        )

    elif "50/30/20" in q_lower or "503020" in q_lower:
        needs_target = income * 0.50
        wants_target = income * 0.30
        savings_target = income * 0.20

        advice_detail = (
            f"📊 **Phân tích theo Quy tắc 50/30/20 (Thu nhập {income:,.0f} VNĐ)**:\n"
            f"- **50% Nhu cầu thiết yếu** (Tối đa {needs_target:,.0f} VNĐ): Chi tiêu sinh hoạt hiện tại {expense:,.0f} VNĐ.\n"
            f"- **30% Cá nhân & Giải trí** (Tối đa {wants_target:,.0f} VNĐ).\n"
            f"- **20% Tiết kiệm & Trả nợ** (Tối thiểu {savings_target:,.0f} VNĐ): Thặng dư hiện tại đạt **{net_cashflow:,.0f} VNĐ** ({savings_rate:.1%})."
        )
    else:
        elderly_text = " Ghi nhận gia đình có phụng dưỡng người già ➔ Nâng ngưỡng quỹ dự phòng y tế lên 6 tháng chi tiêu sinh hoạt." if has_dependents else ""
        advice_detail = (
            f"Dựa trên phân tích dòng tiền và đòn bẩy nợ hiện tại, bạn có số dư thặng dư khả dụng hàng tháng là **{net_cashflow:,.0f} VNĐ** (Đạt tỷ lệ tiết kiệm {savings_rate:.1%}). "
            f"Hạn mức trả nợ mới đề xuất thêm tối đa là **{max_add_payment:,.0f} VNĐ/tháng**.{elderly_text}"
        )

    answer = (
        f"Chào {rep_name}, hệ thống AI tư vấn tài chính đã phân tích hồ sơ của bạn.\n\n"
        f"📌 **Đánh giá tổng quan ({overall_status})**:\n"
        f"- Dòng tiền hàng tháng (RB01): Dư thừa khoảng {net_cashflow:,.0f} VNĐ ({rb01.get('status')}).\n"
        f"- Nợ, Tiết kiệm & Tài sản: Nợ {debt_desc} | Tiết kiệm {savings_desc} | Tài sản: {asset_desc}.\n"
        f"- Sức khỏe tài chính (RB02): {rb02.get('status')} (Tỷ lệ DTI trả nợ: {dti:.1%}, Đệm khẩn cấp: {emerg_months:.1f}/{min_emerg_target:.0f} tháng, Tỷ lệ tiết kiệm: {savings_rate:.1%}).\n"
        f"- Khả năng vay vốn (RB05): Trả nợ đề xuất thêm tối đa {max_add_payment:,.0f} VNĐ/tháng ({rb05.get('status')}).\n\n"
        f"💡 **Trả lời cho câu hỏi: '{question}'**:\n"
        f"{advice_detail}"
    )

    suggested = [
        "Tôi muốn mua nhà giá 3 tỷ thì vay được bao nhiêu?",
        "Tư vấn gói đầu tư tích lũy an toàn",
        "Lập kế hoạch xây dựng quỹ dự phòng 6 tháng",
    ]

    return AdviseResponse(
        response_text=answer,
        model_used="HFML-RuleEngine-v0.2.0",
        suggested_questions=suggested,
        tokens_used=150,
    )
