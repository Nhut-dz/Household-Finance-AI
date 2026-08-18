"""Tầng api — FastAPI endpoints.

    GET  /health    kiểm tra service sống, kèm trạng thái model ML01
    POST /advise    tư vấn — định tuyến theo `intent_code` (F02 + F03 + F04)
    POST /predict   phân loại 4 nhóm khuyến nghị bằng model ML01 (F03)

Định tuyến của `/advise` (PLAN.md §8.2 task 12, chốt lại 15/08/2026)
--------------------------------------------------------------------
Bốn chip gợi ý của màn Chatbot gửi kèm `intent_code`; câu người dùng tự gõ
thì không có, và chỉ khi đó mới đoán bằng từ khoá.

    SAVINGS_PACKAGE              → tầng rule (giữ nguyên logic cũ)
    FINANCIAL_HEALTH_DIAGNOSIS   → ML01 → tầng diễn đạt
    LOAN_RISK_DIAGNOSIS          → ML02 → tầng diễn đạt
    BUDGET_50_30_20              → tầng rule (giữ nguyên logic cũ)

Hai intent ML **không** đoán được bằng từ khoá, và đó là ràng buộc có chủ ý
chứ không phải giới hạn kỹ thuật — xem `hfml.api.intents`.
"""
import re
from typing import Any, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from hfml.api.intents import INTENT_LABELS, IntentCode, resolve_intent
from hfml.api.schemas import (
    Ml01ModelConfidence,
    Ml01PredictRequest,
    Ml01PredictResponse,
    build_probabilities,
)
from hfml.config import CONFIG
from hfml.inference import engine as inference_engine
from hfml.inference.lifecycle import MANAGER, ModelUnavailable
from hfml.inference.payloads import normalize_payload
from hfml.inference.settings import ML01, ML02, SETTINGS
from hfml.llm.narrator import explain_ml01, explain_ml02
from hfml.logger import get_logger
from hfml.ml.ml01_recommendation.labeler import LABELS_VI, RAW_FEATURES, RecommendationGroup
from hfml.rules.engine import RuleEngine
from hfml.rules.rb05_loan_capacity import evaluate_loan_capacity

log = get_logger(__name__)

app = FastAPI(title="Household Finance ML API", version="0.3.0")
rule_engine = RuleEngine()

# Slug artifact và ngưỡng KHÔNG còn khai bằng hằng số ở đây.
#
# Chúng thuộc về cấu hình (Epic AI-03 task 5): `config/config.yaml` → khối
# `inference`, phủ được bằng env `HFML_ML01_SLUG` / `HFML_ML02_SLUG`.
#
# Bản hằng số cũ ở file này đã trôi khỏi bản ở `pipeline/predictor.py` —
# `ML02_SLUG` từng trỏ vào `"ml02_best_reduced_vfinal"`, một artifact KHÔNG hề
# tồn tại trên đĩa. Không ai phát hiện vì nhánh dùng tới nó luôn trả lời "model
# đang huấn luyện" trước khi kịp chạm vào model. Đó là lý do cấu hình phải nằm
# ngoài mã nguồn: hai bản sao trôi khỏi nhau trong im lặng.


def get_ml01_model():
    """Model ML01 đang phục vụ.

    Uỷ quyền cho `hfml.inference.lifecycle` — nơi duy nhất chịu trách nhiệm
    nạp, giữ trong bộ nhớ và đổi model. Giữ lại tên hàm này vì nó là điểm thay
    thế của bộ test định tuyến.

    Ném `FileNotFoundError` khi thiếu artifact để giữ nguyên hợp đồng cũ với
    các nhánh bắt lỗi bên dưới.
    """
    try:
        return MANAGER.get(ML01).model
    except ModelUnavailable as exc:
        raise FileNotFoundError(str(exc)) from exc


class AdviseRequest(BaseModel):
    question: str
    household: dict[str, Any]

    #: Mã ý định do chip gợi ý gửi kèm (Hướng 1 của PLAN.md §8.2 task 12).
    #: Có mã thì tin tuyệt đối; không có thì đoán bằng từ khoá. Hai intent
    #: chạy model CHỈ kích hoạt được qua đường này — xem `hfml.api.intents`.
    intent_code: Optional[str] = None

    #: 17 feature của ML01, do backend quy đổi bằng cùng một hàm đang dùng
    #: cho `/predict`. `None` nghĩa là không dựng được (thường vì hồ sơ thiếu
    #: năm sinh); khi đó nhánh ML01 nói ra lý do chứ không đoán bừa.
    ml_features: Optional[dict[str, Any]] = None

    #: Dữ liệu màn "Thông tin khoản vay". `None` = hộ chưa khai, và nhánh
    #: ML02 phải hướng người dùng đi khai chứ không chạy model trên số rỗng.
    loan_application: Optional[dict[str, Any]] = None


class AdviseResponse(BaseModel):
    response_text: str
    model_used: str = "HFML-RuleEngine-Advisor"
    suggested_questions: Optional[List[str]] = None
    tokens_used: Optional[int] = None

    #: Ý định đã được chốt, trả về để backend/FE log và kiểm chứng được rằng
    #: chip đã đi đúng nhánh. Không có trường này thì "chip có vào đúng engine
    #: không" là câu hỏi phải đọc log server mới trả lời được.
    intent_code: Optional[str] = None

    #: `True` khi câu trả lời cần người dùng làm thêm một việc trước — hiện
    #: chỉ dùng cho ML02 khi hộ chưa khai thông tin khoản vay. FE dựa vào đây
    #: để hiện nút điều hướng thay vì chỉ in ra một đoạn chữ.
    requires_loan_application: bool = False


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
    report = inference_engine.health()
    ml01 = dict(report["models"].get(ML01, {}))
    if ml01.get("loaded"):
        try:
            ml01["classes"] = list(MANAGER.get(ML01).model.classes_)
        except ModelUnavailable:
            pass

    return {
        "status": "ok",
        "service": "Household Finance ML Service",
        "ml01": ml01,
        # Từ Epic AI-03: nói ra trạng thái của CẢ module, không chỉ ML01.
        "inference": report,
    }


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
            detail=f"Chưa có artifact ML01 ({SETTINGS.ml01_slug}). Chạy train + export trước. {exc}",
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
        # Output nghiệp vụ: ĐÚNG MỘT nhóm định hướng.
        prediction=label,
        prediction_vi=LABELS_VI[RecommendationGroup(label)],
        model_confidence=Ml01ModelConfidence(
            confidence=confidence,
            # Dưới ngưỡng thì FE phải nói ra là kết quả chưa chắc chắn, chứ
            # không hiển thị như một kết luận (PLAN.md §8.1).
            low_confidence=confidence < CONFIG.confidence_threshold,
            probabilities=probabilities,
        ),
        model_version=SETTINGS.ml01_slug,
    )


def _ml01_from_features(features: dict[str, Any]) -> dict[str, Any]:
    """Chạy ML01 trên bộ feature đã chuẩn hoá và trả về kết quả dạng dict.

    Dùng chung đúng đường tính với `/predict` — cùng model, cùng thứ tự cột,
    cùng ngưỡng tin cậy. Hai đường tính riêng cho cùng một câu hỏi là chỗ để
    màn "Chẩn đoán" và thẻ dự đoán ở màn Phương án nói hai kết quả khác nhau
    về cùng một hộ.
    """
    model = get_ml01_model()
    frame = pd.DataFrame([[features.get(name) for name in RAW_FEATURES]],
                         columns=list(RAW_FEATURES))

    label = str(model.predict(frame)[0])
    proba = model.predict_proba(frame)[0]
    probabilities = build_probabilities(list(model.classes_), proba)
    confidence = next(p.probability for p in probabilities if p.label == label)

    return {
        "label": label,
        "label_vi": LABELS_VI[RecommendationGroup(label)],
        "confidence": confidence,
        "low_confidence": confidence < CONFIG.confidence_threshold,
        "probabilities": [p.model_dump() for p in probabilities],
    }


def _advise_financial_health(req: AdviseRequest, rule_summary: str) -> AdviseResponse:
    """Nhánh `FINANCIAL_HEALTH_DIAGNOSIS` — dữ liệu hộ → ML01 → LLM giải thích.

    Thứ tự này là ràng buộc kiến trúc, không phải chi tiết cài đặt: **model
    quyết nhãn trước, tầng diễn đạt chỉ nhận nhãn đã quyết**. `explain_ml01()`
    nhận `label` như tham số bắt buộc và không có nhánh nào đổi nó.
    """
    if not req.ml_features:
        return AdviseResponse(
            response_text=(
                "🧭 **Chẩn đoán sức khỏe tài chính**\n\n"
                "Hồ sơ của bạn còn thiếu dữ liệu bắt buộc để chạy mô hình — "
                "thường là **năm sinh**. Vui lòng bổ sung ở màn *Nhập thông tin* "
                "rồi thử lại.\n\n"
                "Hệ thống cố ý không điền tuổi mặc định: điền bừa thì mô hình vẫn "
                "trả về một nhóm trông hợp lý, và không ai biết nó dựa trên dữ "
                "liệu bịa."
            ),
            model_used="HFML-ML01-Advisor",
            intent_code=IntentCode.FINANCIAL_HEALTH_DIAGNOSIS.value,
        )

    try:
        result = _ml01_from_features(req.ml_features)
    except FileNotFoundError as exc:
        log.warning("Chưa có artifact ML01 (%s): %s", SETTINGS.ml01_slug, exc)
        return AdviseResponse(
            response_text=(
                "🧭 **Chẩn đoán sức khỏe tài chính**\n\n"
                "Mô hình phân loại chưa sẵn sàng trên máy chủ nên chưa chẩn đoán "
                "được. Dưới đây là đánh giá theo bộ quy tắc tài chính:\n\n"
                f"{rule_summary}"
            ),
            model_used="HFML-RuleEngine-Fallback",
            intent_code=IntentCode.FINANCIAL_HEALTH_DIAGNOSIS.value,
        )
    except Exception as exc:  # noqa: BLE001 — biên ngoài, không để traceback lọt ra
        log.exception("ML01 lỗi khi chẩn đoán qua chat")
        raise HTTPException(status_code=500, detail=f"Model lỗi khi dự đoán: {exc}") from exc

    return AdviseResponse(
        response_text=explain_ml01(
            label=result["label"],
            label_vi=result["label_vi"],
            confidence=result["confidence"],
            probabilities=result["probabilities"],
            low_confidence=result["low_confidence"],
            rule_summary=rule_summary,
            model_version=SETTINGS.ml01_slug,
        ),
        model_used=f"HFML-ML01/{SETTINGS.ml01_slug}",
        intent_code=IntentCode.FINANCIAL_HEALTH_DIAGNOSIS.value,
        suggested_questions=[
            "Tôi nên bắt đầu từ khoản chi nào?",
            "Quỹ dự phòng của tôi cần bao nhiêu tháng chi tiêu?",
            "Lập kế hoạch tích lũy trong 12 tháng tới",
        ],
    )


def _advise_loan_risk(req: AdviseRequest) -> AdviseResponse:
    """Nhánh `LOAN_RISK_DIAGNOSIS` — thông tin khoản vay → ML02 → LLM giải thích.

    Hai cửa chặn trước khi tới model, theo đúng thứ tự:

        1. Chưa khai thông tin khoản vay → hướng người dùng sang màn nhập.
           ML02 cần 16 trường của màn đó; chạy trên số rỗng thì vẫn ra một xác
           suất, và đó là con số vô nghĩa mà không có gì báo hiệu.
        2. Chưa có artifact ML02 → nói thẳng là đang hoàn thiện.

    Cửa 2 hiện LUÔN đóng: F04 mới xong task 1/15. Viết sẵn đường đi để khi
    task 15 export artifact thì nhánh này tự sống, FE và backend không phải
    sửa lại lần nữa.
    """
    if not req.loan_application:
        return AdviseResponse(
            response_text=(
                "⚖️ **Chẩn đoán rủi ro vay vốn**\n\n"
                "Bạn chưa khai thông tin khoản vay nên chưa đánh giá được. Phần "
                "này cần các dữ liệu mà màn *Thông tin khoản vay* thu thập: "
                "số tiền vay, thời hạn, khoản trả hàng tháng, giá trị tài sản và "
                "lịch sử tín dụng.\n\n"
                "Vui lòng điền màn **Thông tin khoản vay** rồi quay lại đây."
            ),
            model_used="HFML-ML02-Advisor",
            intent_code=IntentCode.LOAN_RISK_DIAGNOSIS.value,
            requires_loan_application=True,
        )

    # Chạy Rule + ML02 qua module inference, KHÔNG gọi LLM.
    #
    # Dùng `analyze()` chứ không phải `chat()`. Đã sập đúng vì chuyện này khi
    # test trên UI: `chat()` gọi Gemini, và khi hết quota thì client tự retry
    # với backoff nên vượt hẳn 30 giây mà Laravel chờ
    # (`PYTHON_ADVISOR_TIMEOUT`). Người dùng chỉ thấy "Không kết nối được
    # service tư vấn AI", còn log Python thì dừng lặng lẽ sau bước dựng context.
    #
    # `/advise` là bề mặt ĐỒNG BỘ với Laravel, và nhánh ML01 bên trên cũng chỉ
    # dùng template (`explain_ml01`). Nhánh này phải đối xứng: `analyze()` chạy
    # ~150ms, không tốn quota, không phụ thuộc mạng ngoài. Đường có LLM là
    # `/api/v1/chat` — nơi timeout khai riêng và client biết mình đang chờ.
    result = inference_engine.analyze(
        normalize_payload(req.household, req.loan_application))
    analysis = result.to_dict().get("analysis") or {}
    ml02 = analysis.get("ml02") or {}

    if not ml02.get("available"):
        reason = ml02.get("error") or "Chưa đủ dữ liệu để đánh giá."
        return AdviseResponse(
            response_text=(
                "⚖️ **Chẩn đoán rủi ro vay vốn**\n\n"
                f"{reason}\n\nVui lòng kiểm tra lại màn **Thông tin khoản vay**."),
            model_used="HFML-ML02-Advisor",
            intent_code=IntentCode.LOAN_RISK_DIAGNOSIS.value,
            requires_loan_application=(
                ml02.get("reason_code") == "missing_input"),
        )

    threshold = MANAGER.threshold_for(ML02)
    rules = analysis.get("rules") or {}
    loan_summary = "\n".join(
        f"- {code}: {rules[code]['details']['summary_vi']}"
        for code in ("RB01", "RB02", "RB05")
        if rules.get(code, {}).get("details", {}).get("summary_vi"))

    return AdviseResponse(
        response_text=explain_ml02(
            label=str(ml02["label"]),
            label_vi=str(ml02["label_vi"]),
            probability=float(ml02["probability"]),
            threshold=float(threshold if threshold is not None else 0.0),
            loan_summary=loan_summary,
            model_version=str(ml02.get("model_version") or SETTINGS.ml02_slug),
        ),
        model_used=f"HFML-ML02/{SETTINGS.ml02_slug}",
        intent_code=IntentCode.LOAN_RISK_DIAGNOSIS.value,
        suggested_questions=[
            "Tôi nên vay tối đa bao nhiêu thì an toàn?",
            "Làm sao giảm tỉ lệ nợ trên thu nhập?",
            "Kéo dài thời hạn vay thì rủi ro thay đổi thế nào?",
        ],
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

    # ----------------------------------------------------------------- intent
    # Chip gợi ý gửi kèm `intent_code` → tin tuyệt đối. Câu tự gõ → đoán bằng
    # từ khoá. Hai intent chạy model chỉ vào được bằng đường thứ nhất.
    intent = resolve_intent(question, req.intent_code)

    # Câu có số tiền mà không nhận ra ý định gì thì gần như chắc chắn là hỏi
    # về khoản vay ("nhà 3 tỷ thì thế nào?").
    #
    # Chỉ nâng cấp từ GENERAL, KHÔNG ghi đè intent đã nhận ra. Bản cũ đặt điều
    # kiện `parsed_price is not None` ngay ở nhánh đầu nên "tiết kiệm 500
    # triệu" bị kéo sang nhánh vay — và giờ thì nó còn có thể cướp cả intent
    # của chip gợi ý.
    if intent is IntentCode.GENERAL and parsed_price is not None:
        intent = IntentCode.LOAN_CAPACITY

    log.info("advise: intent=%s (%s)", intent.value,
             "chip" if req.intent_code else "từ khoá")

    # Tóm tắt tầng rule, dùng lại cho cả câu trả lời thường lẫn phần diễn giải
    # của ML01 — hai nơi nói khác nhau về cùng một hồ sơ là chuyện phải tránh.
    rule_summary = (
        f"📌 **Đánh giá tổng quan ({overall_status})**:\n"
        f"- Dòng tiền hàng tháng (RB01): Dư thừa khoảng {net_cashflow:,.0f} VNĐ ({rb01.get('status')}).\n"
        f"- Nợ, Tiết kiệm & Tài sản: Nợ {debt_desc} | Tiết kiệm {savings_desc} | Tài sản: {asset_desc}.\n"
        f"- Sức khỏe tài chính (RB02): {rb02.get('status')} (Tỷ lệ DTI trả nợ: {dti:.1%}, Đệm khẩn cấp: {emerg_months:.1f}/{min_emerg_target:.0f} tháng, Tỷ lệ tiết kiệm: {savings_rate:.1%}).\n"
        f"- Khả năng vay vốn (RB05): Trả nợ đề xuất thêm tối đa {max_add_payment:,.0f} VNĐ/tháng ({rb05.get('status')})."
    )

    # Hai nhánh ML trả về câu trả lời HOÀN CHỈNH và thoát sớm: chúng có cấu
    # trúc riêng do tầng diễn đạt dựng, không phải một đoạn `advice_detail`
    # ghép vào khung trả lời chung.
    if intent is IntentCode.FINANCIAL_HEALTH_DIAGNOSIS:
        return _advise_financial_health(req, rule_summary)

    if intent is IntentCode.LOAN_RISK_DIAGNOSIS:
        return _advise_loan_risk(req)

    # ------------------------------------------------- nhánh rule, giữ nguyên
    if intent is IntentCode.LOAN_CAPACITY:
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

    elif intent is IntentCode.SAVINGS_PACKAGE:
        buf_min = expense * min_emerg_target
        months_min = (buf_min / net_cashflow) if net_cashflow > 0 else 0
        elderly_note = " (Đã nâng lên 6 tháng do có phụng dưỡng người già)" if has_dependents else ""

        advice_detail = (
            f"🐖 **Gói tư vấn tiết kiệm & Quỹ dự phòng**:\n"
            f"- Số tiền tiết kiệm hiện tại: **{savings_desc}**.\n"
            f"- Số dư thặng dư khả dụng hàng tháng: **{net_cashflow:,.0f} VNĐ/tháng** (Đạt tỷ lệ tiết kiệm {savings_rate:.1%}).\n"
            f"- Mục tiêu quỹ dự phòng an toàn khuyến nghị ({min_emerg_target:.0f} tháng chi tiêu = {buf_min:,.0f} VNĐ){elderly_note}: Cần tích lũy khoảng **{months_min:.1f} tháng**."
        )

    # Chip "Gói đầu tư" đã rút khỏi nhóm gợi ý (15/08/2026), nhưng nhánh này
    # giữ nguyên: người dùng vẫn gõ "tư vấn đầu tư" được, và bỏ nhánh đi thì
    # câu đó rơi xuống trả lời chung.
    elif intent is IntentCode.INVESTMENT:
        advice_detail = (
            f"📈 **Gói tư vấn phân bổ đầu tư**:\n"
            f"- Với thặng dư dòng tiền **{net_cashflow:,.0f} VNĐ/tháng** (Tỷ lệ tiết kiệm {savings_rate:.1%}):\n"
            f"- Trích **30% ({net_cashflow*0.3:,.0f} VNĐ)** cho tiền gửi tiết kiệm thanh khoản cao dự phòng.\n"
            f"- Trích **70% ({net_cashflow*0.7:,.0f} VNĐ)** đầu tư vào tài sản sinh lời an toàn (Trái phiếu / Chứng chỉ quỹ)."
        )

    elif intent is IntentCode.BUDGET_50_30_20:
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

    # Người dùng bấm chip thì "câu hỏi" chỉ là nhãn của chip, nhắc lại nguyên
    # văn nghe như máy đọc lại chính nút vừa bấm.
    heading = (f"💡 **{INTENT_LABELS[intent]}**:" if req.intent_code
               else f"💡 **Trả lời cho câu hỏi: '{question}'**:")

    answer = (
        f"Chào {rep_name}, hệ thống AI tư vấn tài chính đã phân tích hồ sơ của bạn.\n\n"
        f"{rule_summary}\n\n"
        f"{heading}\n"
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
        intent_code=intent.value,
    )


# ==========================================================================
# Epic AI-03 — phơi module inference nguyên vẹn
# ==========================================================================
# Hai endpoint dưới đây KHÔNG chứa nghiệp vụ: chúng nhận JSON, gọi đúng một
# hàm của `hfml.inference`, rồi trả kết quả ra. Toàn bộ điều phối, xử lý lỗi
# và cấu hình nằm trong module — nơi test được mà không cần dựng HTTP client.
#
# Đây là điểm khác biệt so với `/advise` phía trên: `/advise` là bề mặt hợp
# đồng cũ với Laravel và vẫn còn tự dựng câu trả lời bằng chuỗi, còn hai
# endpoint này là đường đi mới, mỏng đúng như một tầng vận chuyển nên mỏng.


class InferenceRequest(BaseModel):
    """Hồ sơ hộ gia đình, đúng dạng `normalize_input` nhận."""

    household: dict[str, Any]


class InferenceChatRequest(InferenceRequest):
    question: str
    intent_code: Optional[str] = None
    history: Optional[List[dict]] = None
    previous_intent: Optional[str] = None


@app.post("/inference/analyze")
def inference_analyze(req: InferenceRequest) -> dict:
    """Input → Validation → Preprocessing → Rule → ML01 → ML02 → Aggregation.

    Không gọi LLM, nên rẻ và nhanh. Dùng cho màn hình chỉ cần chỉ số.
    """
    return inference_engine.analyze(req.household).to_dict()


@app.post("/inference/chat")
def inference_chat(req: InferenceChatRequest) -> dict:
    """Trọn sơ đồ: Input → … → LLM → Output Validation → Response."""
    return inference_engine.chat(
        payload=req.household,
        question=req.question,
        intent_code=req.intent_code,
        history=req.history,
        previous_intent=req.previous_intent,
    ).to_dict()


@app.post("/inference/reload")
def inference_reload(name: Optional[str] = None) -> dict:
    """Bỏ model khỏi bộ nhớ để lượt gọi sau nạp lại từ đĩa.

    Dùng sau khi train xong bản mới và ghi đè lên cùng slug — không phải khởi
    động lại service chỉ để đổi một artifact.
    """
    MANAGER.reload(name)
    return inference_engine.health()
