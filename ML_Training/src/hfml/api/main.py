"""Tầng api — FastAPI endpoints.

GET  /health
POST /advise
POST /predict
"""

import re
from typing import Any, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from hfml.api.intents import INTENT_LABELS, IntentCode, resolve_intent
from hfml.api.schemas import (
    ML02_ORDERED_LABELS,
    Ml01ModelConfidence,
    Ml01PredictRequest,
    Ml01PredictResponse,
    Ml01Probability,
    Ml02ModelConfidence,
    Ml02PredictRequest,
    Ml02PredictResponse,
    build_probabilities,
)
from hfml.config import CONFIG
from hfml.inference import engine as inference_engine
from hfml.inference.lifecycle import MANAGER, ModelUnavailable
from hfml.inference.payloads import normalize_payload
from hfml.inference.settings import ML01, ML02, SETTINGS
from hfml.llm import presentation
from hfml.llm.narrator import explain_ml01, explain_ml02, ml01_title, ml02_title
from hfml.logger import get_logger
from hfml.ml.ml01_recommendation.labeler import (
    LABELS_VI,
    RAW_FEATURES,
    RecommendationGroup,
)
from hfml.rules.engine import RuleEngine
from hfml.rules.rb05_loan_capacity import evaluate_loan_capacity


log = get_logger(__name__)

app = FastAPI(
    title="Household Finance ML API",
    version="0.3.0",
)

rule_engine = RuleEngine()

ML01_SLUG = "ml01_xgboost_v1"


# =========================================================
# ML01 MODEL
# =========================================================

def get_ml01_model():
    try:
        return MANAGER.get(ML01).model

    except ModelUnavailable as exc:
        raise FileNotFoundError(str(exc)) from exc


# =========================================================
# REQUEST / RESPONSE
# =========================================================

class AdviseRequest(BaseModel):
    question: str
    household: dict[str, Any]

    intent_code: Optional[str] = None
    ml_features: Optional[dict[str, Any]] = None
    loan_application: Optional[dict[str, Any]] = None


class AdviseResponse(BaseModel):
    response_text: str

    model_used: str = "HFML-RuleEngine-Advisor"

    suggested_questions: Optional[List[str]] = None
    tokens_used: Optional[int] = None
    intent_code: Optional[str] = None

    requires_loan_application: bool = False


# =========================================================
# TEXT PARSER
# =========================================================

def parse_amount_from_text(text: str) -> float | None:

    match_ty = re.search(
        r"(\d+(?:[\.,]\d+)?)\s*(tỷ|ty|tỉ|ti)",
        text,
        re.IGNORECASE,
    )

    if match_ty:
        val = float(
            match_ty.group(1).replace(",", ".")
        )

        return val * 1_000_000_000


    match_trieu = re.search(
        r"(\d+(?:[\.,]\d+)?)\s*(triệu|trieu|tr)",
        text,
        re.IGNORECASE,
    )

    if match_trieu:
        val = float(
            match_trieu.group(1).replace(",", ".")
        )

        return val * 1_000_000


    match_digits = re.search(
        r"\b(\d{7,12})\b",
        text,
    )

    if match_digits:
        return float(
            match_digits.group(1)
        )

    return None


def parse_term_months_from_text(
    text: str,
) -> int | None:

    match_nam = re.search(
        r"(\d+)\s*(năm|nam|n)\b",
        text,
        re.IGNORECASE,
    )

    if match_nam:

        years = int(
            match_nam.group(1)
        )

        if 1 <= years <= 30:
            return years * 12


    match_thang = re.search(
        r"(\d+)\s*(tháng|thang|th)\b",
        text,
        re.IGNORECASE,
    )

    if match_thang:

        months = int(
            match_thang.group(1)
        )

        if 1 <= months <= 360:
            return months

    return None


# =========================================================
# HEALTH
# =========================================================

@app.get("/health")
def health() -> dict:

    report = inference_engine.health()

    ml01 = dict(
        report["models"].get(
            ML01,
            {},
        )
    )

    if ml01.get("loaded"):

        try:

            ml01["classes"] = list(
                MANAGER.get(
                    ML01
                ).model.classes_
            )

        except ModelUnavailable:
            pass


    return {

        "status": "ok",

        "service":
            "Household Finance ML Service",

        "ml01":
            ml01,

        "inference":
            report,
    }


# =========================================================
# ML01 PREDICT
# =========================================================

@app.post(
    "/predict",
    response_model=Ml01PredictResponse,
)
def predict(
    req: Ml01PredictRequest,
) -> Ml01PredictResponse:

    try:

        model = get_ml01_model()

    except FileNotFoundError as exc:

        raise HTTPException(

            status_code=503,

            detail=(
                f"Chưa có artifact ML01 "
                f"({SETTINGS.ml01_slug}). "
                f"Chạy train + export trước. "
                f"{exc}"
            ),

        ) from exc


    payload = req.model_dump()


    frame = pd.DataFrame(

        [[
            payload[name]
            for name
            in RAW_FEATURES
        ]],

        columns=list(
            RAW_FEATURES
        ),
    )


    try:

        label = str(
            model.predict(
                frame
            )[0]
        )


        proba = (
            model.predict_proba(
                frame
            )[0]
        )


    except Exception as exc:

        log.exception(
            "ML01 dự đoán lỗi"
        )

        raise HTTPException(

            status_code=500,

            detail=(
                f"Model lỗi khi dự đoán: "
                f"{exc}"
            ),

        ) from exc


    probabilities = (
        build_probabilities(
            list(model.classes_),
            proba,
        )
    )


    confidence = next(

        p.probability

        for p in probabilities

        if p.label == label
    )


    return Ml01PredictResponse(

        prediction=label,

        prediction_vi=(
            LABELS_VI[
                RecommendationGroup(
                    label
                )
            ]
        ),

        model_confidence=(
            Ml01ModelConfidence(

                confidence=confidence,

                low_confidence=(
                    confidence
                    < CONFIG.confidence_threshold
                ),

                probabilities=probabilities,
            )
        ),

        model_version=(
            SETTINGS.ml01_slug
        ),
    )


# =========================================================
# LLM NARRATION
# =========================================================

_NARRATED_SOURCES = frozenset({
    "llm",
    "llm_retry",
    "out_of_scope",
})
@app.post("/predict-loan-risk", response_model=Ml02PredictResponse)
def predict_loan_risk(req: Ml02PredictRequest) -> Ml02PredictResponse:
    """ML02 — rủi ro của khoản vay đang xét (F04), cho thẻ trên màn Chẩn đoán hồ sơ.

    Đi cùng đường với chip "Chẩn đoán rủi ro vay vốn": `analyze()` rồi lấy
    `analysis.ml02`. Không tự dựng frame ở đây — làm vậy là có hai đường quy
    đổi cho cùng một model, và thẻ với chatbot có thể nói hai xác suất khác
    nhau về cùng một khoản vay. Không gọi LLM, nên rẻ và không tốn quota.

    Mã trạng thái nói đúng loại lỗi, như `/predict`:
        422  thiếu hoặc sai dữ liệu khoản vay — việc của người dùng, FE hướng
             sang màn nhập
        503  chưa có artifact ML02 — lỗi triển khai
        500  model chạy lỗi
    """
    if not req.loan_application:
        raise HTTPException(
            status_code=422,
            detail="Chưa có thông tin khoản vay — cần màn 'Thông tin khoản vay'.")

    result = inference_engine.analyze(
        normalize_payload(req.household, req.loan_application)).to_dict()
    ml02 = (result.get("analysis") or {}).get("ml02") or {}

    if not ml02.get("available"):
        errors = "; ".join(e.get("message", "") for e in result.get("errors") or [])
        detail = ml02.get("error") or errors or "Chưa đánh giá được khoản vay."
        reason = ml02.get("reason_code")
        if reason == "missing_input" or (reason is None and errors):
            status = 422
        elif reason == "model_unavailable":
            status = 503
        else:
            status = 500
        raise HTTPException(status_code=status, detail=detail)

    confidence = ml02.get("confidence") or {}
    by_label = {p["label"]: p for p in ml02.get("probabilities") or []}

    return Ml02PredictResponse(
        prediction=str(ml02["label"]),
        prediction_vi=str(ml02["label_vi"]),
        risk_probability=float(ml02["probability"]),
        model_confidence=Ml02ModelConfidence(
            confidence=float(confidence.get("confidence", 0.0)),
            low_confidence=bool(confidence.get("low_confidence", False)),
            threshold=float(confidence.get("threshold", 0.0)),
            description=str(confidence.get("description", "")),
            probabilities=[
                Ml01Probability(label=name,
                                label_vi=str(by_label[name]["label_vi"]),
                                probability=float(by_label[name]["probability"]))
                for name in ML02_ORDERED_LABELS if name in by_label
            ],
        ),
        model_version=str(ml02.get("model_version") or SETTINGS.ml02_slug),
    )


#: Nguồn câu trả lời được coi là "LLM đã diễn giải xong".
#:
#: `out_of_scope` nằm trong danh sách dù không hề gọi LLM: đó là câu từ chối
#: cố định, đã là tiếng Việt hoàn chỉnh, và thay nó bằng bản dựng sẵn của rule
#: thì hoá ra lại đi trả lời một câu hỏi vừa từ chối.
_NARRATED_SOURCES: frozenset[str] = frozenset({"llm", "llm_retry", "out_of_scope"})


def _narrate(
    req: AdviseRequest,
    intent: IntentCode,
    fallback_text: str,
    extra_profile: dict[str, Any] | None = None,
) -> tuple[str, str]:

    payload = dict(
        req.household or {}
    )


    if extra_profile:

        payload.update(
            extra_profile
        )


    try:

        result = inference_engine.chat(

            normalize_payload(
                payload,
                req.loan_application,
            ),

            question=req.question,

            intent_code=intent.value,
        )


    except Exception as exc:

        log.warning(

            "Diễn giải bằng LLM lỗi "
            "(%s: %s) — dùng bản dựng sẵn.",

            type(exc).__name__,
            exc,
        )

        return (

            presentation.to_plain_text(
                fallback_text
            ),

            "Template",
        )


    answer = (
        result
        .to_dict()
        .get("answer")
        or {}
    )


    source = str(
        answer.get(
            "source",
            "",
        )
    )


    text = str(
        result.text or ""
    )


    if (
        source in _NARRATED_SOURCES
        and text.strip()
    ):

        log.info(

            "advise: LLM đã diễn giải "
            "(nguồn=%s, model=%s)",

            source,

            answer.get("model")
            or "-",
        )


        return (

            presentation.to_plain_text(
                text
            ),

            f"LLM/"
            f"{answer.get('model') or 'unknown'}",
        )


    log.info(

        "advise: LLM không diễn giải "
        "được (nguồn=%s) — dùng "
        "bản dựng sẵn.",

        source or "-",
    )


    return (

        presentation.to_plain_text(
            fallback_text
        ),

        "Template",
    )


# =========================================================
# ML01 FROM FEATURES
# =========================================================

def _ml01_from_features(
    features: dict[str, Any],
) -> dict[str, Any]:

    model = get_ml01_model()


    frame = pd.DataFrame(

        [[
            features.get(name)

            for name
            in RAW_FEATURES
        ]],

        columns=list(
            RAW_FEATURES
        ),
    )


    label = str(
        model.predict(
            frame
        )[0]
    )


    proba = (
        model.predict_proba(
            frame
        )[0]
    )


    probabilities = (
        build_probabilities(
            list(model.classes_),
            proba,
        )
    )


    confidence = next(

        p.probability

        for p in probabilities

        if p.label == label
    )


    return {

        "label":
            label,

        "label_vi":
            LABELS_VI[
                RecommendationGroup(
                    label
                )
            ],

        "confidence":
            confidence,

        "low_confidence":
            confidence
            < CONFIG.confidence_threshold,

        "probabilities":
            [
                p.model_dump()
                for p
                in probabilities
            ],
    }


def _with_title(text: str, title: str) -> str:
    """Mở đầu câu trả lời của một nhánh ML bằng dòng tiêu đề mang nhãn model.

    Bản dựng sẵn (`explain_ml01/ml02`) đã có sẵn dòng này; bản do LLM viết thì
    không — nó chỉ diễn giải, và nhãn không phải thứ nhờ LLM nhắc lại. Không có
    dòng chung thì hai bản của cùng một nhánh mở đầu khác nhau, và màn Chatbot
    phải tự gắn một nhãn chung chung ("Kết quả đánh giá") cho bản LLM.
    """
    if text.startswith(title):
        return text
    return f"{title}\n\n{text}"


def _advise_financial_health(req: AdviseRequest, rule_summary: str) -> AdviseResponse:
    """Nhánh `FINANCIAL_HEALTH_DIAGNOSIS` — dữ liệu hộ → ML01 → LLM giải thích.

def _advise_financial_health(
    req: AdviseRequest,
    rule_summary: str,
) -> AdviseResponse:

    if not req.ml_features:

        return AdviseResponse(

            response_text=(
                presentation.to_plain_text(

                    "🧭 Chẩn đoán sức khỏe tài chính\n\n"

                    "Hồ sơ của bạn còn thiếu năm sinh "
                    "nên chưa chạy được phần chẩn đoán. "

                    "Vui lòng bổ sung ở màn Nhập thông tin "
                    "rồi thử lại."
                )
            ),

            model_used=
                "HFML-ML01-Advisor",

            intent_code=(
                IntentCode
                .FINANCIAL_HEALTH_DIAGNOSIS
                .value
            ),
        )


    try:

        result = (
            _ml01_from_features(
                req.ml_features
            )
        )


    except FileNotFoundError as exc:

        log.warning(

            "Chưa có artifact ML01 (%s): %s",

            SETTINGS.ml01_slug,
            exc,
        )


        return AdviseResponse(

            response_text=(
                presentation.to_plain_text(

                    "🧭 Chẩn đoán sức khỏe tài chính\n\n"

                    "Phần chẩn đoán bằng mô hình "
                    "chưa sẵn sàng nên chưa có kết quả.\n\n"

                    "Dưới đây là đánh giá theo "
                    "bộ quy tắc tài chính:\n\n"

                    f"{rule_summary}"
                )
            ),

            model_used=
                "HFML-RuleEngine-Fallback",

            intent_code=(
                IntentCode
                .FINANCIAL_HEALTH_DIAGNOSIS
                .value
            ),
        )


    except Exception as exc:

        log.exception(
            "ML01 lỗi khi chẩn đoán qua chat"
        )

        raise HTTPException(

            status_code=500,

            detail=(
                f"Model lỗi khi dự đoán: "
                f"{exc}"
            ),

        ) from exc


    fallback = explain_ml01(

        label=result["label"],

        label_vi=
            result["label_vi"],

        confidence=
            result["confidence"],

        probabilities=
            result["probabilities"],

        low_confidence=
            result["low_confidence"],

        rule_summary=
            rule_summary,

        model_version=
            SETTINGS.ml01_slug,
    )


    text, narrator = _narrate(

        req,

        IntentCode
        .FINANCIAL_HEALTH_DIAGNOSIS,

        fallback,
    )

        req, IntentCode.FINANCIAL_HEALTH_DIAGNOSIS, fallback)
    text = _with_title(text, ml01_title(result["label_vi"]))

    return AdviseResponse(

        response_text=text,

        model_used=(
            f"HFML-ML01/"
            f"{SETTINGS.ml01_slug}"
            f"+{narrator}"
        ),

        intent_code=(
            IntentCode
            .FINANCIAL_HEALTH_DIAGNOSIS
            .value
        ),

        suggested_questions=[

            "Tôi nên bắt đầu từ khoản chi nào?",

            "Quỹ dự phòng của tôi cần "
            "bao nhiêu tháng chi tiêu?",

            "Lập kế hoạch tích lũy "
            "trong 12 tháng tới",
        ],
    )


# =========================================================
# ML02 LOAN RISK
# =========================================================

def _advise_loan_risk(
    req: AdviseRequest,
) -> AdviseResponse:

        1. Chưa khai thông tin khoản vay → hướng người dùng sang màn nhập.
           ML02 cần 15 trường của màn đó; chạy trên số rỗng thì vẫn ra một xác
           suất, và đó là con số vô nghĩa mà không có gì báo hiệu.
        2. Chưa có artifact ML02 → nói thẳng là đang hoàn thiện.

    Cửa 2 hiện LUÔN đóng: F04 mới xong task 1/15. Viết sẵn đường đi để khi
    task 15 export artifact thì nhánh này tự sống, FE và backend không phải
    sửa lại lần nữa.
    """
    if not req.loan_application:

        return AdviseResponse(

            response_text=(
                presentation.to_plain_text(

                    "⚖️ Chẩn đoán rủi ro vay vốn\n\n"

                    "Bạn chưa khai thông tin khoản vay "
                    "nên chưa đánh giá được. "

                    "Phần này cần số tiền vay, thời hạn, "
                    "khoản trả hàng tháng, giá trị tài sản "
                    "và lịch sử tín dụng.\n\n"

                    "Vui lòng điền màn Thông tin khoản vay "
                    "rồi quay lại đây."
                )
            ),

            model_used=
                "HFML-ML02-Advisor",

            intent_code=(
                IntentCode
                .LOAN_RISK_DIAGNOSIS
                .value
            ),

            requires_loan_application=True,
        )


    # =====================================================
    # NORMALIZE HOUSEHOLD
    # =====================================================

    hh_dict = dict(
        req.household
    )


    if (
        "assets" in hh_dict
        and isinstance(
            hh_dict["assets"],
            list,
        )
    ):

        norm_assets = []


        for a in hh_dict["assets"]:

            a_str = str(

                a.value
                if hasattr(
                    a,
                    "value",
                )
                else a

            ).lower()


            if a_str in [
                "house",
                "land",
                "real_estate",
            ]:

                norm_assets.append(
                    "real_estate"
                )


            elif a_str in [
                "car",
                "vehicle",
            ]:

                norm_assets.append(
                    "vehicle"
                )


        hh_dict["assets"] = list(

            dict.fromkeys(
                norm_assets
            )
        )


    # =====================================================
    # NORMALIZE LOAN APPLICATION
    # =====================================================

    loan_dict = dict(
        req.loan_application
    )


    # occupation
    if (
        "occupation" in loan_dict
        and loan_dict["occupation"]
    ):

        occ_str = str(
            loan_dict["occupation"]
        ).lower()


        if (
            "it" in occ_str
            or "công nghệ" in occ_str
        ):

            loan_dict["occupation"] = (
                "it_staff"
            )


        elif (
            "văn phòng" in occ_str
            or "office" in occ_str
        ):

            loan_dict["occupation"] = (
                "office_staff"
            )


    # education
    if (
        "education_level" in loan_dict
        and loan_dict["education_level"]
    ):

        edu_str = str(
            loan_dict["education_level"]
        ).lower()


        if edu_str in [
            "university",
            "đại học",
            "dai_hoc",
        ]:

            loan_dict[
                "education_level"
            ] = "higher"


        elif edu_str in [
            "master",
            "doctor",
            "sau đại học",
            "academic",
        ]:

            loan_dict[
                "education_level"
            ] = "academic_degree"


        elif edu_str in [
            "high_school",
            "trung học",
            "thpt",
        ]:

            loan_dict[
                "education_level"
            ] = "secondary"


    # gender
    if (
        "gender" in loan_dict
        and loan_dict["gender"]
    ):

        g_str = str(
            loan_dict["gender"]
        ).lower()


        if g_str in [
            "male",
            "nam",
            "m",
        ]:

            loan_dict["gender"] = (
                "male"
            )


        elif g_str in [
            "female",
            "nữ",
            "nu",
            "f",
        ]:

            loan_dict["gender"] = (
                "female"
            )


    # marital
    if (
        "marital_status" in loan_dict
        and loan_dict["marital_status"]
    ):

        m_str = str(
            loan_dict["marital_status"]
        ).lower()


        if m_str in [
            "single",
            "độc thân",
            "doc_than",
        ]:

            loan_dict[
                "marital_status"
            ] = "single"


        elif m_str in [
            "married",
            "kết hôn",
            "da_ket_hon",
        ]:

            loan_dict[
                "marital_status"
            ] = "married"


    # purpose
    if (
        "loan_purpose" in loan_dict
        and loan_dict["loan_purpose"]
    ):

        p_str = str(
            loan_dict["loan_purpose"]
        ).lower()


        if (
            "nhà" in p_str
            or "house" in p_str
        ):

            loan_dict[
                "loan_purpose"
            ] = "buy_house"


        elif (
            "xe" in p_str
            or "car" in p_str
        ):

            loan_dict[
                "loan_purpose"
            ] = "buy_car"


    # =====================================================
    # RUN ML02
    # =====================================================

    result = (
        inference_engine.analyze(

            normalize_payload(

                hh_dict,
                loan_dict,
            )
        )
    )


    analysis = (
        result
        .to_dict()
        .get("analysis")
        or {}
    )


    ml02 = (
        analysis.get("ml02")
        or {}
    )


    # =====================================================
    # ML02 NOT AVAILABLE
    # =====================================================

    if not ml02.get(
        "available"
    ):

        reason = (

            ml02.get("error")

            or
            "Chưa đủ dữ liệu để đánh giá."
        )


        return AdviseResponse(

            response_text=(
                presentation.to_plain_text(

                    "⚖️ Chẩn đoán rủi ro vay vốn\n\n"

                    f"{reason}\n\n"

                    "Vui lòng kiểm tra lại màn "
                    "Thông tin khoản vay."
                )
            ),

            model_used=
                "HFML-ML02-Advisor",

            intent_code=(
                IntentCode
                .LOAN_RISK_DIAGNOSIS
                .value
            ),

            requires_loan_application=(

                ml02.get(
                    "reason_code"
                )
                ==
                "missing_input"
            ),
        )


    threshold = (
        MANAGER.threshold_for(
            ML02
        )
    )


    rules = (
        analysis.get("rules")
        or {}
    )


    loan_summary = "\n".join(

        (
            f"• "
            f"{presentation.rule_name(code).capitalize()}: "
            f"{rules[code]['details']['summary_vi']}"
        )

        for code in (
            "RB01",
            "RB02",
            "RB05",
        )

        if (
            rules
            .get(code, {})
            .get("details", {})
            .get("summary_vi")
        )
    )


    fallback = explain_ml02(

        label=str(
            ml02["label"]
        ),

        label_vi=str(
            ml02["label_vi"]
        ),

        probability=float(
            ml02["probability"]
        ),

        threshold=float(

            threshold

            if threshold is not None

            else 0.0
        ),

        loan_summary=
            loan_summary,

        model_version=str(

            ml02.get(
                "model_version"
            )

            or
            SETTINGS.ml02_slug
        ),
    )


    text, narrator = _narrate(

        req,

        IntentCode
        .LOAN_RISK_DIAGNOSIS,

        fallback,
    )


    return AdviseResponse(

        response_text=text,

        model_used=(

            f"HFML-ML02/"
            f"{SETTINGS.ml02_slug}"
            f"+{narrator}"
        ),

        intent_code=(
            IntentCode
            .LOAN_RISK_DIAGNOSIS
            .value
        ),

        suggested_questions=[

            "Tôi nên vay tối đa bao nhiêu thì an toàn?",

            "Làm sao giảm tỉ lệ nợ trên thu nhập?",

            "Kéo dài thời hạn vay thì "
            "rủi ro thay đổi thế nào?",
        ],
    )


# =========================================================
# ADVISE
# =========================================================

@app.post(
    "/advise",
    response_model=AdviseResponse,
)
def advise(
    req: AdviseRequest,
) -> AdviseResponse:

    household = req.household

    res = rule_engine.evaluate(
        household
    )


    overall_status = res.get(
        "overall_status",
        "STABLE",
    )


    rules = res.get(
        "rules",
        {},
    )


    total_debt = float(

        household.get(
            "total_debt"
        )

        or household.get(
            "total_current_debt"
        )

        or 0.0
    )


    debt_payment = float(

        household.get(
            "monthly_debt_payment"
        )

        or 0.0
    )


    savings = float(

        household.get(
            "current_savings"
        )

        or household.get(
            "savings_amount"
        )

        or 0.0
    )


    income = float(

        household.get(
            "monthly_income"
        )

        or household.get(
            "average_monthly_income"
        )

        or 0.0
    )


    expense = float(

        household.get(
            "monthly_living_cost"
        )

        or household.get(
            "average_monthly_expense"
        )

        or 0.0
    )


    has_dependents = bool(

        household.get(
            "supports_elderly"
        )

        or household.get(
            "has_dependents"
        )

        or False
    )


    assets_raw = (
        household.get(
            "assets"
        )
        or []
    )


    assets = [

        a.value
        if hasattr(
            a,
            "value",
        )
        else str(a)

        for a
        in assets_raw
    ]


    # =====================================================
    # RULE DATA
    # =====================================================

    rb01 = rules.get(
        "RB01",
        {},
    )


    rb01_val = rb01.get(
        "value",
        {},
    )


    net_cashflow = rb01_val.get(
        "net_cashflow",
        0.0,
    )


    rb02 = rules.get(
        "RB02",
        {},
    )


    rb02_val = rb02.get(
        "value",
        {},
    )


    dti = rb02_val.get(
        "dti",
        0.0,
    )


    emerg_months = rb02_val.get(
        "emergency_months",
        0.0,
    )


    savings_rate = rb02_val.get(
        "savings_rate",
        0.0,
    )


    min_emerg_target = rb02_val.get(

        "min_recommended_emergency_months",

        3.0,
    )


    rb05 = rules.get(
        "RB05",
        {},
    )


    rb05_val = rb05.get(
        "value",
        {},
    )


    max_add_payment = rb05_val.get(

        "max_allowed_monthly_payment",

        0.0,
    )


    collateral_quality = rb05_val.get(

        "collateral_quality",

        "NONE",
    )


    rep_name = (
        household.get(
            "representative_name"
        )
        or "bạn"
    )


    question = (
        req.question.strip()
    )


    # =====================================================
    # DEBT
    # =====================================================

    if (
        total_debt > 0
        or debt_payment > 0
    ):

        debt_desc = (

            f"{presentation.money(total_debt)} "

            f"(Trả gốc lãi "
            f"{presentation.money(debt_payment)}/tháng)"
        )

    else:

        debt_desc = (
            "Không có nợ"
        )


    # =====================================================
    # SAVINGS
    # =====================================================

    if savings > 0:

        savings_desc = (
            presentation.money(
                savings
            )
        )

    else:

        savings_desc = (
            "0đ (chưa có tích lũy)"
        )


    # =====================================================
    # ASSETS
    # =====================================================

    asset_label_list = []


    for a in assets:

        a_lower = str(
            a
        ).lower()


        if a_lower == "house":

            asset_label_list.append(
                "Nhà ở"
            )


        elif a_lower == "land":

            asset_label_list.append(
                "Đất đai"
            )


        elif a_lower in [
            "real_estate",
            "bất động sản",
        ]:

            asset_label_list.append(
                "Bất động sản"
            )


        elif a_lower in [
            "car",
            "vehicle",
            "xe",
        ]:

            asset_label_list.append(
                "Phương tiện (Xe)"
            )


        elif a_lower in [
            "cash",
            "savings",
        ]:

            asset_label_list.append(
                "Tiền gửi / Tiền mặt"
            )


        elif a_lower == "gold":

            asset_label_list.append(
                "Vàng & Kim loại quý"
            )


        elif a_lower == "insurance":

            asset_label_list.append(
                "Bảo hiểm"
            )


        elif a_lower in [
            "investment",
            "stock",
        ]:

            asset_label_list.append(
                "Đầu tư / Cổ phiếu"
            )


        else:

            asset_label_list.append(
                str(a).upper()
            )


    if asset_label_list:

        asset_desc = ", ".join(

            list(

                dict.fromkeys(
                    asset_label_list
                )
            )
        )

    else:

        asset_desc = (
            "Chưa có tài sản"
        )


    # =====================================================
    # QUESTION VALUES
    # =====================================================

    parsed_price = (
        parse_amount_from_text(
            question
        )
    )


    parsed_term = (
        parse_term_months_from_text(
            question
        )
    )


    term_months = (

        parsed_term

        or int(
            household.get(
                "loan_term_months"
            )
            or 240
        )
    )


    if term_months % 12 == 0:

        term_years_str = (
            f"{term_months // 12} năm"
        )

    else:

        term_years_str = (
            f"{term_months} tháng"
        )


    # =====================================================
    # INTENT
    # =====================================================

    intent = resolve_intent(

        question,

        req.intent_code,
    )


    if (
        intent is IntentCode.GENERAL
        and parsed_price is not None
    ):

        intent = (
            IntentCode.LOAN_CAPACITY
        )


    log.info(

        "advise: intent=%s (%s)",

        intent.value,

        (
            "chip"
            if req.intent_code
            else "từ khoá"
        ),
    )


    # =====================================================
    # RULE SUMMARY
    # =====================================================

    cashflow_word = (

        "Dư khoảng"

        if net_cashflow > 0

        else
        "Thiếu khoảng"

        if net_cashflow < 0

        else
        "Vừa đủ, không dư không thiếu —"
    )


    emergency = (

        f"{emerg_months:.1f}"
        .replace(
            ".",
            ",",
        )
    )


    rule_summary = (

        f"📌 Đánh giá tổng quan: "

        f"{presentation.label_status('OVERALL', overall_status)}\n"

        f"• Dòng tiền hằng tháng: "

        f"{cashflow_word} "

        f"{presentation.money(abs(net_cashflow))}.\n"

        f"• Nợ, tiết kiệm & tài sản: "

        f"Nợ {debt_desc} | "

        f"Tiết kiệm {savings_desc} | "

        f"Tài sản: {asset_desc}.\n"

    # Tóm tắt tầng rule, dùng lại cho cả câu trả lời thường lẫn phần diễn giải
    # của ML01 — hai nơi nói khác nhau về cùng một hồ sơ là chuyện phải tránh.
    #
    # Mã rule và mã trạng thái KHÔNG xuất hiện ở đây nữa. Chuỗi này là nguồn
    # trực tiếp của những dòng `(RB01) … (DEFICIT)` mà người dùng đọc phải: nó
    # đi thẳng vào `response_text` và trước đây không có bước nào đứng giữa.
    #
    # Nhân đây sửa luôn một chỗ nói sai: bản cũ viết cứng "Dư thừa khoảng
    # {net_cashflow}" cho mọi hồ sơ, nên hộ đang âm dòng tiền vẫn được báo là
    # "dư thừa khoảng -2.000.000 VNĐ (DEFICIT)". Câu đó tự mâu thuẫn, và phần
    # duy nhất nói đúng lại chính là mã đang phải bỏ đi.
    # Nói bằng chữ, không kèm mã trạng thái trong ngoặc.
    #
    # "Dư khoảng 3.000.000đ" đã nói đúng thứ mà `(POSITIVE)` nói, nên thêm
    # ngoặc vào chỉ là lặp lại chính mình bằng một thứ tiếng khó hơn.
    cashflow_word = ("Dư khoảng" if net_cashflow > 0
                     else "Thiếu khoảng" if net_cashflow < 0
                     else "Vừa đủ, không dư không thiếu —")
    emergency = f"{emerg_months:.1f}".replace(".", ",")
    household_lines = [
        f"📌 Đánh giá tổng quan: "
        f"{presentation.label_status('OVERALL', overall_status)}",
        f"• Dòng tiền hằng tháng: {cashflow_word} "
        f"{presentation.money(abs(net_cashflow))}.",
        f"• Nợ, tiết kiệm & tài sản: Nợ {debt_desc} | Tiết kiệm {savings_desc} "
        f"| Tài sản: {asset_desc}.",
        f"• Sức khỏe tài chính: "

        f"{presentation.label_status('RB02', rb02.get('status'))} "

        f"(tỉ lệ trả nợ trên thu nhập "

        f"{presentation.percent(dti)}, "

        f"quỹ dự phòng "

        f"{emergency}/{min_emerg_target:.0f} tháng, "

        f"tỉ lệ tiết kiệm "

        f"{presentation.percent(savings_rate)}).\n"

        f"• Khả năng vay: "

        f"có thể gánh thêm tối đa "

        f"{presentation.money(max_add_payment)}/tháng "

        f"({presentation.label_status('RB05', rb05.get('status'))})."
    )

        f"(tỉ lệ trả nợ trên thu nhập {presentation.percent(dti)}, quỹ dự phòng "
        f"{emergency}/{min_emerg_target:.0f} tháng, tỉ lệ tiết kiệm "
        f"{presentation.percent(savings_rate)}).",
    ]
    loan_capacity_line = (
        f"• Khả năng vay: có thể gánh thêm tối đa "
        f"{presentation.money(max_add_payment)}/tháng "
        f"({presentation.label_status('RB05', rb05.get('status'))}).")
    rule_summary = "\n".join(household_lines + [loan_capacity_line])

    # Hai nhánh ML trả về câu trả lời HOÀN CHỈNH và thoát sớm: chúng có cấu
    # trúc riêng do tầng diễn đạt dựng, không phải một đoạn `advice_detail`
    # ghép vào khung trả lời chung.
    #
    # ML01 nhận bản tóm tắt KHÔNG có dòng "Khả năng vay": câu trả lời của một
    # chip ML phải nằm gọn trong phạm vi của model ấy, và hạn mức vay (RB05)
    # thuộc chức năng khác. Cùng ranh giới với context của LLM ở
    # `hfml.llm.context` — bản dựng sẵn và bản LLM phải kể cùng một chuyện.
    if intent is IntentCode.FINANCIAL_HEALTH_DIAGNOSIS:
        return _advise_financial_health(req, "\n".join(household_lines))

    # =====================================================
    # ML01
    # =====================================================

    if (
        intent
        is
        IntentCode.FINANCIAL_HEALTH_DIAGNOSIS
    ):

        return _advise_financial_health(

            req,

            rule_summary,
        )


    # =====================================================
    # ML02
    # =====================================================

    if (
        intent
        is
        IntentCode.LOAN_RISK_DIAGNOSIS
    ):

        return _advise_loan_risk(
            req
        )


    # =====================================================
    # LOAN CAPACITY
    # =====================================================

    if (
        intent
        is
        IntentCode.LOAN_CAPACITY
    ):

        asset_price = (

            parsed_price

            or float(
                household.get(
                    "asset_price"
                )
                or 0.0
            )
        )


        loan_eval = (
            evaluate_loan_capacity(

                household,

                asset_price=
                    asset_price,

                term_months=
                    term_months,
            )
        )


        loan_val = (
            loan_eval.get(
                "value",
                {},
            )
        )


        max_loan = loan_val.get(

            "max_allowed_loan",

            0.0,
        )


        max_pmt = loan_val.get(

            "max_allowed_monthly_payment",

            0.0,
        )


        max_ltv_loan = loan_val.get(

            "max_loan_by_ltv",

            0.0,
        )


        if debt_payment > 0:

            debt_note = (

                " (Đã trừ đi khoản nợ đang trả "

                f"{presentation.money(debt_payment)}/tháng)"
            )

        else:

            debt_note = ""


        if asset_price > 0:

            down_payment = max(

                0.0,

                asset_price
                -
                max_loan,
            )


            advice_detail = (

                f"🏡 **Tư vấn vay mua tài sản "
                f"({presentation.money(asset_price)}, "
                f"kỳ hạn {term_years_str})**:\n"

                f"- Hạn mức vay an toàn tối đa "
                f"dựa trên thu nhập "
                f"(DTI ≤ 40%, {term_years_str}): "

                f"**{presentation.money(max_loan)}**"
                f"{debt_note}.\n"

                f"- Số tiền trả gốc lãi tối đa "
                f"có thể gánh thêm: "

                f"**~{presentation.money(max_pmt)}/tháng**.\n"

                f"- Hạn mức vay tối đa theo tài sản "
                f"thế chấp (LTV 70%): "

                f"**{presentation.money(max_ltv_loan)}**.\n"

                f"- Năng lực thế chấp tài sản sở hữu "
                f"hiện tại ({asset_desc}): "

                f"**{presentation.label_status('COLLATERAL', collateral_quality)}**.\n"

                f"- 💡 **Khuyên dùng**: "

                f"Bạn có thể vay an toàn tối đa "

                f"**{presentation.money(max_loan)}** "

                f"trong thời hạn **{term_years_str}**. "

                f"Vốn tự có cần chuẩn bị tối thiểu "

                f"**{presentation.money(down_payment)}** "

                f"({presentation.percent(down_payment / asset_price)})."
            )


        else:

            advice_detail = (

                f"🏦 **Tư vấn hạn mức vay an toàn "
                f"(kỳ hạn {term_years_str})**:\n"

                f"- Hạn mức vay an toàn tối đa: "

                f"**{presentation.money(max_loan)}**"
                f"{debt_note}.\n"

                f"- Khả năng trả gốc lãi vay mới tối đa: "

                f"**{presentation.money(max_pmt)}/tháng**.\n"

                f"- Năng lực thế chấp từ tài sản sở hữu "
                f"({asset_desc}): "

                f"**{presentation.label_status('COLLATERAL', collateral_quality)}**."
            )


    # =====================================================
    # SAVINGS PACKAGE
    # =====================================================

    elif (
        intent
        is
        IntentCode.SAVINGS_PACKAGE
    ):

        buf_min = (
            expense
            *
            min_emerg_target
        )


        elderly_note = (

            " (Đã nâng lên 6 tháng do có "
            "phụng dưỡng người già)"

            if has_dependents

            else ""
        )


        if net_cashflow <= 0:

            advice_detail = (

                "🐖 **Gói tư vấn tiết kiệm "
                "& Quỹ dự phòng**:\n"

                f"- Số tiền tiết kiệm hiện tại: "
                f"**{savings_desc}**.\n"

                f"- Dòng tiền hằng tháng: "
                f"{cashflow_word} "
                f"{presentation.money(abs(net_cashflow))}.\n"

                f"- Mục tiêu quỹ dự phòng an toàn "
                f"khuyến nghị "

                f"({min_emerg_target:.0f} tháng chi tiêu = "

                f"{presentation.money(buf_min)})"
                f"{elderly_note}.\n"

                f"- ⚠️ Dòng tiền hiện chưa dư nên "
                f"chưa tích lũy thêm được. "

                f"Ưu tiên số 1 là rà soát và cắt giảm "
                f"chi tiêu chưa cấp thiết."
            )


        else:

            months_min = (
                buf_min
                /
                net_cashflow
            )


            advice_detail = (

                "🐖 **Gói tư vấn tiết kiệm "
                "& Quỹ dự phòng**:\n"

                f"- Số tiền tiết kiệm hiện tại: "
                f"**{savings_desc}**.\n"

                f"- Số dư thặng dư khả dụng hàng tháng: "

                f"**{presentation.money(net_cashflow)}/tháng** "

                f"(Đạt tỷ lệ tiết kiệm "

                f"{presentation.percent(savings_rate)}).\n"

                f"- Mục tiêu quỹ dự phòng an toàn "
                f"khuyến nghị "

                f"({min_emerg_target:.0f} tháng chi tiêu = "

                f"{presentation.money(buf_min)})"

                f"{elderly_note}: "

                f"Cần tích lũy khoảng "

                f"**{months_min:.1f} tháng**."
            )


    # =====================================================
    # INVESTMENT
    # =====================================================

    elif (
        intent
        is
        IntentCode.INVESTMENT
    ):

        if net_cashflow <= 0:

            advice_detail = (

                "📈 **Gói tư vấn phân bổ đầu tư**:\n"

                f"- Dòng tiền hằng tháng: "
                f"{cashflow_word} "

                f"{presentation.money(abs(net_cashflow))}.\n"

                f"- ⚠️ Gia đình chưa có dòng tiền "
                f"thặng dư dương nên chưa phù hợp "
                f"để tham gia các kênh đầu tư sinh lời."
            )


        else:

            advice_detail = (

                "📈 **Gói tư vấn phân bổ đầu tư**:\n"

                f"- Với thặng dư dòng tiền "

                f"**{presentation.money(net_cashflow)}/tháng** "

                f"(Tỷ lệ tiết kiệm "

                f"{presentation.percent(savings_rate)}):\n"

                f"- Trích **30% "
                f"({presentation.money(net_cashflow * 0.3)})** "

                f"cho tiền gửi tiết kiệm thanh khoản cao.\n"

                f"- Trích **70% "
                f"({presentation.money(net_cashflow * 0.7)})** "

                f"đầu tư vào tài sản sinh lời an toàn."
            )


    # =====================================================
    # 50 / 30 / 20
    # =====================================================

    elif (
        intent
        is
        IntentCode.BUDGET_50_30_20
    ):

        needs_target = (
            income * 0.50
        )

        wants_target = (
            income * 0.30
        )

        savings_target = (
            income * 0.20
        )


        if net_cashflow < 0:

            advice_detail = (

                f"📊 **Phân tích theo Quy tắc 50/30/20 "
                f"(Thu nhập {presentation.money(income)})**:\n"

                f"- **50% Nhu cầu thiết yếu** "
                f"(Tối đa {presentation.money(needs_target)}): "

                f"Chi tiêu sinh hoạt hiện tại "
                f"{presentation.money(expense)} "

                f"+ trả gốc lãi nợ "
                f"{presentation.money(debt_payment)}/tháng.\n"

                f"- **30% Cá nhân & Giải trí** "
                f"(Tối đa {presentation.money(wants_target)}).\n"

                f"- **20% Tiết kiệm & Trả nợ** "
                f"(Tối thiểu {presentation.money(savings_target)}).\n"

                f"- ⚠️ Dòng tiền hiện đang thâm hụt khoảng "

                f"{presentation.money(abs(net_cashflow))}/tháng."
            )


        else:

            advice_detail = (

                f"📊 **Phân tích theo Quy tắc 50/30/20 "
                f"(Thu nhập {presentation.money(income)})**:\n"

                f"- **50% Nhu cầu thiết yếu** "
                f"(Tối đa {presentation.money(needs_target)}): "

                f"Chi tiêu sinh hoạt hiện tại "
                f"{presentation.money(expense)}.\n"

                f"- **30% Cá nhân & Giải trí** "
                f"(Tối đa {presentation.money(wants_target)}).\n"

                f"- **20% Tiết kiệm & Trả nợ** "
                f"(Tối thiểu {presentation.money(savings_target)}): "

                f"Thặng dư hiện tại đạt "

                f"**{presentation.money(net_cashflow)}** "

                f"({presentation.percent(savings_rate)})."
            )


    # =====================================================
    # GENERAL
    # =====================================================

    else:

        elderly_text = (

            " Ghi nhận gia đình có phụng dưỡng "
            "người già ➔ Nâng ngưỡng quỹ dự phòng "
            "lên 6 tháng chi tiêu sinh hoạt."

            if has_dependents

            else ""
        )


        advice_detail = (

            "Dựa trên phân tích dòng tiền và "
            "đòn bẩy nợ hiện tại, "

            f"bạn có số dư thặng dư khả dụng "
            f"hàng tháng là "

            f"**{presentation.money(net_cashflow)}** "

            f"(Đạt tỷ lệ tiết kiệm "

            f"{presentation.percent(savings_rate)}). "

            f"Hạn mức trả nợ mới đề xuất thêm tối đa là "

            f"**{presentation.money(max_add_payment)}/tháng**."

            f"{elderly_text}"
        )


    # =====================================================
    # FALLBACK + LLM
    # =====================================================

    heading = (

        f"💡 {INTENT_LABELS[intent]}:"

        if req.intent_code

        else

        f"💡 Trả lời cho câu hỏi: "
        f"{question}"
    )


    fallback = (

        f"Chào {rep_name}, "
        f"hệ thống đã phân tích hồ sơ của bạn.\n\n"

        f"{rule_summary}\n\n"

        f"{heading}\n"

        f"{advice_detail}"
    )


    extra: dict[str, Any] = {

        "loan_term_months":
            term_months,
    }


    if parsed_price:

        extra[
            "asset_price"
        ] = parsed_price


    text, narrator = _narrate(

        req,

        intent,

        fallback,

        extra,
    )


    suggested = [

        "Tôi muốn mua nhà giá 3 tỷ "
        "thì vay được bao nhiêu?",

        "Tư vấn gói đầu tư "
        "tích lũy an toàn",

        "Lập kế hoạch xây dựng "
        "quỹ dự phòng 6 tháng",
    ]


    return AdviseResponse(

        response_text=text,

        model_used=(

            "HFML-RuleEngine-v0.2.0"

            f"+{narrator}"
        ),

        suggested_questions=
            suggested,

        tokens_used=150,

        intent_code=
            intent.value,
    )


# =========================================================
# INFERENCE API
# =========================================================

class InferenceRequest(BaseModel):

    household: dict[str, Any]


class InferenceChatRequest(
    InferenceRequest
):

    question: str

    intent_code: Optional[str] = None

    history: Optional[List[dict]] = None

    previous_intent: Optional[str] = None


@app.post(
    "/inference/analyze"
)
def inference_analyze(
    req: InferenceRequest,
) -> dict:

    return (
        inference_engine
        .analyze(
            req.household
        )
        .to_dict()
    )


@app.post(
    "/inference/chat"
)
def inference_chat(
    req: InferenceChatRequest,
) -> dict:

    return (
        inference_engine
        .chat(

            payload=
                req.household,

            question=
                req.question,

            intent_code=
                req.intent_code,

            history=
                req.history,

            previous_intent=
                req.previous_intent,
        )
        .to_dict()
    )


@app.post(
    "/inference/reload"
)
def inference_reload(
    name: Optional[str] = None,
) -> dict:

    MANAGER.reload(
        name
    )

    return (
        inference_engine.health()
    )