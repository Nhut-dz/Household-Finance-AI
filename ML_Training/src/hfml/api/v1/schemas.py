"""AI-04 task 1 — Request/Response schema của API v1 (F05 · M08).

Schema ở đây là **HỢP ĐỒNG CÔNG KHAI** với Backend/Frontend. Nó cố ý khác với
cấu trúc bên trong, và sự khác biệt đó là điểm chính của cả file.

Không phơi chi tiết cài đặt
-----------------------------
`InferenceResult` của AI-03 mang theo hai thứ chỉ dành cho người vận hành:

    trace      tên từng bước bên trong + thời gian từng bước
    settings   slug artifact, đường dẫn thư mục model, tham số LLM

Trả thẳng chúng ra ngoài thì hai chuyện xảy ra. Một, client bắt đầu phụ thuộc
vào tên bước nội bộ, và đổi tên một bước thành thay đổi phá vỡ hợp đồng. Hai,
đường dẫn thư mục và tên artifact là thông tin hạ tầng — không có lý do gì để
trình duyệt của người dùng biết model đang nằm ở đâu.

Nên `AIResult` là một PHÉP CHIẾU có chủ ý, không phải bản sao. Thêm trường vào
bên trong không tự động rò ra ngoài.

Alias sinh TỪ bảng của AI-03, không khai lại
----------------------------------------------
Backend Laravel gọi `monthly_income`, schema chuẩn gọi
`average_monthly_income`. Bảng quy đổi đã có ở `hfml.inference.payloads`, nên ở
đây `AliasChoices` được **sinh ra từ chính bảng đó**. Khai lại bằng tay là tạo
bản sao thứ hai, và hai bản sẽ trôi khỏi nhau đúng như hai hằng `ML02_SLUG`
từng trôi khỏi nhau trước Epic AI-03.
"""
from __future__ import annotations

from decimal import Decimal
from typing import Any, Final, Literal, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from hfml.inference.payloads import FIELD_ALIASES

#: Trần tiền tệ — chặn số vô lý trước khi tới model (1.000 tỉ VNĐ).
MAX_MONEY: Final[Decimal] = Decimal("1000000000000")

#: Trần số người trong hộ. Trên mức này gần như chắc chắn là gõ nhầm.
MAX_HOUSEHOLD: Final[int] = 30


def _aliases(canonical: str) -> AliasChoices:
    """Mọi tên mà client được phép dùng cho một trường.

    Sinh từ `FIELD_ALIASES` của AI-03 để bảng quy đổi chỉ tồn tại ở MỘT nơi.
    """
    names = [canonical]
    names += [outside for outside, inside in FIELD_ALIASES.items()
              if inside == canonical]
    return AliasChoices(*names)


def _money(description: str, **kwargs) -> Any:
    return Field(None, ge=0, le=MAX_MONEY, description=description,
                 **kwargs)


# ==========================================================================
# Đầu vào
# ==========================================================================
class FinancialInput(BaseModel):
    """Hồ sơ tài chính hộ gia đình.

    `extra="forbid"`: gửi thừa trường lạ thì báo lỗi ngay kèm tên field, chứ
    không nuốt im. Trường lạ gần như luôn là dấu hiệu backend map sai tên —
    và nuốt im nghĩa là dữ liệu người dùng nhập bị bỏ mà không ai biết.

    Ràng buộc ở đây chỉ là ràng buộc KIỂU và MIỀN GIÁ TRỊ. Luật nghiệp vụ
    (bắt buộc theo ngữ cảnh, quan hệ chéo giữa các trường) vẫn nằm ở
    `HouseholdProfile` — một nơi duy nhất, và API không cài lại.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    representative_name: Optional[str] = Field(
        None, max_length=200, description="Tên người đại diện hộ")
    birth_year: Optional[int] = Field(
        None, ge=1900, le=2100, description="Năm sinh — model cần để suy ra tuổi")
    residence: Optional[str] = Field(None, max_length=200)

    household_size: Optional[int] = Field(None, ge=1, le=MAX_HOUSEHOLD)
    children_count: Optional[int] = Field(None, ge=0, le=MAX_HOUSEHOLD)
    has_dependents: Optional[bool] = Field(
        None, validation_alias=_aliases("has_dependents"))

    average_monthly_income: Optional[Decimal] = Field(
        None, ge=0, le=MAX_MONEY,
        validation_alias=_aliases("average_monthly_income"),
        description="Thu nhập bình quân tháng (VNĐ)")
    average_monthly_expense: Optional[Decimal] = Field(
        None, ge=0, le=MAX_MONEY,
        validation_alias=_aliases("average_monthly_expense"),
        description="Chi tiêu sinh hoạt bình quân tháng (VNĐ)")

    has_debt: Optional[bool] = None
    total_current_debt: Optional[Decimal] = Field(
        None, ge=0, le=MAX_MONEY,
        validation_alias=_aliases("total_current_debt"))
    monthly_debt_payment: Optional[Decimal] = Field(None, ge=0, le=MAX_MONEY)

    has_savings: Optional[bool] = None
    savings_amount: Optional[Decimal] = Field(
        None, ge=0, le=MAX_MONEY, validation_alias=_aliases("savings_amount"))

    assets: Optional[list[str]] = None
    financial_needs: Optional[list[str]] = None
    occupation: Optional[str] = None
    employment_years: Optional[Decimal] = Field(None, ge=0, le=80)

    asset_price: Optional[Decimal] = Field(None, ge=0, le=MAX_MONEY)
    loan_amount: Optional[Decimal] = Field(None, ge=0, le=MAX_MONEY)
    loan_term_months: Optional[int] = Field(None, ge=1, le=600)

    #: Dữ liệu màn "Thông tin khoản vay". Để trống thì ML02 không chạy và nói
    #: rõ là thiếu — KHÔNG chạy model trên số rỗng.
    loan_application: Optional[dict[str, Any]] = None

    def to_payload(self) -> dict[str, Any]:
        """Payload cho `hfml.inference`, đã mang tên chuẩn.

        `by_alias=False` nên dù client gửi `monthly_income`, thứ đi ra là
        `average_monthly_income`. Bỏ trường `None` để không đè lên mặc định
        của schema lõi bằng một giá trị rỗng mà người dùng không hề nhập.
        """
        return self.model_dump(exclude_none=True, by_alias=False)


class InferenceRequest(BaseModel):
    """Yêu cầu chạy inference — chỉ Rule + ML, không gọi LLM."""

    model_config = ConfigDict(extra="forbid")

    household: FinancialInput


class ChatTurnInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str = Field(..., max_length=4000)


class ChatRequest(InferenceRequest):
    """Yêu cầu hội thoại — chạy trọn pipeline tới câu trả lời."""

    question: str = Field(..., min_length=1, max_length=2000)

    #: Mã ý định do chip gợi ý gửi kèm. Có mã thì tin tuyệt đối; không có thì
    #: đoán theo từ khoá. Hai intent chạy model CHỈ vào được qua đường này.
    intent_code: Optional[str] = Field(None, max_length=64)

    #: Vài lượt gần nhất, để hiểu câu hỏi nối tiếp ("thế còn 2 tỷ?").
    history: Optional[list[ChatTurnInput]] = Field(None, max_length=20)
    previous_intent: Optional[str] = Field(None, max_length=64)


# ==========================================================================
# Đầu ra
# ==========================================================================
class RuleResult(BaseModel):
    """Kết quả một quy tắc RB01–RB05."""

    code: str
    status: str
    summary: str = ""
    #: Các con số quy tắc đã tính. Để nguyên dict vì mỗi rule có bộ số riêng.
    values: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_internal(cls, code: str, rule: dict) -> "RuleResult":
        return cls(
            code=code,
            status=str(rule.get("status") or ""),
            summary=str(rule.get("details", {}).get("summary_vi") or ""),
            values=dict(rule.get("value") or {}),
        )


class MLResult(BaseModel):
    """Kết quả một model. `available=False` là trạng thái BÌNH THƯỜNG.

    Người dùng chưa khai khoản vay thì ML02 không chạy — đó không phải lỗi.
    `reason_code` nói rõ vì sao, để client hiển thị đúng việc cần làm thay vì
    một câu "hệ thống lỗi" chung chung.
    """

    available: bool
    label: Optional[str] = None
    label_vi: Optional[str] = None
    probability: Optional[float] = None
    probabilities: list[dict] = Field(default_factory=list)
    low_confidence: bool = False
    #: `missing_input` · `model_unavailable` · `prediction_error` ·
    #: `invalid_probability`. Bốn mã KHÔNG cùng loại — mã đầu là việc của
    #: người dùng, ba mã sau là việc của người vận hành.
    reason_code: Optional[str] = None
    message: Optional[str] = None
    model_version: Optional[str] = None

    @classmethod
    def from_internal(cls, part: dict) -> "MLResult":
        return cls(
            available=bool(part.get("available")),
            label=part.get("label"),
            label_vi=part.get("label_vi"),
            probability=part.get("probability"),
            probabilities=list(part.get("probabilities") or []),
            low_confidence=bool(
                (part.get("confidence") or {}).get("low_confidence")),
            reason_code=part.get("reason_code"),
            message=part.get("error"),
            model_version=part.get("model_version") or None,
        )


class Diagnostic(BaseModel):
    """Một lỗi hoặc cảnh báo, đủ để client hiển thị đúng chỗ cần sửa."""

    code: str
    message: str
    severity: str = "warning"
    field: str = ""


class AIResult(BaseModel):
    """Kết quả phân tích — Rule + ML, đã lọc bỏ phần nội bộ.

    KHÔNG mang `trace` và `settings` của `InferenceResult`; xem docstring đầu
    file.
    """

    ok: bool
    schema_version: str
    generated_at: str
    overall_status: Optional[str] = None
    rules: list[RuleResult] = Field(default_factory=list)
    ml01: MLResult
    ml02: MLResult
    errors: list[Diagnostic] = Field(default_factory=list)
    warnings: list[Diagnostic] = Field(default_factory=list)
    elapsed_ms: float = 0.0

    @classmethod
    def from_internal(cls, result) -> "AIResult":
        payload = result.to_dict()
        analysis = payload.get("analysis") or {}
        empty = {"available": False, "reason_code": "model_unavailable"}

        return cls(
            ok=payload["ok"],
            schema_version=payload["schema_version"],
            generated_at=payload["generated_at"],
            overall_status=analysis.get("overall_status"),
            rules=[RuleResult.from_internal(code, rule)
                   for code, rule in (analysis.get("rules") or {}).items()
                   if rule],
            ml01=MLResult.from_internal(analysis.get("ml01") or empty),
            ml02=MLResult.from_internal(analysis.get("ml02") or empty),
            errors=[Diagnostic(**_diag(d)) for d in payload["errors"]],
            warnings=[Diagnostic(**_diag(d)) for d in payload["warnings"]],
            elapsed_ms=round(
                sum(step["elapsed_ms"] for step in payload["trace"]), 2),
        )


def _diag(item: dict) -> dict:
    """Giữ đúng bốn trường công khai — `stage` là tên bước nội bộ, bỏ đi."""
    return {"code": item.get("code", ""), "message": item.get("message", ""),
            "severity": item.get("severity", "warning"),
            "field": item.get("field", "")}


class LLMResponse(BaseModel):
    """Câu trả lời đã qua kiểm, kèm đúng phần metadata client cần biết."""

    text: str
    explanation: str = ""
    recommendations: list[dict] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    needs_more_data: list[str] = Field(default_factory=list)
    suggested_questions: list[str] = Field(default_factory=list)

    intent_code: str = ""
    topic: str = ""

    #: `llm` · `llm_retry` · `template` · `out_of_scope`.
    #:
    #: Phơi ra là CÓ CHỦ Ý: client cần phân biệt câu do model viết với câu
    #: dựng sẵn từ dữ liệu. Giấu đi thì một lượt hạ cấp trông y hệt một lượt
    #: bình thường, và không ai biết chất lượng đã tụt.
    source: str = ""
    prompt_version: str = ""

    #: `True` đạt · `False` bị chặn khi kiểm · `None` chưa gọi được LLM.
    #: Ba trạng thái, không phải hai — xem `hfml.llm.client`.
    validated: Optional[bool] = None

    #: Cần người dùng làm thêm một việc trước (thường là điền màn khoản vay).
    requires_more_input: bool = False

    @classmethod
    def from_internal(cls, result) -> "LLMResponse":
        payload = result.to_dict()
        answer = payload.get("answer") or {}
        return cls(
            text=payload.get("text") or "",
            explanation=str(answer.get("explanation") or ""),
            recommendations=list(answer.get("recommendations") or []),
            caveats=[str(c) for c in (answer.get("caveats") or [])],
            needs_more_data=[str(d) for d in (answer.get("needs_more_data") or [])],
            suggested_questions=[
                str(q) for q in (answer.get("suggested_questions") or [])],
            intent_code=payload.get("intent") or "",
            topic=payload.get("topic") or "",
            source=str(answer.get("source") or ""),
            prompt_version=str(answer.get("prompt_version") or ""),
            validated=(answer.get("validation") or {}).get("valid"),
            requires_more_input=bool(answer.get("needs_more_data")),
        )


class ChatResponse(BaseModel):
    """Trả lời của `POST /api/v1/chat` — câu trả lời kèm căn cứ của nó.

    Mang cả `analysis` chứ không chỉ câu chữ: client cần đối chiếu được câu
    trả lời với con số đã tính, và không có nó thì không có cách nào kiểm
    chứng ngoài việc tin.
    """

    ok: bool
    answer: LLMResponse
    analysis: AIResult


class HealthComponent(BaseModel):
    status: str
    detail: str = ""
    info: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """`healthy` · `degraded` · `unhealthy` — xem `health.py` về ranh giới."""

    status: Literal["healthy", "degraded", "unhealthy"]
    service: str
    api_version: str
    components: dict[str, HealthComponent]


class ErrorResponse(BaseModel):
    """Vỏ lỗi DUY NHẤT của API v1.

    Mọi đường hỏng — validate, model, LLM, timeout, thiếu cấu hình, lỗi không
    lường trước — đều ra đúng hình dạng này. Client vì vậy viết một nhánh xử
    lý lỗi, không phải sáu.
    """

    ok: Literal[False] = False
    error: str = Field(..., description="Mã lỗi ổn định, dùng để phân nhánh")
    message: str = Field(..., description="Câu giải thích cho người đọc")
    details: list[Diagnostic] = Field(default_factory=list)
    request_id: Optional[str] = None
