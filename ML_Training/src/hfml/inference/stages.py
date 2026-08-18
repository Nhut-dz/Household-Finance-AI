"""AI-03 task 1, 6 — Mười bước của pipeline (F05 · M07).

Mỗi bước là một hàm nhận `PipelineState`, trả về `StageResult`, và **uỷ quyền**
cho module đã có. File này không chứa phép tính nào.

Đó là ràng buộc quan trọng nhất ở đây
---------------------------------------
Chép một công thức sang đây để "cho tiện" là tạo ra nguồn sự thật thứ hai cho
cùng một con số. Bản chép sẽ không được sửa khi bản gốc sửa, và cái sai sẽ
không lộ ra ở đâu cả — hai bên vẫn chạy, chỉ ra hai kết quả khác nhau. Nên:

    Input · Validation · Preprocessing   → pipeline.normalizer.normalize_input
    Rule-Based                           → pipeline.orchestrator.run_rules
    ML01 · ML02                          → pipeline.predictor.predict_*
    Result Aggregation                   → pipeline.orchestrator (AiResult)
    Intent                               → llm.understanding.understand
    Context                              → llm.context.build_context
    LLM · Output Validation              → llm.client.generate
    Response                             → gói lại, không tính thêm

Ba bước đầu là MỘT lời gọi, và điều đó là đúng
------------------------------------------------
`normalize_input` vừa kiểm tra vừa chuẩn hoá trong một lượt, vì hai việc đó
dùng chung một bản đọc dữ liệu. Tách ra thành ba hàm để sơ đồ nhìn cho cân chỉ
làm cùng một dữ liệu bị duyệt ba lần. Sơ đồ vẫn ghi ba bước vì đó là ba việc;
mã gộp lại vì đó là một lượt đọc.

Bước "LLM" và bước "Output Validation" cũng vậy: `client.generate` kiểm ngay
sau khi gọi để còn SINH LẠI với hint đúng lỗi. Tách kiểm ra sau thì mất khả
năng sinh lại — thứ đang cứu được phần lớn câu trả lời trượt lần đầu.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from hfml.inference.result import ERROR, INFO, WARNING, StageResult
from hfml.inference.settings import SETTINGS
from hfml.logger import get_logger

log = get_logger(__name__)

#: Tên các bước, đúng thứ tự sơ đồ. Dùng làm khoá trong `trace`.
INPUT = "input"
VALIDATION = "validation"
PREPROCESSING = "preprocessing"
RULE_ENGINE = "rule_engine"
ML01_STAGE = "ml01"
ML02_STAGE = "ml02"
AGGREGATION = "aggregation"
INTENT = "intent"
CONTEXT = "context"
LLM = "llm"
OUTPUT_VALIDATION = "output_validation"
RESPONSE = "response"


@dataclass
class PipelineState:
    """Thứ được truyền dọc pipeline. Mỗi bước đọc phần mình cần, ghi phần mình ra.

    Một đối tượng trạng thái thay vì chuỗi tham số dài dần: thêm một bước mới
    không kéo theo sửa chữ ký của mọi bước phía sau.
    """

    payload: dict = field(default_factory=dict)
    question: str = ""
    intent_code: str | None = None
    history: list[dict] = field(default_factory=list)
    previous_intent: str | None = None

    # Kết quả từng bước ghi lại đây.
    normalized: Any = None
    analysis: Any = None          # AiResult
    understanding: Any = None
    context: Any = None
    answer: Any = None


def timed(stage: str) -> Callable:
    """Bọc một bước: đo thời gian và **bắt mọi ngoại lệ tại biên bước đó**.

    Đây là hiện thực của yêu cầu "không làm crash toàn bộ pipeline khi một
    thành phần lỗi". Không có lớp bọc này thì một lỗi lập trình trong
    `build_context` bay thẳng lên biên service thành 500, kéo theo cả phần rule
    và ML đã tính xong — thứ người dùng lẽ ra vẫn đọc được.
    """
    def decorate(func: Callable[[PipelineState], StageResult]):
        def run(state: PipelineState) -> StageResult:
            started = time.perf_counter()
            try:
                result = func(state)
            except Exception as exc:  # noqa: BLE001 — biên của một bước
                log.exception("Bước %s lỗi", stage)
                result = StageResult.failed(
                    stage, f"{stage}_error",
                    f"Bước {stage} lỗi: {type(exc).__name__}: {exc}")
            result.elapsed_ms = (time.perf_counter() - started) * 1000.0
            return result
        run.__name__ = func.__name__
        run.__doc__ = func.__doc__
        return run
    return decorate


# --------------------------------------------------------------------------
# Input · Validation · Preprocessing
# --------------------------------------------------------------------------
@timed(VALIDATION)
def stage_normalize(state: PipelineState) -> StageResult:
    """Đọc payload thô → hồ sơ đã kiểm và chuẩn hoá.

    Gộp ba bước đầu của sơ đồ — xem docstring đầu file về lý do.
    """
    from hfml.inference.payloads import normalize_payload
    from hfml.pipeline.normalizer import normalize_input

    # Quy đổi tên trường của client về schema chuẩn TRƯỚC khi kiểm tra.
    # Không có bước này thì payload dạng Laravel cho ra một loạt lỗi
    # `Field required`, và người dùng bị báo thiếu dữ liệu dù đã điền đủ.
    #
    # GHI ĐÈ `state.payload` bằng bản đã quy đổi, không giữ bản thô. Bước
    # `stage_analyze` phía sau đọc lại `state.payload`, nên nếu để nguyên bản
    # thô thì quy đổi chỉ có tác dụng ở nửa đầu pipeline: bước này báo hồ sơ
    # hợp lệ, bước sau nhận đúng payload đó và báo thiếu trường.
    state.payload = normalize_payload(state.payload)
    normalized = normalize_input(state.payload)
    state.normalized = normalized

    result = StageResult(stage=VALIDATION, data=normalized)
    for issue in normalized.warnings:
        result.add(getattr(issue, "code", "input_warning"), issue.message,
                   WARNING, getattr(issue, "field", ""))

    if not normalized.is_valid:
        # Ca DUY NHẤT làm dừng cả pipeline: không có gì hợp lệ để tính tiếp.
        for issue in normalized.errors:
            result.add("invalid_input", issue.message, ERROR,
                       getattr(issue, "field", ""))
    return result


# --------------------------------------------------------------------------
# Rule-Based · ML01 · ML02 · Aggregation
# --------------------------------------------------------------------------
@timed(AGGREGATION)
def stage_analyze(state: PipelineState) -> StageResult:
    """Chạy 5 rule + hai model rồi gom thành `AiResult`.

    Uỷ quyền trọn vẹn cho `orchestrator.analyze` — nơi đã cài sẵn quy tắc chịu
    lỗi từng phần (rule hỏng không kéo theo ML, và ngược lại) cùng với việc gắn
    cờ low_confidence. Gọi lẻ từng phần ở đây là viết lại logic đó lần thứ hai.
    """
    from hfml.pipeline.orchestrator import analyze

    analysis = analyze(state.payload)
    state.analysis = analysis

    result = StageResult(stage=AGGREGATION, data=analysis, ok=analysis.ok)
    for item in analysis.errors:
        result.add(item.get("code", "analysis_error"), item.get("message", ""),
                   ERROR, item.get("field", ""))
    for item in analysis.warnings:
        result.add(item.get("code", "analysis_warning"), item.get("message", ""),
                   WARNING, item.get("field", ""))
    return result


# --------------------------------------------------------------------------
# Intent
# --------------------------------------------------------------------------
@timed(INTENT)
def stage_intent(state: PipelineState) -> StageResult:
    """Xác định intent và kiểm xem có đủ dữ liệu để trả lời không.

    `intent_code` từ chip THẮNG suy đoán theo từ khoá — ràng buộc nghiệp vụ đã
    chốt cho hai intent ML.
    """
    from hfml.api.intents import IntentCode
    from hfml.llm.chat import _inherit_intent
    from hfml.llm.understanding import understand

    analysis = state.analysis.to_dict() if state.analysis else {}
    understanding = understand(state.question, analysis, state.intent_code)

    # Câu nối tiếp ("thế còn 2 tỷ?") kế thừa intent của lượt trước.
    inherited = _inherit_intent(state.question, understanding.intent,
                                state.previous_intent)
    if inherited:
        understanding = understand(state.question, analysis, inherited)

    state.understanding = understanding

    result = StageResult(stage=INTENT, data=understanding)
    if not understanding.can_answer:
        # Thiếu dữ liệu là chuyện bình thường, KHÔNG phải lỗi: hệ thống hỏi
        # xin thêm rồi đi tiếp. Đánh dấu `error` ở đây sẽ làm cả request bị
        # coi là hỏng chỉ vì người dùng chưa điền màn khoản vay.
        for requirement in understanding.missing:
            result.add("missing_data", requirement.label, WARNING,
                       requirement.path)
    return result


# --------------------------------------------------------------------------
# Context
# --------------------------------------------------------------------------
@timed(CONTEXT)
def stage_context(state: PipelineState) -> StageResult:
    """Lọc theo intent và niêm yết mọi con số LLM được phép dùng."""
    from hfml.llm.context import build_context

    context = build_context(
        state.question,
        state.analysis.to_dict() if state.analysis else {},
        state.understanding,
        state.history)
    state.context = context

    result = StageResult(stage=CONTEXT, data=context)
    if not context.numeric_facts:
        result.add(
            "empty_numeric_facts",
            "Không có con số nào được phép dùng — câu trả lời sẽ không dẫn "
            "được số liệu nào.", WARNING)
    return result


# --------------------------------------------------------------------------
# LLM · Output Validation
# --------------------------------------------------------------------------
@timed(LLM)
def stage_generate(state: PipelineState) -> StageResult:
    """Sinh câu trả lời rồi kiểm ngay — hai việc trong một bước.

    Kiểm nằm trong `client.generate` để còn sinh lại với hint đúng lỗi; xem
    docstring đầu file.
    """
    from hfml.llm import client

    if not SETTINGS.llm_enabled:
        # Tắt LLM bằng cấu hình vẫn phải ra câu trả lời dùng được, chỉ là bản
        # dựng từ template — không được biến thành lỗi.
        answer = client._template_answer(state.context)
        answer.validation = {"valid": None, "issues": [], "ungrounded_numbers": [],
                             "note": "tầng LLM đang tắt theo cấu hình"}
        state.answer = answer
        return StageResult(stage=LLM, data=answer).add(
            "llm_disabled", "Tầng LLM đang tắt theo cấu hình.", INFO)

    answer = client.generate(state.context, state.understanding)
    state.answer = answer

    result = StageResult(stage=LLM, data=answer)
    check = answer.validation or {}
    valid = check.get("valid")

    # Ba trạng thái, không phải hai — xem `client.generate`.
    if valid is None and check.get("note"):
        result.add("llm_unreachable", check["note"], WARNING)
    elif valid is False:
        for issue in check.get("issues", []):
            result.add(f"output_{issue.get('check', 'invalid')}",
                       issue.get("message", ""), WARNING)
        if check.get("ungrounded_numbers"):
            result.add(
                "ungrounded_numbers",
                "Câu trả lời chứa số không có căn cứ: "
                + ", ".join(check["ungrounded_numbers"]) + ". Đã hạ cấp về "
                "câu trả lời dựng từ dữ liệu đã tính.", WARNING)
    return result
