"""AI-03 task 2 — Điều phối toàn bộ pipeline (F05 · M07).

Hai điểm vào, và chỉ hai:

    analyze(payload)              Input → … → Aggregation. Không gọi LLM.
    chat(payload, question, …)    Input → … → Response. Trọn sơ đồ.

Vì sao tách làm hai chứ không một
-----------------------------------
Chúng phục vụ hai nhu cầu khác hẳn nhau về chi phí. `analyze` chạy hết trong
vài chục mili-giây và không tốn quota; `chat` gọi ra mạng ngoài, tốn tiền, và
có thể mất vài giây. Màn hình chỉ cần hiện chỉ số tài chính mà phải trả giá
của một lượt gọi LLM là lãng phí, nên `analyze` phải gọi được riêng.

`chat` KHÔNG cài lại `analyze` — nó gọi đúng hàm đó rồi đi tiếp.

Vì sao điều phối nằm ở đây, không nằm trong FastAPI
-----------------------------------------------------
Module này không import FastAPI, không biết HTTP là gì, và chạy được từ script,
notebook hay test. Trước AI-03, `api/main.py` tự nạp model, tự gọi RuleEngine,
tự ghép câu trả lời — nghĩa là muốn kiểm nghiệp vụ thì phải dựng cả một HTTP
client, và mọi thứ ở đó không dùng lại được ở nơi khác.

Không ném ngoại lệ ra ngoài
-----------------------------
Cả hai hàm luôn trả về `InferenceResult`. Ở biên một service, ngoại lệ nghĩa là
500 và người dùng không biết mình sai chỗ nào; một cấu trúc có `errors` thì nói
được "thiếu trường năm sinh" thay vì "Internal Server Error".
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from hfml.inference import stages
from hfml.inference.lifecycle import MANAGER
from hfml.inference.result import ERROR, InferenceResult, StageResult
from hfml.inference.settings import SETTINGS
from hfml.inference.stages import PipelineState
from hfml.logger import get_logger

log = get_logger(__name__)

#: Phiên bản schema của `InferenceResult`. Tầng tiêu thụ đọc khoá này để biết
#: mình đang nhận cấu trúc nào — đổi shape mà không đổi số là cách chắc chắn
#: nhất để phía dùng hỏng âm thầm.
SCHEMA_VERSION = "1.0"

#: Các bước chạy trước khi cần tới câu hỏi của người dùng.
ANALYSIS_STAGES = (stages.stage_normalize, stages.stage_analyze)

#: Các bước cần câu hỏi.
CHAT_STAGES = (stages.stage_intent, stages.stage_context, stages.stage_generate)


def _new_result() -> InferenceResult:
    return InferenceResult(
        schema_version=SCHEMA_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        settings=SETTINGS.to_dict())


def _record(result: InferenceResult, stage: StageResult) -> StageResult:
    """Gộp kết quả một bước vào kết quả chung."""
    result.trace.append(stage.to_dict())
    result.diagnostics.extend(stage.diagnostics)
    return stage


def _run(result: InferenceResult, state: PipelineState, steps) -> bool:
    """Chạy lần lượt các bước. Trả về False nếu phải dừng sớm.

    Chỉ dừng khi một bước báo `error` — `warning` không làm dừng gì cả. Ranh
    giới đó là điều giữ cho hồ sơ thiếu khoản vay vẫn nhận được phân tích đầy
    đủ về dòng tiền, thay vì bị chặn ngay từ bước đầu.
    """
    for step in steps:
        stage = _record(result, step(state))
        if not stage.ok:
            log.warning("Dừng pipeline tại bước %s: %s", stage.stage,
                        "; ".join(d.message for d in stage.errors))
            return False
    return True


# --------------------------------------------------------------------------
# Điểm vào 1 — chỉ phân tích
# --------------------------------------------------------------------------
def analyze(payload: dict[str, Any]) -> InferenceResult:
    """Chạy Input → Validation → Preprocessing → Rule → ML01 → ML02 → Aggregation.

    Không gọi LLM. Dùng cho màn hình chỉ cần chỉ số, và cho mọi chỗ cần
    structured result mà không cần câu chữ.
    """
    result = _new_result()
    state = PipelineState(payload=payload)

    _run(result, state, ANALYSIS_STAGES)

    if state.analysis is not None:
        result.analysis = state.analysis.to_dict()
    result.ok = not result.errors
    return result


# --------------------------------------------------------------------------
# Điểm vào 2 — trọn pipeline
# --------------------------------------------------------------------------
def chat(
    payload: dict[str, Any],
    question: str,
    intent_code: str | None = None,
    history: list[dict] | None = None,
    previous_intent: str | None = None,
) -> InferenceResult:
    """Chạy trọn sơ đồ: Input → … → LLM → Output Validation → Response.

    `payload` là hồ sơ hộ gia đình, `question` là câu người dùng hỏi. Câu hỏi
    đến từ chip thì truyền `intent_code` — nó THẮNG suy đoán theo từ khoá.
    """
    from hfml.llm import client, guardrails

    result = _new_result()

    # Chặn câu ngoài phạm vi TRƯỚC khi chạy bất cứ thứ gì.
    #
    # Không phải để tiết kiệm: chạy cả pipeline rồi mới từ chối vẫn cho ra câu
    # từ chối đúng. Nhưng nó khiến prompt mang theo toàn bộ hồ sơ tài chính của
    # người dùng cho một câu hỏi về bitcoin — dữ liệu đã gửi đi rồi thì không
    # rút lại được.
    verdict = guardrails.check_scope(question)
    if not verdict.allowed:
        answer = client.out_of_scope_answer(verdict)
        stage = StageResult(stage=stages.RESPONSE, data=answer)
        stage.add("out_of_scope", verdict.reason, "info")
        _record(result, stage)
        return _finish(result, question, answer,
                       intent="GENERAL", topic="general")

    state = PipelineState(
        payload=payload, question=question, intent_code=intent_code,
        history=history or [], previous_intent=previous_intent)

    if not _run(result, state, ANALYSIS_STAGES + CHAT_STAGES):
        # Dừng sớm — vẫn phải trả về thứ đọc được, không phải một vỏ rỗng.
        if state.analysis is not None:
            result.analysis = state.analysis.to_dict()
        result.ok = False
        result.text = _failure_text(result)
        return result

    result.analysis = state.analysis.to_dict()

    answer = state.answer
    # Người dùng đòi một lời cam kết → thêm lời nhắc, không chặn cả câu hỏi.
    overreach = guardrails.detect_overreach(question)
    if overreach and answer is not None:
        answer.caveats.append(
            "Hệ thống không cam kết được mức lợi nhuận hay kết quả chắc chắn "
            "nào — mọi con số ở trên là ước lượng tham khảo.")

    return _finish(result, question, answer,
                   intent=state.understanding.intent.value,
                   topic=state.understanding.topic)


def _finish(result: InferenceResult, question: str, answer,
            intent: str, topic: str) -> InferenceResult:
    """Bước Response — gói lại, không tính thêm gì."""
    if answer is not None:
        result.answer = answer.to_dict()
        result.text = answer.as_text()
    result.intent = intent
    result.topic = topic
    result.ok = not result.errors
    return result


def _failure_text(result: InferenceResult) -> str:
    """Câu nói cho người dùng khi pipeline dừng sớm.

    Nêu đích danh trường sai. "Đã xảy ra lỗi" không cho người dùng cách nào
    sửa; "Thiếu năm sinh" thì họ điền được ngay.
    """
    lines = ["Mình chưa xử lý được hồ sơ này."]
    for item in result.errors:
        lines.append(f"- {item.message}" + (f" ({item.field})" if item.field else ""))
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Trạng thái vận hành
# --------------------------------------------------------------------------
def health() -> dict:
    """Trạng thái module — model nào nạp được, cấu hình đang là gì.

    `ok` chỉ xét model: thiếu API key LLM KHÔNG phải là hỏng, vì pipeline vẫn
    trả lời được bằng câu dựng từ dữ liệu đã tính.
    """
    from hfml.llm import client

    models = MANAGER.status()
    return {
        "ok": all(item.get("loaded") for item in models.values()),
        "schema_version": SCHEMA_VERSION,
        "models": models,
        "llm": {
            "enabled": SETTINGS.llm_enabled,
            "available": client.is_llm_available(),
            "model": SETTINGS.llm.get("model", ""),
        },
        "settings": SETTINGS.to_dict(),
    }
