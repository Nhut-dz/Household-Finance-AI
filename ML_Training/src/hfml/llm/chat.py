"""AI-02 task 6 — Hội thoại nhiều lượt và điểm vào của tầng LLM (F05 · M06).

`answer()` là điểm vào DUY NHẤT của Epic AI-02. Nó nối bảy bước còn lại thành
một đường:

    guardrail phạm vi → hiểu câu hỏi → dựng context → gọi LLM
    → kiểm → (sinh lại) → hạ cấp nếu cần

Câu hỏi nối tiếp
-----------------
"Thế còn nếu vay 2 tỷ?" không có chủ ngữ, không có chủ đề — hiểu nó cần lượt
trước. Hai cơ chế, đơn giản và tách bạch:

    · `history` đưa vào context để LLM đọc được mạch hội thoại
    · intent của lượt trước được **kế thừa** khi lượt này không nhận ra được

Cơ chế thứ hai quan trọng hơn vẻ ngoài của nó: không kế thừa thì "thế còn nếu
vay 2 tỷ?" rơi vào `GENERAL`, và context sẽ thiếu đúng phần RB05 mà câu hỏi
cần. LLM khi đó vẫn trả lời — bằng dữ liệu không liên quan.

Kế thừa CÓ ĐIỀU KIỆN, không phải luôn luôn
--------------------------------------------
Chỉ kế thừa khi lượt này ra `GENERAL` và câu hỏi NGẮN (dấu hiệu của câu nối
tiếp). Kế thừa vô điều kiện thì người dùng đổi chủ đề sẽ bị kẹt lại ở chủ đề
cũ — và đó là kiểu hỏng khó chịu hơn nhiều, vì họ hỏi rõ ràng mà vẫn bị hiểu
sai.

Không lặp lại thông tin
------------------------
Prompt đã dặn "không lặp lại số liệu đã nói ở lượt trước trừ khi được hỏi
lại", và lịch sử được cắt còn 3 lượt để LLM còn thấy mình đã nói gì.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final

from hfml.api.intents import IntentCode
from hfml.llm import client, guardrails
from hfml.llm.context import build_context
from hfml.llm.understanding import understand
from hfml.logger import get_logger

log = get_logger(__name__)

#: Câu ngắn hơn ngần này ký tự được coi là có thể nối tiếp lượt trước.
#:
#: "Thế còn 2 tỷ?" (14) nối tiếp; "Tôi muốn biết cách xây quỹ dự phòng cho
#: gia đình 4 người" (56) là câu hỏi mới đủ nghĩa.
FOLLOW_UP_CHAR_LIMIT: Final[int] = 60


@dataclass
class ChatTurn:
    """Một lượt hỏi đáp hoàn chỉnh."""

    question: str
    answer: client.Answer
    intent: str
    topic: str
    understanding: dict = field(default_factory=dict)
    context_summary: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "intent": self.intent,
            "topic": self.topic,
            "answer": self.answer.to_dict(),
            "text": self.answer.as_text(),
            "understanding": self.understanding,
            "context_summary": self.context_summary,
        }


def _looks_like_follow_up(question: str) -> bool:
    """Câu này có vẻ nối tiếp lượt trước không.

    Chỉ dựa vào độ dài và vài từ nối — cố ý thô sơ. Một bộ nhận diện tinh vi
    hơn sẽ đoán sai theo những cách khó lường, mà cái giá của đoán sai ở đây
    là trả lời đúng chủ đề cũ cho một câu hỏi mới.
    """
    text = question.strip().lower()
    if len(text) > FOLLOW_UP_CHAR_LIMIT:
        return False
    return True


def _inherit_intent(
    question: str,
    current: IntentCode,
    previous: str | None,
) -> str | None:
    """Có nên dùng lại intent của lượt trước không.

    Trả về `intent_code` để truyền lại cho `understand()`, hoặc `None`.
    """
    if previous is None or current is not IntentCode.GENERAL:
        return None
    if not _looks_like_follow_up(question):
        return None
    try:
        inherited = IntentCode(previous)
    except ValueError:
        return None
    if inherited is IntentCode.GENERAL:
        return None

    log.info("Câu nối tiếp — kế thừa intent %s của lượt trước", inherited.value)
    return inherited.value


def answer(
    question: str,
    result: dict,
    intent_code: str | None = None,
    history: list[dict] | None = None,
    previous_intent: str | None = None,
) -> ChatTurn:
    """Trả lời một lượt hỏi. Điểm vào duy nhất của Epic AI-02.

    `result` là `AiResult.to_dict()` của Epic AI-01 — tầng này KHÔNG tự chạy
    rule hay model, nó chỉ diễn đạt thứ đã được tính.

    `previous_intent` là mã intent của lượt trước, dùng cho câu hỏi nối tiếp.
    """
    # -- Task 7: chặn ngoài phạm vi TRƯỚC khi gọi LLM ---------------------
    verdict = guardrails.check_scope(question)
    if not verdict.allowed:
        return ChatTurn(
            question=question,
            answer=client.out_of_scope_answer(verdict),
            intent=IntentCode.GENERAL.value,
            topic="general",
            understanding={"blocked": True, **verdict.to_dict()},
        )

    # -- Task 1: hiểu câu hỏi ---------------------------------------------
    understanding = understand(question, result, intent_code)

    # -- Task 6: câu nối tiếp thì kế thừa intent của lượt trước ------------
    inherited = _inherit_intent(question, understanding.intent, previous_intent)
    if inherited:
        understanding = understand(question, result, inherited)

    # -- Task 2: dựng context ---------------------------------------------
    context = build_context(question, result, understanding, history)

    # -- Task 4, 5, 8: sinh — kiểm — hạ cấp -------------------------------
    generated = client.generate(context, understanding)

    # Người dùng đòi một lời cam kết → thêm lời nhắc, không chặn cả câu hỏi.
    overreach = guardrails.detect_overreach(question)
    if overreach:
        generated.caveats.append(
            "Hệ thống không cam kết được mức lợi nhuận hay kết quả chắc chắn "
            "nào — mọi con số ở trên là ước lượng tham khảo.")

    return ChatTurn(
        question=question,
        answer=generated,
        intent=understanding.intent.value,
        topic=understanding.topic,
        understanding=understanding.to_dict(),
        context_summary={
            "n_rules": len(context.rules),
            "n_numeric_facts": len(context.numeric_facts),
            "n_history_turns": len(context.history),
            "has_ml01": bool(context.ml01.get("available")),
            "has_ml02": bool(context.ml02.get("available")),
            "low_confidence": context.has_low_confidence,
        },
    )
