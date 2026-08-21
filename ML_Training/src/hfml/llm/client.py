"""AI-02 task 4, 5 — Sinh giải thích và khuyến nghị (F05 · M06).

Gọi Gemini với context đã dựng, kiểm câu trả lời, và **hạ cấp về template khi
không đạt**. Ba chế độ, theo thứ tự ưu tiên:

    LLM        có API key và câu trả lời qua được `validator`
    SINH LẠI   lần đầu không đạt → nhắc lại lỗi cụ thể, gọi thêm MỘT lần
    TEMPLATE   không có key, LLM lỗi, hoặc cả hai lần đều không đạt

Template không phải phương án chữa cháy
-----------------------------------------
`narrator.py` sinh câu trả lời từ chính `AiResult` bằng chuỗi f-string, nên
theo cấu trúc nó **không thể** bịa số. Đó là lý do nó là đích hạ cấp: khi
không tin được câu chữ của LLM, thứ thay thế phải là thứ không cần tin.

Chỉ sinh lại MỘT lần
---------------------
Lần hai được nhắc đúng lỗi đã mắc. Không đạt tiếp thì dừng — vòng lặp sinh lại
nhiều lần vừa tốn thời gian chờ của người dùng vừa hiếm khi cứu được: nếu model
đã bịa số hai lần liên tiếp trên cùng một context thì lần ba cũng vậy.
"""
from __future__ import annotations

import functools
import json
from dataclasses import dataclass, field
from typing import Any, Final

from hfml.config import CONFIG
from hfml.llm import guardrails, prompts
from hfml.llm.narrator import DISCLAIMER, LOAN_RISK_DISCLAIMER
from hfml.llm.understanding import TOPIC_LABELS
from hfml.llm.validator import ValidationReport, validate
from hfml.logger import get_logger

log = get_logger(__name__)

#: Số lần gọi lại tối đa khi câu trả lời không qua kiểm — xem docstring.
MAX_RETRIES: Final[int] = 1

#: Nguồn của câu trả lời cuối cùng. Đi kèm mọi kết quả để về sau còn truy được.
SOURCE_LLM: Final[str] = "llm"
SOURCE_LLM_RETRY: Final[str] = "llm_retry"
SOURCE_TEMPLATE: Final[str] = "template"
SOURCE_OUT_OF_SCOPE: Final[str] = "out_of_scope"


@dataclass
class Answer:
    """Câu trả lời cuối cùng, đã qua kiểm."""

    explanation: str
    recommendations: list[dict] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    needs_more_data: list[str] = field(default_factory=list)
    suggested_questions: list[str] = field(default_factory=list)
    source: str = SOURCE_TEMPLATE
    prompt_version: str = prompts.PROMPT_VERSION
    model: str = ""
    validation: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "explanation": self.explanation,
            "recommendations": self.recommendations,
            "caveats": self.caveats,
            "needs_more_data": self.needs_more_data,
            "suggested_questions": self.suggested_questions,
            "source": self.source,
            "prompt_version": self.prompt_version,
            "model": self.model,
            "validation": self.validation,
        }

    def as_text(self) -> str:
        """Gộp thành một đoạn văn cho tầng hiển thị."""
        parts = [self.explanation]
        if self.recommendations:
            parts.append("\n**Việc nên làm:**")
            order = {"high": "🔴", "medium": "🟠", "low": "🟢"}
            for item in self.recommendations:
                mark = order.get(str(item.get("priority", "")).lower(), "·")
                parts.append(f"{mark} {item.get('action', '')}"
                             + (f" — {item['reason']}" if item.get("reason") else ""))
        if self.caveats:
            parts.append("\n**Lưu ý:**")
            parts += [f"- {c}" for c in self.caveats]
        if self.needs_more_data:
            parts.append("\n**Cần bổ sung:**")
            parts += [f"- {d}" for d in self.needs_more_data]
        return "\n".join(parts)


# --------------------------------------------------------------------------
# Gọi Gemini
# --------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def _client():
    """Client Gemini, nạp một lần. `None` khi chưa cấu hình API key."""
    if not CONFIG.llm_api_key:
        log.info("Chưa có GEMINI_API_KEY — tầng LLM chạy chế độ template.")
        return None
    try:
        from google import genai
    except ImportError:
        log.warning("Chưa cài google-genai — chạy chế độ template.")
        return None
    return genai.Client(api_key=CONFIG.llm_api_key)


def is_llm_available() -> bool:
    return _client() is not None


def _call(system: str, user: str) -> dict | None:
    """Một lượt gọi Gemini, trả về JSON đã parse. `None` khi hỏng.

    Ép `response_mime_type='application/json'` thay vì xin lịch sự trong
    prompt: model trả văn xuôi thì `validator` phải đoán đâu là khuyến nghị,
    đâu là giải thích, và phép kiểm mất độ chính xác.
    """
    client = _client()
    if client is None:
        return None

    try:
        from google.genai import types

        response = client.models.generate_content(
            model=CONFIG.llm["model"],
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=CONFIG.llm["temperature"],
                max_output_tokens=CONFIG.llm["max_tokens"],
                response_mime_type="application/json",
            ),
        )

        # Bị cắt vì hết hạn mức token là một ca RIÊNG, phải nói rõ.
        #
        # JSON cắt giữa chừng sẽ ném `JSONDecodeError` và rơi vào nhánh chung
        # bên dưới — khi đó log chỉ báo "JSON hỏng" và người vận hành sẽ đi
        # tìm lỗi ở prompt, trong khi thứ cần sửa là `max_tokens`. Đã mất một
        # lượt gỡ lỗi đúng vì chuyện này: model dùng ~2150 token suy luận cộng
        # ~640 token đầu ra, mà hạn mức đang đặt 1500.
        finish = response.candidates[0].finish_reason
        if finish is not None and finish.name == "MAX_TOKENS":
            usage = response.usage_metadata
            log.warning(
                "LLM bị cắt vì hết hạn mức token (đầu ra %s, suy luận %s, "
                "hạn mức %s). Tăng `llm.max_tokens` trong config.yaml.",
                usage.candidates_token_count,
                getattr(usage, "thoughts_token_count", "?"),
                CONFIG.llm["max_tokens"])
            return None

        return json.loads(response.text)
    except json.JSONDecodeError as exc:
        log.warning("LLM trả về JSON hỏng: %s", exc)
    except Exception as exc:  # noqa: BLE001 — biên ngoài, mạng/quota/model
        log.warning("Gọi LLM lỗi: %s: %s", type(exc).__name__, exc)
    return None


def _retry_hint(report: ValidationReport) -> str:
    """Phần nhắc thêm cho lần gọi thứ hai, nêu ĐÚNG lỗi đã mắc.

    Nhắc chung chung ("hãy cẩn thận hơn") gần như không đổi được gì; nêu đích
    danh con số bị bịa thì model có cái để sửa.
    """
    lines = ["", "LẦN TRƯỚC BẠN ĐÃ SAI Ở NHỮNG ĐIỂM SAU — SỬA LẠI:"]
    for issue in report.errors:
        lines.append(f"  · {issue.message}")
    if report.ungrounded_numbers:
        lines.append(
            "  · Những con số trên KHÔNG có trong dữ liệu được cấp. Chỉ dùng "
            "số trong mục 'CÁC CON SỐ BẠN ĐƯỢC PHÉP DÙNG'.")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Đích hạ cấp — template
# --------------------------------------------------------------------------
def _template_answer(context) -> Answer:
    """Câu trả lời dựng từ chính `AiResult` bằng f-string.

    Theo cấu trúc nó không thể bịa số: mọi con số đều lấy trực tiếp từ
    context. Đó là lý do nó là đích hạ cấp an toàn.
    """
    parts: list[str] = []
    recommendations: list[dict] = []
    caveats: list[str] = []

    ml01 = context.ml01 or {}
    if ml01.get("available"):
        parts.append(
            f"Nhóm định hướng tài chính của gia đình bạn: **{ml01['label_vi']}** "
            f"(mức tin cậy {ml01['probability']:.1%}).")
        if ml01.get("confidence", {}).get("low_confidence"):
            caveats.append(
                "Hồ sơ nằm gần ranh giới giữa các nhóm nên kết quả này chưa "
                "chắc chắn — nên đọc kèm phần đánh giá theo quy tắc.")

    ml02 = context.ml02 or {}
    if ml02.get("available"):
        parts.append(
            f"Mức rủi ro của khoản vay: **{ml02['label_vi']}** — xác suất gặp "
            f"khó khăn trả nợ ước tính {ml02['probability']:.1%}.")
        caveats.append(LOAN_RISK_DISCLAIMER)

    for code, rule in (context.rules or {}).items():
        # Câu tóm tắt của rule nằm ở `details.summary_vi` — KHÔNG phải
        # `message_vi`. Đọc nhầm khoá thì `.get()` trả None, vòng lặp bỏ qua
        # lặng lẽ, và với intent không có phần ML (ví dụ hỏi 50/30/20) câu trả
        # lời hạ cấp rỗng trơn, chỉ còn mỗi dòng miễn trừ. Đó là kiểu hỏng tệ
        # nhất của một lưới an toàn: nhìn từ ngoài tưởng nó đã đỡ.
        message = (rule.get("details", {}).get("summary_vi")
                   or rule.get("message_vi") or rule.get("message") or "")
        status = rule.get("status", "")
        if message:
            parts.append(f"- {code} ({status}): {message}")
        elif status:
            parts.append(f"- {code}: {status}")

    for warning in context.warnings or []:
        caveats.append(warning.get("message", ""))

    parts.append("")
    parts.append(DISCLAIMER)

    return Answer(
        explanation="\n".join(p for p in parts if p is not None),
        recommendations=recommendations,
        caveats=[c for c in caveats if c],
        source=SOURCE_TEMPLATE,
        model="",
    )


# --------------------------------------------------------------------------
# Điểm vào
# --------------------------------------------------------------------------
def generate(context, understanding) -> Answer:
    """Sinh giải thích + khuyến nghị cho một lượt hỏi.

    Trình tự: che dữ liệu nhạy cảm → gọi LLM → kiểm → sinh lại nếu cần → hạ
    cấp về template nếu vẫn không đạt.
    """
    # Task 7: che danh tính TRƯỚC khi dựng prompt.
    guardrails.apply(context)

    # Thiếu dữ liệu bắt buộc thì HỎI, không gọi LLM với context rỗng.
    if not understanding.can_answer:
        return Answer(
            explanation=(
                "Mình chưa đủ dữ liệu để trả lời câu hỏi này.\n\n"
                f"{understanding.ask_message()}"),
            needs_more_data=[r.label for r in understanding.missing],
            source=SOURCE_TEMPLATE,
        )

    system = prompts.SYSTEM_PROMPT
    user = prompts.render_user_prompt(
        context, TOPIC_LABELS.get(context.topic, context.topic))

    for attempt in range(MAX_RETRIES + 1):
        payload = _call(system, user)
        if payload is None:
            # KHÔNG GỌI ĐƯỢC ≠ GỌI ĐƯỢC NHƯNG BỊ ĐÁNH TRƯỢT.
            #
            # Cả hai đều hạ cấp về template, nhưng nguyên nhân khác hẳn: một
            # bên là mạng/quota/model hỏng, một bên là câu trả lời vi phạm
            # guardrail. Trả cùng một `validation` rỗng thì người vận hành đọc
            # log thấy "kiểm không đạt" và đi sửa prompt, trong khi thứ hỏng
            # là hạn mức API. Đã suýt lạc hướng đúng vì chuyện này khi gặp 429.
            answer = _template_answer(context)
            answer.validation = {
                "valid": None, "issues": [], "ungrounded_numbers": [],
                "note": "không gọi được LLM — chưa có câu trả lời nào để kiểm",
            }
            return answer

        report = validate(payload, context)
        if report.is_valid:
            return Answer(
                explanation=str(payload.get("explanation", "")),
                recommendations=list(payload.get("recommendations") or []),
                caveats=[str(c) for c in (payload.get("caveats") or [])],
                needs_more_data=[str(d) for d in
                                 (payload.get("needs_more_data") or [])],
                source=SOURCE_LLM if attempt == 0 else SOURCE_LLM_RETRY,
                model=CONFIG.llm["model"],
                validation=report.to_dict(),
            )

        log.warning("Câu trả lời lần %d không đạt: %s", attempt + 1,
                    [i.check for i in report.errors])
        if attempt < MAX_RETRIES:
            user = user + _retry_hint(report)
        else:
            # Hạ cấp, nhưng GIỮ LẠI báo cáo kiểm để về sau còn truy được vì sao.
            answer = _template_answer(context)
            answer.validation = report.to_dict()
            return answer

    # Chỉ tới được đây nếu `MAX_RETRIES` bị đặt âm — vòng lặp không chạy lần
    # nào. Giữ lại để hàm luôn trả về một câu trả lời dùng được.
    return _template_answer(context)


def out_of_scope_answer(verdict) -> Answer:
    """Câu trả lời cố định cho câu hỏi ngoài phạm vi — KHÔNG gọi LLM."""
    payload = prompts.OUT_OF_SCOPE_REPLY
    return Answer(
        explanation=payload["explanation"],
        suggested_questions=list(payload["suggested_questions"]),
        source=SOURCE_OUT_OF_SCOPE,
        validation={"valid": True, "issues": [], "ungrounded_numbers": []},
    )
