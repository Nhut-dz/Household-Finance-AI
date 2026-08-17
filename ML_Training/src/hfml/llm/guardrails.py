"""AI-02 task 7 — Safety guardrails (F05 · M06).

Guardrail có hai phía, và cả hai đều cần
------------------------------------------
    TRƯỚC khi gọi LLM  chặn câu hỏi ngoài phạm vi, che dữ liệu nhạy cảm
    SAU khi gọi LLM    kiểm câu trả lời — nằm ở `validator.py`

File này lo phía trước. Chặn sớm rẻ hơn và an toàn hơn: một câu hỏi ngoài phạm
vi mà vẫn gửi cho LLM là mời nó nói về chủ đề đó trước khi từ chối, và phần
"nói về" mới là thứ cần tránh.

Che dữ liệu nhạy cảm — hạn chế, không phải xoá
------------------------------------------------
Prompt gửi ra ngoài hệ thống nên nó không nên mang theo thứ không cần cho việc
diễn đạt. Nhưng cắt quá tay thì câu trả lời mất căn cứ.

Ranh giới: **con số tài chính thì CẦN** (không có chúng thì không giải thích
được gì), **danh tính thì KHÔNG**. Tên riêng, nơi ở, mã phiên đều bị bỏ trước
khi dựng prompt — chúng không giúp gì cho việc giải thích một tỉ lệ DTI.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

from hfml.logger import get_logger

log = get_logger(__name__)

#: Trường định danh — bỏ khỏi context trước khi gửi đi.
#:
#: Chúng không đóng góp gì cho việc diễn đạt kết quả tài chính, nên giữ lại
#: chỉ là gửi thừa dữ liệu cá nhân ra ngoài hệ thống.
SENSITIVE_FIELDS: Final[frozenset[str]] = frozenset({
    "representative_name", "residence", "guest_session_id", "session_token",
    "household_id", "user_id", "id",
})

#: Chủ đề rõ ràng nằm ngoài tài chính hộ gia đình.
#:
#: Danh sách này CỐ Ý ngắn và cụ thể. Bộ lọc rộng sẽ chặn nhầm câu hỏi tài
#: chính hợp lệ, mà chặn nhầm thì người dùng không có cách nào diễn đạt lại
#: cho đúng — họ chỉ thấy hệ thống từ chối một câu hỏi bình thường.
OUT_OF_SCOPE_TOPICS: Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
    ("y tế", ("chẩn đoán bệnh", "triệu chứng", "thuốc gì", "khám ở đâu")),
    ("pháp lý", ("kiện tụng", "luật sư", "khởi kiện", "tố cáo")),
    ("chính trị", ("bầu cử", "đảng phái")),
    ("đầu cơ", ("bitcoin", "crypto", "tiền ảo", "forex", "cá độ", "lô đề",
                "đánh bạc")),
    ("mã chứng khoán", ("mã nào sẽ tăng", "nên mua mã", "cổ phiếu nào",
                        "khuyến nghị mã")),
)

#: Yêu cầu người dùng đòi hệ thống làm điều nó không được phép làm.
#:
#: Khác "ngoài phạm vi": đây là câu hỏi ĐÚNG chủ đề tài chính nhưng đòi một
#: lời cam kết. Từ chối phần cam kết, vẫn trả lời phần còn lại.
OVERREACH_PATTERNS: Final[tuple[str, ...]] = (
    "chắc chắn lãi", "đảm bảo lợi nhuận", "cam kết sinh lời",
    "bao nhiêu phần trăm lãi", "lãi bao nhiêu là chắc",
)


@dataclass(frozen=True)
class GuardrailVerdict:
    """Kết luận của phần kiểm trước khi gọi LLM."""

    allowed: bool
    reason: str = ""
    topic: str = ""

    def to_dict(self) -> dict:
        return {"allowed": self.allowed, "reason": self.reason,
                "topic": self.topic}


def check_scope(question: str) -> GuardrailVerdict:
    """Câu hỏi có thuộc phạm vi tài chính hộ gia đình không.

    Chặn ở đây thay vì để LLM tự từ chối: gọi LLM để nó từ chối vẫn tốn một
    lượt gọi, và tệ hơn là nó thường nói vài câu về chủ đề đó trước khi từ
    chối — phần "nói vài câu" chính là thứ cần tránh.
    """
    lowered = question.lower()

    for topic, phrases in OUT_OF_SCOPE_TOPICS:
        hit = next((p for p in phrases if p in lowered), None)
        if hit:
            log.info("Chặn câu hỏi ngoài phạm vi (%s): %r", topic, hit)
            return GuardrailVerdict(
                allowed=False, topic=topic,
                reason=f"Câu hỏi thuộc chủ đề {topic}, ngoài phạm vi tư vấn "
                       "tài chính hộ gia đình.")

    return GuardrailVerdict(allowed=True)


def detect_overreach(question: str) -> str:
    """Người dùng có đang đòi một lời cam kết không.

    Trả về cụm từ bắt được, rỗng nếu không. Không chặn cả câu hỏi — chỉ để
    tầng trên thêm một lời nhắc rằng hệ thống không cam kết được điều đó.
    """
    lowered = question.lower()
    return next((p for p in OVERREACH_PATTERNS if p in lowered), "")


def redact(payload: dict) -> dict:
    """Bỏ trường định danh khỏi context trước khi dựng prompt.

    Đệ quy vì `AiContext` lồng nhiều tầng. Giữ nguyên MỌI con số tài chính —
    không có chúng thì không giải thích được gì, và đó mới là việc của LLM.
    """
    if isinstance(payload, dict):
        return {key: redact(value) for key, value in payload.items()
                if key not in SENSITIVE_FIELDS}
    if isinstance(payload, list):
        return [redact(item) for item in payload]
    return payload


def apply(context) -> None:
    """Che dữ liệu nhạy cảm NGAY TRÊN context, tại chỗ.

    Sửa tại chỗ chứ không trả bản sao: nếu trả bản sao thì chỗ gọi có thể vô
    tình dựng prompt từ bản gốc chưa che, và lỗi đó không lộ ra ở đâu cả.
    """
    context.profile = redact(context.profile)
    context.rules = redact(context.rules)
    context.ml01 = redact(context.ml01)
    context.ml02 = redact(context.ml02)
    # Bảng con số dựng lại sau khi che, để nó không còn chứa mục đã bị bỏ.
    from hfml.llm.context import build_numeric_facts

    context.numeric_facts = build_numeric_facts(context)
