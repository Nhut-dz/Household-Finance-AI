"""AI-02 task 2 — Xây dựng AI context (F05 · M06).

Gom đúng phần cần thiết của `AiResult` (Epic AI-01) thành một cấu trúc gọn cho
LLM. Hai việc, và việc thứ hai là trụ cột của toàn bộ guardrail:

    1. LỌC       chỉ đưa phần liên quan tới intent vào context
    2. NIÊM YẾT  liệt kê MỌI con số LLM được phép dùng, thành một danh sách

`numeric_facts` — danh sách trắng của những con số hợp lệ
-----------------------------------------------------------
§8.2 guardrail 2 yêu cầu "trích mọi số trong câu trả lời bằng regex, đối chiếu
với JSON đầu vào". Đối chiếu với JSON thô thì rất khó làm cho chặt: số nằm rải
trong cấu trúc lồng nhau, có cái là id, có cái là chỉ số kỹ thuật, và không
biết cái nào LLM được phép nhắc tới.

Nên context mang theo `numeric_facts` — một bảng phẳng `tên → giá trị` liệt kê
**đúng** những con số LLM được nói ra. Prompt nói rõ: chỉ dùng số trong bảng
này. `validator.py` kiểm ngược lại: mọi số trong câu trả lời phải khớp một mục
trong bảng.

Nhờ vậy guardrail trở thành một phép kiểm chính xác, không phải một phép đoán
"số này trông có vẻ có trong dữ liệu".

Lọc theo intent, không đổ hết
------------------------------
Đổ cả `AiResult` vào prompt vừa tốn token vừa làm LLM lạc: hỏi về ngân sách mà
context có cả xác suất vỡ nợ thì câu trả lời dễ lôi thứ không ai hỏi vào. Mỗi
intent khai phần mình cần ở `understanding.REQUIREMENTS`; context lấy theo đó,
cộng một phần lõi luôn có mặt (tổng quan hồ sơ, cảnh báo).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final

from hfml.api.intents import IntentCode
from hfml.llm.understanding import REQUIREMENTS, Understanding
from hfml.logger import get_logger

log = get_logger(__name__)

#: Số lượt hỏi đáp gần nhất đưa vào context.
#:
#: 3 lượt là đủ cho câu hỏi nối tiếp ("thế còn nếu vay 2 tỷ?") mà không kéo
#: theo cả phiên. Lịch sử dài làm LLM lặp lại thông tin đã nói và làm loãng
#: phần dữ liệu thật.
HISTORY_TURNS: Final[int] = 3

#: Cắt bớt tin nhắn cũ — chỉ giữ phần đầu để nhắc chủ đề, không giữ cả bài.
HISTORY_CHAR_LIMIT: Final[int] = 300

#: Phần luôn có trong context, bất kể intent.
CORE_PATHS: Final[tuple[str, ...]] = ("input_summary", "overall_status")


@dataclass
class AiContext:
    """Context đưa cho LLM. Chỉ chứa dữ liệu ĐÃ TÍNH SẴN."""

    question: str
    intent: str
    topic: str
    overall_status: str | None = None
    #: Đánh giá tổng quan đã dịch. Đi kèm `overall_status` chứ không thay nó:
    #: tầng kiểm chứng đối chiếu mã, tầng câu chữ dùng bản dịch.
    overall_status_vi: str = ""
    profile: dict = field(default_factory=dict)
    rules: dict = field(default_factory=dict)
    ml01: dict = field(default_factory=dict)
    ml02: dict = field(default_factory=dict)
    warnings: list[dict] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)
    #: Bảng phẳng mọi con số LLM được phép dùng — xem docstring đầu file.
    numeric_facts: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "intent": self.intent,
            "topic": self.topic,
            "overall_status": self.overall_status,
            "overall_status_vi": self.overall_status_vi,
            "profile": self.profile,
            "rules": self.rules,
            "ml01": self.ml01,
            "ml02": self.ml02,
            "warnings": self.warnings,
            "errors": self.errors,
            "history": self.history,
            "numeric_facts": self.numeric_facts,
        }

    @property
    def has_low_confidence(self) -> bool:
        return any(part.get("confidence", {}).get("low_confidence")
                   for part in (self.ml01, self.ml02) if part.get("available"))


# --------------------------------------------------------------------------
# Niêm yết con số
# --------------------------------------------------------------------------
#: Khoá KHÔNG được đưa vào `numeric_facts`.
#:
#: Chúng là số kỹ thuật, không phải con số nghiệp vụ: nhắc `schema_version`
#: hay `model_version` trong câu tư vấn là vô nghĩa, mà để chúng trong danh
#: sách trắng thì LLM có thể dùng như một con số tài chính.
_SKIP_KEYS: Final[frozenset[str]] = frozenset({
    "schema_version", "model_version", "generated_at", "threshold",
    "n_errors", "n_warnings",
})


def _collect_numbers(node: Any, prefix: str, out: dict[str, float]) -> None:
    """Duyệt đệ quy, gom mọi số thành bảng phẳng `đường.dẫn → giá trị`."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key in _SKIP_KEYS:
                continue
            _collect_numbers(value, f"{prefix}.{key}" if prefix else key, out)
    elif isinstance(node, (list, tuple)):
        for index, value in enumerate(node):
            _collect_numbers(value, f"{prefix}[{index}]", out)
    elif isinstance(node, bool):
        return                      # True/False không phải con số để nhắc tới
    elif isinstance(node, (int, float)):
        out[prefix] = float(node)


def _question_numbers(context: AiContext) -> dict[str, float]:
    """Những con số CHÍNH NGƯỜI DÙNG nói ra, trong câu hỏi và các lượt trước.

    Guardrail sinh ra để chặn LLM **bịa ra dữ kiện**, không phải để cấm nó
    nhắc lại lời người dùng. Không có phần này thì "thế còn nếu vay 2 tỷ?" trở
    thành câu không thể trả lời: LLM buộc phải nhắc lại "2 tỷ" để hỏi cho rõ
    hoặc để đặt giả định, và ngay khi nhắc thì bị đánh là bịa số rồi hạ cấp về
    template. Mọi câu hỏi giả định đều hỏng theo đúng kiểu đó — đã thấy khi
    chạy demo.

    Rủi ro mở ra: LLM có thể dùng số của người dùng như thể đó là số hệ thống
    tính ra. Nhỏ hơn hẳn cái giá phải trả, và đã có hai lớp chặn khác — prompt
    cấm tự tính, và các con số này mang tiền tố `question.` nên phân biệt được
    với `rules.` hay `ml01.` khi đọc lại.
    """
    from hfml.llm.validator import extract_numbers

    sources = [context.question]
    sources += [turn["content"] for turn in context.history
                if turn.get("role") == "user"]

    facts: dict[str, float] = {}
    numbers = [item for text in sources for item in extract_numbers(text)]
    for index, (_token, values) in enumerate(numbers, start=1):
        # `values` là các cách đọc của cùng một chuỗi ("1,234" → 1,234 hoặc
        # 1234). Cho cả hai vào danh sách trắng, vì bản thân người dùng viết ra
        # nó nên cách đọc nào cũng là số của họ.
        for order, value in enumerate(values):
            name = f"question.so_{index}" if order == 0 \
                else f"question.so_{index}_doc_khac"
            facts[name] = value
    return facts


def build_numeric_facts(context: AiContext) -> dict[str, float]:
    """Bảng mọi con số LLM được phép nói ra.

    Gom từ CHÍNH các phần đã lọc vào context, không phải từ `AiResult` đầy đủ.
    Nếu gom từ bản đầy đủ thì LLM được phép nhắc một con số mà nó không hề
    nhìn thấy trong prompt — tức được phép nói về thứ nó không có căn cứ.
    """
    facts: dict[str, float] = {}
    for name, node in (("profile", context.profile), ("rules", context.rules),
                       ("ml01", context.ml01), ("ml02", context.ml02)):
        _collect_numbers(node, name, facts)
    facts.update(_question_numbers(context))
    return facts


# --------------------------------------------------------------------------
# Dựng context
# --------------------------------------------------------------------------
def _relevant_rules(result: dict, intent: IntentCode) -> dict:
    """Chỉ các rule mà intent này cần, cộng RB01 và RB02 làm nền.

    RB01 (dòng tiền) và RB02 (sức khỏe) luôn có mặt vì gần như mọi lời khuyên
    tài chính đều phải đặt trên hai con số đó — nói "nên tiết kiệm thêm" mà
    không biết hộ còn dư bao nhiêu là lời khuyên rỗng.

    Mỗi rule được gắn thêm `name_vi` và `status_vi` — tên nghiệp vụ và trạng
    thái đã dịch. Prompt cấm LLM chép lại mã nội bộ, nhưng cấm không thôi thì
    chưa đủ: nếu trong prompt chỉ có `"status": "CRITICAL"` thì model không có
    từ nào khác để dùng và nó sẽ chép, hoặc tự dịch mỗi lần một kiểu. Đưa sẵn
    bản tiếng Việt vào là biến lệnh cấm thành một lựa chọn dễ hơn.

    Bản sao NÔNG chứ không sửa tại chỗ: `rules[code]` là chính dict nằm trong
    `AiResult`, và nhét khoá trình bày vào đó là để tầng hiển thị đổi cấu trúc
    dữ liệu mà tầng kiểm chứng đang đọc.
    """
    from hfml.llm import presentation

    wanted = {"RB01", "RB02"}
    for requirement in REQUIREMENTS.get(intent, ()):
        if requirement.path.startswith("rules."):
            wanted.add(requirement.path.split(".", 1)[1])

    rules = result.get("rules") or {}
    return {
        code: {
            **rules[code],
            "name_vi": presentation.rule_name(code),
            "status_vi": presentation.label_status(code, rules[code].get("status")),
        }
        for code in sorted(wanted) if code in rules
    }


def _trim_history(history: list[dict] | None) -> list[dict]:
    """Giữ vài lượt gần nhất, cắt ngắn nội dung.

    Lịch sử dài làm LLM lặp lại thứ đã nói và làm loãng phần dữ liệu thật.
    """
    if not history:
        return []
    return [
        {"role": turn.get("role", "user"),
         "content": str(turn.get("content", ""))[:HISTORY_CHAR_LIMIT]}
        for turn in history[-HISTORY_TURNS * 2:]
    ]


def build_context(
    question: str,
    result: dict,
    understanding: Understanding,
    history: list[dict] | None = None,
) -> AiContext:
    """Dựng context cho một lượt hỏi.

    `result` là `AiResult.to_dict()` của AI-01. Không tính thêm gì ở đây — mọi
    con số đều được sao chép nguyên vẹn, vì đó là điều làm cho phép kiểm số ở
    `validator.py` có nghĩa.
    """
    from hfml.llm import presentation

    intent = understanding.intent

    context = AiContext(
        question=question,
        intent=intent.value,
        topic=understanding.topic,
        overall_status=result.get("overall_status"),
        overall_status_vi=presentation.label_status(
            "OVERALL", result.get("overall_status")),
        profile=result.get("input_summary") or {},
        rules=_relevant_rules(result, intent),
        # Chỉ đưa phần ML mà intent thật sự cần. Hỏi về ngân sách mà context
        # có xác suất vỡ nợ thì câu trả lời dễ lôi thứ không ai hỏi vào.
        ml01=(result.get("ml01") or {}) if _needs(intent, "ml01") else {},
        ml02=(result.get("ml02") or {}) if _needs(intent, "ml02") else {},
        warnings=result.get("warnings") or [],
        errors=result.get("errors") or [],
        history=_trim_history(history),
    )
    context.numeric_facts = build_numeric_facts(context)

    log.info("Context: intent=%s · %d rule · %d con số được phép dùng",
             intent.value, len(context.rules), len(context.numeric_facts))
    return context


def _needs(intent: IntentCode, part: str) -> bool:
    """Intent này có cần phần ML đó không."""
    return any(r.path.split(".")[0] == part
               for r in REQUIREMENTS.get(intent, ()))
