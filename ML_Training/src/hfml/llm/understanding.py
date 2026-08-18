"""AI-02 task 1 — Hiểu câu hỏi và xác định dữ liệu cần (F05 · M06).

Trả lời hai câu, theo đúng thứ tự đó:

    1. Người dùng đang hỏi về CHUYỆN GÌ  → intent
    2. Để trả lời được thì CẦN GÌ        → yêu cầu dữ liệu

Câu thứ hai mới là phần dễ bỏ sót. Không có nó thì LLM nhận một context thiếu
và vẫn viết ra một câu trả lời trôi chảy — chỉ là dựa trên khoảng trống.

LLM KHÔNG tham gia bước này
----------------------------
Phân loại intent ở đây là **keyword + rule**, dùng lại `hfml.api.intents` mà
màn Chatbot đã dùng từ 15/08. Hai lý do, cả hai đều không phải chuyện tiết
kiệm chi phí:

1. **Một vốn từ vựng duy nhất.** Chip gợi ý gửi `intent_code` xuống, engine
   định tuyến theo nó. Nếu tầng LLM lại có bảng intent riêng thì cùng một câu
   hỏi có thể được hai tầng hiểu thành hai chuyện khác nhau.
2. **Hai intent chạy model không được đoán.** `FINANCIAL_HEALTH_DIAGNOSIS` và
   `LOAN_RISK_DIAGNOSIS` chỉ kích hoạt bằng chip. Để LLM tự quyết "câu này nên
   chạy ML02" là mở đúng cánh cửa mà `api.intents` đóng lại có chủ ý.

Yêu cầu dữ liệu là RÀNG BUỘC CỨNG, không phải gợi ý
-----------------------------------------------------
Mỗi intent khai rõ nó cần phần nào của `AiResult`. Thiếu phần bắt buộc thì
`DataRequirement.missing` liệt kê ra, và tầng trên phải **hỏi người dùng bổ
sung** thay vì gọi LLM. Đây là chỗ §8.2 guardrail 1 bắt đầu: prompt chỉ được
chứa dữ liệu đã tính sẵn, nên thiếu dữ liệu là thiếu thật, không phải chuyện
LLM "cố gắng suy luận".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

from hfml.api.intents import INTENT_LABELS, IntentCode, resolve_intent
from hfml.logger import get_logger

log = get_logger(__name__)

#: Nhóm chủ đề theo yêu cầu AI-02, ánh xạ từ `IntentCode`.
#:
#: Bảy nhóm của đề bài không trùng 1-1 với tám `IntentCode`: cả
#: `LOAN_CAPACITY` (vay được bao nhiêu) lẫn `LOAN_RISK_DIAGNOSIS` (rủi ro
#: khoản vay) đều thuộc nhóm "Loan". Giữ hai mã riêng vì chúng đi vào hai
#: nhánh engine khác nhau, nhưng gộp khi trình bày chủ đề.
TOPIC_OF: Final[dict[IntentCode, str]] = {
    IntentCode.FINANCIAL_HEALTH_DIAGNOSIS: "financial_health",
    IntentCode.BUDGET_50_30_20: "budgeting",
    IntentCode.SAVINGS_PACKAGE: "saving",
    IntentCode.DEBT: "debt",
    IntentCode.LOAN_CAPACITY: "loan",
    IntentCode.LOAN_RISK_DIAGNOSIS: "loan",
    IntentCode.INVESTMENT: "investment",
    IntentCode.GENERAL: "general",
}

TOPIC_LABELS: Final[dict[str, str]] = {
    "financial_health": "Sức khỏe tài chính",
    "budgeting": "Ngân sách",
    "saving": "Tiết kiệm",
    "debt": "Xử lý nợ",
    "loan": "Vay vốn",
    "investment": "Đầu tư",
    "general": "Tư vấn chung",
}


@dataclass(frozen=True)
class Requirement:
    """Một mẩu dữ liệu mà intent cần.

    `path` trỏ vào `AiResult.to_dict()` theo dạng chấm — `rules.RB02`,
    `ml01.available`. Dùng đường dẫn thay vì tên tự do để kiểm được bằng mã,
    chứ không phải bằng cách đọc tài liệu.
    """

    path: str
    label: str
    #: Thiếu nó thì KHÔNG gọi LLM, phải hỏi người dùng trước.
    required: bool = True
    #: Người dùng cần làm gì để bổ sung — đưa thẳng ra màn hình.
    ask_user: str = ""


#: Dữ liệu từng intent cần. Rule luôn có mặt vì nó chạy được với mọi hồ sơ
#: hợp lệ; chỉ hai intent ML mới cần model tương ứng.
REQUIREMENTS: Final[dict[IntentCode, tuple[Requirement, ...]]] = {
    IntentCode.FINANCIAL_HEALTH_DIAGNOSIS: (
        Requirement("rules.RB02", "Đánh giá sức khỏe tài chính (RB02)"),
        Requirement("rules.RB01", "Dòng tiền (RB01)"),
        Requirement("ml01", "Nhóm khuyến nghị ML01",
                    ask_user="Vui lòng bổ sung năm sinh ở màn Nhập thông tin."),
    ),
    IntentCode.LOAN_RISK_DIAGNOSIS: (
        Requirement("rules.RB05", "Khả năng đáp ứng khoản vay (RB05)"),
        Requirement("ml02", "Ước lượng rủi ro ML02",
                    ask_user="Vui lòng điền màn Thông tin khoản vay."),
    ),
    IntentCode.LOAN_CAPACITY: (
        Requirement("rules.RB05", "Khả năng đáp ứng khoản vay (RB05)"),
        Requirement("rules.RB02", "Tỉ lệ trả nợ trên thu nhập (RB02)"),
    ),
    IntentCode.BUDGET_50_30_20: (
        Requirement("rules.RB04", "Phân bổ 50/30/20 (RB04)"),
        Requirement("rules.RB01", "Dòng tiền (RB01)"),
    ),
    IntentCode.SAVINGS_PACKAGE: (
        Requirement("rules.RB03", "Tiến độ mục tiêu tiết kiệm (RB03)"),
        Requirement("rules.RB01", "Dòng tiền (RB01)"),
        Requirement("rules.RB02", "Quỹ dự phòng (RB02)", required=False),
    ),
    IntentCode.DEBT: (
        Requirement("rules.RB02", "Tỉ lệ trả nợ trên thu nhập (RB02)"),
        Requirement("rules.RB01", "Dòng tiền (RB01)"),
        Requirement("ml01", "Nhóm khuyến nghị ML01", required=False),
    ),
    IntentCode.INVESTMENT: (
        Requirement("rules.RB01", "Dòng tiền (RB01)"),
        Requirement("rules.RB02", "Quỹ dự phòng (RB02)"),
    ),
    IntentCode.GENERAL: (
        Requirement("rules.RB01", "Dòng tiền (RB01)"),
        Requirement("rules.RB02", "Sức khỏe tài chính (RB02)"),
    ),
}


@dataclass
class Understanding:
    """Kết quả hiểu một câu hỏi."""

    intent: IntentCode
    topic: str
    intent_label: str
    from_chip: bool
    missing: list[Requirement] = field(default_factory=list)
    available: list[str] = field(default_factory=list)

    @property
    def can_answer(self) -> bool:
        """Đủ dữ liệu để gọi LLM chưa.

        `False` nghĩa là phải HỎI người dùng, không phải gọi LLM với context
        thiếu — LLM sẽ vẫn viết ra một câu trả lời trôi chảy dựa trên khoảng
        trống, và không có gì trong câu đó để lộ ra điều ấy.
        """
        return not self.missing

    def ask_message(self) -> str:
        """Câu yêu cầu bổ sung dữ liệu, ghép từ các mục còn thiếu."""
        if self.can_answer:
            return ""
        parts = [r.ask_user or f"Thiếu {r.label}." for r in self.missing]
        # Bỏ trùng nhưng giữ thứ tự — hai requirement có thể cùng một hành động.
        seen: list[str] = []
        for part in parts:
            if part not in seen:
                seen.append(part)
        return " ".join(seen)

    def to_dict(self) -> dict:
        return {
            "intent": self.intent.value,
            "topic": self.topic,
            "topic_label": TOPIC_LABELS[self.topic],
            "intent_label": self.intent_label,
            "from_chip": self.from_chip,
            "can_answer": self.can_answer,
            "missing": [{"path": r.path, "label": r.label,
                         "ask_user": r.ask_user} for r in self.missing],
            "available": list(self.available),
        }


def _resolve_path(data: dict, path: str):
    """Lấy giá trị theo đường dẫn chấm. `None` khi không có."""
    current = data
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _is_present(value) -> bool:
    """Một mẩu dữ liệu có dùng được không.

    Phần ML có khoá `available` — nó luôn tồn tại kể cả khi model không chạy
    (AI-01 cố ý giữ đủ khoá), nên chỉ kiểm "khoá có mặt" là chưa đủ: một
    `ml02` với `available: false` vẫn trông như dữ liệu đã có.
    """
    if value is None:
        return False
    if isinstance(value, dict):
        if "available" in value:
            return bool(value["available"])
        return bool(value)
    return True


def understand(
    question: str,
    result: dict,
    intent_code: str | None = None,
) -> Understanding:
    """Hiểu một câu hỏi trong bối cảnh kết quả AI-01.

    `result` là `AiResult.to_dict()` của Epic AI-01. Truyền cả kết quả vào chứ
    không chỉ truyền câu hỏi: "cần dữ liệu gì" chỉ trả lời được khi biết đang
    có sẵn những gì.
    """
    intent = resolve_intent(question, intent_code)
    requirements = REQUIREMENTS.get(intent, REQUIREMENTS[IntentCode.GENERAL])

    missing, available = [], []
    for requirement in requirements:
        if _is_present(_resolve_path(result, requirement.path)):
            available.append(requirement.path)
        elif requirement.required:
            missing.append(requirement)

    understanding = Understanding(
        intent=intent,
        topic=TOPIC_OF[intent],
        intent_label=INTENT_LABELS[intent],
        from_chip=bool(intent_code),
        missing=missing,
        available=available,
    )
    log.info("Hiểu câu hỏi: intent=%s topic=%s đủ_dữ_liệu=%s",
             intent.value, understanding.topic, understanding.can_answer)
    return understanding
