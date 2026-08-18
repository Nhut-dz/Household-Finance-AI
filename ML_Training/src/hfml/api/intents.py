"""Định tuyến ý định của Chatbot (PLAN.md §8.2 task 12).

Hai đường vào, KHÁC HẲN nhau về độ tin cậy — và đây là toàn bộ lý do file này
tồn tại:

    Hướng 1  Người dùng bấm một chip gợi ý → FE gửi kèm `intent_code`.
             Ý định là ĐÃ BIẾT CHẮC, không phải đoán. Đi thẳng vào engine.
    Hướng 2  Người dùng tự gõ câu hỏi → không có `intent_code`, phải đoán
             bằng từ khoá. Đoán thì có lúc sai.

Vì sao hai intent ML không bao giờ được đoán bằng từ khoá
----------------------------------------------------------
Thử đoán nhãn tiếng Việt của chúng bằng chuỗi `if/elif` cũ thì hỏng ngay ở
cả hai:

    "Chẩn đoán rủi ro vay vốn"       chứa chữ "vay"  → rơi vào nhánh RB05
                                     (hạn mức vay), KHÔNG bao giờ tới ML02
    "Chẩn đoán sức khỏe tài chính"   không chứa từ khoá nào → rơi xuống
                                     nhánh trả lời chung, KHÔNG tới ML01

Cả hai đều vẫn trả lời trôi chảy, chỉ là trả lời bằng nhánh sai — loại lỗi
không ai phát hiện được khi nhìn màn hình. Nên `classify_by_keyword()` được
viết sao cho **không thể** trả về một intent ML: đó là bất biến có test canh
(`test_keyword_classifier_never_returns_an_ml_intent`), không phải quy ước
suông.

Muốn chạy ML thì phải bấm chip, hoặc gọi thẳng `/predict`. Không có đường
vòng nào khác.
"""
from __future__ import annotations

from enum import Enum
from typing import Final


class IntentCode(str, Enum):
    """Ý định của một lượt hỏi.

    Bốn giá trị đầu ứng với đúng bốn chip gợi ý trên màn Chatbot; ba giá trị
    sau chỉ sinh ra từ câu người dùng tự gõ.
    """

    # -- Bốn chip gợi ý --------------------------------------------------
    SAVINGS_PACKAGE = "SAVINGS_PACKAGE"
    FINANCIAL_HEALTH_DIAGNOSIS = "FINANCIAL_HEALTH_DIAGNOSIS"   # → ML01
    LOAN_RISK_DIAGNOSIS = "LOAN_RISK_DIAGNOSIS"                 # → ML02
    BUDGET_50_30_20 = "BUDGET_50_30_20"

    # -- Chỉ đến từ câu tự gõ ---------------------------------------------
    LOAN_CAPACITY = "LOAN_CAPACITY"     # RB05 — "vay được bao nhiêu"
    DEBT = "DEBT"                       # xử lý nợ đang có
    INVESTMENT = "INVESTMENT"           # gói đầu tư (đã rút khỏi chip gợi ý)
    GENERAL = "GENERAL"                 # không nhận ra ý định


#: Hai intent chạy model. Chúng CHỈ được kích hoạt bằng `intent_code` tường
#: minh từ chip gợi ý — xem docstring đầu file.
ML_INTENTS: Final[frozenset[IntentCode]] = frozenset({
    IntentCode.FINANCIAL_HEALTH_DIAGNOSIS,
    IntentCode.LOAN_RISK_DIAGNOSIS,
})

#: Nhãn tiếng Việt của chip, để log và câu trả lời gọi đúng tên chức năng.
INTENT_LABELS: Final[dict[IntentCode, str]] = {
    IntentCode.SAVINGS_PACKAGE: "Gói tiết kiệm",
    IntentCode.FINANCIAL_HEALTH_DIAGNOSIS: "Chẩn đoán sức khỏe tài chính",
    IntentCode.LOAN_RISK_DIAGNOSIS: "Chẩn đoán rủi ro vay vốn",
    IntentCode.BUDGET_50_30_20: "Quy tắc 50/30/20",
    IntentCode.LOAN_CAPACITY: "Khả năng vay vốn",
    IntentCode.DEBT: "Xử lý nợ",
    IntentCode.INVESTMENT: "Phân bổ đầu tư",
    IntentCode.GENERAL: "Tư vấn chung",
}

#: Từ khoá → intent, xét theo ĐÚNG thứ tự này. Thứ tự có ý nghĩa: câu "vay
#: mua nhà 3 tỷ" khớp cả `LOAN_CAPACITY` lẫn `SAVINGS_PACKAGE`, và ý chính
#: của nó là khoản vay.
#:
#: Bảng này CỐ TÌNH không chứa intent nào trong `ML_INTENTS`.
_KEYWORD_RULES: Final[tuple[tuple[IntentCode, tuple[str, ...]], ...]] = (
    (IntentCode.BUDGET_50_30_20, ("50/30/20", "503020", "50 30 20")),
    (IntentCode.LOAN_CAPACITY, ("vay", "mua nhà", "mua xe", "mua đất", "thế chấp")),
    (IntentCode.DEBT, ("trả nợ", "nợ nần", "dư nợ", "giảm nợ", "gộp nợ",
                       "lãi suất cao", "nợ xấu")),
    (IntentCode.INVESTMENT, ("đầu tư", "sinh lời", "chứng chỉ quỹ", "trái phiếu")),
    (IntentCode.SAVINGS_PACKAGE, ("tiết kiệm", "tích lũy", "tích luỹ", "quỹ dự phòng")),
)


def classify_by_keyword(question: str) -> IntentCode:
    """Đoán ý định của câu người dùng TỰ GÕ.

    Không bao giờ trả về một intent trong `ML_INTENTS` — chạy model là việc
    chỉ được kích hoạt tường minh, không phải kết quả của một phép đoán.

    Không nhận ra thì trả `GENERAL`, chứ không đoán liều intent gần nhất:
    trả lời sai chủ đề tệ hơn hẳn trả lời chung chung, vì người dùng không
    nhận ra là mình đã bị hiểu nhầm.
    """
    text = question.lower()

    for intent, keywords in _KEYWORD_RULES:
        if any(keyword in text for keyword in keywords):
            return intent

    return IntentCode.GENERAL


def resolve_intent(question: str, intent_code: str | None) -> IntentCode:
    """Chốt ý định cuối cùng cho một lượt hỏi.

    Có `intent_code` thì tin nó tuyệt đối — đó là Hướng 1, người dùng đã nói
    rõ mình muốn gì bằng cách bấm chip. Không có thì mới đoán.

    `intent_code` lạ (FE cũ, ai đó gọi API tay) bị coi như không có và rơi về
    đoán từ khoá, thay vì ném lỗi: một mã intent sai không đáng để cả câu hỏi
    của người dùng bị từ chối. Nhưng nó cũng không thể vô tình mở đường vào
    ML, vì đường đó chỉ mở khi mã khớp chính xác.
    """
    if intent_code:
        try:
            return IntentCode(intent_code.strip().upper())
        except ValueError:
            pass

    return classify_by_keyword(question)
