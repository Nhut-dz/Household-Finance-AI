"""Test định tuyến ý định Chatbot (`hfml.api.intents`).

Bài test quan trọng nhất file này là
`test_keyword_classifier_never_returns_an_ml_intent`. Nó canh một bất biến mà
nếu vỡ thì hệ thống KHÔNG báo lỗi: câu hỏi vẫn được trả lời, chỉ là bằng nhánh
sai. Loại hỏng đó không nhìn màn hình mà thấy được.
"""
from __future__ import annotations

import pytest

from hfml.api.intents import (
    INTENT_LABELS,
    ML_INTENTS,
    IntentCode,
    classify_by_keyword,
    resolve_intent,
)

#: Nhãn tiếng Việt in trên bốn chip gợi ý, kèm intent mà chip đó phải chạy.
CHIP_LABELS = {
    "Gói tiết kiệm": IntentCode.SAVINGS_PACKAGE,
    "Chẩn đoán sức khỏe tài chính": IntentCode.FINANCIAL_HEALTH_DIAGNOSIS,
    "Chẩn đoán rủi ro vay vốn": IntentCode.LOAN_RISK_DIAGNOSIS,
    "Quy tắc 50/30/20": IntentCode.BUDGET_50_30_20,
}


# ------------------------------------------------- bất biến của đường từ khoá
def test_keyword_classifier_never_returns_an_ml_intent():
    """Đoán từ khoá KHÔNG BAO GIỜ được mở đường vào model.

    Chạy model phải là hành động tường minh của người dùng (bấm chip), không
    phải kết quả của một phép đoán chuỗi. Quét cả những câu cố tình mô tả đúng
    hai chức năng đó.
    """
    tempting = [
        "chẩn đoán sức khỏe tài chính giúp tôi",
        "chẩn đoán rủi ro vay vốn",
        "financial health diagnosis",
        "đánh giá rủi ro tín dụng của tôi",
        "tôi có nguy cơ vỡ nợ không",
        "phân loại nhóm tài chính của gia đình tôi",
        "chạy mô hình ML01 đi",
        "dùng ML02 để chấm điểm hồ sơ",
    ]

    for question in tempting:
        assert classify_by_keyword(question) not in ML_INTENTS, question


def test_ml_intents_are_absent_from_the_keyword_table():
    """Chặn ngay từ cấu hình, không chỉ chặn ở kết quả.

    Test trên chỉ quét được những câu nghĩ ra được. Test này canh chính bảng
    luật: thêm từ khoá cho một intent ML là hỏng ở đây, kể cả khi chưa ai nghĩ
    ra câu hỏi kích hoạt được nó.
    """
    from hfml.api.intents import _KEYWORD_RULES

    for intent, _ in _KEYWORD_RULES:
        assert intent not in ML_INTENTS


def test_the_two_chip_labels_would_be_routed_wrong_by_keyword_alone():
    """Ghi lại LÝ DO phải có `intent_code`, bằng số đo chứ không bằng lời.

    Đây chính là hai ca hỏng đã thấy khi đọc lại luồng cũ:

        "Chẩn đoán rủi ro vay vốn"      chứa "vay"    → LOAN_CAPACITY (RB05)
        "Chẩn đoán sức khỏe tài chính"  không từ khoá → GENERAL

    Nếu một ngày test này fail vì hai câu đó đã đoán đúng, thì phải xem lại
    `classify_by_keyword` — nhiều khả năng ai đó vừa mở đường tắt vào ML.
    """
    assert classify_by_keyword("Chẩn đoán rủi ro vay vốn") is IntentCode.LOAN_CAPACITY
    assert classify_by_keyword("Chẩn đoán sức khỏe tài chính") is IntentCode.GENERAL


# ----------------------------------------------------- đường chip (Hướng 1)
@pytest.mark.parametrize(("label", "intent"), list(CHIP_LABELS.items()))
def test_chip_intent_wins_over_any_keyword_in_its_label(label: str, intent: IntentCode):
    """Có `intent_code` thì tin tuyệt đối, kể cả khi nhãn chứa từ khoá gây nhiễu."""
    assert resolve_intent(label, intent.value) is intent


def test_chip_intent_wins_even_when_the_text_says_something_else():
    """Mã intent thắng nội dung câu chữ — nội dung chỉ là nhãn nút, không phải ý định."""
    assert resolve_intent(
        "tôi muốn hỏi về đầu tư chứng chỉ quỹ",
        IntentCode.FINANCIAL_HEALTH_DIAGNOSIS.value,
    ) is IntentCode.FINANCIAL_HEALTH_DIAGNOSIS


def test_intent_code_is_case_and_space_insensitive():
    """FE gửi thừa khoảng trắng hay viết thường không được làm hỏng định tuyến."""
    assert resolve_intent("", "  financial_health_diagnosis  ") \
        is IntentCode.FINANCIAL_HEALTH_DIAGNOSIS


def test_unknown_intent_code_falls_back_to_keyword_instead_of_failing():
    """Mã lạ (FE cũ, gọi API tay) không được làm hỏng cả câu hỏi.

    Nhưng nó cũng KHÔNG được vô tình mở đường vào ML — rơi về đoán từ khoá,
    mà đường đó theo định nghĩa không dẫn tới model.
    """
    intent = resolve_intent("tôi muốn tiết kiệm", "KHONG_TON_TAI")

    assert intent is IntentCode.SAVINGS_PACKAGE
    assert intent not in ML_INTENTS


def test_empty_intent_code_is_treated_as_free_text():
    """Chuỗi rỗng = không có mã, không phải một mã hợp lệ tên là rỗng."""
    assert resolve_intent("quy tắc 50/30/20", "") is IntentCode.BUDGET_50_30_20
    assert resolve_intent("quy tắc 50/30/20", None) is IntentCode.BUDGET_50_30_20


# ----------------------------------------------------- đường tự gõ (Hướng 2)
@pytest.mark.parametrize(("question", "expected"), [
    ("Tôi muốn mua nhà giá 3 tỷ thì vay được bao nhiêu?", IntentCode.LOAN_CAPACITY),
    ("thế chấp sổ đỏ được bao nhiêu", IntentCode.LOAN_CAPACITY),
    ("tư vấn gói đầu tư an toàn", IntentCode.INVESTMENT),
    ("nên mua trái phiếu hay chứng chỉ quỹ", IntentCode.INVESTMENT),
    ("làm sao tích lũy nhanh hơn", IntentCode.SAVINGS_PACKAGE),
    ("quỹ dự phòng cần bao nhiêu tháng", IntentCode.SAVINGS_PACKAGE),
    ("giải thích quy tắc 50/30/20", IntentCode.BUDGET_50_30_20),
    ("503020 là gì", IntentCode.BUDGET_50_30_20),
])
def test_free_text_is_classified_by_keyword(question: str, expected: IntentCode):
    assert classify_by_keyword(question) is expected


def test_loan_keyword_wins_over_savings_in_a_mixed_question():
    """Thứ tự luật có ý nghĩa: câu vừa nhắc vay vừa nhắc tiết kiệm là hỏi về vay."""
    assert classify_by_keyword(
        "tôi đang tiết kiệm, có nên vay mua nhà không") is IntentCode.LOAN_CAPACITY


def test_unrecognised_question_falls_back_to_general_not_a_guess():
    """Không nhận ra thì trả lời chung, KHÔNG đoán liều intent gần nhất.

    Trả lời sai chủ đề tệ hơn trả lời chung chung: người dùng không nhận ra là
    mình vừa bị hiểu nhầm.
    """
    assert classify_by_keyword("hôm nay trời đẹp quá") is IntentCode.GENERAL
    assert classify_by_keyword("") is IntentCode.GENERAL


# ---------------------------------------------------------------- hợp đồng
def test_every_intent_has_a_vietnamese_label():
    """Thiếu nhãn thì câu trả lời in ra mã máy — lỗi lộ ra tận mặt người dùng."""
    for intent in IntentCode:
        assert INTENT_LABELS.get(intent), intent


def test_ml_intents_are_exactly_the_two_model_backed_ones():
    assert ML_INTENTS == {
        IntentCode.FINANCIAL_HEALTH_DIAGNOSIS,
        IntentCode.LOAN_RISK_DIAGNOSIS,
    }


def test_intent_values_are_stable_strings():
    """Giá trị enum là HỢP ĐỒNG với backend và FE, đổi tên là hỏng cả ba nơi."""
    assert [i.value for i in IntentCode] == [
        "SAVINGS_PACKAGE",
        "FINANCIAL_HEALTH_DIAGNOSIS",
        "LOAN_RISK_DIAGNOSIS",
        "BUDGET_50_30_20",
        "LOAN_CAPACITY",
        # Thêm ở Epic AI-02 cùng lúc tại cả ba nơi:
        # Front-end/src/api/messages.ts · Back-end IntentCodeEnum.php · đây.
        "DEBT",
        "INVESTMENT",
        "GENERAL",
    ]
