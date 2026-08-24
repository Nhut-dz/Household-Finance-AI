"""Test định tuyến của `POST /advise` (F05 · màn Chatbot).

Test ở đây đi qua endpoint thật bằng `TestClient` chứ không gọi hàm con: chỗ
hay hỏng không phải logic từng nhánh mà là **câu hỏi vào có tới đúng nhánh
không**, và câu đó chỉ trả lời được ở mức endpoint.

Không test nào cần artifact ML01 trên đĩa. Nhánh ML01 được kiểm bằng cách
thay `get_ml01_model` — mục tiêu là kiểm ĐỊNH TUYẾN và HỢP ĐỒNG, còn chất
lượng dự đoán đã có bộ test riêng của F03.

Không test nào gọi ra mạng ngoài
----------------------------------
Từ 24/08/2026 mọi nhánh của `/advise` kết thúc ở tầng LLM, nên nếu để nguyên
thì cả file này gọi Gemini thật: chậm (đo được 247 giây cho một lượt chạy),
tốn quota, và tệ nhất là KHÔNG TẤT ĐỊNH — câu chữ đổi mỗi lần chạy nên không
có gì để khẳng định.

Fixture `llm_offline` tắt lượt gọi đó cho toàn file. Đây không phải né tránh:
file này kiểm định tuyến, và định tuyến xảy ra TRƯỚC bước diễn giải. Phần "LLM
có thật sự được nối vào không" được kiểm riêng ở cuối file, nơi lượt gọi được
thay bằng một bản trả lời cố định.
"""
from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from hfml.api import main
from hfml.api.intents import IntentCode
from hfml.llm import client as llm_client
from hfml.llm import presentation

HOUSEHOLD = {
    "representative_name": "Nguyễn Văn A",
    "birth_year": 1991,
    "household_size": 4,
    "children_count": 2,
    "supports_elderly": False,
    "monthly_income": 35_000_000,
    "monthly_living_cost": 17_000_000,
    "has_debt": True,
    "total_debt": 500_000_000,
    "monthly_debt_payment": 5_000_000,
    "has_savings": True,
    "current_savings": 150_000_000,
}

ML_FEATURES = {
    "average_monthly_income": 35_000_000.0,
    "average_monthly_expense": 17_000_000.0,
    "savings_amount": 150_000_000.0,
    "total_current_debt": 500_000_000.0,
    "monthly_debt_payment": 5_000_000.0,
    "household_size": 4,
    "children_count": 2,
    "age": 35,
    "has_debt": True,
    "has_savings": True,
    "has_dependents": False,
    "has_asset_cash": True,
    "has_asset_vehicle": False,
    "has_asset_real_estate": True,
    "has_asset_insurance": False,
    "has_asset_gold": False,
    "has_asset_investment": False,
}

LOAN_APPLICATION = {
    "borrower_age": 35,
    "gender": "male",
    "marital_status": "married",
    "children_count": 2,
    "education_level": "higher",
    "occupation": "office_staff",
    "employment_years": 8.5,
    "loan_amount": 1_400_000_000.0,
    "loan_term_months": 240,
    "monthly_payment": 12_000_000.0,
    "asset_price": 2_000_000_000.0,
    "loan_purpose": "buy_house",
    "previous_loan_count": 3,
    "late_payment_count": 1,
    "has_overdue_loan": False,
    "total_overdue_amount": 0.0,
}


@pytest.fixture(autouse=True)
def llm_offline(monkeypatch):
    """Không lượt gọi Gemini nào trong file này — xem docstring đầu file.

    `_call` trả `None` là đúng ca "không gọi được LLM", nên `/advise` hạ cấp
    về bản dựng sẵn của từng nhánh. Đó chính là thứ các test dưới đây khẳng
    định, và cũng là sàn chất lượng mà người dùng thật sẽ nhận khi mạng hỏng.
    """
    monkeypatch.setattr(llm_client, "_call", lambda *_args, **_kwargs: None)


@pytest.fixture
def client() -> TestClient:
    return TestClient(main.app)


def ask(client: TestClient, question: str, **extra) -> dict:
    response = client.post("/advise",
                           json={"question": question, "household": HOUSEHOLD, **extra})
    assert response.status_code == 200, response.text
    return response.json()


class _FakeMl01Model:
    """Model giả trả về nhãn cố định, đủ để kiểm đường đi và hợp đồng."""

    feature_names_ = ()
    classes_ = np.array(["BUILD_BUFFER", "DEBT_FOCUS", "EMERGENCY", "GROWTH"])

    def __init__(self, proba=(0.10, 0.70, 0.05, 0.15)):
        self._proba = np.array([proba])

    def predict(self, frame):
        return np.array([self.classes_[int(self._proba[0].argmax())]])

    def predict_proba(self, frame):
        return self._proba


@pytest.fixture
def fake_ml01(monkeypatch):
    """Thay model ML01 bằng bản giả. Trả về hàm để test tự chọn phân phối."""
    def install(proba=(0.10, 0.70, 0.05, 0.15)):
        model = _FakeMl01Model(proba)
        monkeypatch.setattr(main, "get_ml01_model", lambda: model)
        return model

    return install


# ------------------------------------------------------ chip đi đúng nhánh
def test_chip_intent_is_echoed_back_so_routing_is_verifiable(client):
    """Response phải nói ra nhánh nào đã chạy.

    Không có trường này thì câu "chip có vào đúng engine không" chỉ trả lời
    được bằng cách đọc log server — tức trên thực tế là không ai kiểm.
    """
    body = ask(client, "Gói tiết kiệm", intent_code="SAVINGS_PACKAGE")

    assert body["intent_code"] == IntentCode.SAVINGS_PACKAGE.value


def test_savings_chip_keeps_its_old_answer(client):
    """"Gói tiết kiệm" giữ nguyên logic cũ — nội dung phải là quỹ dự phòng."""
    body = ask(client, "Gói tiết kiệm", intent_code="SAVINGS_PACKAGE")

    assert "Gói tư vấn tiết kiệm" in body["response_text"]
    assert "quỹ dự phòng" in body["response_text"].lower()


def test_budget_chip_keeps_its_old_answer(client):
    """"Quy tắc 50/30/20" giữ nguyên logic cũ."""
    body = ask(client, "Quy tắc 50/30/20", intent_code="BUDGET_50_30_20")

    assert body["intent_code"] == IntentCode.BUDGET_50_30_20.value
    assert "50/30/20" in body["response_text"]


def test_loan_risk_chip_does_not_fall_into_the_rb05_branch(client):
    """Ca hỏng chính mà thay đổi này sinh ra để chặn.

    Nhãn "Chẩn đoán rủi ro vay vốn" chứa chữ "vay". Không có `intent_code` thì
    nó rơi vào nhánh hạn mức vay RB05 và trả lời rất trôi chảy — về đúng một
    chủ đề khác hẳn.
    """
    body = ask(client, "Chẩn đoán rủi ro vay vốn", intent_code="LOAN_RISK_DIAGNOSIS")

    assert body["intent_code"] == IntentCode.LOAN_RISK_DIAGNOSIS.value
    assert "Hạn mức vay an toàn" not in body["response_text"]


def test_health_chip_does_not_fall_into_the_generic_branch(client, fake_ml01):
    """Nhãn "Chẩn đoán sức khỏe tài chính" không chứa từ khoá nào của bảng luật."""
    fake_ml01()

    body = ask(client, "Chẩn đoán sức khỏe tài chính",
               intent_code="FINANCIAL_HEALTH_DIAGNOSIS", ml_features=ML_FEATURES)

    assert body["intent_code"] == IntentCode.FINANCIAL_HEALTH_DIAGNOSIS.value
    assert "Chẩn đoán sức khỏe tài chính" in body["response_text"]


# ------------------------------------------------------------- nhánh ML01
def test_ml01_branch_reports_the_label_the_model_chose(client, fake_ml01):
    """Nhãn trong câu trả lời phải là nhãn model chọn, không phải suy từ rule."""
    fake_ml01(proba=(0.05, 0.80, 0.05, 0.10))     # DEBT_FOCUS thắng

    body = ask(client, "Chẩn đoán sức khỏe tài chính",
               intent_code="FINANCIAL_HEALTH_DIAGNOSIS", ml_features=ML_FEATURES)

    assert "Cần tập trung xử lý nợ" in body["response_text"]
    assert body["model_used"].startswith("HFML-ML01/")


def test_ml01_branch_says_out_loud_when_confidence_is_low(client, fake_ml01):
    """Xác suất cao nhất dưới ngưỡng thì phải NÓI RA, không im lặng (§8.1 task 7)."""
    fake_ml01(proba=(0.28, 0.30, 0.22, 0.20))     # cao nhất 0,30 < 0,60

    body = ask(client, "Chẩn đoán sức khỏe tài chính",
               intent_code="FINANCIAL_HEALTH_DIAGNOSIS", ml_features=ML_FEATURES)

    assert "chưa chắc chắn" in body["response_text"]


def test_ml01_branch_shows_all_four_probabilities(client, fake_ml01):
    """Hiện đủ bốn nhóm: 0,41/0,39 rất khác 0,95/0,02 mà nhìn nhãn thì giống hệt."""
    fake_ml01()

    text = ask(client, "Chẩn đoán sức khỏe tài chính",
               intent_code="FINANCIAL_HEALTH_DIAGNOSIS",
               ml_features=ML_FEATURES)["response_text"]

    for label_vi in ("Cần xử lý khẩn cấp dòng tiền", "Cần tập trung xử lý nợ",
                     "Cần xây dựng quỹ dự phòng", "Có thể hướng tới tăng trưởng"):
        assert label_vi in text


def test_ml01_branch_explains_what_is_missing_instead_of_guessing(client):
    """Thiếu feature (thường vì thiếu năm sinh) thì nói rõ, không điền bừa tuổi."""
    body = ask(client, "Chẩn đoán sức khỏe tài chính",
               intent_code="FINANCIAL_HEALTH_DIAGNOSIS")

    assert "năm sinh" in body["response_text"]
    assert body["intent_code"] == IntentCode.FINANCIAL_HEALTH_DIAGNOSIS.value


def test_ml01_branch_falls_back_to_rules_when_the_artifact_is_missing(client, monkeypatch):
    """Thiếu artifact thì hạ cấp xuống tầng rule, không ném 500 vào mặt người dùng."""
    def missing():
        raise FileNotFoundError("không tìm thấy artifact")

    monkeypatch.setattr(main, "get_ml01_model", missing)

    body = ask(client, "Chẩn đoán sức khỏe tài chính",
               intent_code="FINANCIAL_HEALTH_DIAGNOSIS", ml_features=ML_FEATURES)

    assert body["model_used"] == "HFML-RuleEngine-Fallback"
    assert "Đánh giá tổng quan" in body["response_text"]


# ------------------------------------------------------------- nhánh ML02
def test_ml02_branch_asks_for_the_loan_form_when_nothing_was_declared(client):
    """Chưa khai khoản vay → hướng đi khai, và bật cờ để FE hiện nút điều hướng."""
    body = ask(client, "Chẩn đoán rủi ro vay vốn", intent_code="LOAN_RISK_DIAGNOSIS")

    assert body["requires_loan_application"] is True
    assert "Thông tin khoản vay" in body["response_text"]


def test_ml02_branch_runs_the_model_once_the_artifact_exists(client):
    """Có đủ dữ liệu + artifact → chạy ML02 thật, không còn câu chờ đợi.

    Test này TỪNG khẳng định điều ngược lại: nhánh phải trả lời "mô hình đang
    huấn luyện". Đúng vào lúc nó được viết — F04 mới xong task 1/15. Nhưng
    ML02 đã train, calibrate, chốt ngưỡng và export từ lâu, còn nhánh này thì
    không biết, nên chip "Chẩn đoán rủi ro vay vốn" vẫn nói với người dùng một
    điều không còn đúng.

    Từ Epic AI-03 nhánh uỷ quyền cho `hfml.inference`, nên hợp đồng mới là:
    có chạy model, và ngưỡng lấy từ artifact.
    """
    body = ask(client, "Chẩn đoán rủi ro vay vốn",
               intent_code="LOAN_RISK_DIAGNOSIS", loan_application=LOAN_APPLICATION)

    assert body["requires_loan_application"] is False
    assert body["model_used"].startswith("HFML-ML02/")
    assert "đang trong quá trình huấn luyện" not in body["response_text"]
    assert body["response_text"].strip()


def test_ml02_branch_still_asks_for_the_form_when_data_is_missing(client):
    """Thiếu dữ liệu vẫn phải hướng đi khai, KHÔNG chạy model trên số rỗng."""
    body = ask(client, "Chẩn đoán rủi ro vay vốn",
               intent_code="LOAN_RISK_DIAGNOSIS")

    assert body["requires_loan_application"] is True
    assert "Thông tin khoản vay" in body["response_text"]


def test_risk_threshold_comes_from_the_artifact_not_a_default(client):
    """Ngưỡng phải là ngưỡng đã chốt, và tuyệt đối không phải 0,5.

    Với tỉ lệ vỡ nợ nền 8,07%, ngưỡng 0,5 xếp gần như mọi hồ sơ vào LOW_RISK —
    hệ thống trông như đang chạy trong khi nó không phân loại gì. Trước AI-03
    điều đó được canh bằng `ML02_RISK_THRESHOLD is None`; nay ngưỡng thật đã
    có nên phép canh chuyển sang chính con số đang phục vụ.
    """
    from hfml.inference.settings import ML02, SETTINGS
    from hfml.inference.lifecycle import MANAGER

    assert SETTINGS.ml02_threshold is None      # không ghi đè trong cấu hình

    threshold = MANAGER.threshold_for(ML02)
    assert threshold is not None
    assert threshold != 0.5
    assert 0.0 < threshold < 0.5


# --------------------------------------------------- câu tự gõ vẫn như cũ
def test_free_text_still_routes_by_keyword(client):
    body = ask(client, "Tôi muốn tiết kiệm cho con đi học")

    assert body["intent_code"] == IntentCode.SAVINGS_PACKAGE.value
    assert "Gói tư vấn tiết kiệm" in body["response_text"]


def test_free_text_can_never_reach_a_model_branch(client):
    """Không có `intent_code` thì không có đường nào vào ML, kể cả câu mô tả đúng."""
    for question in ("chẩn đoán sức khỏe tài chính của tôi",
                     "chẩn đoán rủi ro vay vốn giúp tôi"):
        body = ask(client, question)

        assert body["intent_code"] not in {
            IntentCode.FINANCIAL_HEALTH_DIAGNOSIS.value,
            IntentCode.LOAN_RISK_DIAGNOSIS.value,
        }, question


def test_amount_in_free_text_still_implies_a_loan_question(client):
    """Câu chỉ có số tiền vẫn hiểu là hỏi về vay — hành vi cũ, giữ nguyên."""
    body = ask(client, "nhà 3 tỷ thì sao")

    assert body["intent_code"] == IntentCode.LOAN_CAPACITY.value


def test_amount_no_longer_hijacks_an_explicit_chip_intent(client):
    """Số tiền trong câu KHÔNG được cướp intent của chip.

    Bản cũ đặt `parsed_price is not None` ngay ở nhánh đầu, nên "tiết kiệm 500
    triệu" bị kéo sang nhánh vay. Giờ số tiền chỉ nâng cấp intent khi engine
    chưa nhận ra ý định nào.
    """
    body = ask(client, "Gói tiết kiệm 500 triệu", intent_code="SAVINGS_PACKAGE")

    assert body["intent_code"] == IntentCode.SAVINGS_PACKAGE.value
    assert "Gói tư vấn tiết kiệm" in body["response_text"]


# ==========================================================================
# Tầng LLM phải thật sự nằm trên đường đi
# ==========================================================================
# Đây là phần chống tái phát cho sự cố 24/08/2026: `/advise` ghép
# `response_text` bằng f-string và không hề gọi LLM, nên người dùng đọc được
# nguyên `RB01`, `CRITICAL`, `DEFICIT`, `REJECTED`. Tầng diễn giải vẫn tồn tại
# và vẫn có test riêng đầy đủ — nó chỉ không được nối vào endpoint mà backend
# thật sự gọi. Bộ test cũ không bắt được vì mọi khẳng định đều đặt trên chính
# chuỗi f-string đó.

#: Câu trả lời cố định thay cho một lượt gọi Gemini đạt yêu cầu.
#:
#: Không chứa con số nào, và đó là chủ ý: `validator` đối chiếu mọi số với
#: `numeric_facts`, nên một câu trả lời giả có số sẽ bị đánh trượt vì bịa số
#: và test hoá ra lại đi kiểm nhánh hạ cấp.
_LLM_REPLY = {
    "explanation": "Đây là phần giải thích do tầng diễn giải viết ra.",
    "recommendations": [
        {"priority": "high", "action": "Ưu tiên gom đủ quỹ dự phòng",
         "reason": "Đệm chi tiêu hiện còn mỏng"},
    ],
    "caveats": [],
    "needs_more_data": [],
}


@pytest.fixture
def llm_online(monkeypatch):
    """Thay lượt gọi Gemini bằng một câu trả lời cố định, đạt yêu cầu kiểm."""
    monkeypatch.setattr(llm_client, "_call",
                        lambda *_args, **_kwargs: dict(_LLM_REPLY))


def test_advise_actually_sends_the_result_through_the_llm(client, llm_online):
    """Câu chữ trả về phải là của tầng diễn giải, không phải chuỗi f-string.

    Chống tái phát trực tiếp: nếu ai đó nối lại `response_text` thẳng từ
    RuleEngine, câu của tầng diễn giải sẽ biến mất và test này đỏ ngay.
    """
    body = ask(client, "Gói tiết kiệm", intent_code="SAVINGS_PACKAGE")

    assert "Đây là phần giải thích do tầng diễn giải viết ra." in body["response_text"]
    assert "Ưu tiên gom đủ quỹ dự phòng" in body["response_text"]
    # `model_used` nói ra CẢ engine lẫn người viết câu chữ.
    assert "+LLM/" in body["model_used"]


def test_advise_falls_back_to_prewritten_text_when_the_llm_fails(client):
    """LLM hỏng thì vẫn phải có câu trả lời đúng số liệu, không phải lỗi.

    `llm_offline` đang bật (autouse), nên đây chính là ca LLM không gọi được.
    """
    body = ask(client, "Gói tiết kiệm", intent_code="SAVINGS_PACKAGE")

    assert "Gói tư vấn tiết kiệm" in body["response_text"]
    assert body["model_used"].endswith("+Template")


@pytest.mark.parametrize("question,extra", [
    ("Gói tiết kiệm", {"intent_code": "SAVINGS_PACKAGE"}),
    ("Quy tắc 50/30/20", {"intent_code": "BUDGET_50_30_20"}),
    ("Chẩn đoán rủi ro vay vốn",
     {"intent_code": "LOAN_RISK_DIAGNOSIS", "loan_application": LOAN_APPLICATION}),
    ("Tôi muốn mua nhà giá 3 tỷ thì vay được bao nhiêu?", {}),
    ("Tôi muốn tiết kiệm cho con đi học", {}),
])
def test_no_branch_ever_leaks_internal_vocabulary(client, question, extra):
    """Không nhánh nào được để lọt mã rule, mã trạng thái hay slug model.

    Chạy trên nhánh HẠ CẤP (LLM đang tắt) vì đó là nhánh dựng câu bằng
    f-string — chỗ mà mọi lần rò trước đây đều bắt nguồn. Nhánh có LLM còn có
    thêm prompt cấm và một lượt kiểm nữa, nên nó là ca dễ hơn.
    """
    text = ask(client, question, **extra)["response_text"]

    assert not presentation.has_internal_vocabulary(text), text


@pytest.mark.parametrize("marker", ["**", "```", "__"])
def test_no_branch_ever_leaks_markdown(client, marker):
    """Màn Chatbot render chữ nguyên trạng, nên Markdown chỉ là rác trên màn hình.

    Nó còn bị nút "Đọc bằng giọng nói" đọc thành tiếng — người dùng nghe máy
    đọc cả dấu sao.
    """
    text = ask(client, "Gói tiết kiệm", intent_code="SAVINGS_PACKAGE")["response_text"]

    assert marker not in text


def test_ml01_branch_keeps_saying_which_model_answered(client, fake_ml01, llm_online):
    """LLM diễn giải rồi thì `model_used` vẫn phải nói ra nhánh nào đã chạy.

    Hai mẩu thông tin khác nhau: engine trả lời "chip vào đúng nhánh chưa",
    narrator trả lời "câu chữ này ai viết". Thay engine bằng narrator là mất
    đúng vế mà `intent_code` và `model_used` được thêm vào để kiểm chứng.
    """
    fake_ml01()

    body = ask(client, "Chẩn đoán sức khỏe tài chính",
               intent_code="FINANCIAL_HEALTH_DIAGNOSIS", ml_features=ML_FEATURES)

    assert body["model_used"].startswith("HFML-ML01/")
    assert "+LLM/" in body["model_used"]
