"""Test API v1 — Epic AI-04 (F05 · M08).

Test ở đây đi qua HTTP thật bằng `TestClient`, không gọi thẳng hàm route: thứ
cần kiểm là **hợp đồng** — mã trạng thái, hình dạng thân response, tên trường —
và những thứ đó chỉ tồn tại ở mức endpoint. Gọi hàm trực tiếp thì bỏ qua đúng
phần đang được kiểm: validate của pydantic và các exception handler.

Nghiệp vụ KHÔNG kiểm lại ở đây
--------------------------------
Rule, model, tầng diễn đạt đã có bộ test riêng (F02–F04, AI-02, AI-03). Lặp lại
ở tầng API là nhân đôi chi phí bảo trì để kiểm cùng một thứ hai lần — và bản ở
đây sẽ chậm hơn nhiều vì phải qua HTTP.

Không lượt nào gọi Gemini thật
--------------------------------
`client._call` luôn bị thay. Test gọi mạng thì chậm, không ổn định và tốn quota.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from hfml.api.v1 import create_app
from hfml.api.v1.config import API_PREFIX
from hfml.api.v1.errors import (
    INTERNAL_ERROR,
    MODEL_UNAVAILABLE,
    TIMEOUT,
    VALIDATION_ERROR,
)
from hfml.inference.settings import SETTINGS


HOUSEHOLD = {
    "representative_name": "Nguyễn Văn A",
    "birth_year": 1991,
    "household_size": 4,
    "children_count": 2,
    "has_dependents": False,
    "average_monthly_income": 35_000_000,
    "average_monthly_expense": 17_000_000,
    "has_debt": True,
    "total_current_debt": 500_000_000,
    "monthly_debt_payment": 5_000_000,
    "has_savings": True,
    "savings_amount": 150_000_000,
    "assets": ["cash", "real_estate"],
    "financial_needs": ["home_loan"],
    "loan_application": {
        "borrower_age": 35, "gender": "male", "marital_status": "married",
        "children_count": 2, "education_level": "higher",
        "occupation": "office_staff", "employment_years": 8.5,
        "loan_amount": 1_400_000_000, "loan_term_months": 240,
        "monthly_payment": 12_000_000, "asset_price": 2_000_000_000,
        "loan_purpose": "buy_house", "previous_loan_count": 3,
        "late_payment_count": 1, "has_overdue_loan": False,
        "total_overdue_amount": 0,
    },
}


@pytest.fixture(autouse=True)
def stub_llm(monkeypatch):
    """Không lượt test nào được gọi Gemini thật."""
    from hfml.llm import client

    monkeypatch.setattr(client, "_call", lambda *_a, **_k: {
        "explanation": "Dòng tiền ròng của bạn là 13.000.000đ mỗi tháng.",
        "recommendations": [{"action": "Duy trì quỹ dự phòng",
                             "reason": "Đã đủ nhiều tháng chi tiêu",
                             "priority": "medium"}],
        "caveats": [], "needs_more_data": [],
    })


@pytest.fixture
def client():
    # `with` để lifespan chạy — đó chính là phần nạp model lúc khởi động.
    with TestClient(create_app()) as test_client:
        yield test_client


@pytest.fixture
def lenient_client():
    """Client KHÔNG ném lại ngoại lệ của server.

    Mặc định `TestClient` bật `raise_server_exceptions`, nên nó ném ngoại lệ ra
    cho test thay vì trả response — tiện khi gỡ lỗi, nhưng khi đó không kiểm
    được chính cái đang cần kiểm: handler 500 có trả đúng vỏ `ErrorResponse`
    không. Ở môi trường thật thì handler luôn chạy, nên đây mới là hành vi
    tương ứng.
    """
    with TestClient(create_app(), raise_server_exceptions=False) as test_client:
        yield test_client


def post(client, path: str, **body):
    return client.post(f"{API_PREFIX}{path}", json=body)


# ==========================================================================
# Task 3 — nạp model lúc khởi động
# ==========================================================================
class TestStartup:
    def test_model_duoc_nap_luc_khoi_dong(self, monkeypatch):
        """Nạp lúc khởi động, không chờ request đầu tiên."""
        loaded = []
        from hfml.inference import lifecycle

        original = lifecycle.MANAGER.get
        monkeypatch.setattr(
            lifecycle.MANAGER, "get",
            lambda name: (loaded.append(name), original(name))[1])

        with TestClient(create_app()):
            pass

        assert "ml01" in loaded and "ml02" in loaded

    def test_thieu_artifact_khong_lam_chet_service(self, monkeypatch):
        """Service vẫn phải lên được để còn nói ra là mình đang thiếu gì.

        Ném ngoại lệ trong lifespan thì tiến trình không lên nổi, và khi đó
        không còn gì để hỏi trạng thái — người vận hành chỉ thấy container
        restart liên tục mà không biết vì sao.
        """
        monkeypatch.setattr(SETTINGS, "ml01_slug", "ml01_khong_ton_tai")
        monkeypatch.setattr(SETTINGS, "ml02_slug", "ml02_khong_ton_tai")
        from hfml.inference import lifecycle
        monkeypatch.setattr(lifecycle, "MANAGER", lifecycle.ModelManager())

        with TestClient(create_app()) as test_client:
            response = test_client.get("/health")

        assert response.status_code == 503
        assert response.json()["status"] == "unhealthy"

    def test_model_khong_nap_lai_moi_request(self, client, monkeypatch):
        """Đệm phải giữ giữa các request, nếu không mỗi lượt tốn thêm ~200ms."""
        from hfml.ml import registry

        calls = []
        monkeypatch.setattr(
            registry, "load_model",
            lambda *a, **k: calls.append(a) or pytest.fail("nạp lại model"))

        for _ in range(3):
            assert post(client, "/inference", household=HOUSEHOLD).status_code == 200
        assert calls == []


# ==========================================================================
# Task 4 — health check
# ==========================================================================
class TestHealth:
    def test_day_du_thi_healthy(self, client):
        body = client.get("/health").json()
        assert body["status"] == "healthy"
        for name in ("api", "ml01", "ml02", "preprocessing", "llm"):
            assert body["components"][name]["status"] == "healthy", name

    def test_hai_duong_dan_cho_cung_ket_qua(self, client):
        """`/health` cho hạ tầng, `/api/v1/health` cho client có phiên bản."""
        assert client.get("/health").json() == client.get(
            f"{API_PREFIX}/health").json()

    def test_thieu_llm_la_degraded_chu_khong_phai_unhealthy(self, client, monkeypatch):
        """Mất LLM thì câu trả lời khô hơn nhưng vẫn đúng — đó là `degraded`.

        Gộp vào `unhealthy` thì bộ cân bằng tải rút service ra trong khi nó
        vẫn trả lời tốt phần lớn câu hỏi.
        """
        from hfml.llm import client as llm_client
        monkeypatch.setattr(llm_client, "is_llm_available", lambda: False)

        response = client.get("/health")
        body = response.json()

        assert response.status_code == 200      # vẫn phục vụ
        assert body["status"] == "degraded"
        assert body["components"]["llm"]["status"] == "degraded"

    def test_mat_mot_model_la_degraded(self, monkeypatch):
        """Còn một model thì còn trả lời được một nửa số câu hỏi."""
        monkeypatch.setattr(SETTINGS, "ml02_slug", "ml02_khong_ton_tai")
        from hfml.inference import lifecycle
        monkeypatch.setattr(lifecycle, "MANAGER", lifecycle.ModelManager())

        with TestClient(create_app()) as test_client:
            response = test_client.get("/health")

        assert response.status_code == 200
        assert response.json()["status"] == "degraded"

    def test_mat_ca_hai_model_la_unhealthy_kem_503(self, monkeypatch):
        """503 để bộ cân bằng tải rút service mà không phải đọc thân response."""
        monkeypatch.setattr(SETTINGS, "ml01_slug", "ml01_khong_ton_tai")
        monkeypatch.setattr(SETTINGS, "ml02_slug", "ml02_khong_ton_tai")
        from hfml.inference import lifecycle
        monkeypatch.setattr(lifecycle, "MANAGER", lifecycle.ModelManager())

        with TestClient(create_app()) as test_client:
            response = test_client.get("/health")

        assert response.status_code == 503
        assert response.json()["status"] == "unhealthy"
        assert response.json()["components"]["preprocessing"]["status"] == "unhealthy"


# ==========================================================================
# Task 2 — POST /api/v1/inference
# ==========================================================================
class TestInferenceEndpoint:
    def test_request_hop_le(self, client):
        response = post(client, "/inference", household=HOUSEHOLD)
        body = response.json()

        assert response.status_code == 200
        assert body["ok"] is True
        assert [r["code"] for r in body["rules"]] == [
            "RB01", "RB02", "RB03", "RB04", "RB05"]
        assert body["ml01"]["available"] is True
        assert body["ml02"]["available"] is True

    def test_khong_lo_chi_tiet_noi_bo(self, client):
        """`trace` và `settings` là của người vận hành, không phải của client.

        Rò ra thì client bắt đầu phụ thuộc vào tên bước nội bộ, và đường dẫn
        thư mục artifact không có lý do gì để trình duyệt người dùng biết.
        """
        body = post(client, "/inference", household=HOUSEHOLD).json()

        assert "trace" not in body
        assert "settings" not in body
        assert "artifact_dir" not in str(body)
        # Chẩn đoán cũng không được mang tên bước nội bộ.
        for item in body["errors"] + body["warnings"]:
            assert "stage" not in item

    def test_thieu_khoan_vay_van_200(self, client):
        """Chưa khai khoản vay là bình thường, không phải lỗi request."""
        household = {k: v for k, v in HOUSEHOLD.items()
                     if k != "loan_application"}
        household["financial_needs"] = ["saving"]

        response = post(client, "/inference", household=household)
        body = response.json()

        assert response.status_code == 200
        assert body["ml01"]["available"] is True
        assert body["ml02"]["available"] is False
        assert body["ml02"]["reason_code"] == "missing_input"

    def test_nhan_ca_ten_truong_cua_backend(self, client):
        """Laravel gọi `monthly_income`, schema chuẩn gọi khác — nhận cả hai.

        Alias được sinh từ bảng của AI-03 nên không có bản sao thứ hai.
        """
        response = post(client, "/inference", household={
            "representative_name": "A", "birth_year": 1991,
            "household_size": 4, "children_count": 2, "has_dependents": False,
            "monthly_income": 35_000_000, "monthly_living_cost": 17_000_000,
            "has_debt": True, "total_debt": 500_000_000,
            "monthly_debt_payment": 5_000_000,
            "has_savings": True, "current_savings": 150_000_000,
        })

        assert response.status_code == 200, response.json()
        assert response.json()["ok"] is True


# ==========================================================================
# Task 2 — POST /api/v1/chat
# ==========================================================================
class TestChatEndpoint:
    def test_request_hop_le(self, client):
        response = post(client, "/chat", household=HOUSEHOLD,
                        question="Sức khỏe tài chính của tôi thế nào?",
                        intent_code="FINANCIAL_HEALTH_DIAGNOSIS")
        body = response.json()

        assert response.status_code == 200
        assert body["answer"]["text"].strip()
        assert body["answer"]["intent_code"] == "FINANCIAL_HEALTH_DIAGNOSIS"
        assert body["answer"]["source"] == "llm"
        assert body["answer"]["validated"] is True

    def test_tra_ve_ca_can_cu_cua_cau_tra_loi(self, client):
        """Không có phần phân tích thì client không kiểm chứng được gì."""
        body = post(client, "/chat", household=HOUSEHOLD,
                    question="Sức khỏe tài chính?").json()

        assert body["analysis"]["ml01"]["available"] is True
        assert body["analysis"]["rules"]

    def test_ngoai_pham_vi_van_200_kem_goi_y(self, client):
        """Từ chối lịch sự là một câu trả lời hợp lệ, không phải lỗi."""
        body = post(client, "/chat", household=HOUSEHOLD,
                    question="Tôi nên mua bitcoin không?").json()

        assert body["answer"]["source"] == "out_of_scope"
        assert body["answer"]["suggested_questions"]

    def test_lich_su_hoi_thoai_duoc_nhan(self, client):
        response = post(client, "/chat", household=HOUSEHOLD,
                        question="Thế còn 2 tỷ?",
                        previous_intent="LOAN_RISK_DIAGNOSIS",
                        history=[{"role": "user", "content": "Tôi vay được bao nhiêu?"},
                                 {"role": "assistant", "content": "Khoảng 900 triệu."}])

        assert response.status_code == 200
        assert response.json()["answer"]["intent_code"] == "LOAN_RISK_DIAGNOSIS"


# ==========================================================================
# Task 5 — lỗi thống nhất
# ==========================================================================
class TestErrorHandling:
    def _assert_shape(self, body: dict, error: str):
        """Mọi lỗi phải ra ĐÚNG một hình dạng — client viết một nhánh, không sáu."""
        assert body["ok"] is False
        assert body["error"] == error
        assert isinstance(body["message"], str) and body["message"]
        assert isinstance(body["details"], list)

    def test_thieu_truong_bat_buoc(self, client):
        response = post(client, "/chat", household=HOUSEHOLD)   # thiếu question

        assert response.status_code == 422
        body = response.json()
        self._assert_shape(body, VALIDATION_ERROR)
        assert any("question" in item["field"] for item in body["details"])

    def test_sai_kieu_du_lieu(self, client):
        response = post(client, "/inference", household={
            **HOUSEHOLD, "household_size": "bốn"})

        assert response.status_code == 422
        self._assert_shape(response.json(), VALIDATION_ERROR)

    def test_gia_tri_ngoai_mien_cho_phep(self, client):
        response = post(client, "/inference", household={
            **HOUSEHOLD, "average_monthly_income": -1})

        assert response.status_code == 422
        body = response.json()
        assert any("average_monthly_income" in i["field"] for i in body["details"])

    def test_truong_la_bi_tu_choi_chu_khong_nuot_im(self, client):
        """Nuốt im nghĩa là dữ liệu người dùng nhập bị bỏ mà không ai biết."""
        response = post(client, "/inference", household={
            **HOUSEHOLD, "thu_nhap_phu": 5_000_000})

        assert response.status_code == 422
        self._assert_shape(response.json(), VALIDATION_ERROR)

    def test_loi_nam_ngoai_du_lieu_thanh_500_co_ma_truy_vet(self, lenient_client, monkeypatch):
        """Traceback vào log, client chỉ nhận mã — kèm `request_id` để nối lại."""
        from hfml.api.v1 import routes

        def no_tung(*_a, **_k):
            raise RuntimeError("hỏng có chủ ý")

        monkeypatch.setattr("hfml.inference.analyze", no_tung)
        monkeypatch.setattr(routes, "_run",
                            lambda work, timeout, label: work())

        response = post(lenient_client, "/inference", household=HOUSEHOLD)

        assert response.status_code == 500
        body = response.json()
        self._assert_shape(body, INTERNAL_ERROR)
        assert body["request_id"]
        assert "hỏng có chủ ý" not in body["message"]      # không lộ nội bộ

    def test_qua_han_gio_thanh_504(self, client, monkeypatch):
        """504 nói rõ là chậm — khác hẳn 500 vốn nghĩa là lỗi lập trình."""
        from hfml.api.v1 import config

        monkeypatch.setattr(config.API_SETTINGS, "inference_timeout", 0.01)

        import time
        monkeypatch.setattr("hfml.inference.analyze",
                            lambda *_a, **_k: time.sleep(1))

        response = post(client, "/inference", household=HOUSEHOLD)

        assert response.status_code == 504
        self._assert_shape(response.json(), TIMEOUT)

    def test_model_khong_kha_dung_thanh_503(self, client, monkeypatch):
        """503 = người vận hành sửa, không phải người gọi sửa."""
        from hfml.api.v1 import routes
        from hfml.inference.lifecycle import ModelUnavailable

        def no_tung(*_a, **_k):
            raise ModelUnavailable("Chưa có artifact ML01.")

        monkeypatch.setattr("hfml.inference.analyze", no_tung)
        monkeypatch.setattr(routes, "_run",
                            lambda work, timeout, label: work())

        response = post(client, "/inference", household=HOUSEHOLD)

        assert response.status_code == 503
        self._assert_shape(response.json(), MODEL_UNAVAILABLE)

    def test_duong_dan_khong_ton_tai_cung_dung_vo_do(self, client):
        """404 mặc định của FastAPI trả `{"detail": ...}` — hình dạng thứ hai."""
        response = client.get("/api/v1/khong-co-duong-nay")

        assert response.status_code == 404
        body = response.json()
        assert body["ok"] is False
        assert body["error"] == "not_found"


# ==========================================================================
# LLM không khả dụng
# ==========================================================================
class TestLlmUnavailable:
    def test_mat_llm_van_tra_loi_duoc(self, client, monkeypatch):
        """Pipeline hạ cấp về câu dựng từ dữ liệu — vẫn 200, vẫn có nội dung."""
        from hfml.llm import client as llm_client
        monkeypatch.setattr(llm_client, "_call", lambda *_a, **_k: None)

        response = post(client, "/chat", household=HOUSEHOLD,
                        question="Sức khỏe tài chính?")
        body = response.json()

        assert response.status_code == 200
        assert body["answer"]["source"] == "template"
        assert body["answer"]["validated"] is None      # chưa gọi được, khác bị chặn
        assert body["answer"]["text"].strip()

    def test_tat_llm_bang_cau_hinh(self, client, monkeypatch):
        monkeypatch.setattr(SETTINGS, "llm_enabled", False)

        body = post(client, "/chat", household=HOUSEHOLD,
                    question="Sức khỏe tài chính?").json()

        assert body["answer"]["source"] == "template"
        assert body["answer"]["text"].strip()


# ==========================================================================
# Task 6 — cấu hình
# ==========================================================================
class TestApiConfiguration:
    def test_cors_mac_dinh_tat(self):
        """Mặc định `"*"` là mở hồ sơ tài chính cho mọi trang web bất kỳ."""
        from hfml.api.v1.config import load_api_settings

        assert load_api_settings().cors_origins == []

    def test_cors_bat_duoc_bang_env(self, monkeypatch):
        monkeypatch.setenv("HFML_API_CORS_ORIGINS",
                           "http://localhost:5173, https://app.example.com")
        from hfml.api.v1.config import load_api_settings

        assert load_api_settings().cors_origins == [
            "http://localhost:5173", "https://app.example.com"]

    def test_timeout_doi_duoc_bang_env(self, monkeypatch):
        monkeypatch.setenv("HFML_API_CHAT_TIMEOUT", "12.5")
        from hfml.api.v1.config import load_api_settings

        assert load_api_settings().chat_timeout == 12.5

    def test_endpoint_deu_nam_duoi_tien_to_phien_ban(self, client):
        """Không được có đường đi vòng qua phiên bản.

        Client lỡ dùng đường không phiên bản sẽ hỏng đúng vào lúc v2 ra đời,
        mà không ai lường trước.
        """
        assert client.post("/inference", json={"household": HOUSEHOLD}).status_code == 404
        assert client.post("/chat", json={"household": HOUSEHOLD,
                                          "question": "x"}).status_code == 404

    def test_openapi_khai_bao_vo_loi(self, client):
        """Client sinh mã từ OpenAPI phải thấy được hình dạng lỗi."""
        schema = client.get("/openapi.json").json()
        assert "ErrorResponse" in schema["components"]["schemas"]

        responses = schema["paths"][f"{API_PREFIX}/chat"]["post"]["responses"]
        for code in ("422", "500", "503", "504"):
            assert code in responses, code
