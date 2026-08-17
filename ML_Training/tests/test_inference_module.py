"""Test module inference — Epic AI-03 (F05 · M07).

Phủ từng tầng của sơ đồ cộng một lượt chạy đầu-cuối:

    Input · Validation · Preprocessing → TestValidationStage
    Rule · ML01 · ML02 · Aggregation   → TestAnalysisStage
    Intent                             → TestIntentStage
    Context                            → TestContextStage
    LLM · Output Validation            → TestLlmStage
    Cả pipeline                        → TestEndToEnd

Hai nguyên tắc của cả file
---------------------------
1. **Không gọi Gemini thật.** `client._call` luôn bị thay. Test gọi mạng thì
   chậm, không ổn định, tốn quota, và kiểm một thứ không kiểm được — câu chữ
   của model đổi mỗi lần chạy.
2. **Không train, không đọc dataset.** Model thật được nạp từ artifact đã
   export; ca "thiếu model" thì dựng bằng cấu hình trỏ vào slug không tồn tại,
   không phải bằng cách xoá file.

Vì sao có nguyên một lớp test cho việc "không sập"
----------------------------------------------------
Yêu cầu của epic là một thành phần lỗi không được kéo sập cả pipeline. Đó là
loại tính chất chỉ tồn tại nếu có ai đó cố tình phá từng bước một rồi kiểm lại
— nên `TestFailureIsolation` bơm lỗi vào từng bước và đòi các bước còn lại vẫn
cho ra kết quả đọc được.
"""
from __future__ import annotations

import pytest

from hfml.inference import analyze, chat, health
from hfml.inference import engine as engine_mod
from hfml.inference import stages
from hfml.inference.lifecycle import ModelManager, ModelUnavailable
from hfml.inference.result import ERROR, WARNING, Diagnostic, StageResult
from hfml.inference.settings import ML01, ML02, SETTINGS, load_settings
from hfml.inference.stages import PipelineState


# --------------------------------------------------------------------------
# Fixture
# --------------------------------------------------------------------------
@pytest.fixture
def household() -> dict:
    """Hồ sơ đủ dữ liệu cho cả ba tầng — rule, ML01 và ML02."""
    return {
        "representative_name": "Nguyễn Văn A",
        "birth_year": 1991,
        "residence": "TP. Hồ Chí Minh",
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


@pytest.fixture
def no_loan(household) -> dict:
    """Hồ sơ hợp lệ nhưng chưa khai khoản vay — ca rất phổ biến."""
    profile = dict(household)
    profile.pop("loan_application")
    profile["financial_needs"] = ["saving"]
    return profile


@pytest.fixture
def fake_llm(monkeypatch):
    """Thay lượt gọi Gemini bằng một JSON cố định."""
    def install(payload=None):
        reply = payload if payload is not None else {
            "explanation": "Dòng tiền ròng của bạn là 13.000.000đ mỗi tháng.",
            "recommendations": [{"action": "Duy trì quỹ dự phòng",
                                 "reason": "Đã đủ nhiều tháng chi tiêu",
                                 "priority": "medium"}],
            "caveats": [], "needs_more_data": [],
        }
        from hfml.llm import client
        monkeypatch.setattr(client, "_call", lambda *_a, **_k: reply)
        return reply

    return install


@pytest.fixture
def no_llm(monkeypatch):
    """Không có LLM — mọi câu trả lời phải dựng từ template."""
    from hfml.llm import client
    monkeypatch.setattr(client, "_call", lambda *_a, **_k: None)


# ==========================================================================
# Cấu hình
# ==========================================================================
class TestSettings:
    def test_slug_khong_con_hardcode_trong_ma(self):
        """Slug phải đọc được từ cấu hình — đây là lý do epic yêu cầu tách ra.

        Trước AI-03 slug khai bằng hằng số ở hai file, và hai bản đã trôi khỏi
        nhau: `api/main.py` trỏ vào `ml02_best_reduced_vfinal`, một artifact
        không tồn tại trên đĩa. Không ai phát hiện vì nhánh dùng nó luôn trả
        lời "model đang huấn luyện" trước khi kịp chạm tới model.
        """
        assert SETTINGS.ml01_slug and SETTINGS.ml02_slug
        assert SETTINGS.slug_for(ML01) == SETTINGS.ml01_slug
        assert SETTINGS.slug_for(ML02) == SETTINGS.ml02_slug

    def test_env_phu_len_cau_hinh(self, monkeypatch):
        monkeypatch.setenv("HFML_ML02_SLUG", "ml02_thu_nghiem")
        assert load_settings().ml02_slug == "ml02_thu_nghiem"

    def test_overrides_thang_tat_ca(self):
        assert load_settings({"history_turns": 9}).history_turns == 9

    def test_tham_so_la_khong_ton_tai_thi_bao_ngay(self):
        """Gõ sai tên tham số phải hỏng NGAY, không âm thầm bỏ qua.

        Bỏ qua lặng lẽ thì một dòng cấu hình sai chính tả trông y như một dòng
        cấu hình có tác dụng.
        """
        with pytest.raises(ValueError, match="khong_co_tham_so_nay"):
            load_settings({"khong_co_tham_so_nay": 1})

    def test_nguong_ml02_mac_dinh_lay_tu_artifact(self):
        """`None` = dùng ngưỡng đã chốt ở F04 task 14, không phải 0,5.

        Tỉ lệ vỡ nợ nền chỉ 8,07%, nên ngưỡng 0,5 xếp gần như mọi hồ sơ vào
        LOW_RISK — model trông như đang chạy trong khi nó không phân loại gì.
        """
        assert load_settings().ml02_threshold is None


# ==========================================================================
# Vòng đời model
# ==========================================================================
class TestLifecycle:
    def test_nap_mot_lan_roi_giu_lai(self):
        manager = ModelManager()
        first = manager.get(ML01)
        assert manager.get(ML01) is first

    def test_thieu_artifact_khong_phai_loi_lap_trinh(self, monkeypatch):
        monkeypatch.setattr(SETTINGS, "ml01_slug", "ml01_khong_ton_tai")
        with pytest.raises(ModelUnavailable):
            ModelManager().get(ML01)

    def test_doi_slug_trong_cau_hinh_thi_nap_lai(self, monkeypatch):
        """Đệm phải theo slug, không chỉ theo tên model.

        Nếu đệm chỉ khoá theo `"ml01"` thì đổi slug xong vẫn nhận bản cũ — tức
        thao tác đổi model có vẻ thành công mà thực tế không đổi gì.
        """
        manager = ModelManager()
        manager.get(ML01)
        monkeypatch.setattr(SETTINGS, "ml01_slug", "ml01_xgboost_v1")
        assert manager.get(ML01).slug == "ml01_xgboost_v1"

    def test_swap_hong_thi_giu_nguyen_ban_dang_chay(self, monkeypatch):
        """Slug mới hỏng thì bản cũ phải còn nguyên.

        Thay trước rồi nạp sau thì một slug gõ sai làm service mất luôn model
        đang chạy được — hỏng nặng hơn hẳn việc từ chối đổi.
        """
        monkeypatch.setattr(SETTINGS, "ml01_slug", "ml01_xgboost_vfinal")
        manager = ModelManager()
        manager.get(ML01)

        with pytest.raises(ModelUnavailable):
            manager.swap(ML01, "ml01_khong_ton_tai")

        assert SETTINGS.ml01_slug == "ml01_xgboost_vfinal"
        assert manager.get(ML01).slug == "ml01_xgboost_vfinal"

    def test_reload_bo_ban_dang_giu(self):
        manager = ModelManager()
        first = manager.get(ML01)
        manager.reload(ML01)
        assert manager.get(ML01) is not first

    def test_status_noi_ra_model_nao_thieu(self, monkeypatch):
        monkeypatch.setattr(SETTINGS, "ml02_slug", "ml02_khong_ton_tai")
        report = ModelManager().status()
        assert report[ML01]["loaded"] is True
        assert report[ML02]["loaded"] is False
        assert "ml02_khong_ton_tai" in report[ML02]["error"]

    def test_nguong_ghi_de_duoc_bao_ra_ngoai(self, monkeypatch):
        """Ghi đè ngưỡng phải hiện rõ trong `status`, không lặng lẽ.

        Người vận hành nhìn `/health` phải biết nhãn đang được cắt ở ngưỡng
        khác với ngưỡng đã được đánh giá.
        """
        monkeypatch.setattr(SETTINGS, "ml02_threshold", 0.25)
        report = ModelManager().status()
        assert report[ML02]["threshold"] == 0.25
        assert report[ML02]["threshold_overridden"] is True


# ==========================================================================
# Vỏ kết quả
# ==========================================================================
class TestStageResult:
    def test_them_error_thi_buoc_thanh_hong(self):
        result = StageResult(stage="x")
        assert result.ok
        result.add("code", "hỏng", ERROR)
        assert not result.ok

    def test_warning_khong_lam_hong_buoc(self):
        """Ranh giới này giữ cho hồ sơ thiếu khoản vay vẫn được phân tích.

        Gộp warning vào error là cách chắc chắn nhất để mọi hồ sơ bình thường
        bị báo là hỏng.
        """
        result = StageResult(stage="x")
        result.add("code", "lưu ý", WARNING)
        assert result.ok
        assert result.warnings and not result.errors


# ==========================================================================
# Input — quy đổi tên trường của client
# ==========================================================================
class TestPayloadAdapter:
    """Payload dạng Laravel phải đi hết pipeline, không chỉ nửa đầu."""

    def test_quy_doi_ten_truong(self):
        from hfml.inference.payloads import normalize_payload

        out = normalize_payload({
            "monthly_income": 35_000_000, "monthly_living_cost": 17_000_000,
            "total_debt": 500_000_000, "current_savings": 150_000_000,
            "supports_elderly": True,
        })
        assert out["average_monthly_income"] == 35_000_000
        assert out["average_monthly_expense"] == 17_000_000
        assert out["total_current_debt"] == 500_000_000
        assert out["savings_amount"] == 150_000_000
        assert out["has_dependents"] is True

    def test_bo_khoa_la_thay_vi_chuyen_tiep(self):
        """Schema từ chối khoá lạ, nên một trường phụ của backend làm hỏng cả
        hồ sơ hợp lệ — người dùng bị báo "thiếu dữ liệu" vì thứ họ không nhập.
        """
        from hfml.inference.payloads import normalize_payload

        out = normalize_payload({"household_size": 4, "id": 7,
                                 "created_at": "2026-01-01"})
        assert out == {"household_size": 4}

    def test_ten_chuan_thang_khi_client_gui_ca_hai(self):
        from hfml.inference.payloads import normalize_payload

        out = normalize_payload({"average_monthly_income": 40_000_000,
                                 "monthly_income": 35_000_000})
        assert out["average_monthly_income"] == 40_000_000

    def test_payload_laravel_di_het_pipeline(self, household):
        """Chống tái phát: quy đổi từng chỉ áp ở bước kiểm, không áp ở bước sau.

        `stage_analyze` đọc lại `state.payload`, nên nếu bước kiểm không ghi
        đè bản đã quy đổi thì bước kiểm báo hợp lệ còn bước phân tích báo
        thiếu trường — trên cùng một hồ sơ.
        """
        laravel = {
            "representative_name": "Nguyễn Văn A", "birth_year": 1991,
            "household_size": 4, "children_count": 2, "supports_elderly": False,
            "monthly_income": 35_000_000, "monthly_living_cost": 17_000_000,
            "has_debt": True, "total_debt": 500_000_000,
            "monthly_debt_payment": 5_000_000,
            "has_savings": True, "current_savings": 150_000_000,
            "id": 7,
        }
        result = analyze(laravel)
        assert result.ok, [d.message for d in result.errors]
        assert result.analysis["ml01"]["available"] is True


# ==========================================================================
# Validation · Preprocessing
# ==========================================================================
class TestValidationStage:
    def test_ho_so_du_thi_qua(self, household):
        result = stages.stage_normalize(PipelineState(payload=household))
        assert result.ok
        assert result.data.is_valid

    def test_thieu_truong_bat_buoc_thi_neu_dich_danh(self):
        """Lỗi phải nêu TÊN TRƯỜNG, không chỉ nói "hồ sơ sai".

        Người dùng đọc "đã xảy ra lỗi" thì không có cách nào sửa; đọc tên
        trường thì điền lại được ngay.
        """
        result = stages.stage_normalize(PipelineState(payload={"a": 1}))
        assert not result.ok
        assert all(d.stage == stages.VALIDATION for d in result.errors)
        assert any(d.field for d in result.errors)

    def test_do_thoi_gian_tung_buoc(self, household):
        result = stages.stage_normalize(PipelineState(payload=household))
        assert result.elapsed_ms >= 0.0


# ==========================================================================
# Rule · ML01 · ML02 · Aggregation
# ==========================================================================
class TestAnalysisStage:
    def test_gom_du_ca_rule_va_hai_model(self, household):
        result = analyze(household)
        assert result.ok
        assert set(result.analysis["rules"]) >= {"RB01", "RB02", "RB05"}
        assert result.analysis["ml01"]["available"] is True
        assert result.analysis["ml02"]["available"] is True

    def test_thieu_khoan_vay_van_phan_tich_day_du(self, no_loan):
        """Không khai khoản vay là chuyện bình thường, KHÔNG phải lỗi.

        Đây là ca dễ làm sai nhất: đánh dấu cả request là hỏng chỉ vì thiếu
        ML02 sẽ chặn mất phần dòng tiền mà người dùng thật sự đang hỏi.
        """
        result = analyze(no_loan)
        assert result.ok
        assert result.analysis["ml01"]["available"] is True
        assert result.analysis["ml02"]["available"] is False
        assert result.analysis["ml02"]["reason_code"] == "missing_input"
        assert result.analysis["rules"]["RB01"]

    def test_input_hong_thi_dung_va_khong_co_ket_qua_bia(self):
        result = analyze({"representative_name": "X"})
        assert not result.ok
        assert result.errors
        assert result.analysis.get("ml01", {}).get("available") is not True

    def test_trace_ghi_lai_tung_buoc(self, household):
        result = analyze(household)
        assert [t["stage"] for t in result.trace] == [
            stages.VALIDATION, stages.AGGREGATION]

    def test_khong_nem_ngoai_le_ra_ngoai(self):
        """Điểm vào phải luôn trả về `InferenceResult`.

        Ở biên một service, ngoại lệ nghĩa là 500 và người dùng không biết
        mình sai chỗ nào.
        """
        for payload in ({}, {"household_size": "sai kiểu"}, {"a": None}):
            assert analyze(payload).ok is False


# ==========================================================================
# Intent
# ==========================================================================
class TestIntentStage:
    def test_chip_thang_keyword(self, household):
        result = chat(household, "tôi muốn vay mua nhà, rủi ro ra sao?",
                      intent_code="FINANCIAL_HEALTH_DIAGNOSIS")
        assert result.intent == "FINANCIAL_HEALTH_DIAGNOSIS"

    def test_thieu_du_lieu_la_canh_bao_chu_khong_phai_loi(self, no_loan, no_llm):
        """Thiếu dữ liệu thì hỏi xin, không đánh sập request."""
        result = chat(no_loan, "Khoản vay của tôi rủi ro thế nào?",
                      intent_code="LOAN_RISK_DIAGNOSIS")
        assert result.ok
        assert any(d.code == "missing_data" for d in result.warnings)
        assert result.answer["needs_more_data"]

    def test_cau_noi_tiep_ke_thua_intent(self, household, fake_llm):
        fake_llm()
        result = chat(household, "Thế còn 2 tỷ?",
                      previous_intent="LOAN_RISK_DIAGNOSIS")
        assert result.intent == "LOAN_RISK_DIAGNOSIS"


# ==========================================================================
# Context
# ==========================================================================
class TestContextStage:
    def test_context_niem_yet_con_so_duoc_phep_dung(self, household, fake_llm):
        fake_llm()
        state = PipelineState(payload=household, question="Sức khỏe tài chính?",
                              intent_code="FINANCIAL_HEALTH_DIAGNOSIS")
        stages.stage_normalize(state)
        stages.stage_analyze(state)
        stages.stage_intent(state)
        result = stages.stage_context(state)

        assert result.ok
        assert state.context.numeric_facts
        assert any(k.startswith("rules.") for k in state.context.numeric_facts)

    def test_context_chi_mang_phan_ml_ma_intent_can(self, household, fake_llm):
        fake_llm()
        state = PipelineState(payload=household, question="Sức khỏe tài chính?",
                              intent_code="FINANCIAL_HEALTH_DIAGNOSIS")
        for step in (stages.stage_normalize, stages.stage_analyze,
                     stages.stage_intent, stages.stage_context):
            step(state)
        assert state.context.ml01.get("available") is True
        assert state.context.ml02 == {}


# ==========================================================================
# LLM · Output Validation
# ==========================================================================
class TestLlmStage:
    def test_cau_tra_loi_dat_thi_dung(self, household, fake_llm):
        fake_llm()
        result = chat(household, "Sức khỏe tài chính?",
                      intent_code="FINANCIAL_HEALTH_DIAGNOSIS")
        assert result.answer["source"] == "llm"
        assert result.answer["validation"]["valid"] is True

    def test_so_bia_thi_ha_cap_va_noi_ra_ly_do(self, household, fake_llm):
        """Bịa số thì hạ cấp, và lý do phải nằm trong `warnings`.

        Hạ cấp im lặng thì nhìn từ ngoài không phân biệt được với một lượt
        chạy bình thường.
        """
        fake_llm({"explanation": "Bạn dư 999.111.222đ mỗi tháng.",
                  "recommendations": [{"action": "x", "reason": "y",
                                       "priority": "low"}]})
        result = chat(household, "Sức khỏe tài chính?",
                      intent_code="FINANCIAL_HEALTH_DIAGNOSIS")

        assert result.answer["source"] == "template"
        assert any(d.code == "ungrounded_numbers" for d in result.warnings)

    def test_khong_goi_duoc_llm_khac_voi_bi_danh_truot(self, household, no_llm):
        """Hai nguyên nhân hạ cấp phải phân biệt được.

        Cùng rơi về template, nhưng một bên là mạng/quota hỏng, một bên là câu
        trả lời vi phạm guardrail. Trả cùng một tín hiệu thì người đọc log đi
        sửa prompt trong khi thứ hỏng là hạn mức API.
        """
        result = chat(household, "Sức khỏe tài chính?",
                      intent_code="FINANCIAL_HEALTH_DIAGNOSIS")
        assert result.answer["validation"]["valid"] is None
        assert any(d.code == "llm_unreachable" for d in result.warnings)

    def test_tat_llm_bang_cau_hinh_van_ra_cau_tra_loi(self, household, monkeypatch):
        """Tắt LLM phải cho ra câu trả lời dùng được, không phải một lỗi."""
        monkeypatch.setattr(SETTINGS, "llm_enabled", False)
        result = chat(household, "Sức khỏe tài chính?",
                      intent_code="FINANCIAL_HEALTH_DIAGNOSIS")
        assert result.ok
        assert result.text.strip()
        assert result.answer["source"] == "template"

    def test_ngoai_pham_vi_khong_cham_toi_ho_so(self, household, monkeypatch):
        """Câu ngoài phạm vi phải bị chặn TRƯỚC khi dựng prompt.

        Không phải để tiết kiệm: chạy hết rồi mới từ chối vẫn ra câu từ chối
        đúng, nhưng prompt khi đó đã mang toàn bộ hồ sơ tài chính đi cho một
        câu hỏi về bitcoin — dữ liệu gửi đi rồi thì không rút lại được.
        """
        from hfml.llm import client

        def _boom(*_a, **_k):
            raise AssertionError("Không được gọi LLM cho câu ngoài phạm vi.")

        monkeypatch.setattr(client, "_call", _boom)
        result = chat(household, "Tôi nên mua bitcoin không?")

        assert result.answer["source"] == "out_of_scope"
        assert result.answer["suggested_questions"]
        assert result.analysis == {}


# ==========================================================================
# Chịu lỗi từng phần
# ==========================================================================
class TestFailureIsolation:
    """Một thành phần lỗi không được kéo sập cả pipeline."""

    def test_buoc_nem_ngoai_le_thi_thanh_ket_qua_hong_co_cau_truc(self, monkeypatch):
        """Ngoại lệ trong một bước phải thành `StageResult`, không bay lên biên."""
        @stages.timed("thu_nghiem")
        def no_tung(_state):
            raise RuntimeError("hỏng có chủ ý")

        result = no_tung(PipelineState())
        assert not result.ok
        assert "hỏng có chủ ý" in result.errors[0].message
        assert result.errors[0].stage == "thu_nghiem"

    def test_ml_hong_thi_rule_van_bao_cao_duoc(self, household, monkeypatch):
        """Mất cả hai model vẫn phải còn nguyên phần quy tắc.

        Bốn nguồn kết quả độc lập với nhau; gộp số phận của chúng lại là biến
        một sự cố artifact thành mất trắng cả câu trả lời.
        """
        monkeypatch.setattr(SETTINGS, "ml01_slug", "ml01_khong_ton_tai")
        monkeypatch.setattr(SETTINGS, "ml02_slug", "ml02_khong_ton_tai")
        from hfml.inference import lifecycle
        monkeypatch.setattr(lifecycle, "MANAGER", ModelManager())

        result = analyze(household)

        assert result.analysis["ml01"]["available"] is False
        assert result.analysis["ml02"]["available"] is False
        assert result.analysis["rules"]["RB01"]["status"]
        assert result.analysis["overall_status"]

    def test_context_hong_thi_van_giu_duoc_phan_phan_tich(self, household, monkeypatch):
        """Bước sau hỏng không được xoá kết quả các bước trước."""
        def no_tung(_state):
            raise RuntimeError("context hỏng")

        monkeypatch.setattr(stages, "stage_context",
                            stages.timed(stages.CONTEXT)(no_tung))
        monkeypatch.setattr(engine_mod, "CHAT_STAGES",
                            (stages.stage_intent, stages.stage_context,
                             stages.stage_generate))

        result = chat(household, "Sức khỏe tài chính?",
                      intent_code="FINANCIAL_HEALTH_DIAGNOSIS")

        assert not result.ok
        assert result.analysis["rules"]           # phần đã tính vẫn còn
        assert result.text.strip()                # vẫn nói được gì đó


# ==========================================================================
# Đầu-cuối
# ==========================================================================
class TestEndToEnd:
    def test_tron_pipeline(self, household, fake_llm):
        fake_llm()
        result = chat(household, "Sức khỏe tài chính của tôi thế nào?",
                      intent_code="FINANCIAL_HEALTH_DIAGNOSIS")

        assert result.ok
        assert [t["stage"] for t in result.trace] == [
            stages.VALIDATION, stages.AGGREGATION, stages.INTENT,
            stages.CONTEXT, stages.LLM]
        assert result.text.strip()
        assert result.analysis["ml01"]["available"] is True

    def test_ket_qua_luon_du_khoa(self, household, no_llm):
        """Một khoá biến mất rất dễ bị đọc nhầm thành "không có vấn đề gì"."""
        payload = chat(household, "Sức khỏe tài chính?").to_dict()
        for key in ("ok", "schema_version", "generated_at", "intent", "topic",
                    "text", "answer", "analysis", "errors", "warnings",
                    "trace", "settings"):
            assert key in payload, key

    def test_health_khong_coi_thieu_llm_la_hong(self, monkeypatch):
        """Thiếu API key KHÔNG phải là hỏng — pipeline vẫn trả lời được."""
        from hfml.llm import client
        monkeypatch.setattr(client, "is_llm_available", lambda: False)
        report = health()
        assert report["ok"] is True
        assert report["llm"]["available"] is False

    def test_hai_diem_vao_cho_cung_mot_ket_qua_phan_tich(self, household, no_llm):
        """`chat` KHÔNG cài lại `analyze` — nó gọi đúng hàm đó rồi đi tiếp."""
        only = analyze(household).analysis
        full = chat(household, "Sức khỏe tài chính?").analysis

        assert only["ml01"]["label"] == full["ml01"]["label"]
        assert only["overall_status"] == full["overall_status"]
