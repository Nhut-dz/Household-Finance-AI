"""Test tầng LLM — Epic AI-02 (F05 · M06).

Nguyên tắc bao trùm cả file: **không lượt test nào được gọi Gemini thật**.
`_call` luôn bị monkeypatch. Test gọi mạng thì chậm, không ổn định, tốn quota,
và tệ nhất là nó kiểm thứ không kiểm được — câu chữ của model đổi mỗi lần chạy.

Thứ file này kiểm là phần MÌNH viết: hiểu câu hỏi, lọc context, niêm yết con
số, kiểm câu trả lời, che dữ liệu, hạ cấp. Model chỉ là một hàm trả về JSON.

Hai lớp test chống tái phát ở phần `validator`
-----------------------------------------------
Cả hai lỗi đó đều đã xảy ra thật khi chạy đầu-cuối, và cả hai đều có cùng một
kiểu hỏng: **bộ canh bắt nhầm câu trả lời đúng**. Kiểu hỏng này nguy hiểm âm
thầm, vì nhìn từ ngoài chỉ thấy "LLM lại hạ cấp về template" chứ không thấy
nguyên nhân. Chúng được khoá lại bằng test để không quay lại.
"""
from __future__ import annotations

import pytest

from hfml.api.intents import IntentCode
from hfml.llm import chat, client, context as context_mod, guardrails, prompts
from hfml.llm import presentation, validator
from hfml.llm.context import build_context, build_numeric_facts
from hfml.llm.understanding import understand


# --------------------------------------------------------------------------
# Fixture — một `AiResult` rút gọn nhưng đúng hình dạng
# --------------------------------------------------------------------------
@pytest.fixture
def result() -> dict:
    """Bản rút gọn của `AiResult.to_dict()` (Epic AI-01).

    Rút gọn chứ không bịa hình dạng: các khoá và mức lồng nhau giữ y hệt bản
    thật, vì `build_context` đọc theo đường dẫn. Đổi hình dạng ở đây thì test
    vẫn xanh trong khi mã thật đã hỏng.
    """
    return {
        "ok": True,
        "schema_version": "1.0",
        "overall_status": "GOOD",
        "input_summary": {
            "valid": True,
            "representative_name": "Nguyễn Văn A",
            "household_size": 4,
            "monthly_income": 35_000_000.0,
            "monthly_expense": 17_000_000.0,
            "has_debt": True,
            "has_loan_application": True,
        },
        "rules": {
            "RB01": {"code": "RB01", "status": "POSITIVE",
                     "value": {"net_cashflow": 13_000_000.0,
                               "savings_rate": 0.3714},
                     "details": {"summary_vi": "Dòng tiền ròng dương."}},
            "RB02": {"code": "RB02", "status": "EXCELLENT",
                     "value": {"dti": 0.1429, "emergency_months": 8.82},
                     "details": {"summary_vi": "Sức khỏe tài chính tốt."}},
            "RB04": {"code": "RB04", "status": "BALANCED",
                     "value": {"needs_budget": 17_500_000.0,
                               "wants_budget": 10_500_000.0,
                               "savings_budget": 7_000_000.0},
                     "details": {"summary_vi": "Ngân sách 50/30/20 cân bằng."}},
            "RB05": {"code": "RB05", "status": "OK",
                     "value": {"max_loan_amount": 1_200_000_000.0},
                     "details": {"summary_vi": "Khả năng vay ổn."}},
        },
        "ml01": {
            "available": True, "label": "GROWTH",
            "label_vi": "Có thể hướng tới tăng trưởng",
            "probability": 0.9982,
            "confidence": {"low_confidence": False},
        },
        "ml02": {
            "available": True, "label": "LOW_RISK", "label_vi": "Rủi ro thấp",
            "probability": 0.0412, "threshold": 0.1303,
            "confidence": {"low_confidence": False},
        },
        "warnings": [],
        "errors": [],
    }


@pytest.fixture
def ai_context(result):
    u = understand("Sức khỏe tài chính thế nào?", result,
                   IntentCode.FINANCIAL_HEALTH_DIAGNOSIS.value)
    return build_context("Sức khỏe tài chính thế nào?", result, u)


@pytest.fixture
def loan_context(result):
    """Context của intent chẩn đoán rủi ro vay — phần ML02 mới có mặt ở đây."""
    q = "Khoản vay của tôi rủi ro thế nào?"
    u = understand(q, result, IntentCode.LOAN_RISK_DIAGNOSIS.value)
    return build_context(q, result, u)


@pytest.fixture
def health(result):
    """Cặp `(context, understanding)` dựng từ CÙNG một `result`.

    Phải cùng nguồn: `client.generate` đọc `understanding.can_answer` để quyết
    định có gọi LLM không, nên ghép context của hồ sơ này với understanding của
    hồ sơ khác là dựng một trạng thái không bao giờ xảy ra thật.
    """
    q = "Sức khỏe tài chính thế nào?"
    u = understand(q, result, IntentCode.FINANCIAL_HEALTH_DIAGNOSIS.value)
    return build_context(q, result, u), u


@pytest.fixture
def no_llm(monkeypatch):
    """Chặn mọi lượt gọi ra ngoài. Gọi thật thì test này sai thiết kế."""
    def _boom(*_args, **_kwargs):
        raise AssertionError("Test không được gọi Gemini thật.")

    monkeypatch.setattr(client, "_call", _boom)


def _reply(**over) -> dict:
    """Một câu trả lời JSON hợp lệ của LLM, cho phép ghi đè từng khoá."""
    payload = {"explanation": "Dòng tiền ròng của bạn là 13.000.000đ mỗi tháng.",
               "recommendations": [{"action": "Duy trì quỹ dự phòng",
                                    "reason": "Đã đủ 8.82 tháng chi tiêu",
                                    "priority": "medium"}],
               "caveats": [], "needs_more_data": []}
    payload.update(over)
    return payload


# ==========================================================================
# Task 1 — hiểu câu hỏi
# ==========================================================================
class TestUnderstanding:
    def test_chip_intent_thang_keyword(self, result):
        """`intent_code` từ chip phải THẮNG suy đoán theo từ khoá.

        Đây là ràng buộc nghiệp vụ đã chốt: hai intent ML không bao giờ được
        phân luồng bằng keyword. Câu hỏi dưới đây đầy từ khoá của chủ đề vay,
        nhưng chip nói rõ là chẩn đoán sức khỏe — chip phải thắng.
        """
        u = understand("tôi muốn vay mua nhà, rủi ro khoản vay ra sao?", result,
                       IntentCode.FINANCIAL_HEALTH_DIAGNOSIS.value)
        assert u.intent is IntentCode.FINANCIAL_HEALTH_DIAGNOSIS
        assert u.from_chip is True

    def test_khong_co_chip_thi_dung_keyword(self, result):
        u = understand("Tôi nên tiết kiệm bao nhiêu mỗi tháng?", result)
        assert u.from_chip is False

    def test_thieu_du_lieu_thi_hoi_lai(self, result):
        """Thiếu dữ liệu bắt buộc → không trả lời, mà hỏi xin."""
        result["ml02"] = {"available": False}
        u = understand("Rủi ro khoản vay?", result,
                       IntentCode.LOAN_RISK_DIAGNOSIS.value)
        assert not u.can_answer
        assert u.missing
        assert u.ask_message().strip()

    def test_du_du_lieu_thi_tra_loi_duoc(self, result):
        u = understand("Rủi ro khoản vay?", result,
                       IntentCode.LOAN_RISK_DIAGNOSIS.value)
        assert u.can_answer


# ==========================================================================
# Task 2 — dựng context
# ==========================================================================
class TestContext:
    def test_chi_dua_phan_ml_ma_intent_can(self, result):
        """Hỏi sức khỏe tài chính thì context KHÔNG được mang xác suất vỡ nợ."""
        u = understand("Sức khỏe tài chính?", result,
                       IntentCode.FINANCIAL_HEALTH_DIAGNOSIS.value)
        ctx = build_context("Sức khỏe tài chính?", result, u)
        assert ctx.ml01.get("available") is True
        assert ctx.ml02 == {}

    def test_rb01_rb02_luon_co_mat(self, result):
        u = understand("Rủi ro khoản vay?", result,
                       IntentCode.LOAN_RISK_DIAGNOSIS.value)
        ctx = build_context("Rủi ro khoản vay?", result, u)
        assert {"RB01", "RB02"} <= set(ctx.rules)

    def test_numeric_facts_gom_tu_context_da_loc(self, ai_context):
        """Bảng số phải gom từ context ĐÃ LỌC, không phải từ AiResult đầy đủ.

        Gom từ bản đầy đủ thì LLM được phép nhắc một con số nó không hề nhìn
        thấy trong prompt — tức được nói về thứ nó không có căn cứ.
        """
        facts = ai_context.numeric_facts
        assert facts["rules.RB01.value.net_cashflow"] == 13_000_000.0
        assert facts["ml01.probability"] == pytest.approx(0.9982)
        # ml02 bị lọc khỏi context → con số của nó KHÔNG được nằm trong bảng.
        assert not any(k.startswith("ml02.") for k in facts)

    def test_bo_qua_so_ky_thuat(self, ai_context):
        assert not any("schema_version" in k for k in ai_context.numeric_facts)

    def test_bool_khong_phai_con_so(self, ai_context):
        """`True` là 1.0 với Python — nhưng không phải con số để LLM nhắc tới."""
        assert "profile.valid" not in ai_context.numeric_facts
        assert "profile.has_debt" not in ai_context.numeric_facts

    def test_lich_su_bi_cat(self, result):
        u = understand("Thế còn?", result)
        history = [{"role": "user", "content": f"câu {i}"} for i in range(20)]
        ctx = build_context("Thế còn?", result, u, history)
        assert len(ctx.history) == context_mod.HISTORY_TURNS * 2


# ==========================================================================
# Task 3 — prompt
# ==========================================================================
class TestPrompts:
    def test_prompt_liet_ke_con_so_duoc_phep(self, ai_context):
        text = prompts.render_user_prompt(ai_context, "sức khỏe tài chính")
        assert "13" in text            # net_cashflow có mặt để LLM dùng
        assert ai_context.question in text

    def test_prompt_khai_moi_khoa_bat_buoc(self, ai_context):
        """Schema đầu ra phải được khai bằng chính JSON, ngay trong prompt.

        Mô tả bằng lời thì model tự đặt tên khoá theo ý nó, và `check_schema`
        đánh trượt một câu trả lời đúng nội dung chỉ vì sai tên khoá.
        """
        rendered = prompts.render_user_prompt(ai_context, "sức khỏe tài chính")
        for key in prompts.REQUIRED_KEYS:
            assert key in prompts.RESPONSE_SCHEMA
            assert key in rendered

    def test_prompt_co_phien_ban(self):
        """Prompt phải đánh phiên bản để về sau còn truy được câu trả lời cũ."""
        assert prompts.PROMPT_VERSION


# ==========================================================================
# Task 8 — validator
# ==========================================================================
class TestValidatorNumbers:
    def test_chap_nhan_moi_cach_viet_cua_cung_mot_so(self):
        facts = {"a": 35_000_000.0}
        for text in ("35.000.000đ", "35,000,000 đồng", "35 triệu"):
            assert validator.check_numbers(text, facts) == [], text

    def test_bat_so_bia(self):
        assert validator.check_numbers("để dành 88.888.888đ", {"a": 35e6})

    def test_chap_nhan_dang_phan_tram_cua_ti_le(self):
        assert validator.check_numbers("DTI 14,29%", {"r": 0.1429}) == []

    def test_bo_qua_so_dem_nho(self):
        """"3 việc nên làm" không phải số liệu tài chính."""
        assert validator.check_numbers("Có 3 việc nên làm ngay", {}) == []

    def test_dau_cau_khong_bi_nuot_vao_con_so(self):
        """Chống tái phát: dấu phẩy cuối câu từng bị đọc thành dấu ngăn nghìn.

        `"0.9982,"` khớp cả dấu phẩy, rồi bước chuẩn hoá coi dấu phẩy cuối là
        dấu ngăn nghìn → 9982. Con số hoàn toàn hợp lệ bị báo là bịa và cả câu
        trả lời đúng bị vứt về template. Đã xảy ra thật khi chạy đầu-cuối.
        """
        assert validator.check_numbers("mức tin cậy 0.9982, rất cao",
                                       {"ml01.probability": 0.9982}) == []

    def test_duoc_nhac_lai_so_cua_chinh_nguoi_dung(self, result):
        """Chống tái phát: số trong CÂU HỎI phải được phép nhắc lại.

        Guardrail sinh ra để chặn LLM bịa dữ kiện, không phải để cấm nó nhắc
        lại lời người dùng. Không có phần này thì mọi câu hỏi giả định đều
        hỏng: "thế còn 2 tỷ?" buộc LLM phải nhắc "2 tỷ", và ngay khi nhắc thì
        bị đánh là bịa số. Đã xảy ra thật khi chạy demo.
        """
        q = "Thế còn nếu vay 2 tỷ?"
        u = understand(q, result)
        ctx = build_context(q, result, u)
        assert validator.check_numbers(
            "Với khoản vay 2 tỷ bạn hỏi, số tiền trả hàng tháng sẽ tăng.",
            ctx.numeric_facts) == []

    def test_van_bat_so_bia_khi_co_so_trong_cau_hoi(self, result):
        """Nới cho số của người dùng KHÔNG được nới cho số LLM tự nghĩ ra."""
        q = "Thế còn nếu vay 2 tỷ?"
        ctx = build_context(q, result, understand(q, result))
        assert validator.check_numbers("Bạn sẽ phải trả 77.777.777đ mỗi tháng.",
                                       ctx.numeric_facts)

    def test_so_mo_ho_khop_theo_bat_ky_cach_doc_nao(self):
        """`0,163` là 0,163 (VN) hay 163 (Anh Mỹ) — chấp nhận cả hai.

        Với một bộ canh, hướng sai lệch phải là "chấp nhận khi còn nghi ngờ":
        từ chối nhầm thì mất câu trả lời đúng, còn khả năng một số bịa tình cờ
        khớp dưới MỘT trong hai cách đọc là rất nhỏ.
        """
        assert validator.check_numbers("tỉ lệ 0,163", {"r": 0.163}) == []
        assert validator.check_numbers("khoảng 1,234", {"r": 1234.0}) == []


class TestValidatorCertainty:
    def test_bat_loi_hua_chac_chan(self):
        assert validator.check_certainty("Khoản này chắc chắn sẽ sinh lời.")
        assert validator.check_certainty("Chúng tôi đảm bảo lợi nhuận 12%.")

    def test_khong_bat_cau_mien_tru(self):
        """Chống tái phát: câu miễn trừ từng bị tính là vi phạm.

        Prompt YÊU CẦU LLM nói rõ kết quả chưa chắc chắn khi độ tin cậy thấp.
        Bộ canh khi đó bắt trần chuỗi "chắc chắn"/"đảm bảo" nên nó đánh dấu vi
        phạm cho chính câu mình bắt model phải viết — hai lượt gọi liên tiếp
        đều trượt và hệ thống hạ cấp về template. Đã xảy ra thật.
        """
        assert validator.check_certainty(
            "Các dự đoán mang tính xác suất và không đảm bảo chắc chắn "
            "diễn biến trong tương lai.") == []
        assert validator.check_certainty(
            "Kết quả này chưa chắc chắn, nên đọc kèm đánh giá theo quy tắc.") == []

    def test_khong_bat_dong_tu_thong_thuong(self):
        """"đảm bảo quỹ dự phòng" là lời khuyên đúng, không phải lời hứa."""
        assert validator.check_certainty(
            "Nên đảm bảo quỹ dự phòng đủ 6 tháng chi tiêu.") == []


class TestValidatorOther:
    def test_bat_mau_thuan_voi_model(self, loan_context):
        """LLM không được nói ngược nhãn mà model đã trả về.

        Đây là dạng sai tệ nhất: người dùng tin vào câu chữ chứ không đọc JSON,
        nên một câu "rủi ro cao" đè lên nhãn LOW_RISK là đã đổi kết luận của
        hệ thống mà không để lại dấu vết nào.
        """
        assert validator.check_contradiction(
            "Hồ sơ của bạn thuộc nhóm rủi ro cao, cần thận trọng.",
            loan_context)

    def test_khong_bat_khi_noi_dung_nhan(self, loan_context):
        assert validator.check_contradiction(
            "Khoản vay của bạn được xếp mức rủi ro thấp.", loan_context) == []

    def test_khong_bat_khi_neu_ca_xac_suat_bu(self, loan_context):
        """Chống tái phát: câu trả lời TỐT NHẤT từng bị đánh trượt.

        Model trả LOW_RISK và LLM nêu đúng nhãn kèm xác suất của nhãn ngược
        lại — chính xác hơn hẳn mức tối thiểu. Bộ canh khi đó chỉ dò chuỗi
        "rủi ro cao" nên nó bắt nhầm, hai lượt liền, rồi hạ cấp về template.
        Đã xảy ra thật khi chạy demo.
        """
        assert validator.check_contradiction(
            "Mô hình phân loại hồ sơ của bạn ở nhãn 'Rủi ro thấp' với xác "
            "suất 0.9796 (xác suất rủi ro cao chỉ 0.0204).", loan_context) == []

    def test_bat_khuyen_nghi_vuot_pham_vi(self):
        assert validator.check_scope("Bạn nên mua cổ phiếu VNM ngay bây giờ.")

    def test_bat_thieu_khoa_bat_buoc(self):
        assert validator.check_schema({"explanation": "..."})

    def test_cau_tra_loi_dung_thi_qua(self, ai_context):
        report = validator.validate(_reply(), ai_context)
        assert report.is_valid, report.to_dict()
        assert report.ungrounded_numbers == []


# ==========================================================================
# Task 7 — guardrail phía trước
# ==========================================================================
class TestGuardrails:
    def test_chan_cau_hoi_ngoai_pham_vi(self):
        assert not guardrails.check_scope("Tôi nên mua bitcoin không?").allowed

    def test_khong_chan_cau_hoi_tai_chinh_binh_thuong(self):
        for q in ("Tôi nên tiết kiệm bao nhiêu?",
                  "Gia đình tôi có nên vay mua nhà không?",
                  "Quy tắc 50/30/20 áp dụng thế nào?"):
            assert guardrails.check_scope(q).allowed, q

    def test_doi_cam_ket_thi_nhac_chu_khong_chan(self):
        """Câu hỏi ĐÚNG chủ đề nhưng đòi cam kết — trả lời, kèm lời nhắc."""
        q = "Gửi tiết kiệm thì đảm bảo lợi nhuận bao nhiêu?"
        assert guardrails.check_scope(q).allowed
        assert guardrails.detect_overreach(q)

    def test_che_danh_tinh_nhung_giu_con_so(self):
        clean = guardrails.redact({
            "representative_name": "Nguyễn Văn A",
            "household_id": 7,
            "monthly_income": 35_000_000.0,
            "nested": {"residence": "Hà Nội", "dti": 0.1429},
        })
        assert "representative_name" not in clean
        assert "household_id" not in clean
        assert "residence" not in clean["nested"]
        # Con số tài chính phải còn — không có chúng thì không giải thích được gì.
        assert clean["monthly_income"] == 35_000_000.0
        assert clean["nested"]["dti"] == 0.1429

    def test_apply_dung_lai_bang_so_sau_khi_che(self, ai_context):
        """Che xong phải dựng lại `numeric_facts`, không giữ bảng cũ.

        Giữ bảng cũ thì một trường vừa bị che vẫn nằm trong danh sách trắng —
        tức LLM vẫn được phép nói ra con số mà nó không còn nhìn thấy.
        """
        ai_context.profile["household_id"] = 12345
        ai_context.numeric_facts = build_numeric_facts(ai_context)
        assert "profile.household_id" in ai_context.numeric_facts

        guardrails.apply(ai_context)
        assert "profile.household_id" not in ai_context.numeric_facts


# ==========================================================================
# Task 4, 5 — sinh, kiểm, hạ cấp
# ==========================================================================
class TestGenerate:
    def test_cau_tra_loi_dat_thi_dung_luon(self, health, monkeypatch):
        monkeypatch.setattr(client, "_call", lambda *_: _reply())
        answer = client.generate(*health)
        assert answer.source == client.SOURCE_LLM

    def test_lan_dau_truot_thi_sinh_lai_mot_lan(self, health, monkeypatch):
        calls = []

        def _fake(system, user):
            calls.append(user)
            # Lần đầu bịa số, lần sau viết đúng.
            return _reply(explanation="Bạn dư 999.111.222đ") if len(calls) == 1 \
                else _reply()

        monkeypatch.setattr(client, "_call", _fake)
        answer = client.generate(*health)

        assert answer.source == client.SOURCE_LLM_RETRY
        assert len(calls) == 2
        # Lượt hai phải được nhắc ĐÚNG lỗi đã mắc, không nhắc chung chung.
        assert "999.111.222" in calls[1]

    def test_truot_ca_hai_lan_thi_ha_cap(self, health, monkeypatch):
        monkeypatch.setattr(
            client, "_call",
            lambda *_: _reply(explanation="Bạn dư 999.111.222đ"))
        answer = client.generate(*health)

        assert answer.source == client.SOURCE_TEMPLATE
        # Giữ lại báo cáo kiểm để về sau còn truy được vì sao đã hạ cấp.
        assert answer.validation["valid"] is False

    def test_llm_hong_thi_ha_cap(self, health, monkeypatch):
        monkeypatch.setattr(client, "_call", lambda *_: None)
        assert client.generate(*health).source == client.SOURCE_TEMPLATE

    def test_phan_biet_khong_goi_duoc_voi_bi_danh_truot(self, health, monkeypatch):
        """Hai nguyên nhân hạ cấp phải phân biệt được trong kết quả trả về.

        Cả hai đều rơi về template, nhưng một bên là mạng/quota hỏng, một bên
        là câu trả lời vi phạm guardrail. Trả cùng một `validation` thì người
        đọc log thấy "kiểm không đạt" và đi sửa prompt, trong khi thứ hỏng là
        hạn mức API — đã suýt lạc hướng đúng vì chuyện này khi gặp lỗi 429.
        """
        monkeypatch.setattr(client, "_call", lambda *_: None)
        assert client.generate(*health).validation["valid"] is None

        monkeypatch.setattr(
            client, "_call",
            lambda *_: _reply(explanation="Bạn dư 999.111.222đ"))
        assert client.generate(*health).validation["valid"] is False

    def test_template_noi_ra_ket_qua_rule(self, result, monkeypatch):
        """Chống tái phát: lưới an toàn từng trả về câu RỖNG.

        `_template_answer` đọc `message_vi`, nhưng rule dict không hề có khoá
        đó — câu tóm tắt nằm ở `details.summary_vi`. `.get()` trả None, vòng
        lặp bỏ qua lặng lẽ, nên với intent không có phần ML (hỏi 50/30/20) câu
        trả lời hạ cấp chỉ còn mỗi dòng miễn trừ. Kiểu hỏng tệ nhất của một
        lưới an toàn: nhìn từ ngoài tưởng nó đã đỡ. Đã xảy ra thật khi chạy demo.
        """
        monkeypatch.setattr(client, "_call", lambda *_: None)
        q = "Tôi nên phân bổ thu nhập theo quy tắc 50/30/20 thế nào?"
        u = understand(q, result)
        ctx = build_context(q, result, u)
        # Đúng ca đã hỏng: intent này không mang theo phần ML nào.
        assert not ctx.ml01 and not ctx.ml02

        answer = client.generate(ctx, u)
        # Gọi rule bằng TÊN NGHIỆP VỤ, không bằng mã.
        #
        # Test này từng khẳng định `"RB01" in answer.explanation` — nó chốt
        # đúng cái rò rỉ mà 24/08/2026 phải đi bịt: người dùng đọc được
        # `- RB01 (CRITICAL): …` ngay trên màn chat. Điều cần canh vẫn là điều
        # cũ (lưới an toàn không được rỗng), chỉ khác ở chỗ nội dung phải là
        # thứ người dùng đọc hiểu được.
        assert "Dòng tiền hằng tháng" in answer.explanation
        assert "Dòng tiền ròng dương." in answer.explanation
        assert not presentation.has_internal_vocabulary(answer.explanation)

    def test_template_khong_the_bia_so(self, health, monkeypatch):
        """Đích hạ cấp phải là thứ KHÔNG CẦN TIN.

        `_template_answer` dựng câu bằng f-string từ chính context, nên theo
        cấu trúc nó không thể bịa. Đó là lý do nó là đích hạ cấp an toàn — và
        test này kiểm chính tính chất đó bằng cùng bộ canh dùng cho LLM.
        """
        monkeypatch.setattr(client, "_call", lambda *_: None)
        ctx, _ = health
        answer = client.generate(*health)
        assert validator.check_numbers(
            answer.explanation, ctx.numeric_facts) == []

    def test_thieu_du_lieu_thi_hoi_chu_khong_goi_llm(self, result, no_llm):
        result["ml02"] = {"available": False}
        u = understand("Rủi ro khoản vay?", result,
                       IntentCode.LOAN_RISK_DIAGNOSIS.value)
        ctx = build_context("Rủi ro khoản vay?", result, u)
        answer = client.generate(ctx, u)
        assert answer.needs_more_data


# ==========================================================================
# Task 6 — hội thoại nhiều lượt
# ==========================================================================
class TestChat:
    def test_ngoai_pham_vi_thi_khong_goi_llm(self, result, no_llm):
        turn = chat.answer("Tôi nên mua bitcoin không?", result)
        assert turn.answer.source == client.SOURCE_OUT_OF_SCOPE
        assert turn.answer.suggested_questions

    def test_cau_noi_tiep_ke_thua_intent(self, result, monkeypatch):
        monkeypatch.setattr(client, "_call", lambda *_: _reply())
        # Câu này tự nó ra GENERAL — đúng ca mà cơ chế kế thừa sinh ra để lo.
        assert understand("Thế còn 2 tỷ?", result).intent is IntentCode.GENERAL

        turn = chat.answer("Thế còn 2 tỷ?", result,
                           previous_intent=IntentCode.LOAN_RISK_DIAGNOSIS.value)
        assert turn.intent == IntentCode.LOAN_RISK_DIAGNOSIS.value

    def test_cau_hoi_moi_du_nghia_thi_khong_ke_thua(self, result, monkeypatch):
        """Đổi chủ đề mà bị kẹt lại chủ đề cũ là kiểu hỏng khó chịu hơn nhiều.

        Người dùng hỏi rõ ràng, đủ nghĩa, mà vẫn bị trả lời sai chủ đề — nên
        kế thừa phải CÓ ĐIỀU KIỆN, chỉ áp cho câu ngắn.
        """
        long_question = ("Tôi muốn biết cách xây dựng quỹ dự phòng khẩn cấp "
                         "cho gia đình bốn người thì nên bắt đầu từ đâu?")
        assert len(long_question) > chat.FOLLOW_UP_CHAR_LIMIT
        monkeypatch.setattr(client, "_call", lambda *_: _reply())
        turn = chat.answer(long_question, result,
                           previous_intent=IntentCode.LOAN_RISK_DIAGNOSIS.value)
        assert turn.intent != IntentCode.LOAN_RISK_DIAGNOSIS.value

    def test_doi_cam_ket_thi_them_loi_nhac(self, result, monkeypatch):
        monkeypatch.setattr(client, "_call", lambda *_: _reply())
        turn = chat.answer("Gửi tiết kiệm đảm bảo lợi nhuận bao nhiêu?", result)
        assert any("không cam kết" in c for c in turn.answer.caveats)

    def test_tra_ve_du_thong_tin_truy_vet(self, result, monkeypatch):
        monkeypatch.setattr(client, "_call", lambda *_: _reply())
        payload = chat.answer("Sức khỏe tài chính?", result,
                              IntentCode.FINANCIAL_HEALTH_DIAGNOSIS.value).to_dict()
        assert payload["answer"]["prompt_version"] == prompts.PROMPT_VERSION
        assert payload["context_summary"]["n_numeric_facts"] > 0
        assert payload["text"].strip()
