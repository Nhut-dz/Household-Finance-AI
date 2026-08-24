"""Test tầng trình bày — biên cuối trước mắt người dùng (F05 · M06).

File này canh MỘT lời hứa: không có chuỗi nào của tầng trong lọt ra ngoài.

Vì sao lời hứa đó cần một bộ test riêng
-----------------------------------------
Nó không thuộc về một nhánh nghiệp vụ nào cả. Sự cố 24/08/2026 rò qua ba
đường khác nhau cùng lúc — chuỗi f-string ở `/advise`, câu dựng sẵn ở
`llm/client.py`, và chính LLM chép lại mã nó đọc được trong prompt. Test theo
từng nhánh thì mỗi nhánh phải tự nhớ mà kiểm, và đường thứ tư sẽ lại lọt.

Nên phép kiểm đặt ở chính hàm mà mọi câu chữ đều phải đi qua.
"""
from __future__ import annotations

import pytest

from hfml.llm import presentation
from hfml.llm.presentation import (
    has_internal_vocabulary,
    label_status,
    money,
    percent,
    rule_name,
    to_plain_text,
)


# --------------------------------------------------------------- bảng tra
class TestBangTra:
    def test_cung_mot_trang_thai_o_hai_rule_cho_hai_nghia_khac_nhau(self):
        """`WARNING` của RB02 và của RB05 KHÔNG được dịch giống nhau.

        RB02 nói về sức khỏe tài chính, RB05 nói về khoản vay. Gộp chung một
        bảng tra theo riêng tên trạng thái thì một nửa số câu sẽ nói sai ý —
        đó là lý do khoá tra là CẶP (mã rule, trạng thái).
        """
        assert label_status("RB02", "WARNING") != label_status("RB05", "WARNING")
        assert label_status("RB01", "BALANCED") != label_status("RB04", "BALANCED")

    def test_khong_tra_duoc_thi_tra_rong_chu_khong_tra_ma_goc(self):
        """Trả lại mã gốc là mở đúng đường rò mà module này sinh ra để bịt."""
        assert label_status("RB01", "MOT_TRANG_THAI_LA") == ""
        assert label_status("RB01", None) == ""

    def test_moi_ma_trang_thai_that_deu_dich_duoc(self):
        """Bảng tra phải phủ hết trạng thái mà 5 rule thực sự sinh ra.

        Thiếu một mã thì nó rơi xuống bước quét cuối và bị BỎ HẲN khỏi câu —
        người dùng mất luôn phần thông tin đó mà không có gì báo.
        """
        that = {
            "OVERALL": ("CRITICAL", "WARNING", "STABLE", "EXCELLENT"),
            "RB01": ("POSITIVE", "BALANCED", "DEFICIT"),
            "RB02": ("CRITICAL", "WARNING", "GOOD", "EXCELLENT"),
            "RB03": ("COMPLETED", "FEASIBLE", "STRETCHED", "INFEASIBLE"),
            "RB04": ("OVERBUDGET", "UNDER_SAVING", "BALANCED"),
            "RB05": ("REJECTED", "APPROVED", "WARNING", "ELIGIBLE"),
        }
        for code, statuses in that.items():
            for status in statuses:
                assert label_status(code, status), f"{code}/{status}"

    def test_nam_rule_deu_co_ten_nghiep_vu(self):
        for code in ("RB01", "RB02", "RB03", "RB04", "RB05"):
            assert rule_name(code)


# ------------------------------------------------------------- định dạng
class TestDinhDang:
    def test_xac_suat_ra_phan_tram_kieu_viet_nam(self):
        """0.9449 là thứ không ai đọc ra "gần như chắc chắn"."""
        assert percent(0.9449) == "94,5%"
        assert percent(0.0) == "0,0%"

    def test_tien_dung_dau_cham_ngan_nghin(self):
        assert money(3_000_000) == "3.000.000đ"


# ------------------------------------------------------------ quét Markdown
class TestBoMarkdown:
    @pytest.mark.parametrize("goc,mong_doi", [
        ("**đậm**", "đậm"),
        ("*nghiêng*", "nghiêng"),
        ("__đậm__", "đậm"),
        ("_nghiêng_", "nghiêng"),
        ("`mã`", "mã"),
        ("## Tiêu đề", "Tiêu đề"),
        ("> trích dẫn", "trích dẫn"),
        ("[nhãn](https://vi.dụ)", "nhãn"),
    ])
    def test_bo_het_ky_hieu(self, goc, mong_doi):
        assert to_plain_text(goc) == mong_doi

    def test_gach_dau_dong_thanh_dau_cham_tron(self):
        assert to_plain_text("- một\n- hai") == "• một\n• hai"

    def test_dong_mo_dau_bang_dau_sao_khong_bi_hieu_la_chu_nghieng(self):
        """`* mục` là gạch đầu dòng, không phải dấu nghiêng đang mở.

        Đây là lý do bước gạch đầu dòng phải chạy TRƯỚC bước đậm/nghiêng.
        """
        assert to_plain_text("* một\n* hai") == "• một\n• hai"

    def test_gach_duoi_giua_ten_truong_khong_bi_cat(self):
        """`has_asset_cash` phải nguyên vẹn — nó không phải chữ nghiêng."""
        assert "has_asset_cash" in to_plain_text("cột has_asset_cash bật")


# -------------------------------------------------------- quét từ vựng nội bộ
class TestBoTuVungNoiBo:
    def test_ma_rule_trong_ngoac_bi_bo_han(self):
        assert to_plain_text("Dòng tiền hàng tháng (RB01): dư 3.000.000đ") == (
            "Dòng tiền hàng tháng: dư 3.000.000đ")

    def test_ma_rule_mo_dau_muc_bi_bo(self):
        assert to_plain_text("- RB05: hạn mức 1 tỷ") == "• hạn mức 1 tỷ"

    def test_ma_trang_thai_trong_ngoac_duoc_dich(self):
        assert to_plain_text("Đánh giá tổng quan (CRITICAL)") == (
            "Đánh giá tổng quan (cần xử lý ngay)")

    @pytest.mark.parametrize("nhan", [
        "EMERGENCY", "DEBT_FOCUS", "BUILD_BUFFER", "GROWTH",
        "LOW_RISK", "HIGH_RISK",
    ])
    def test_nhan_model_khong_lot_ra(self, nhan):
        assert nhan not in to_plain_text(f"Nhóm của bạn: {nhan}.")

    def test_slug_artifact_bi_bo(self):
        assert "ml01_xgboost_vfinal" not in to_plain_text(
            "Mô hình: ml01_xgboost_vfinal")

    def test_duong_dan_khoa_context_bi_bo(self):
        """`rules.RB02.value.dti` phải biến mất NGUYÊN CỤM.

        Chống tái phát: bước dịch mã rule từng chạy trước và biến cụm này
        thành `rules.sức khỏe tài chính.value.dti`, khiến phép bắt đường dẫn
        chỉ ăn được `rules.s` và để lại một mảnh chữ cụt giữa câu.
        """
        out = to_plain_text("Xem rules.RB02.value.dti để biết thêm")
        assert "dti" not in out
        assert "khỏe tài chính.value" not in out

    def test_cho_trong_fstring_chua_dien_bi_bo(self):
        assert "{" not in to_plain_text("Bạn dư {net_cashflow} mỗi tháng")

    def test_gia_tri_rong_thanh_chu_doc_duoc(self):
        assert "None" not in to_plain_text("Năng lực thế chấp: None")

    def test_thuat_ngu_tai_chinh_that_KHONG_bi_quet(self):
        """`DTI`, `LTV` là thuật ngữ thật, người dùng đọc hiểu được.

        Xoá mọi cụm viết hoa là quét quá tay: "tỷ lệ DTI 34,3%" mà mất chữ
        DTI thì mất luôn thứ đang được đo. Từ vựng nội bộ là tập ĐÓNG, nên
        chặn theo danh sách chứ không theo hình dạng.
        """
        out = to_plain_text("Tỷ lệ DTI 34,3% và LTV 70%")
        assert "DTI" in out and "LTV" in out


# ------------------------------------------------------------------ tổng thể
class TestBienCuoi:
    #: Đúng dạng câu trả lời mà người dùng đã đọc phải trên màn chat.
    RO_RI_THAT = (
        "📌 **Đánh giá tổng quan (CRITICAL)**:\n"
        "- Dòng tiền hàng tháng (RB01): Dư thừa khoảng 3.000.000 VNĐ (DEFICIT).\n"
        "- Sức khỏe tài chính (RB02): CRITICAL (Tỷ lệ DTI: 34.3%).\n"
        "- Khả năng vay vốn (RB05): tối đa 2.000.000 VNĐ/tháng (REJECTED).\n"
        "- Năng lực thế chấp: Mức NONE.\n"
        "_Mô hình: `ml01_xgboost_vfinal`._"
    )

    def test_quet_sach_ca_doan_ro_ri_that(self):
        assert not has_internal_vocabulary(to_plain_text(self.RO_RI_THAT))

    @pytest.mark.parametrize("marker", ["**", "__", "`", "RB01", "RB02", "RB05",
                                        "CRITICAL", "DEFICIT", "REJECTED"])
    def test_khong_con_ky_hieu_hay_ma_nao(self, marker):
        assert marker not in to_plain_text(self.RO_RI_THAT)

    def test_con_so_khong_bi_dong_toi(self):
        """Quét chỉ BỎ thứ không nên có, không được sửa số liệu đã tính đúng."""
        out = to_plain_text(self.RO_RI_THAT)
        for so in ("3.000.000", "34.3", "2.000.000"):
            assert so in out

    def test_quet_hai_lan_van_ra_ket_qua_cu(self):
        """Hàm phải bất biến khi chạy lại — nó được gọi ở nhiều lớp chồng nhau.

        `Answer.as_text()` đã quét từng phần rồi quét cả đoạn, và `_narrate`
        quét thêm lần nữa. Không bất biến thì mỗi lớp lại gặm thêm một ít.
        """
        mot_lan = to_plain_text(self.RO_RI_THAT)
        assert to_plain_text(mot_lan) == mot_lan

    @pytest.mark.parametrize("rong", [None, "", "   ", "\n\n"])
    def test_dau_vao_rong_khong_lam_no(self, rong):
        assert to_plain_text(rong) == ""

    def test_has_internal_vocabulary_bat_duoc_truoc_khi_quet(self):
        """Bộ dò phải báo có, nếu không thì test dùng nó chỉ luôn xanh."""
        assert has_internal_vocabulary(self.RO_RI_THAT)


def test_moi_ma_trong_bang_tra_deu_bien_mat_sau_khi_quet():
    """Không mã nào trong bảng tra được phép sống sót qua bước quét.

    Duyệt chính bảng thay vì liệt kê tay: thêm một mã mới mà quên cập nhật
    phép quét thì test này đỏ ngay, chứ không đợi tới lúc người dùng đọc phải.
    """
    for ma in presentation._GENERIC_VI:
        assert ma not in to_plain_text(f"Kết quả đánh giá: {ma}."), ma


def test_cau_tra_loi_cua_llm_cung_duoc_quet_o_dang_co_cau_truc():
    """`Answer.explanation` phải sạch, không chỉ `Answer.as_text()`.

    `/api/v1/chat` trả về CẢ hai: `text` (đã gộp) và `explanation` (thô).
    Quét mỗi lúc gộp thì client nào đọc `explanation` vẫn nhận nguyên mã —
    một đường rò y hệt đường cũ, chỉ khác cửa ra.
    """
    from types import SimpleNamespace

    from hfml.llm import client
    from hfml.llm.context import AiContext

    ctx = AiContext(question="Tình hình tài chính của tôi thế nào?",
                    intent="GENERAL", topic="general")
    # `generate` chỉ đọc `can_answer` trên đường thành công; phần còn lại của
    # `Understanding` thuộc nhánh thiếu dữ liệu, không liên quan ở đây.
    u = SimpleNamespace(can_answer=True)

    ban_ban = {
        "explanation": "Sức khỏe tài chính của bạn: **CRITICAL** (RB02).",
        "recommendations": [
            {"priority": "high", "action": "Cắt **chi tiêu** không cần thiết",
             "reason": "Dòng tiền đang ở mức DEFICIT"},
        ],
        "caveats": ["_Đây là ước lượng._"],
        "needs_more_data": [],
    }

    original_call = client._call
    try:
        client._call = lambda *_args, **_kwargs: dict(ban_ban)
        answer = client.generate(ctx, u)
    finally:
        client._call = original_call

    assert not has_internal_vocabulary(answer.explanation), answer.explanation
    assert "**" not in answer.explanation
    assert "**" not in answer.recommendations[0]["action"]
    assert not has_internal_vocabulary(answer.recommendations[0]["reason"])
    assert "_" not in answer.caveats[0]
    # `priority` là mã nội bộ nhưng KHÔNG hiển thị trực tiếp — `as_text()`
    # dịch nó sang chữ, nên nó phải còn nguyên trong cấu trúc.
    assert answer.recommendations[0]["priority"] == "high"
    assert "ưu tiên cao" in answer.as_text()
