"""AI-02 task 3 — System prompt và prompt template (F05 · M06).

Prompt là MÃ NGUỒN, không phải văn bản trang trí
-------------------------------------------------
Nó quyết định hành vi của hệ thống ở mức ngang với một hàm, nên nó được đánh
số phiên bản (`PROMPT_VERSION`) và con số đó đi kèm mọi câu trả lời. Không có
phiên bản thì khi chất lượng đổi, không ai biết là do prompt đổi, do model
đổi, hay do dữ liệu đổi.

Bốn ranh giới prompt phải dựng, xếp theo mức nguy hiểm nếu vỡ
---------------------------------------------------------------
1. **Không được tính toán.** Mọi con số đến từ tầng rule và ML. LLM tính lại
   thì hệ thống có hai nguồn sự thật cho cùng một con số, và nguồn sai không
   để lại dấu vết nào.
2. **Không được đổi kết quả model.** Nhãn và xác suất là dữ kiện. "Model nói
   rủi ro thấp nhưng tôi thấy hồ sơ này đáng lo" là câu tuyệt đối cấm.
3. **Không biến dự đoán thành sự chắc chắn.** Xác suất 85% vẫn để ngỏ 15%.
4. **Không cam kết lợi nhuận, không khuyến nghị mã cụ thể.** Chỉ nói theo lớp
   tài sản (§8.2 guardrail 3).

Ba loại thông tin phải được phân biệt trong câu trả lời
--------------------------------------------------------
    ĐÃ TÍNH      kết quả rule — xác định, đúng theo công thức đã công bố
    DỰ ĐOÁN      nhãn + xác suất của model — có sai số, có độ tin cậy
    KHUYẾN NGHỊ  suy ra từ hai loại trên — là gợi ý tham khảo

Gộp ba loại lại thành một giọng đều đều là cách chắc chắn nhất để người dùng
đọc một xác suất như một lời hứa.

Đầu ra có cấu trúc
-------------------
Bắt LLM trả JSON theo `RESPONSE_SCHEMA` thay vì văn xuôi tự do. Văn xuôi thì
`validator.py` phải đoán đâu là khuyến nghị, đâu là giải thích; JSON thì kiểm
được từng phần, và phần nào hỏng thì thay phần đó chứ không bỏ cả câu trả lời.
"""
from __future__ import annotations

import json
from typing import Any, Final

#: Phiên bản prompt. ĐỔI KHI SỬA NỘI DUNG BÊN DƯỚI — nó đi kèm mọi câu trả
#: lời để về sau còn truy được câu nào sinh ra bởi bản prompt nào.
PROMPT_VERSION: Final[str] = "ai02-v1"

#: Khoá bắt buộc của JSON mà LLM phải trả về.
REQUIRED_KEYS: Final[tuple[str, ...]] = ("explanation", "recommendations")

#: Schema đầu ra, mô tả cho LLM bằng chính JSON để khỏi diễn giải bằng lời.
RESPONSE_SCHEMA: Final[dict[str, Any]] = {
    "explanation": "string — giải thích kết quả bằng tiếng Việt tự nhiên",
    "recommendations": [
        {
            "priority": "high | medium | low",
            "action": "string — việc cụ thể nên làm",
            "reason": "string — dựa trên số liệu nào trong dữ liệu đã cho",
        }
    ],
    "caveats": ["string — điều người dùng cần lưu ý (tuỳ chọn)"],
    "needs_more_data": ["string — dữ liệu còn thiếu, nếu có (tuỳ chọn)"],
}

SYSTEM_PROMPT: Final[str] = """\
Bạn là trợ lý tư vấn tài chính hộ gia đình cho người dùng Việt Nam.

VAI TRÒ CỦA BẠN
Bạn DIỄN ĐẠT kết quả đã được hệ thống tính sẵn. Bạn không phải người phân
tích, không phải người tính toán, và không phải người quyết định.

Hệ thống có ba tầng, và bạn đứng ở tầng cuối:
  · Tầng quy tắc (RB01–RB05): tính toán xác định theo công thức đã công bố
  · Tầng mô hình (ML01, ML02): phân loại, trả về nhãn kèm xác suất
  · Bạn: đọc kết quả hai tầng trên và giải thích cho người dùng

QUY TẮC TUYỆT ĐỐI
1. CHỈ dùng con số có trong `numeric_facts` của dữ liệu được cấp. Không tự
   tính, không cộng trừ, không ước lượng, không làm tròn thành số khác. Cần
   một con số không có trong đó thì nói là chưa có, đừng tạo ra nó.
2. KHÔNG thay đổi, không phản bác, không "điều chỉnh" nhãn hay xác suất của
   mô hình. Chúng là dữ kiện đầu vào của bạn.
3. KHÔNG biến dự đoán thành điều chắc chắn. Xác suất 85% vẫn còn 15% khả năng
   ngược lại. Dùng "khả năng cao", không dùng "sẽ", "chắc chắn", "đảm bảo".
4. KHÔNG cam kết lợi nhuận. KHÔNG khuyến nghị mã cổ phiếu, quỹ hay sản phẩm
   cụ thể của một tổ chức nào. Chỉ nói theo LỚP tài sản: tiền gửi, trái phiếu,
   chứng chỉ quỹ.

PHÂN BIỆT BA LOẠI THÔNG TIN
Khi viết, phải cho người đọc thấy rõ đâu là loại nào:
  · Kết quả TÍNH TOÁN (tầng quy tắc) — xác định, nói thẳng
  · Kết quả DỰ ĐOÁN (mô hình) — kèm mức xác suất và độ tin cậy
  · KHUYẾN NGHỊ của bạn — là gợi ý tham khảo suy từ hai loại trên

ĐỘ TIN CẬY
Nếu dữ liệu có `low_confidence: true`, bạn PHẢI nói ra rằng kết quả đó chưa
chắc chắn và nên đọc kèm phần đánh giá theo quy tắc. Không được im lặng bỏ
qua và trình bày nó như một kết luận chắc.

THIẾU DỮ LIỆU
Thiếu phần nào thì nói rõ thiếu gì và người dùng cần bổ sung ở đâu. Không suy
đoán thay họ.

NGOÀI PHẠM VI
Câu hỏi không thuộc tài chính hộ gia đình: từ chối lịch sự, nói rõ phạm vi
của bạn, và gợi ý vài câu hỏi bạn trả lời được.

GIỌNG VĂN
Tiếng Việt tự nhiên, ngắn gọn, xưng "bạn". Không dùng thuật ngữ tiếng Anh khi
đã có từ tiếng Việt. Không lặp lại số liệu đã nói ở lượt trước trừ khi người
dùng hỏi lại.

Kết thúc phần giải thích luôn kèm: đây là thông tin tham khảo, không phải tư
vấn tài chính chuyên nghiệp.
"""

#: Khuôn cho một lượt hỏi. `{...}` được điền bằng `render_user_prompt`.
USER_PROMPT_TEMPLATE: Final[str] = """\
CÂU HỎI CỦA NGƯỜI DÙNG
{question}

CHỦ ĐỀ ĐÃ XÁC ĐỊNH: {topic_label} (mã {intent})

{history_block}
DỮ LIỆU ĐÃ TÍNH SẴN
{payload}

CÁC CON SỐ BẠN ĐƯỢC PHÉP DÙNG
Chỉ những con số dưới đây. Bất kỳ số nào khác đều bị coi là bịa đặt.
{numeric_facts}
{warning_block}
YÊU CẦU
Trả lời bằng JSON đúng cấu trúc sau, không kèm giải thích nào ngoài JSON:
{schema}
"""


def _format_number(value: float) -> str:
    """Định dạng số theo kiểu Việt Nam để LLM chép lại đúng dạng đó.

    Cho LLM thấy sẵn "35.000.000" thì nó chép lại nguyên dạng, và bước kiểm số
    ở `validator.py` khớp ngay. Đưa số thô "35000000.0" thì LLM tự định dạng
    theo kiểu của nó và mỗi lần một khác.
    """
    if value != value:                       # NaN
        return "không có"
    if abs(value) >= 1000:
        return f"{value:,.0f}".replace(",", ".")
    if value == int(value):
        return str(int(value))
    return f"{value:.4f}".rstrip("0").rstrip(".")


def render_numeric_facts(facts: dict[str, float]) -> str:
    """Bảng con số dưới dạng danh sách gạch đầu dòng."""
    if not facts:
        return "  (không có con số nào — đừng nhắc tới bất kỳ số liệu nào)"
    return "\n".join(f"  · {name} = {_format_number(value)}"
                     for name, value in sorted(facts.items()))


def render_user_prompt(context, topic_label: str) -> str:
    """Dựng prompt cho một lượt hỏi từ `AiContext`."""
    data = context.to_dict()
    # Bảng con số đã có mục riêng nên bỏ khỏi phần payload cho gọn.
    payload = {k: v for k, v in data.items()
               if k not in ("numeric_facts", "history", "question")}

    history_block = ""
    if context.history:
        lines = "\n".join(
            f"  {'Người dùng' if turn['role'] == 'user' else 'Bạn'}: "
            f"{turn['content']}" for turn in context.history)
        history_block = (
            "HỘI THOẠI TRƯỚC ĐÓ (để hiểu câu hỏi nối tiếp, đừng lặp lại)\n"
            f"{lines}\n\n")

    warning_block = ""
    if context.warnings:
        lines = "\n".join(f"  · [{w.get('code')}] {w.get('message')}"
                          for w in context.warnings)
        warning_block = (
            "\nCẢNH BÁO BẮT BUỘC NÓI RA\n"
            "Những điều sau phải xuất hiện trong câu trả lời của bạn:\n"
            f"{lines}\n")

    return USER_PROMPT_TEMPLATE.format(
        question=context.question,
        topic_label=topic_label,
        intent=context.intent,
        history_block=history_block,
        payload=json.dumps(payload, ensure_ascii=False, indent=2),
        numeric_facts=render_numeric_facts(context.numeric_facts),
        warning_block=warning_block,
        schema=json.dumps(RESPONSE_SCHEMA, ensure_ascii=False, indent=2),
    )


#: Câu trả lời cho câu hỏi ngoài phạm vi — cố định, không gọi LLM.
#:
#: Gọi LLM để nó tự từ chối là mời nó nói về chủ đề ngoài phạm vi trước khi
#: từ chối, và đó chính là thứ cần tránh.
OUT_OF_SCOPE_REPLY: Final[dict] = {
    "explanation": (
        "Xin lỗi, mình chỉ hỗ trợ các câu hỏi về tài chính hộ gia đình: dòng "
        "tiền, ngân sách, tiết kiệm, xử lý nợ, khả năng vay và phân bổ tiền "
        "nhàn rỗi. Câu hỏi của bạn nằm ngoài phạm vi này.\n\n"
        "_Thông tin mình đưa ra là tham khảo, không phải tư vấn tài chính "
        "chuyên nghiệp._"),
    "recommendations": [],
    "caveats": [],
    "needs_more_data": [],
    "suggested_questions": [
        "Sức khỏe tài chính của gia đình tôi hiện thế nào?",
        "Tôi nên phân bổ thu nhập theo quy tắc 50/30/20 ra sao?",
        "Với thu nhập hiện tại, tôi vay được tối đa bao nhiêu?",
    ],
}
