"""Tầng llm — diễn đạt kết quả ra tiếng Việt (F05 · M06 · Tuần 6).

LLM đóng đúng một vai: NGƯỜI DIỄN ĐẠT. Nó nhận structured result JSON đã
tính sẵn từ `hfml.pipeline` và viết lại thành lời giải thích + khuyến nghị.
TUYỆT ĐỐI không tính toán số.

    prompts.py      task 8       Prompt giải thích kết quả
    context.py      task 9       Dựng context từ Rule + ML
    client.py       task 10, 11  Gọi model, sinh giải thích + khuyến nghị
    chat.py         task 12      Hội thoại chatbot (mức demo)
    guardrails.py   task 13      Safety guardrails
    validator.py    task 14      Validate output

Bốn guardrail bắt buộc (PLAN.md §8.2):

1.  Prompt CHỈ chứa JSON đã tính sẵn từ Rule/ML — cấm LLM tự tính.
2.  Validate hậu kiểm: trích mọi số trong câu trả lời bằng regex, đối chiếu
    với JSON đầu vào; số nào không khớp → chặn và sinh lại.
3.  Disclaimer bắt buộc: chỉ khuyến nghị PHÂN BỔ THEO LỚP TÀI SẢN (tiền gửi
    / trái phiếu / quỹ), KHÔNG khuyến nghị mã cụ thể. Ghi rõ "thông tin tham
    khảo, không phải tư vấn tài chính chuyên nghiệp".
4.  Kết quả ML02 phải diễn đạt là ước lượng tham khảo, không phải cam kết.
    Câu hỏi ngoài phạm vi → từ chối lịch sự + gợi ý bộ câu hỏi mẫu.

Chatbot (task 12) đi theo hai hướng, KHÔNG dùng ML router:

    Hướng 1 (chính)  Bộ câu hỏi mẫu gắn sẵn `intent_code` → vào thẳng engine,
                     không qua phân loại. Nhanh, chính xác, ít phụ thuộc LLM.
    Hướng 2          Người dùng tự nhập → phân loại bằng keyword + rule thành
                     BUSINESS / FINANCE_GENERAL / OUT_OF_SCOPE.
                     OUT_OF_SCOPE fallback về Hướng 1.

Bản v3 từng dự kiến intent router bằng ML. Kế hoạch hiện tại chốt KHÔNG mở
rộng bài toán ML thứ ba, nên router là keyword + rule (PLAN.md §8.2).
"""
