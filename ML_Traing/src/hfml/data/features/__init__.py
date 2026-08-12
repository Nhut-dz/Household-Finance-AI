"""Feature engineering & selection (F01 task 12, 13).

    builder.py     task 12   Sinh feature tài chính dạng TỈ LỆ
    selection.py   task 13   Chọn feature

Quy tắc feature tỉ lệ (PLAN.md §2.1) — nền tảng để model train trên Home
Credit dùng được cho người dùng Việt Nam. Home Credit không dùng VNĐ:
`AMT_INCOME_TOTAL` trung vị ≈ 147.150 trong khi người dùng VN nhập
50.000.000 — lệch ~340 lần. Gặp giá trị ngoài phân phối huấn luyện, model
trả về số vô nghĩa mà KHÔNG báo lỗi.

    Loại bỏ mọi giá trị tiền tuyệt đối khỏi feature set của ML02.

    dti                  = trả nợ tháng ÷ thu nhập tháng
    ltv                  = tiền vay ÷ giá nhà
    credit_income_ratio  = tiền vay ÷ thu nhập năm
    income_per_capita    = thu nhập ÷ nhân khẩu
    savings_months       = tiết kiệm ÷ chi tiêu tháng
    debt_income_ratio    = dư nợ ÷ thu nhập năm

Feature phi tiền tệ giữ nguyên: tuổi, số con, số nhân khẩu, số năm đi làm,
nghề nghiệp, sở hữu nhà/xe/đất, vùng.

Trong báo cáo, mục này trình bày dưới tên "xử lý bài toán chuyển miền
(domain transfer) giữa dataset nghiên cứu và dữ liệu người dùng Việt Nam".
"""
