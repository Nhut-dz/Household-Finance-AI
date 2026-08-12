"""Tiền xử lý — làm sạch, encode, scale (F01 task 8–11, 14).

    cleaner.py    task 8, 9    Missing values, duplicate, giá trị bất hợp lệ
    encoders.py   task 10, 11  Encoding categorical + scaling numerical
    pipeline.py   task 14      Đóng gói toàn bộ vào sklearn Pipeline

Hai bẫy dữ liệu đã biết, xử lý ở `cleaner.py` (PLAN.md §4.2):

    - Home Credit: `DAYS_EMPLOYED` có sentinel 365243 (≈1000 năm) → NaN.
    - Form người dùng: cảnh báo khi tỉ lệ tiết kiệm > 60% hoặc chi > thu.

`pipeline.py` là thành phẩm quan trọng nhất của F01 — nó được `joblib.dump`
cùng model và dùng lại nguyên vẹn lúc inference (`hfml.pipeline`).
Đây là mục "không được cắt" trong PLAN.md §13.
"""
