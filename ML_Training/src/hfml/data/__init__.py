"""Tầng data — chuẩn bị dữ liệu & feature (F01 · M01 · Tuần 1).

Đầu vào của tầng: form onboarding của người dùng, và `application_train.csv`
của Home Credit. Đầu ra: ma trận feature sạch, sẵn sàng cho tầng `ml`.

Module trong tầng này (số hiệu = task trong PLAN.md §4):

    schema.py           task 4      Data contract của form đầu vào (pydantic)
    loader.py           task 5      Nạp Home Credit + quản lý nguồn dữ liệu
    quality.py          task 6, 7   Kiểm tra chất lượng + hash phiên bản dataset
    synthetic.py        F03 task 2  Sinh dân số hộ gia đình cho ML01
    preprocessing/      task 8–11, 14
    features/           task 12, 13

Hai quy tắc bắt buộc của tầng này:

1.  Mọi bước impute → encode → scale phải nằm trong một `sklearn.Pipeline`
    được `fit` CHỈ trên tập train (PLAN.md §4.4). Chạy rời trước khi split
    là rò rỉ dữ liệu, và inference sẽ lệch so với training.

2.  Không sinh feature dạng lag / rolling / xu hướng — dữ liệu là snapshot
    một thời điểm, không có lịch sử (PLAN.md §2).
"""
