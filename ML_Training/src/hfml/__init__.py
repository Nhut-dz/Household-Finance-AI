"""hfml — Household Finance ML.

Pipeline Python cho hệ thống "AI tư vấn tài chính hộ gia đình" (xem PLAN.md).

Kiến trúc 5 tầng, mỗi tầng là một package, xếp đúng thứ tự dữ liệu chảy qua:

    data      → chuẩn bị dữ liệu & feature      (F01 · M01)
    rules     → 5 rule tài chính xác định        (F02 · M02)
    ml        → ML01 + ML02, train & đánh giá    (F03, F04 · M03, M04)
    pipeline  → inference, tổng hợp Rule + ML    (F05 · M05)
    llm       → diễn đạt kết quả ra tiếng Việt   (F05 · M06)
    api       → FastAPI demo                     (F05 · M08)

Ranh giới cứng giữa các tầng (PLAN.md §1) — vi phạm là hỏng kiến trúc:

    rules  không "học" gì cả, chỉ tính toán xác định.
    ml     chỉ trả nhãn + xác suất, KHÔNG tự sinh khuyến nghị.
    llm    chỉ diễn đạt JSON đã tính sẵn, TUYỆT ĐỐI không tính toán số.

Ràng buộc dữ liệu chi phối toàn bộ thiết kế (PLAN.md §2): dữ liệu chỉ thu
một lần qua form onboarding → mọi mô hình là cross-sectional (một dòng hồ sơ
vào, một kết quả ra). Không lag, không rolling, không dự báo chuỗi thời gian.
"""

__version__ = "0.2.0"
