"""Tầng rules — Rule-Based Financial Analysis Engine (F02 · M02 · Tuần 2).

Năm rule tài chính xác định. Tầng này KHÔNG "học" gì cả — mỗi rule là một
hàm thuần: cùng đầu vào luôn cho cùng đầu ra.

    rb01_cashflow.py       Thu nhập, chi tiêu, số dư = thu − chi
    rb02_health.py         Sức khỏe tài chính: dti, savings_months, savings_rate → 4 mức
    rb03_savings_goal.py   Tiến độ mục tiêu: cần/tháng = (mục tiêu − tích lũy) ÷ tháng còn lại
    rb04_503020.py         Phân bổ 50/30/20 đề xuất từ thu nhập
    rb05_loan_capacity.py  Khả năng vay: DTI ≤ ngưỡng và LTV ≤ ngưỡng → hạn mức tối đa
    thresholds.py          Nạp ngưỡng từ config/rules.yaml
    engine.py              Chạy cả 5 rule, gom kết quả

Bốn rule ánh xạ 1-1 với `goal_type` trong `tblfinancial_goals` — mỗi nhu cầu
người dùng chọn sẽ kích hoạt đúng một rule (xem `schema.NEED_TO_RULE`):

    saving          → RB03        home_loan  → RB05 + ML02
    budget_50_30_20 → RB04        investment → chưa có rule

RB04 là rule KÊ ĐƠN, không phải rule CHẨN ĐOÁN
----------------------------------------------
Đây là điểm dựng lại sau khi đọc dữ liệu thật (600 dòng `tblcalculation_results`
của `Household_Finance_V2_Dev.sql`, 11/08/2026). Backend đang tính:

    budget_needs   = 50% × monthly_income
    budget_wants   = 30% × monthly_income
    budget_savings = 20% × monthly_income      tổng = đúng 100% thu nhập

Tức RB04 trả về **mức phân bổ ĐỀ XUẤT** từ thu nhập, KHÔNG so sánh với chi
tiêu thực tế theo từng nhóm. Nhờ vậy nó **không cần thêm ô nhập nào** — form
giữ nguyên. Đây là lý do RB04 quay lại phạm vi sau khi từng bị loại vì tưởng
phải hỏi thêm 3 ô Needs/Wants/Savings.

Phần chẩn đoán vẫn làm được một nửa từ dữ liệu sẵn có: so `monthly_living_cost`
(tổng chi thực tế) với mức đề xuất cho needs+wants (80% thu nhập), và so tỉ lệ
tiết kiệm thực tế `(thu − chi) ÷ thu` với mốc 20%. Không tách được chi thực tế
thành needs và wants — phải ghi rõ giới hạn đó, đừng phát biểu như thể tách được.

Hai quy tắc bắt buộc:

1.  KHÔNG hardcode hệ số / ngưỡng. Toàn bộ đặt trong `config/rules.yaml`,
    mỗi ngưỡng kèm cột nguồn trích dẫn (50/30/20, DTI ≤ 36–40%, LTV ≤ 70–80%)
    — để trả lời được câu hỏi "sao lấy 36%?" (PLAN.md §5).

2.  Mỗi rule trả về dict cùng cấu trúc, để tầng `llm` chỉ việc diễn đạt
    chứ không phải suy luận:

        {"code": ..., "status": ..., "value": ..., "threshold": ..., "message_key": ...}

Cách đánh giá đúng cho tầng này (PLAN.md §5) — rule là hàm xác định, không
có ground truth độc lập, nên KHÔNG chấm accuracy / F1 ở đây. Thành phẩm là:
bảng ngưỡng có nguồn + ma trận case biên (mỗi ngưỡng test 3 điểm: dưới /
đúng bằng / trên) + unit test pass 100%.
"""
from hfml.rules.thresholds import RuleThresholds, load_rule_thresholds
from hfml.rules.rb01_cashflow import evaluate_cashflow
from hfml.rules.rb02_health import evaluate_financial_health
from hfml.rules.rb03_savings_goal import evaluate_savings_goal
from hfml.rules.rb04_503020 import evaluate_503020
from hfml.rules.rb05_loan_capacity import evaluate_loan_capacity
from hfml.rules.engine import RuleEngine

__all__ = [
    "RuleThresholds",
    "load_rule_thresholds",
    "evaluate_cashflow",
    "evaluate_financial_health",
    "evaluate_savings_goal",
    "evaluate_503020",
    "evaluate_loan_capacity",
    "RuleEngine",
]
