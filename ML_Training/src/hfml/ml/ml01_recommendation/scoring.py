"""ML01 — Sinh nhãn bằng điểm số đa chiều (F03 · redesign 17/08/2026).

Thay cho thang `if/else` cũ. Bốn nhóm được cho ĐIỂM song song, nhãn là nhóm
điểm cao nhất — không có nhóm nào được quyết định bởi một phép so sánh.

Vì sao bản cũ phải bỏ
-----------------------
Đo được trên 20.000 hồ sơ:

    P(EMERGENCY | net_cashflow < 0) = 1.000
    P(dti ≥ 0.40 | DEBT_FOCUS)      = 1.000
    cây quyết định sâu 5 trên 3 tỉ số → accuracy 1.0000

Nhãn cũ là hàm bậc thang của ba tỉ số, nên "huấn luyện" chỉ là nội suy lại
chính công thức đó. Không có phần dư ngẫu nhiên nào để học.

Hai cơ chế tạo ra bài toán học thật
-------------------------------------
**1. Điểm liên tục, nhiều thành phần.** Mỗi điểm là tổng có trọng số của 4–5
tín hiệu, mỗi tín hiệu là một hàm logistic TRƠN của một chỉ số tài chính.
Không có bậc thang, nên không có nhát cắt nào tách trọn một nhóm. Hai hộ cùng
`dti = 45%` rơi vào hai nhóm khác nhau nếu đệm và dòng tiền của họ khác nhau —
đúng như yêu cầu.

**2. Nhãn tính trên giá trị THẬT, feature là giá trị KHAI BÁO.** Đây mới là
chỗ tạo ra sai số Bayes. Hộ gia đình khai thu nhập và chi tiêu có sai lệch —
chi tiêu thường bị khai thiếu, thu nhập bất thường hay bị bỏ sót. `synthetic.py`
sinh giá trị thật, chấm điểm trên đó, rồi mới thêm sai số khai báo vào phần
feature. Model vì vậy nhìn thấy một bản nhiễu của thứ đã quyết định nhãn, và
**không mô hình nào đạt được accuracy tuyệt đối** — giới hạn đó là thật, không
phải do cố tình làm khó.

Điều KHÔNG được làm
---------------------
Không đảo nhãn ngẫu nhiên để cân bằng lớp. Đảo nhãn là phá ground truth: nó
tạo ra những hộ mà nhãn mâu thuẫn với chính hoàn cảnh của họ, và model học
được từ đó chỉ là học cách bắt chước một phép tung đồng xu. Mất cân bằng lớp
xử lý bằng `class_weight` và chia tầng khi tách tập.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

from hfml.rules import indicators as rule_indicators

#: Bốn nhóm, đúng thứ tự cột của ma trận điểm.
GROUPS: Final[tuple[str, ...]] = (
    "EMERGENCY", "DEBT_FOCUS", "BUILD_BUFFER", "GROWTH")


def _logistic(x, midpoint: float, scale: float):
    """Chuyển một chỉ số thành tín hiệu trơn trong khoảng (0, 1).

    `midpoint` là chỗ tín hiệu bằng 0,5; `scale` quyết định độ dốc. Dùng hàm
    trơn thay cho phép so sánh nhị phân là điều làm nên khác biệt: một bậc
    thang tạo ra ranh giới mà một nhát cắt học được trọn vẹn, còn hàm trơn
    buộc model phải kết hợp nhiều chiều mới định được nhóm.

    `scale` càng nhỏ thì càng gần bậc thang — không đặt quá nhỏ.
    """
    return 1.0 / (1.0 + np.exp(-(np.asarray(x, dtype=float) - midpoint) / scale))


@dataclass(frozen=True)
class ScoreWeights:
    """Trọng số của từng tín hiệu trong bốn điểm.

    Ràng buộc thiết kế: trong mỗi nhóm, KHÔNG tín hiệu nào vượt 0,35 tổng
    trọng số. Vượt qua mức đó thì tín hiệu ấy một mình định đoạt nhóm, và ta
    quay lại đúng bài toán bậc thang cũ. `test_scoring.py` canh ràng buộc này.
    """

    emergency: dict = None
    debt: dict = None
    buffer: dict = None
    growth: dict = None

    @staticmethod
    def default() -> "ScoreWeights":
        return ScoreWeights(
            # Không đủ sống qua tháng — mức độ cấp thiết nhất.
            emergency={
                "deficit_depth": 0.30,    # thâm hụt sâu tới đâu so với thu nhập
                "no_runway": 0.28,        # đệm gần như không có
                "expense_pressure": 0.20,  # chi tiêu nuốt gần hết thu nhập
                "debt_strain": 0.12,      # trả nợ góp phần vào thâm hụt
                "dependent_load": 0.10,   # miệng ăn trên mỗi đồng thu nhập
            },
            # Nợ là vấn đề trội — còn sống được nhưng nợ đang siết.
            debt={
                "dti_level": 0.28,        # trả nợ trên thu nhập
                "debt_stock": 0.24,       # dư nợ so với thu nhập năm
                "payment_share": 0.20,    # trả nợ chiếm bao nhiêu dòng ra
                "thin_after_debt": 0.16,  # còn lại bao nhiêu sau khi trả nợ
                "buffer_relief": -0.12,   # có đệm dày thì bớt cấp thiết
            },
            # Sống ổn nhưng chưa có đệm.
            buffer={
                "buffer_gap": 0.30,       # còn thiếu bao nhiêu so với 3–6 tháng
                "low_saving_rate": 0.24,  # tích luỹ chậm
                "has_capacity": 0.20,     # có dư để tích luỹ
                "small_stock": 0.16,      # số dư tuyệt đối còn mỏng
                "debt_relief": -0.10,     # nợ nặng thì ưu tiên nợ trước
            },
            # Không còn việc cấp thiết nào chắn đường.
            growth={
                "strong_surplus": 0.26,
                "deep_buffer": 0.24,
                "low_debt": 0.20,
                "saving_momentum": 0.18,
                "per_capita_room": 0.12,
            },
        )


DEFAULT_WEIGHTS: Final[ScoreWeights] = ScoreWeights.default()


def build_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Các tín hiệu trơn trong (0, 1), làm nguyên liệu cho bốn điểm.

    Nhận khung có các cột tiền THÔ. Mọi chỉ số lấy từ `rules.indicators` —
    nguồn định nghĩa duy nhất của cả hệ thống.
    """
    ind = rule_indicators.compute_frame(df)

    income = df["average_monthly_income"].astype(float).clip(lower=1.0)
    expense = df["average_monthly_expense"].astype(float).fillna(0.0)
    savings = df["savings_amount"].astype(float).fillna(0.0)
    payment = df["monthly_debt_payment"].astype(float).fillna(0.0)
    total_debt = df["total_current_debt"].astype(float).fillna(0.0)
    size = df["household_size"].astype(float).clip(lower=1.0)

    net = ind["net_cashflow"]
    # `emergency_months` có thể là inf khi hộ không khai chi tiêu; cắt trần để
    # phép nhân trọng số phía sau không lan ra NaN.
    months = ind["emergency_months"].replace(np.inf, 60.0).clip(upper=60.0)
    dti = ind["dti"]
    rate = ind["savings_rate"]

    expense_ratio = expense / income
    debt_years = total_debt / (income * 12.0)
    outflow = (expense + payment).clip(lower=1.0)
    per_capita_surplus = net / size / income   # dư trên mỗi người, chuẩn hoá

    return pd.DataFrame({
        # -- nguyên liệu của EMERGENCY
        "deficit_depth": _logistic(-net / income, 0.02, 0.09),
        "no_runway": _logistic(-months, -1.6, 0.9),
        "expense_pressure": _logistic(expense_ratio, 0.82, 0.11),
        "debt_strain": _logistic(dti, 0.34, 0.11),
        "dependent_load": _logistic(size / (income / 1e7), 2.2, 1.1),

        # -- nguyên liệu của DEBT_FOCUS
        "dti_level": _logistic(dti, 0.29, 0.085),
        "debt_stock": _logistic(debt_years, 1.5, 0.75),
        "payment_share": _logistic(payment / outflow, 0.30, 0.12),
        "thin_after_debt": _logistic(-rate, -0.10, 0.11),
        "buffer_relief": _logistic(months, 5.0, 2.2),

        # -- nguyên liệu của BUILD_BUFFER
        "buffer_gap": _logistic(-months, -3.4, 1.5),
        "low_saving_rate": _logistic(-rate, -0.16, 0.10),
        "has_capacity": _logistic(net / income, 0.04, 0.08),
        "small_stock": _logistic(-savings / income, -3.0, 2.0),
        "debt_relief": _logistic(dti, 0.32, 0.10),

        # -- nguyên liệu của GROWTH
        "strong_surplus": _logistic(rate, 0.22, 0.09),
        "deep_buffer": _logistic(months, 5.5, 2.0),
        "low_debt": _logistic(-dti, -0.20, 0.09),
        "saving_momentum": _logistic(savings / income, 5.0, 2.5),
        "per_capita_room": _logistic(per_capita_surplus, 0.06, 0.05),
    }, index=df.index)


def compute_scores(
    df: pd.DataFrame,
    weights: ScoreWeights = DEFAULT_WEIGHTS,
) -> pd.DataFrame:
    """Bốn điểm cho mỗi hộ. Cột theo đúng thứ tự `GROUPS`."""
    signals = build_signals(df)
    blocks = {"EMERGENCY": weights.emergency, "DEBT_FOCUS": weights.debt,
              "BUILD_BUFFER": weights.buffer, "GROWTH": weights.growth}

    scores = {}
    for group, block in blocks.items():
        total = pd.Series(0.0, index=df.index)
        for name, weight in block.items():
            total = total + weight * signals[name]
        scores[group] = total
    return pd.DataFrame(scores, index=df.index)[list(GROUPS)]


def label_from_scores(scores: pd.DataFrame) -> pd.Series:
    """Nhãn = nhóm có điểm cao nhất.

    Không có ngưỡng nào ở đây. Nhóm thắng phụ thuộc vào TƯƠNG QUAN giữa bốn
    điểm, mà mỗi điểm lại là tổ hợp của năm tín hiệu — nên không phép so sánh
    đơn lẻ nào tái tạo được nhãn.
    """
    return pd.Series(scores.values.argmax(axis=1), index=scores.index).map(
        dict(enumerate(GROUPS))).rename("recommendation_group")


def score_margin(scores: pd.DataFrame) -> pd.Series:
    """Khoảng cách giữa nhóm nhất và nhóm nhì.

    Nhỏ nghĩa là hồ sơ nằm ở vùng giao giữa hai nhóm — chính là vùng mà một
    bộ phân loại tốt phải trả về xác suất chia đều thay vì tỏ ra chắc chắn.
    Dùng để đo mức độ chồng lấn của dữ liệu, không đưa vào `X`.
    """
    top2 = np.sort(scores.values, axis=1)[:, -2:]
    return pd.Series(top2[:, 1] - top2[:, 0], index=scores.index)


def label_frame(df: pd.DataFrame,
                weights: ScoreWeights = DEFAULT_WEIGHTS) -> pd.Series:
    """Gán nhãn cho cả khung — điểm số rồi lấy nhóm cao nhất."""
    return label_from_scores(compute_scores(df, weights))
