"""Hàm sinh nhãn `g(·)` cho ML01 (F03 task 2).

Bản chốt 11/08/2026, đặc tả đầy đủ ở PLAN.md §6.1b. File này là hiện thực
duy nhất của `g(·)` — không nơi nào khác được định nghĩa lại bốn nhóm.

Thang mức độ CHÍNH LÀ thứ tự kiểm tra
-------------------------------------
    EMERGENCY    🔴 → DEBT_FOCUS 🟠 → BUILD_BUFFER 🟡 → GROWTH 🟢

Hộ thỏa nhiều điều kiện nhận nhãn nặng nhất. Nhờ vậy `g(·)` đơn trị mà không
cần luật phá hòa riêng.

⚠️ Ba chỉ số `savings_months`, `dti`, `savings_rate` mà `g(·)` dùng KHÔNG BAO
GIỜ được đưa vào `X`. Chúng là biến `g(·)` đặt ngưỡng lên; cho vào feature
set thì một cây sâu 3 tầng học thuộc nguyên `g(·)`. Danh sách `X` hợp lệ là
`RAW_FEATURES` ở cuối file, và `test_labeler.py` có test chặn việc này.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final

import numpy as np
import pandas as pd

from hfml.data.schema import ASSET_COLUMNS


class RecommendationGroup(str, Enum):
    """Bốn nhóm khuyến nghị, xếp theo mức độ nghiêm trọng GIẢM DẦN.

    Thứ tự khai báo có ý nghĩa: `severity` dùng nó để so sánh, và bước nhiễu
    nhãn chỉ đảo sang nhóm liền kề trong thang này.
    """

    EMERGENCY = "EMERGENCY"          # 🔴 Tài chính nguy cấp
    DEBT_FOCUS = "DEBT_FOCUS"        # 🟠 Cần tập trung xử lý nợ
    BUILD_BUFFER = "BUILD_BUFFER"    # 🟡 Cần xây dựng quỹ dự phòng
    GROWTH = "GROWTH"                # 🟢 Tài chính tương đối tốt

    @property
    def severity(self) -> int:
        """0 = nặng nhất. Dùng để đảo nhãn sang nhóm liền kề."""
        return ORDERED_GROUPS.index(self)


#: Thang mức độ, nặng → nhẹ. Đừng đổi thứ tự — nhiễu nhãn dựa vào nó.
ORDERED_GROUPS: Final[tuple[RecommendationGroup, ...]] = (
    RecommendationGroup.EMERGENCY,
    RecommendationGroup.DEBT_FOCUS,
    RecommendationGroup.BUILD_BUFFER,
    RecommendationGroup.GROWTH,
)

LABELS_VI: Final[dict[RecommendationGroup, str]] = {
    RecommendationGroup.EMERGENCY: "Tài chính nguy cấp",
    RecommendationGroup.DEBT_FOCUS: "Cần tập trung xử lý nợ",
    RecommendationGroup.BUILD_BUFFER: "Cần xây dựng quỹ dự phòng",
    RecommendationGroup.GROWTH: "Tài chính tương đối tốt, có thể tăng trưởng",
}


@dataclass(frozen=True)
class LabelThresholds:
    """Ngưỡng của `g(·)`, kèm nguồn. Chốt ở PLAN.md §6.1b.

    KHÔNG chỉnh các số này để "vừa" phân bố lớp. Khi một lớp dưới 10%, chỗ
    được chỉnh là tham số sinh dân số (`data/synthetic.py`) — sửa ngưỡng cho
    vừa dữ liệu là đảo ngược quan hệ nhân quả, và câu hỏi "sao lấy 0,40?" sẽ
    không có đáp án.
    """

    #: Dưới mốc này coi như không có đệm nào. Nguồn: khuyến nghị quỹ dự phòng
    #: 3–6 tháng chi tiêu.
    emergency_savings_months: float = 1.0
    #: Quy tắc 28/36 — DTI back-end ≤ 36%; nới lên 40% làm mức "cần xử lý".
    debt_focus_dti: float = 0.40
    #: Cận dưới của khuyến nghị quỹ dự phòng 3–6 tháng.
    buffer_savings_months: float = 3.0
    #: Mức tiết kiệm tối thiểu thường được khuyến nghị.
    buffer_savings_rate: float = 0.10


DEFAULT_THRESHOLDS: Final[LabelThresholds] = LabelThresholds()


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Ba chỉ số mà `g(·)` dùng. **Không phải feature** — xem cảnh báo đầu file.

    Đầu vào là các cột THÔ của form. Trả về khung riêng để không ai vô tình
    nối chúng vào `X`.
    """
    income = df["average_monthly_income"].astype(float)
    expense = df["average_monthly_expense"].astype(float)
    savings = df["savings_amount"].astype(float).fillna(0.0)
    payment = df["monthly_debt_payment"].astype(float).fillna(0.0)

    safe_income = income.where(income > 0)
    safe_expense = expense.where(expense > 0)

    return pd.DataFrame({
        # Không có chi tiêu thì đệm là vô hạn — dùng inf, không phải NaN, để
        # so sánh "< 1" vẫn cho kết quả đúng.
        "savings_months": (savings / safe_expense).replace(np.nan, np.inf),
        "dti": (payment / safe_income).fillna(0.0),
        "savings_rate": ((income - expense) / safe_income).fillna(0.0),
    }, index=df.index)


def label_frame(
    df: pd.DataFrame,
    thresholds: LabelThresholds = DEFAULT_THRESHOLDS,
) -> pd.Series:
    """Áp `g(·)` cho cả khung. Trả về `Series` nhãn dạng chuỗi.

    Thứ tự `np.select` chính là thang mức độ — điều kiện đầu tiên khớp sẽ
    thắng, nên hộ vừa nợ nặng vừa không có đệm rơi vào `EMERGENCY`.
    """
    ind = compute_indicators(df)
    t = thresholds

    conditions = [
        (ind["savings_rate"] < 0) | (ind["savings_months"] < t.emergency_savings_months),
        ind["dti"] >= t.debt_focus_dti,
        (ind["savings_months"] < t.buffer_savings_months)
        | (ind["savings_rate"] < t.buffer_savings_rate),
    ]
    choices = [
        RecommendationGroup.EMERGENCY.value,
        RecommendationGroup.DEBT_FOCUS.value,
        RecommendationGroup.BUILD_BUFFER.value,
    ]
    labels = np.select(conditions, choices, default=RecommendationGroup.GROWTH.value)
    return pd.Series(labels, index=df.index, name="label")


def distance_to_boundary(
    df: pd.DataFrame,
    thresholds: LabelThresholds = DEFAULT_THRESHOLDS,
) -> pd.Series:
    """Khoảng cách TƯƠNG ĐỐI tới ngưỡng gần nhất của `g(·)`.

    Dùng để chọn ra hồ sơ "sát biên" khi thêm nhiễu nhãn: chỉ đảo nhãn ở vùng
    mà ngay cả con người cũng phân vân, chứ không đảo bừa một hộ rõ ràng.

    Vì sao cả năm số hạng phải cùng đơn vị
    --------------------------------------
    `add_label_noise` so cả năm với MỘT con số `boundary_width`. Số hạng nào
    lệch đơn vị thì nó "sát biên" dễ hơn hẳn bốn cái kia, và vùng nhiễu nhãn
    bị nó chi phối.

    Đúng chỗ này từng sai (sửa 11/08/2026): `sr_0` để đơn vị tuyệt đối, nên
    với `boundary_width = 0,10` nó quét cả dải `savings_rate ∈ ±0,10` —
    **15,95%** dân số, nhiều hơn bốn số hạng kia cộng lại (2,9% · 5,8% · 5,9%
    · 2,6%). Hộ để dành 9,5% thu nhập bị coi là sát ranh giới *dòng tiền âm*,
    và nhiễu nhãn đổ vào đúng nhóm đó.

    Ngưỡng `savings_rate < 0` không chia tương đối được (chia cho 0), nên lấy
    `buffer_savings_rate` làm thước — đó là ngưỡng khác duy nhất trên cùng
    trục `savings_rate`, nên hai số hạng `sr_*` dùng chung mẫu số là nhất
    quán.
    """
    ind = compute_indicators(df)
    t = thresholds
    gaps = pd.DataFrame({
        "sm_1": (ind["savings_months"] - t.emergency_savings_months).abs()
                / t.emergency_savings_months,
        "sm_3": (ind["savings_months"] - t.buffer_savings_months).abs()
                / t.buffer_savings_months,
        "dti": (ind["dti"] - t.debt_focus_dti).abs() / t.debt_focus_dti,
        # Ngưỡng là 0 → lấy ngưỡng cùng trục làm thước, xem docstring.
        "sr_0": ind["savings_rate"].abs() / t.buffer_savings_rate,
        "sr_10": (ind["savings_rate"] - t.buffer_savings_rate).abs()
                 / t.buffer_savings_rate,
    })
    return gaps.replace(np.inf, np.nan).min(axis=1).fillna(np.inf)


def add_label_noise(
    labels: pd.Series,
    df: pd.DataFrame,
    rate: float = 0.03,
    boundary_width: float = 0.10,
    seed: int = 42,
    thresholds: LabelThresholds = DEFAULT_THRESHOLDS,
) -> pd.Series:
    """Đảo `rate` phần nhãn sang nhóm LIỀN KỀ về mức độ (PLAN.md §6.2).

    Hai ràng buộc, cả hai đều có lý do:

    1.  Chỉ đảo trong vùng sát biên (`boundary_width`). Đảo một hộ nằm sâu
        trong `GROWTH` thành `EMERGENCY` là nhiễu vô nghĩa, không mô phỏng
        hoàn cảnh nào có thật.
    2.  Chỉ đảo sang nhóm liền kề. `GROWTH` → `BUILD_BUFFER` hợp lý;
        `GROWTH` → `EMERGENCY` thì không.

    Không có bước này thì ranh giới sạch tuyệt đối và mọi thuật toán đạt
    ~100%, bảng so sánh 4 thuật toán mất hết ý nghĩa.
    """
    rng = np.random.default_rng(seed)
    out = labels.copy()

    near = distance_to_boundary(df, thresholds) <= boundary_width
    candidates = out.index[near]
    n_flip = min(int(round(len(out) * rate)), len(candidates))
    if n_flip == 0:
        return out

    chosen = rng.choice(candidates, size=n_flip, replace=False)
    for idx in chosen:
        current = RecommendationGroup(out.loc[idx])
        pos = current.severity
        neighbours = [p for p in (pos - 1, pos + 1) if 0 <= p < len(ORDERED_GROUPS)]
        out.loc[idx] = ORDERED_GROUPS[rng.choice(neighbours)].value
    return out


def class_distribution(labels: pd.Series) -> pd.DataFrame:
    """Bảng phân bố 4 lớp — cổng kiểm chứng 1 của PLAN.md §6.2."""
    counts = labels.value_counts()
    rows = [
        {
            "label": g.value,
            "label_vi": LABELS_VI[g],
            "n": int(counts.get(g.value, 0)),
            "share": float(counts.get(g.value, 0) / len(labels)) if len(labels) else 0.0,
        }
        for g in ORDERED_GROUPS
    ]
    return pd.DataFrame(rows)


#: Feature hợp lệ của ML01 — CHỈ biến thô của form (PLAN.md §6.1c, chốt lại
#: 11/08/2026 sau khi đối chiếu `HouseholdProfile`).
#:
#: Tiêu chí nhận một trường vào `X` — hai điều kiện, phải thỏa CẢ HAI:
#:
#:   1. **Luôn thu được** với mọi người dùng ML01. ML01 chấm sức khỏe tài
#:      chính cho mọi hồ sơ, không riêng người tính vay.
#:   2. **Mã hóa được** mà không phải tự bịa ra vocabulary.
#:
#: Trường bị loại và lý do nằm ở `EXCLUDED_FROM_X` — không xóa nó đi, đó là
#: phần trả lời cho câu "sao không dùng nghề nghiệp?" của hội đồng.
RAW_FEATURES: Final[tuple[str, ...]] = (
    # -- Dòng tiền, nợ, tiết kiệm
    "average_monthly_income",
    "average_monthly_expense",
    "savings_amount",
    "total_current_debt",
    "monthly_debt_payment",
    # -- Nhân khẩu
    "household_size",
    "children_count",
    "age",
    # -- Cờ tình trạng
    "has_debt",
    "has_savings",
    "has_dependents",
    # -- Tài sản sở hữu, multi-hot từ `HouseholdProfile.assets`
    *ASSET_COLUMNS,
)

#: Trường CÓ trong form nhưng cố ý không vào `X` của ML01, kèm lý do.
#:
#: Khác hẳn `FORBIDDEN_IN_X` bên dưới: chỗ này là **không dùng được**, chỗ
#: kia là **rò rỉ nhãn**. Hai loại lý do khác nhau, đừng gộp.
EXCLUDED_FROM_X: Final[dict[str, str]] = {
    "occupation":
        "Nằm trong KHỐI VAY của schema — chỉ hiện khi người dùng chọn "
        "`home_loan`, nên phần lớn hồ sơ ML01 để trống. Train trên dân số "
        "ai cũng có nghề rồi suy luận cho người bỏ trống là lệch phân phối "
        "train/inference. Trường này phục vụ ML02, đúng vai schema đã định.",
    "employment_years":
        "Cùng KHỐI VAY với `occupation`, cùng một lý do.",
    "residence":
        "`str | None` tự do tới 255 ký tự, không có tập giá trị chuẩn. Mã "
        "hóa được thì phải tự dựng vocabulary — mà một vocabulary bịa ra "
        "không đại diện cho phân bố địa lý nào có thật.",
    "representative_name":
        "Định danh cá nhân, không mang thông tin tài chính.",
    "financial_needs":
        "Là điều người dùng MUỐN, không phải hoàn cảnh họ ĐANG có. Đưa vào "
        "thì model học 'ai chọn mục tiêu tiết kiệm thì đang thiếu tiết kiệm'.",
    "asset_price": "Khối vay — xem `occupation`.",
    "loan_amount": "Khối vay — xem `occupation`.",
    "loan_term_months": "Khối vay — xem `occupation`.",
    "guest_session_id":
        "Mã phiên của backend, không phải thuộc tính của hộ. Chính schema "
        "đã ghi `ML service không dùng`.",
}

#: Cột tuyệt đối không được vào `X` vì RÒ RỈ NHÃN — chúng là biến `g(·)` đặt
#: ngưỡng lên. Cây sâu 3 tầng học thuộc nguyên `g(·)`, mọi thuật toán đạt
#: ~100%, bảng so sánh 4 thuật toán mất sạch ý nghĩa (PLAN.md §14).
#:
#: `test_labeler.py` kiểm tra giao với `RAW_FEATURES` là rỗng.
FORBIDDEN_IN_X: Final[tuple[str, ...]] = ("savings_months", "dti", "savings_rate", "label")

#: Trường tiền có điều kiện: `None` trong form NGHĨA LÀ 0, không phải "thiếu".
#:
#: `savings_amount` để trống khi `has_savings = False` — đó là số 0 đã biết
#: chắc, không phải giá trị chưa biết. Điền trung vị vào đây là bịa cho hộ
#: không tiết kiệm một khoản tiết kiệm bằng nửa dân số.
#:
#: Vì vậy `data.synthetic` sinh thẳng `0.0`, và `pipeline.normalizer` lúc
#: inference phải đổi `None → 0.0` cho đúng ba cột này TRƯỚC khi vào
#: Pipeline preprocessing. Cờ `has_debt` / `has_savings` đã mang sẵn thông
#: tin "có hay không", nên không mất gì.
ZERO_WHEN_ABSENT: Final[tuple[str, ...]] = (
    "savings_amount",
    "total_current_debt",
    "monthly_debt_payment",
)
