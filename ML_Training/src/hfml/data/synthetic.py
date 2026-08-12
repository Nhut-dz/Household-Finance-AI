"""Sinh dân số hộ gia đình cho ML01 (F03 task 3).

Vì sao không dùng thẳng 300 hộ trong DB
---------------------------------------
Đã thử: áp `g(·)` lên 300 hộ thật của `Household_Finance_V2_Dev` cho ra
**EMERGENCY 4,0% · DEBT_FOCUS 2,0% · BUILD_BUFFER 11,3% · GROWTH 82,7%** —
ba lớp dưới mốc 10% của cổng kiểm chứng PLAN.md §6.2.

Lý do nằm ở bản chất dữ liệu đó: nó là **300 dòng demo được seed**, và được
seed rất lành —

    thu nhập trung vị     49.000.000 đ/tháng   (cao hơn hẳn hộ VN điển hình)
    chi tiêu / thu nhập   trung vị 0,607
    số hộ chi ≥ thu       0
    có tiết kiệm          100%
    savings_months        trung vị 11,94
    dti                   trung vị 0,099

Một dân số không có hộ nào chi vượt thu thì không thể có nhóm `EMERGENCY`.

PLAN.md §6.2 nói rõ hướng xử lý: khi một lớp dưới 10% thì chỉnh **tham số
sinh dân số**, KHÔNG chỉnh ngưỡng `g(·)`. File này là chỗ chứa các tham số đó.

Cách sinh
---------
Mỗi hộ có một biến ẩn `health` ∈ [0, 1] chi phối đồng thời tỉ lệ chi tiêu,
mức đệm tiết kiệm và gánh nặng nợ — vì trong thực tế ba thứ này tương quan
với nhau (hộ khó khăn thường vừa chi sát thu, vừa ít tiết kiệm, vừa nợ nhiều).

Nhưng mỗi chỉ số còn có nhiễu riêng đủ lớn, nên một hộ vẫn có thể chi tiêu
gọn mà vẫn nợ nặng. Nếu để `health` quyết định hoàn toàn thì bốn nhóm xếp
thành một trật tự một chiều và bài toán mất chiều sâu.

`health` KHÔNG nằm trong dữ liệu trả ra — nó chỉ là công cụ sinh.

Dân số này KHÔNG phải ước lượng về hộ gia đình Việt Nam
-------------------------------------------------------
`population_summary()` in ra bảng đối chiếu, và bảng đó lệch khá xa:

    thu nhập trung vị    21,6 tr  ↔  49,0 tr của DB
    tỉ lệ hộ có nợ       70,0%    ↔  93,3%
    tỉ lệ hộ có tiết kiệm 85,2%   ↔  100%

Phải nói đúng bản chất chỗ lệch này, vì đây là câu hỏi hội đồng dễ đặt ra
nhất ("sao dân số của em nghèo hơn DB một nửa?"):

1.  **300 hộ trong DB không phải mẫu khảo sát.** Đó là dữ liệu demo được
    seed để chạy thử ứng dụng. Trung vị 49 triệu/tháng với 0 hộ chi vượt thu
    và 100% có tiết kiệm không mô tả tổng thể nào có thật. Nó không phải cái
    mốc để bám vào, nên "lệch" so với nó tự thân không phải lỗi.

2.  **Dạng phân phối có căn cứ, tham số cụ thể thì không.** Log-normal là
    lựa chọn tiêu chuẩn cho thu nhập, và các tỉ lệ nhân khẩu (`household_size`,
    `dependents_prob`) cùng tỉ lệ sở hữu tài sản (`ASSET_BASE_RATES`) thì đo
    thẳng từ DB. Nhưng `income_log_mu = 16,9` và `σ = 0,75` là **tham số của
    một thí nghiệm**, không phải ước lượng khớp từ vi dữ liệu GSO. Đừng viết
    trong báo cáo rằng dân số này "theo phân phối thu nhập GSO" — đó là câu
    không đỡ được khi bị hỏi nguồn.

3.  **Điều duy nhất tham số này cần đạt** là ba cổng kiểm chứng ở PLAN.md
    §6.2: bốn lớp đều ≥ 10%, ranh giới đủ dày, mọi model thắng baseline. Vai
    trò của ML01 (PLAN.md §6.2) là *đo năng lực thuật toán khôi phục một ranh
    giới đã biết*, không phải chứng minh điều gì về hộ gia đình Việt Nam —
    nên tính đại diện của dân số không nằm trong những gì ML01 khẳng định.

Ngược lại, nếu để tham số bám sát 300 hộ demo thì mới hỏng thật: dân số ra
82,7% `GROWTH` và bảng per-class của ba lớp còn lại không đọc được.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import numpy as np
import pandas as pd

from hfml.data.schema import ASSET_COLUMNS, AssetType, asset_column
from hfml.logger import get_logger

log = get_logger(__name__)

#: Phân bố số nhân khẩu, đo từ 300 hộ thật trong DB.
HOUSEHOLD_SIZE_WEIGHTS: Final[dict[int, float]] = {
    1: 45, 2: 80, 3: 74, 4: 59, 5: 35, 6: 7,
}

#: Tỉ lệ hộ sở hữu từng loại tài sản — đếm từ 450 dòng `tblassets` của
#: `Household_Finance_V2_Dev` chia cho 300 hộ. Không phải số bịa: mỗi giá trị
#: là `số dòng / 300` lấy thẳng từ chú thích của `AssetType`.
ASSET_BASE_RATES: Final[dict[AssetType, float]] = {
    AssetType.CASH: 116 / 300,
    AssetType.VEHICLE: 112 / 300,
    AssetType.REAL_ESTATE: 72 / 300,
    AssetType.INSURANCE: 50 / 300,
    AssetType.GOLD: 50 / 300,
    AssetType.INVESTMENT: 50 / 300,
}


@dataclass
class PopulationParams:
    """Tham số sinh dân số. **Đây là chỗ được phép chỉnh** khi một lớp < 10%.

    Ngưỡng của `g(·)` thì không — xem `labeler.LabelThresholds`.
    """

    n: int = 20_000

    # -- Thu nhập: log-normal. Tham chiếu hình dạng từ DB (mu 17,62 σ 0,60)
    # nhưng hạ trung vị và nới độ lệch để phủ cả hộ thu nhập thấp.
    income_log_mu: float = 16.9          # trung vị ≈ 21,8 triệu/tháng
    income_log_sigma: float = 0.75

    # -- Chi tiêu / thu nhập. DB có trung vị 0,607 và KHÔNG hộ nào ≥ 1,0.
    # Ở đây cho phép vượt 1,0 để nhóm EMERGENCY tồn tại.
    expense_ratio_mean: float = 0.72
    expense_ratio_spread: float = 0.20
    expense_ratio_health_effect: float = 0.30   # health cao → chi ít hơn

    # -- Đệm tiết kiệm (số tháng chi tiêu)
    savings_prob: float = 0.85           # DB có 100%, thực tế thấp hơn
    savings_months_log_mu: float = 1.35  # trung vị ≈ 3,9 tháng
    savings_months_log_sigma: float = 1.15
    savings_months_health_effect: float = 1.60

    # -- Nợ
    debt_prob: float = 0.70              # DB có 93,3%
    dti_log_mu: float = -1.35            # trung vị ≈ 0,259
    dti_log_sigma: float = 0.85
    # Cố ý ĐỂ YẾU. Hộ có đệm tiết kiệm tốt vẫn có thể gánh khoản vay lớn —
    # đó chính là nhóm DEBT_FOCUS. Bản đầu để 0,70 khiến hai thứ gần như loại
    # trừ nhau và DEBT_FOCUS chỉ còn 2,9%, không qua cổng 10%.
    dti_health_effect: float = 0.30
    debt_years_of_income: float = 0.60   # dư nợ ≈ 0,6 năm thu nhập

    # -- Nhân thân
    age_min: int = 22
    age_max: int = 62
    dependents_prob: float = 0.13        # DB: 13,0%

    # -- Tài sản sở hữu
    #: Hộ khá giả sở hữu nhiều loại tài sản hơn. Để YẾU (0,60) vì tỉ lệ nền
    #: đã lấy từ DB — đẩy mạnh nữa thì tỉ lệ sở hữu trung bình lệch khỏi số
    #: đã đo, mà đó là chỗ duy nhất phần này bám vào dữ liệu thật.
    asset_health_effect: float = 0.60
    asset_rates: dict[AssetType, float] = field(
        default_factory=lambda: dict(ASSET_BASE_RATES))


def _weighted_choice(rng: np.random.Generator, weights: dict, size: int) -> np.ndarray:
    keys = list(weights)
    p = np.array([weights[k] for k in keys], dtype=float)
    return rng.choice(keys, size=size, p=p / p.sum())


def generate_households(
    params: PopulationParams | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Sinh dân số hộ. Trả về CHỈ các cột thô của form — không có nhãn.

    Nhãn sinh riêng bằng `labeler.label_frame()` để hai việc tách bạch: sinh
    dân số không biết gì về `g(·)`, và `g(·)` không biết gì về cách sinh.
    """
    p = params or PopulationParams()
    rng = np.random.default_rng(seed)
    n = p.n

    # Biến ẩn chi phối chung — KHÔNG trả ra ngoài.
    health = rng.beta(2.0, 2.0, n)

    income = rng.lognormal(p.income_log_mu, p.income_log_sigma, n)
    income = np.round(income / 100_000) * 100_000          # làm tròn 100k
    income = np.clip(income, 4_000_000, 400_000_000)

    # Chi tiêu: hộ khỏe hơn chi ít hơn, cộng nhiễu riêng đủ lớn.
    expense_ratio = (p.expense_ratio_mean
                     - p.expense_ratio_health_effect * (health - 0.5)
                     + rng.normal(0, p.expense_ratio_spread, n))
    expense_ratio = np.clip(expense_ratio, 0.25, 1.45)
    expense = np.round(income * expense_ratio / 100_000) * 100_000

    # Đệm tiết kiệm theo số tháng chi tiêu, rồi quy ngược ra số tiền.
    has_savings = rng.random(n) < p.savings_prob
    months = rng.lognormal(
        p.savings_months_log_mu + p.savings_months_health_effect * (health - 0.5),
        p.savings_months_log_sigma, n)
    savings = np.where(has_savings, np.round(months * expense / 100_000) * 100_000, 0.0)

    # Nợ: dti log-normal, hộ khỏe hơn gánh nhẹ hơn.
    has_debt = rng.random(n) < p.debt_prob
    dti = rng.lognormal(
        p.dti_log_mu - p.dti_health_effect * (health - 0.5), p.dti_log_sigma, n)
    dti = np.clip(dti, 0.0, 0.95)
    payment = np.where(has_debt, np.round(income * dti / 100_000) * 100_000, 0.0)
    debt = np.where(
        has_debt,
        np.round(income * 12 * p.debt_years_of_income
                 * rng.lognormal(0, 0.6, n) / 1_000_000) * 1_000_000,
        0.0)
    # Có nợ mà tiền trả làm tròn về 0 thì mâu thuẫn với `has_debt` — kéo lên
    # mức tối thiểu thay vì để schema từ chối cả hồ sơ.
    payment = np.where(has_debt & (payment <= 0), 100_000.0, payment)
    debt = np.where(has_debt & (debt <= 0), 1_000_000.0, debt)

    size = _weighted_choice(rng, HOUSEHOLD_SIZE_WEIGHTS, n).astype(int)
    # Số con luôn nhỏ hơn số nhân khẩu (ràng buộc của schema).
    children = np.array([rng.integers(0, max(1, s)) for s in size])

    age = rng.integers(p.age_min, p.age_max + 1, n)

    df = pd.DataFrame({
        "average_monthly_income": income,
        "average_monthly_expense": expense,
        "savings_amount": savings,
        "total_current_debt": debt,
        "monthly_debt_payment": payment,
        "household_size": size,
        "children_count": children,
        "age": age,
        "has_debt": has_debt,
        "has_savings": has_savings & (savings > 0),
        "has_dependents": rng.random(n) < p.dependents_prob,
    })

    # Tài sản: multi-hot, mỗi loại rút độc lập quanh tỉ lệ nền đo từ DB.
    # Độc lập chứ không rút "số loại rồi chọn loại nào" — vì hai hộ cùng sở
    # hữu 2 loại tài sản không có lý do gì phải sở hữu cùng bộ.
    for asset, base in p.asset_rates.items():
        prob = np.clip(base * (1 + p.asset_health_effect * 2 * (health - 0.5)), 0.0, 1.0)
        df[asset_column(asset)] = rng.random(n) < prob

    log.info("Sinh %d hộ (seed=%d)", len(df), seed)
    return df


def population_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Bảng đối chiếu dân số synthetic với 300 hộ thật — dùng cho báo cáo."""
    income = df["average_monthly_income"]
    expense = df["average_monthly_expense"]
    ratio = expense / income
    rows = [
        ("thu nhập trung vị (tr/tháng)", income.median() / 1e6, 49.0),
        ("chi tiêu / thu nhập trung vị", ratio.median(), 0.607),
        ("tỉ lệ hộ chi ≥ thu", float((ratio >= 1).mean()), 0.0),
        ("tỉ lệ hộ có nợ", float(df["has_debt"].mean()), 0.933),
        ("tỉ lệ hộ có tiết kiệm", float(df["has_savings"].mean()), 1.0),
        ("số nhân khẩu trung bình", df["household_size"].mean(), 2.86),
        # 450 dòng tblassets / 300 hộ = 1,5 loại mỗi hộ.
        ("số loại tài sản trung bình",
         float(df[list(ASSET_COLUMNS)].sum(axis=1).mean()), 450 / 300),
    ]
    return pd.DataFrame(rows, columns=["chỉ số", "synthetic", "DB thật (300 hộ)"])
