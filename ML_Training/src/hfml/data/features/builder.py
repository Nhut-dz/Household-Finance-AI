"""Feature engineering — feature tỉ lệ (F01 task 12).

Đây là nơi giải quyết bài toán chuyển miền (domain transfer) ở PLAN.md §2.1:
model train trên Home Credit phải dùng được cho người dùng Việt Nam, trong
khi hai bên khác đơn vị tiền tệ tới ~340 lần.

Nguyên tắc: mọi feature phải KHÔNG THỨ NGUYÊN
---------------------------------------------
Một feature an toàn khi tử số và mẫu số triệt tiêu đơn vị:

    tiền ÷ tiền   → an toàn   (dti, ltv, savings_months, debt_income_ratio)
    người ÷ người → an toàn   (children_ratio)
    năm ÷ năm     → an toàn   (employment_ratio)
    tiền ÷ người  → KHÔNG an toàn — kết quả vẫn là tiền

⚠️ Sửa so với PLAN.md §2.1: bảng ở mục đó liệt kê
`income_per_capita = thu nhập ÷ nhân khẩu` như một feature tỉ lệ. Nó không
phải: hộ VN 50.000.000 ÷ 4 = 12.500.000 còn Home Credit 147.150 ÷ 2 = 73.575,
vẫn lệch 170 lần. Đưa nó vào feature set của ML02 là tái tạo lại đúng vấn đề
mà mục 2.1 muốn diệt.

Cách xử lý ở đây: giữ ý nghĩa "mức sống" nhưng chia thêm cho **thu nhập bình
quân đầu người tham chiếu của chính quần thể đó** — Home Credit dùng trung vị
của tập train, người dùng VN dùng số liệu GSO. Khi ấy feature trở thành
"gấp mấy lần mức trung bình của quần thể mình", và đại lượng đó mới so sánh
được giữa hai miền. Nếu chưa có số tham chiếu thì feature này KHÔNG được sinh
ra, thay vì sinh ra một con số vô nghĩa (xem `reference_income_per_capita`).

Về kỳ thu nhập của Home Credit
------------------------------
Kaggle chỉ ghi "Income of the client", không nói theo tháng hay theo năm.
Loại trừ bằng số liệu: nếu `AMT_INCOME_TOTAL` là thu nhập NĂM còn
`AMT_ANNUITY` là tiền trả nợ THÁNG thì DTI trung vị = 0,163 × 12 = **196%**,
bất khả. Vậy hai cột chắc chắn cùng kỳ, và `dti` so sánh trực tiếp được với
phía form. Các feature tiền ÷ tiền khác cũng vậy — đơn vị lẫn kỳ đều triệt
tiêu.

Feature nào có ở phía nào
-------------------------
`application_train.csv` KHÔNG có cột nào về tiết kiệm hay dư nợ hiện tại (đã
kiểm: chỉ có 4 cột `AMT_*` là INCOME_TOTAL, CREDIT, ANNUITY, GOODS_PRICE).
Nên `savings_months` và `debt_income_ratio` chỉ dùng được cho ML01 — đó là
ranh giới thật giữa hai bài toán, không phải thiếu sót.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Final

import numpy as np
import pandas as pd

from hfml.data.schema import HouseholdProfile
from hfml.logger import get_logger

log = get_logger(__name__)

#: Số tháng trong một năm — dùng để quy đổi khi form hỏi theo tháng còn
#: feature định nghĩa theo năm.
MONTHS_PER_YEAR: Final[int] = 12
#: Home Credit lưu ngày dưới dạng số âm; đổi sang năm.
DAYS_PER_YEAR: Final[float] = 365.25


class Availability(str, Enum):
    """Feature này lấy được từ nguồn nào — và quan trọng hơn, có CÙNG NGHĨA
    ở cả hai nguồn hay không."""

    #: Lấy được từ cả hai VÀ cùng nghĩa → bộ rút gọn của ML02 (deploy được).
    BOTH = "both"
    #: Chỉ form → ML01 và tầng rule.
    FORM_ONLY = "form_only"
    #: Chỉ Home Credit → chỉ dùng cho bộ Full của ML02, không deploy được.
    HOME_CREDIT_ONLY = "home_credit_only"


class Unit(str, Enum):
    """Đơn vị của tử/mẫu. Dùng để kiểm bằng máy chứ không đọc bằng mắt."""

    MONEY = "tiền"
    PEOPLE = "người"
    YEARS = "năm"
    #: Mẫu số bằng 1 — feature là một đại lượng tuyệt đối.
    NONE = "1"


@dataclass(frozen=True)
class RatioFeature:
    """Một feature, kèm chứng minh vì sao nó bất biến với đơn vị tiền tệ."""

    name: str
    description: str
    availability: Availability
    #: Công thức phía Home Credit, `None` nếu không lấy được.
    home_credit: str | None
    #: Công thức phía form người dùng, `None` nếu không lấy được.
    form: str | None
    numerator_unit: Unit
    denominator_unit: Unit
    #: Ghi chú khi hai nguồn cùng tên nhưng lệch nghĩa — phải viết vào
    #: `docs/model_card.md` như một giới hạn của model.
    caveat: str = ""

    @property
    def currency_invariant(self) -> bool:
        """Tử số là tiền thì mẫu số cũng phải là tiền, nếu không kết quả vẫn
        mang đơn vị tiền tệ và domain gap quay lại."""
        if self.numerator_unit is Unit.MONEY:
            return self.denominator_unit is Unit.MONEY
        return True

    @property
    def dimension(self) -> str:
        return f"{self.numerator_unit.value} ÷ {self.denominator_unit.value}"


#: Đăng ký feature. Thứ tự cố định — thứ tự cột sai là lỗi im lặng
#: (xem `hfml.ml.registry`).
RATIO_FEATURES: Final[tuple[RatioFeature, ...]] = (
    RatioFeature(
        "dti", "Tỉ trọng thu nhập dùng để trả nợ", Availability.BOTH,
        "AMT_ANNUITY / AMT_INCOME_TOTAL",
        "monthly_debt_payment / average_monthly_income",
        Unit.MONEY, Unit.MONEY,
        caveat="Hai bên đo hai khoản nợ khác nhau: Home Credit là kỳ trả của "
               "khoản đang XIN VAY, form là khoản nợ ĐANG CÓ. Cùng là 'phần thu "
               "nhập dành trả nợ' và phân phối khớp nhau (trung vị 0,163 vs "
               "0,20), nhưng phải ghi vào model_card như một giới hạn.",
    ),
    RatioFeature(
        "credit_income_ratio", "Số năm thu nhập để trả hết khoản vay",
        Availability.BOTH,
        "AMT_CREDIT / AMT_INCOME_TOTAL",
        "loan_amount / (average_monthly_income × 12)",
        Unit.MONEY, Unit.MONEY,
        caveat="Feature này cũng là bằng chứng cho thấy AMT_INCOME_TOTAL là thu "
               "nhập NĂM: trung vị Home Credit 3,27 khớp với 3,89 của một hồ sơ "
               "mua nhà điển hình ở VN tính theo thu nhập năm. Nếu là thu nhập "
               "tháng thì hai bên đã lệch nhau 12 lần.",
    ),
    RatioFeature(
        "children_ratio", "Tỉ lệ trẻ em trong hộ", Availability.BOTH,
        "CNT_CHILDREN / CNT_FAM_MEMBERS",
        "children_count / household_size",
        Unit.PEOPLE, Unit.PEOPLE,
    ),
    RatioFeature(
        "age_years", "Tuổi của người đại diện hộ", Availability.BOTH,
        "-DAYS_BIRTH / 365.25",
        "năm hiện tại − birth_year",
        Unit.YEARS, Unit.NONE,
    ),
    RatioFeature(
        "employment_years", "Số năm đi làm liên tục", Availability.BOTH,
        "-DAYS_EMPLOYED / 365.25",
        "employment_years",
        Unit.YEARS, Unit.NONE,
    ),
    RatioFeature(
        "employment_ratio", "Tỉ lệ đời người đã đi làm", Availability.BOTH,
        "DAYS_EMPLOYED / DAYS_BIRTH",
        "employment_years / tuổi",
        Unit.YEARS, Unit.YEARS,
    ),
    RatioFeature(
        "income_per_capita_ratio",
        "Thu nhập đầu người so với mức trung bình của quần thể",
        Availability.BOTH,
        "(AMT_INCOME_TOTAL / CNT_FAM_MEMBERS) ÷ trung vị của tập train",
        "(thu nhập năm / household_size) ÷ mức tham chiếu GSO",
        Unit.MONEY, Unit.MONEY,
        caveat="Chỉ sinh ra khi có mức tham chiếu của ĐÚNG quần thể đó. "
               "Thiếu tham chiếu → NaN, không bịa số.",
    ),

    # ---- Chỉ Home Credit: không lấy được từ form ----
    RatioFeature(
        "credit_goods_markup", "Tỉ lệ đội giá của khoản vay so với giá hàng",
        Availability.HOME_CREDIT_ONLY,
        "AMT_CREDIT / AMT_GOODS_PRICE",
        None,
        Unit.MONEY, Unit.MONEY,
        caveat="ĐÂY KHÔNG PHẢI LTV. Home Credit cộng phí và bảo hiểm vào "
               "AMT_CREDIT nên tỉ lệ này LUÔN ≥ 1,0 (p1 = 1,000, trung vị "
               "1,119). Nó đo mức đội giá, không đo tỉ lệ vay trên tài sản. "
               "Trùng tên với `ltv` của form là cái bẫy đã suýt mắc.",
    ),

    # ---- Chỉ form ----
    RatioFeature(
        "ltv", "Tỉ lệ vay trên giá trị tài sản (vay bao nhiêu phần trăm)",
        Availability.FORM_ONLY,
        None,
        "loan_amount / asset_price",
        Unit.MONEY, Unit.MONEY,
        caveat="Không dùng AMT_CREDIT/AMT_GOODS_PRICE của Home Credit làm "
               "tương ứng — xem `credit_goods_markup`. LTV này phục vụ RB05.",
    ),
    RatioFeature(
        "savings_months", "Số tháng sống được bằng tiết kiệm",
        Availability.FORM_ONLY,
        None,
        "savings_amount / average_monthly_expense",
        Unit.MONEY, Unit.MONEY,
        caveat="application_train.csv không có cột nào về tiết kiệm (đã kiểm: "
               "chỉ 4 cột AMT_* là INCOME_TOTAL, CREDIT, ANNUITY, GOODS_PRICE).",
    ),
    RatioFeature(
        "debt_income_ratio", "Đòn bẩy hiện tại", Availability.FORM_ONLY,
        None,
        "total_current_debt / (average_monthly_income × 12)",
        Unit.MONEY, Unit.MONEY,
        caveat="Home Credit không có dư nợ hiện tại trong application_train.",
    ),
    RatioFeature(
        "savings_rate", "Tỉ lệ tiết kiệm trên thu nhập", Availability.FORM_ONLY,
        None,
        "(thu − chi) / thu",
        Unit.MONEY, Unit.MONEY,
    ),
    RatioFeature(
        "expense_income_ratio", "Tỉ lệ chi trên thu", Availability.FORM_ONLY,
        None,
        "average_monthly_expense / average_monthly_income",
        Unit.MONEY, Unit.MONEY,
    ),
)


def _names(availability: Availability) -> tuple[str, ...]:
    return tuple(f.name for f in RATIO_FEATURES if f.availability is availability)


#: Bộ RÚT GỌN của ML02 — cùng nghĩa ở cả hai nguồn nên deploy được.
SHARED_FEATURES: Final[tuple[str, ...]] = _names(Availability.BOTH)
#: Chỉ form — ML01 và tầng rule.
FORM_ONLY_FEATURES: Final[tuple[str, ...]] = _names(Availability.FORM_ONLY)
#: Chỉ Home Credit — bộ FULL của ML02, không deploy được.
HOME_CREDIT_ONLY_FEATURES: Final[tuple[str, ...]] = _names(Availability.HOME_CREDIT_ONLY)
#: Toàn bộ feature form sinh ra được.
ALL_FORM_FEATURES: Final[tuple[str, ...]] = SHARED_FEATURES + FORM_ONLY_FEATURES
#: Toàn bộ feature Home Credit sinh ra được (bộ Full).
ALL_HOME_CREDIT_FEATURES: Final[tuple[str, ...]] = SHARED_FEATURES + HOME_CREDIT_ONLY_FEATURES


def safe_divide(
    numerator: pd.Series | float,
    denominator: pd.Series | float,
) -> pd.Series | float:
    """Chia, mẫu số ≤ 0 hoặc thiếu → `NaN` chứ không phải `inf`.

    `inf` chảy xuống dưới sẽ làm scaler nổ và `SimpleImputer` không bắt được
    (nó chỉ xử lý NaN). Trả `NaN` để bước impute trong Pipeline lo tiếp, và
    để cờ `_MISSING` phản ánh đúng rằng giá trị này không tính được.
    """
    num = pd.Series(numerator) if not isinstance(numerator, pd.Series) else numerator
    den = pd.Series(denominator) if not isinstance(denominator, pd.Series) else denominator
    den = den.where(den > 0)                      # ≤ 0 và NaN đều thành NaN
    result = num / den
    return result.replace([np.inf, -np.inf], np.nan)


def build_from_home_credit(
    df: pd.DataFrame,
    reference_income_per_capita: float | None = None,
    feature_set: str = "full",
) -> pd.DataFrame:
    """Sinh feature tỉ lệ từ `application_train.csv`.

    `feature_set="reduced"` chỉ trả về `SHARED_FEATURES` — bộ mà form người
    dùng cũng sinh ra được, tức model DEPLOY được (PLAN.md §7.2).
    `"full"` thêm các feature chỉ Home Credit mới có.

    `reference_income_per_capita` là trung vị thu nhập đầu người của chính
    tập train. Để `None` thì `income_per_capita_ratio` là NaN — thà thiếu
    giá trị còn hơn có một cột mang đơn vị tiền tệ lẫn vào feature set.
    """
    if feature_set not in ("full", "reduced"):
        raise ValueError(f"feature_set không hợp lệ: {feature_set!r}")

    out = pd.DataFrame(index=df.index)

    out["dti"] = safe_divide(df["AMT_ANNUITY"], df["AMT_INCOME_TOTAL"])
    out["credit_income_ratio"] = safe_divide(df["AMT_CREDIT"], df["AMT_INCOME_TOTAL"])
    out["children_ratio"] = safe_divide(df["CNT_CHILDREN"], df["CNT_FAM_MEMBERS"])

    # DAYS_* là số âm (số ngày trước ngày nộp đơn). Sentinel của
    # DAYS_EMPLOYED đã thành NaN ở task 8 nên phép chia dưới đây an toàn.
    out["age_years"] = -df["DAYS_BIRTH"] / DAYS_PER_YEAR
    out["employment_years"] = -df["DAYS_EMPLOYED"] / DAYS_PER_YEAR
    out["employment_ratio"] = safe_divide(-df["DAYS_EMPLOYED"], -df["DAYS_BIRTH"])

    if reference_income_per_capita:
        per_capita = safe_divide(df["AMT_INCOME_TOTAL"], df["CNT_FAM_MEMBERS"])
        out["income_per_capita_ratio"] = per_capita / reference_income_per_capita
    else:
        log.warning("Thiếu reference_income_per_capita → income_per_capita_ratio = NaN")
        out["income_per_capita_ratio"] = np.nan

    if feature_set == "full":
        # KHÔNG đặt tên là `ltv`: đại lượng này luôn ≥ 1,0 vì AMT_CREDIT đã
        # cộng phí lên giá hàng. Xem caveat của `credit_goods_markup`.
        out["credit_goods_markup"] = safe_divide(df["AMT_CREDIT"], df["AMT_GOODS_PRICE"])
        return out[list(ALL_HOME_CREDIT_FEATURES)]

    return out[list(SHARED_FEATURES)]


def median_income_per_capita(df: pd.DataFrame) -> float:
    """Trung vị thu nhập đầu người của tập train — mức tham chiếu Home Credit.

    Phải tính TRÊN TẬP TRAIN. Tính trên toàn bộ dữ liệu rồi mới split là rò
    rỉ (PLAN.md §4.4).
    """
    per_capita = safe_divide(df["AMT_INCOME_TOTAL"], df["CNT_FAM_MEMBERS"])
    return float(per_capita.median())


def _f(value: Decimal | float | None) -> float:
    """`Decimal | None` → `float | NaN`. Schema dùng Decimal cho tiền."""
    return float("nan") if value is None else float(value)


def build_from_profile(
    profile: HouseholdProfile,
    reference_income_per_capita: float | None = None,
) -> pd.DataFrame:
    """Sinh feature tỉ lệ từ một hồ sơ form — một dòng, dùng lúc inference.

    Trả về DataFrame một dòng với ĐẦY ĐỦ cột `ALL_FORM_FEATURES`, kể cả khi
    người dùng bỏ trống (giá trị `NaN`). Thiếu cột là thứ tự feature lệch so
    với lúc train, và đó là lỗi im lặng — model vẫn chạy, xác suất vô nghĩa.
    """
    monthly_income = _f(profile.average_monthly_income)
    annual_income = monthly_income * MONTHS_PER_YEAR
    monthly_expense = _f(profile.average_monthly_expense)

    def div(num: float, den: float) -> float:
        if not np.isfinite(num) or not np.isfinite(den) or den <= 0:
            return float("nan")
        return num / den

    age = float(profile.age) if profile.age is not None else float("nan")
    employment = _f(profile.employment_years)

    values: dict[str, float] = {
        # -- SHARED: cùng nghĩa với phía Home Credit --
        "dti": div(_f(profile.monthly_debt_payment), monthly_income),
        "credit_income_ratio": div(_f(profile.loan_amount), annual_income),
        "children_ratio": div(profile.children_count, profile.household_size),
        "age_years": age,
        "employment_years": employment,
        "employment_ratio": div(employment, age),
        "income_per_capita_ratio": float("nan"),
        # -- Chỉ form --
        "ltv": div(_f(profile.loan_amount), _f(profile.asset_price)),
        "savings_months": div(_f(profile.savings_amount), monthly_expense),
        "debt_income_ratio": div(_f(profile.total_current_debt), annual_income),
        "savings_rate": div(monthly_income - monthly_expense, monthly_income),
        "expense_income_ratio": div(monthly_expense, monthly_income),
    }

    if reference_income_per_capita:
        values["income_per_capita_ratio"] = div(
            annual_income / profile.household_size, reference_income_per_capita)

    # Người không khai nợ thì DTI là 0, không phải "không biết".
    if not profile.has_debt:
        values["dti"] = 0.0
        values["debt_income_ratio"] = 0.0
    if not profile.has_savings:
        values["savings_months"] = 0.0

    return pd.DataFrame([[values[name] for name in ALL_FORM_FEATURES]],
                        columns=list(ALL_FORM_FEATURES))


def feature_catalog() -> pd.DataFrame:
    """Bảng tra cứu feature — dùng cho `docs/model_card.md` (F07 task 3).

    Cột `caveat` là phần quan trọng nhất: nó ghi những chỗ hai nguồn dữ liệu
    cùng tên mà lệch nghĩa, tức giới hạn thật của model.
    """
    return pd.DataFrame([
        {
            "name": f.name,
            "description": f.description,
            "availability": f.availability.value,
            "dimension": f.dimension,
            "currency_invariant": f.currency_invariant,
            "home_credit": f.home_credit or "—",
            "form": f.form or "—",
            "caveat": f.caveat,
        }
        for f in RATIO_FEATURES
    ])
