"""Data contract của form đầu vào (F01 task 4).

Đây là RANH GIỚI của hệ thống: mọi thứ đi vào pipeline đều phải qua
`HouseholdProfile`. Sai kiểu, thiếu trường, số vô lý — chặn ngay ở đây chứ
không để chảy xuống rule/ML rồi ra khuyến nghị sai mà không ai biết.

Quy ước đặt tên
---------------
Tên trường lấy theo cột `Field input (API)` của backend, KHÔNG theo tên cột
DB, vì payload mà ML service nhận đến từ API. Bảy chỗ hai bên lệch nhau được
ghi trong `API_TO_DB_COLUMN` để khỏi phải đoán.

Mỗi trường mang kèm `title` là nhãn tiếng Việt. Một nguồn sự thật duy nhất:
form hiển thị nhãn này, thông báo lỗi validate dùng nhãn này, và tầng `llm`
cũng gọi tên trường bằng nhãn này khi diễn đạt kết quả.

Phân biệt "lỗi" và "cảnh báo"
-----------------------------
Lỗi (raise ValidationError):  dữ liệu không thể dùng được — số con ≥ số nhân
    khẩu, có nợ mà không khai dư nợ, chọn nhu cầu vay mà không khai khoản vay.
Cảnh báo (`data_quality_flags`):  dữ liệu hợp lệ nhưng đáng ngờ — chi > thu,
    tỉ lệ tiết kiệm > 60%. Không chặn, nhưng phải nổi lên trong structured
    result để tầng `llm` nói ra (PLAN.md §4.2).

Không tính toán tài chính trong file này. Các tỉ lệ dẫn xuất (dti, ltv,
savings_months…) thuộc `hfml.data.features.builder` — task 12.
"""
from __future__ import annotations

from datetime import date
from decimal import Decimal
from enum import Enum
from typing import Annotated, Final

from pydantic import BaseModel, ConfigDict, Field, model_validator

# --------------------------------------------------------------------------
# Ánh xạ tên: API (dùng trong schema này) → cột DB. Chỉ liệt kê chỗ LỆCH nhau.
# --------------------------------------------------------------------------
API_TO_DB_COLUMN: dict[str, str] = {
    "residence": "location",
    "average_monthly_income": "monthly_income",
    "average_monthly_expense": "monthly_living_cost",
    "has_dependents": "supports_elderly",
    "total_current_debt": "total_debt",
    "savings_amount": "current_savings",
    "guest_session_id": "session_token",
}


# --------------------------------------------------------------------------
# Enum
# --------------------------------------------------------------------------
class AssetType(str, Enum):
    """`tblassets.asset_type` — lấy theo ĐÚNG giá trị đang có trong DB.

    Đối chiếu 450 dòng của `Household_Finance_V2_Dev.sql` (11/08/2026). Danh
    sách này giàu hơn hẳn bản mô tả ban đầu (`house`/`car`/`land`/`other`):
    DB coi tài sản gồm cả tài sản tài chính, nên có `cash`, `gold`,
    `insurance`, `investment`.
    """
    CASH = "cash"                 # 116 dòng
    VEHICLE = "vehicle"           # 112
    REAL_ESTATE = "real_estate"   # 72
    INSURANCE = "insurance"       # 50
    GOLD = "gold"                 # 50
    INVESTMENT = "investment"     # 50


def asset_column(asset: AssetType) -> str:
    """Tên cột multi-hot của một loại tài sản.

    `assets` là danh sách nhiều lựa chọn, không phải một hạng mục — không
    one-hot được. Trải thành 6 cột nhị phân, đặt tên cùng tiền tố `has_` với
    `has_debt` / `has_savings` / `has_dependents` để tầng ml đọc là hiểu.

    Đặt ở đây chứ không ở tầng `ml`: đây là quy ước của data contract, và cả
    `data.synthetic` (sinh ra cột) lẫn `ml01.labeler` (khai báo feature set)
    đều phải nói cùng một tên. Hai nơi tự đặt tên riêng là lỗi im lặng —
    train xong mới phát hiện cột không khớp.
    """
    return f"has_asset_{asset.value}"


#: Sáu cột multi-hot của `assets`. Thứ tự theo `AssetType` và cố định —
#: feature set của model phụ thuộc thứ tự này.
ASSET_COLUMNS: Final[tuple[str, ...]] = tuple(asset_column(a) for a in AssetType)


class FinancialNeed(str, Enum):
    """`tblfinancial_goals.goal_type` — lấy theo ĐÚNG giá trị đang có trong DB.

    Đối chiếu 350 dòng của dump. Bốn giá trị này ánh xạ 1-1 sang bốn rule,
    và `raw_json.scope` của backend cũng ghép đúng bốn cái tên này:

        saving           → RB03  tiến độ mục tiêu tiết kiệm
        home_loan        → RB05 + ML02  khả năng đáp ứng khoản vay
        budget_50_30_20  → RB04  quy tắc 50/30/20
        investment       → chưa có rule; hiện thuộc nhóm GROWTH của ML01
    """
    SAVING = "saving"                    # 87 dòng
    HOME_LOAN = "home_loan"              # 88
    INVESTMENT = "investment"            # 87
    BUDGET_50_30_20 = "budget_50_30_20"  # 88


#: Chỉ `home_loan` mới bật KHỐI VAY và kích hoạt RB05 + ML02.
#:
#: Trước đây tập này gồm 4 giá trị (`buy_house`/`buy_car`/`buy_land`/`loan`)
#: theo bản mô tả ban đầu. Dữ liệu thật cho thấy DB chỉ có một loại nhu cầu
#: liên quan vay là `home_loan` — "Chuẩn bị vốn tự có và đánh giá khoản vay
#: mua căn hộ".
LOAN_TRIGGER_NEEDS: frozenset[FinancialNeed] = frozenset({
    FinancialNeed.HOME_LOAN,
})


class OccupationType(str, Enum):
    """Trường BỔ SUNG — ánh xạ sang `OCCUPATION_TYPE` của Home Credit.

    Danh sách rút gọn cho người dùng Việt Nam. Home Credit có 18 giá trị
    nhưng nhiều giá trị quá hẹp (Waiters/barmen staff, Realty agents…);
    gộp lại còn 14 để form chọn được nhanh. Bảng ánh xạ ở
    `OCCUPATION_TO_HOME_CREDIT`.
    """
    OFFICE_STAFF = "office_staff"
    MANAGER = "manager"
    ACCOUNTANT = "accountant"
    IT_STAFF = "it_staff"
    TEACHER = "teacher"
    MEDICAL_STAFF = "medical_staff"
    SALES_STAFF = "sales_staff"
    DRIVER = "driver"
    SECURITY_STAFF = "security_staff"
    SERVICE_STAFF = "service_staff"
    LABORER = "laborer"
    FARMER = "farmer"
    SELF_EMPLOYED = "self_employed"
    RETIRED = "retired"
    UNEMPLOYED = "unemployed"
    OTHER = "other"


#: Ánh xạ sang giá trị gốc của Home Credit. `None` = Home Credit để trống
#: `OCCUPATION_TYPE` cho nhóm này (nghỉ hưu / thất nghiệp / tự kinh doanh),
#: và đó cũng là NaN hợp lệ — không được điền bừa một nghề vào.
#: `hfml.ml.ml02_credit_risk.features` dùng bảng này.
OCCUPATION_TO_HOME_CREDIT: dict[OccupationType, str | None] = {
    OccupationType.OFFICE_STAFF: "Core staff",
    OccupationType.MANAGER: "Managers",
    OccupationType.ACCOUNTANT: "Accountants",
    OccupationType.IT_STAFF: "IT staff",
    OccupationType.TEACHER: "High skill tech staff",
    OccupationType.MEDICAL_STAFF: "Medicine staff",
    OccupationType.SALES_STAFF: "Sales staff",
    OccupationType.DRIVER: "Drivers",
    OccupationType.SECURITY_STAFF: "Security staff",
    OccupationType.SERVICE_STAFF: "Private service staff",
    OccupationType.LABORER: "Laborers",
    OccupationType.FARMER: "Low-skill Laborers",
    OccupationType.SELF_EMPLOYED: None,
    OccupationType.RETIRED: None,
    OccupationType.UNEMPLOYED: None,
    OccupationType.OTHER: None,
}

#: Kỳ hạn vay cho chọn (tháng). 12→60 cho vay tiêu dùng/mua xe,
#: 120→300 cho vay mua nhà/đất.
LOAN_TERM_CHOICES: tuple[int, ...] = (12, 24, 36, 60, 120, 180, 240, 300)


# --------------------------------------------------------------------------
# Nhãn tiếng Việt của giá trị enum
#
# `vietnamese_labels()` chỉ trả nhãn của TÊN TRƯỜNG. Ba bảng dưới đây là nhãn
# của GIÁ TRỊ — thứ form dùng để dựng dropdown, và tầng `llm` dùng để viết
# "nghề nghiệp: Nhân viên văn phòng" thay vì "office_staff".
#
# Giữ ở đây để chỉ có một nguồn sự thật: sửa một chỗ thì form, thông báo lỗi
# và câu chữ LLM cùng đổi theo.
# --------------------------------------------------------------------------
OCCUPATION_LABELS: dict[OccupationType, str] = {
    OccupationType.OFFICE_STAFF: "Nhân viên văn phòng",
    OccupationType.MANAGER: "Quản lý, lãnh đạo",
    OccupationType.ACCOUNTANT: "Kế toán, tài chính",
    OccupationType.IT_STAFF: "Công nghệ thông tin",
    OccupationType.TEACHER: "Giáo viên, giảng viên",
    OccupationType.MEDICAL_STAFF: "Y tế",
    OccupationType.SALES_STAFF: "Kinh doanh, bán hàng",
    OccupationType.DRIVER: "Lái xe",
    OccupationType.SECURITY_STAFF: "Bảo vệ",
    OccupationType.SERVICE_STAFF: "Dịch vụ, giúp việc",
    OccupationType.LABORER: "Công nhân, lao động phổ thông",
    OccupationType.FARMER: "Nông, lâm, ngư nghiệp",
    OccupationType.SELF_EMPLOYED: "Tự kinh doanh, tự do",
    OccupationType.RETIRED: "Nghỉ hưu",
    OccupationType.UNEMPLOYED: "Chưa có việc làm",
    OccupationType.OTHER: "Khác",
}

ASSET_LABELS: dict[AssetType, str] = {
    AssetType.CASH: "Tiền mặt và tiền gửi",
    AssetType.VEHICLE: "Phương tiện",
    AssetType.REAL_ESTATE: "Bất động sản",
    AssetType.INSURANCE: "Bảo hiểm",
    AssetType.GOLD: "Vàng",
    AssetType.INVESTMENT: "Đầu tư",
}

FINANCIAL_NEED_LABELS: dict[FinancialNeed, str] = {
    FinancialNeed.SAVING: "Xây quỹ dự phòng",
    FinancialNeed.HOME_LOAN: "Đánh giá khoản vay mua nhà",
    FinancialNeed.INVESTMENT: "Phân bổ tiền nhàn rỗi",
    FinancialNeed.BUDGET_50_30_20: "Theo dõi ngân sách 50/30/20",
}

#: Nhu cầu nào kích hoạt rule nào. `raw_json.scope` của backend ghép đúng
#: bốn cái tên này lại.
NEED_TO_RULE: dict[FinancialNeed, str] = {
    FinancialNeed.SAVING: "RB03",
    FinancialNeed.HOME_LOAN: "RB05",
    FinancialNeed.BUDGET_50_30_20: "RB04",
    FinancialNeed.INVESTMENT: "",   # chưa có rule — thuộc nhóm GROWTH của ML01
}


def loan_term_label(months: int) -> str:
    """`240` → `"20 năm (240 tháng)"`. Dropdown đọc bằng năm dễ hình dung hơn."""
    if months % 12 == 0 and months >= 12:
        return f"{months // 12} năm ({months} tháng)"
    return f"{months} tháng"


class DataQualityFlag(str, Enum):
    """Cảnh báo dữ liệu bất thường — KHÔNG chặn request (PLAN.md §4.2).

    Các cờ này đi vào structured result để tầng `llm` nói ra thành lời.
    Im lặng bỏ qua là đưa khuyến nghị dựa trên dữ liệu mà chính mình nghi ngờ.
    """
    EXPENSE_EXCEEDS_INCOME = "expense_exceeds_income"
    SAVINGS_RATE_TOO_HIGH = "savings_rate_too_high"
    DEBT_PAYMENT_EXCEEDS_INCOME = "debt_payment_exceeds_income"
    LOAN_EXCEEDS_ASSET_PRICE = "loan_exceeds_asset_price"
    MISSING_EXPENSE = "missing_expense"
    MISSING_DEBT_PAYMENT = "missing_debt_payment"


#: Ngưỡng cảnh báo tỉ lệ tiết kiệm (PLAN.md §4.2).
SAVINGS_RATE_WARN = Decimal("0.60")

Money = Annotated[Decimal, Field(ge=0, description="VNĐ")]


# --------------------------------------------------------------------------
# Contract chính
# --------------------------------------------------------------------------
class HouseholdProfile(BaseModel):
    """Một hồ sơ hộ gia đình — ĐÚNG MỘT DÒNG, không có lịch sử.

    Dữ liệu chỉ thu một lần qua form onboarding (PLAN.md §2), nên đây là
    toàn bộ những gì hệ thống biết về một hộ. Không có tháng trước, không có
    xu hướng, không có lag.
    """

    model_config = ConfigDict(
        extra="forbid",          # gửi thừa trường lạ → báo lỗi, đừng nuốt im
        str_strip_whitespace=True,
        use_enum_values=False,
    )

    # -- Nhân thân ---------------------------------------------------------
    representative_name: str = Field(
        ..., max_length=150, title="Họ và tên người đại diện")
    birth_year: int | None = Field(
        None, ge=1900, le=date.today().year, title="Năm sinh")
    residence: str | None = Field(
        None, max_length=255, title="Nơi ở")
    household_size: int = Field(
        ..., ge=1, le=50, title="Số người trong nhà")
    children_count: int = Field(
        ..., ge=0, title="Số con")
    has_dependents: bool = Field(
        ..., title="Tình trạng phụng dưỡng người già")

    # -- Dòng tiền ---------------------------------------------------------
    average_monthly_income: Money = Field(
        ..., title="Thu nhập trung bình tháng")
    average_monthly_expense: Money | None = Field(
        None, title="Chi tiêu trung bình tháng")

    # -- Nợ ----------------------------------------------------------------
    has_debt: bool = Field(..., title="Tình trạng nợ")
    total_current_debt: Money | None = Field(
        None, title="Tổng dư nợ hiện tại")
    monthly_debt_payment: Money | None = Field(
        None, title="Số tiền trả nợ hàng tháng")

    # -- Tiết kiệm ---------------------------------------------------------
    has_savings: bool = Field(..., title="Tình trạng tiết kiệm")
    savings_amount: Money | None = Field(
        None, title="Số tiền tiết kiệm")

    # -- Tài sản & mục tiêu ------------------------------------------------
    assets: list[AssetType] = Field(
        default_factory=list, title="Tài sản đang sở hữu")
    financial_needs: list[FinancialNeed] = Field(
        default_factory=list, title="Nhu cầu tài chính")

    # -- KHỐI VAY (BỔ SUNG) ------------------------------------------------
    # Năm trường dưới đây chỉ hiện khi người dùng chọn một nhu cầu trong
    # `LOAN_TRIGGER_NEEDS`, và khi đó đều bắt buộc. Xem `_check_loan_request`.
    #
    # `occupation` và `employment_years` nằm ở đây chứ không ở phần nhân thân
    # là một quyết định về ma sát (chốt 11/08/2026): chúng phục vụ ML02 — dự
    # báo rủi ro tín dụng — mà ML02 chỉ có ý nghĩa với người đang tính vay.
    # Người chỉ muốn xem sức khỏe tài chính không phải nhập thêm ô nào.
    occupation: OccupationType | None = Field(
        None, title="Nghề nghiệp",
        description="BỔ SUNG — ánh xạ OCCUPATION_TYPE của Home Credit")
    employment_years: Decimal | None = Field(
        None, ge=0, le=60, title="Số năm đi làm",
        description="BỔ SUNG — ánh xạ DAYS_EMPLOYED của Home Credit")
    asset_price: Money | None = Field(
        None, title="Giá tài sản dự định mua",
        description="BỔ SUNG — ánh xạ AMT_GOODS_PRICE, dùng tính LTV")
    loan_amount: Money | None = Field(
        None, title="Số tiền dự định vay",
        description="BỔ SUNG — ánh xạ AMT_CREDIT")
    loan_term_months: int | None = Field(
        None, title="Kỳ hạn vay (tháng)",
        description=f"BỔ SUNG — chọn trong {LOAN_TERM_CHOICES}, suy ra AMT_ANNUITY")

    # -- Phiên ------------------------------------------------------------
    guest_session_id: str | None = Field(
        None, max_length=255, title="Mã phiên khách",
        description="Backend bắt buộc khi chưa đăng nhập; ML service không dùng")

    # ---------------------------------------------------------------- rules
    @model_validator(mode="after")
    def _check_household_composition(self) -> "HouseholdProfile":
        if self.children_count >= self.household_size:
            raise ValueError(
                f"'{type(self).model_fields['children_count'].title}' ({self.children_count}) "
                f"phải nhỏ hơn '{type(self).model_fields['household_size'].title}' "
                f"({self.household_size})"
            )
        return self

    @model_validator(mode="after")
    def _check_debt_consistency(self) -> "HouseholdProfile":
        """Khai có nợ thì phải có dư nợ VÀ tiền trả hàng tháng.

        `monthly_debt_payment` bắt buộc là thay đổi so với backend hiện tại
        (đang Tùy chọn). Lý do: DTI = trả nợ tháng ÷ thu nhập tháng là trục
        chính của RB02, RB05 và ML02 — thiếu nó thì ba thứ đó đều không chạy.
        PLAN.md §4.2 cũng yêu cầu bỏ cách ước lượng 1%/tháng trong UI.
        """
        if self.has_debt:
            if self.total_current_debt is None:
                raise ValueError(
                    "Đã khai có nợ thì bắt buộc nhập 'Tổng dư nợ hiện tại'")
            if self.monthly_debt_payment is None:
                raise ValueError(
                    "Đã khai có nợ thì bắt buộc nhập 'Số tiền trả nợ hàng tháng' "
                    "— cần cho DTI, không được ước lượng thay người dùng")
        else:
            if self.total_current_debt:
                raise ValueError(
                    "Khai không có nợ nhưng 'Tổng dư nợ hiện tại' lại khác 0")
            if self.monthly_debt_payment:
                raise ValueError(
                    "Khai không có nợ nhưng 'Số tiền trả nợ hàng tháng' lại khác 0")
        return self

    @model_validator(mode="after")
    def _check_savings_consistency(self) -> "HouseholdProfile":
        if self.has_savings and self.savings_amount is None:
            raise ValueError(
                "Đã khai có tiết kiệm thì bắt buộc nhập 'Số tiền tiết kiệm'")
        if not self.has_savings and self.savings_amount:
            raise ValueError(
                "Khai không có tiết kiệm nhưng 'Số tiền tiết kiệm' lại khác 0")
        return self

    @model_validator(mode="after")
    def _check_unique_arrays(self) -> "HouseholdProfile":
        for field_name in ("assets", "financial_needs"):
            values = getattr(self, field_name)
            if len(values) != len(set(values)):
                raise ValueError(
                    f"'{type(self).model_fields[field_name].title}' có giá trị trùng lặp")
        return self

    @model_validator(mode="after")
    def _check_loan_request(self) -> "HouseholdProfile":
        """Chọn nhu cầu cần vay thì phải khai đủ giá tài sản, số vay, kỳ hạn."""
        if self.loan_term_months is not None and self.loan_term_months not in LOAN_TERM_CHOICES:
            raise ValueError(
                f"'Kỳ hạn vay (tháng)' phải là một trong {LOAN_TERM_CHOICES}")

        if not self.needs_loan_analysis:
            return self

        loan_fields = {
            "occupation": self.occupation,
            "employment_years": self.employment_years,
            "asset_price": self.asset_price,
            "loan_amount": self.loan_amount,
            "loan_term_months": self.loan_term_months,
        }
        missing = [type(self).model_fields[k].title for k, v in loan_fields.items() if v is None]
        if missing:
            triggers = sorted(n.value for n in self.loan_triggers)
            raise ValueError(
                f"Nhu cầu {triggers} cần khai đủ thông tin khoản vay, còn thiếu: "
                + ", ".join(f"'{m}'" for m in missing))
        return self

    # ----------------------------------------------------------- tiện ích
    @property
    def age(self) -> int | None:
        """Tuổi suy từ năm sinh. `None` khi người dùng bỏ trống."""
        return None if self.birth_year is None else date.today().year - self.birth_year

    @property
    def loan_triggers(self) -> set[FinancialNeed]:
        """Các nhu cầu đã chọn có kéo theo phân tích khoản vay."""
        return set(self.financial_needs) & LOAN_TRIGGER_NEEDS

    @property
    def needs_loan_analysis(self) -> bool:
        """True → form hiện KHỐI VAY (5 ô), và RB05 + ML02 mới chạy.

        False → hệ thống chỉ trả về 4 rule + nhóm khuyến nghị ML01, KHÔNG có
        xác suất vỡ nợ. Đó là chủ ý: đưa "xác suất vỡ nợ 8%" cho người không
        hề định vay là một con số vô nghĩa với họ và dễ bị hiểu nhầm.
        """
        return bool(self.loan_triggers)

    def data_quality_flags(self) -> list[DataQualityFlag]:
        """Cảnh báo dữ liệu đáng ngờ. KHÔNG chặn — chỉ để nổi lên trong kết quả.

        Trả về danh sách rỗng nghĩa là dữ liệu không có gì bất thường.
        """
        flags: list[DataQualityFlag] = []
        income = self.average_monthly_income

        if self.average_monthly_expense is None:
            flags.append(DataQualityFlag.MISSING_EXPENSE)
        elif self.average_monthly_expense > income:
            flags.append(DataQualityFlag.EXPENSE_EXCEEDS_INCOME)
        elif income > 0:
            savings_rate = (income - self.average_monthly_expense) / income
            if savings_rate > SAVINGS_RATE_WARN:
                flags.append(DataQualityFlag.SAVINGS_RATE_TOO_HIGH)

        if self.has_debt:
            if self.monthly_debt_payment is None:
                flags.append(DataQualityFlag.MISSING_DEBT_PAYMENT)
            elif self.monthly_debt_payment > income:
                flags.append(DataQualityFlag.DEBT_PAYMENT_EXCEEDS_INCOME)

        if (self.loan_amount is not None and self.asset_price is not None
                and self.loan_amount > self.asset_price):
            flags.append(DataQualityFlag.LOAN_EXCEEDS_ASSET_PRICE)

        return flags

    @classmethod
    def vietnamese_labels(cls) -> dict[str, str]:
        """`{tên_trường: nhãn tiếng Việt}` — cho form, thông báo lỗi và tầng llm."""
        return {name: (f.title or name) for name, f in cls.model_fields.items()}
