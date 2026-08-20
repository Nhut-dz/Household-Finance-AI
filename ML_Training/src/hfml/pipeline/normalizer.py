"""AI-01 task 1 — Chuẩn hoá input inference (F05 · M05).

Đây là RANH GIỚI của pipeline inference. Mọi thứ đi vào đều qua đây, và cái
gì không qua được thì dừng lại ở đây chứ không chảy xuống rule/ML rồi ra một
kết quả sai mà không ai biết.

Nguyên tắc chi phối: KHÔNG TỰ SUY DIỄN DỮ LIỆU THIẾU
------------------------------------------------------
Thiếu trường bắt buộc thì báo thiếu, không điền trung vị, không đoán. Điền bừa
một con số thì rule vẫn tính ra kết quả và model vẫn trả về xác suất — cả hai
đều trông bình thường, và không có gì trong output để lộ ra rằng một phần đầu
vào là bịa.

Ngoại lệ DUY NHẤT, và nó không phải suy diễn: ba trường tiền có điều kiện
(`savings_amount`, `total_current_debt`, `monthly_debt_payment`) để trống nghĩa
là **0 đã biết chắc**, vì cờ `has_savings` / `has_debt` đã mang thông tin
có/không. Quy ước này do `ZERO_WHEN_ABSENT` của §6.1c định nghĩa, không phải
do tầng này tự nghĩ ra.

Hai loại vấn đề, xử lý khác nhau
---------------------------------
    LỖI (error)      Dữ liệu không dùng được → dừng, trả về danh sách lỗi.
                     Số con ≥ số nhân khẩu, có nợ mà không khai dư nợ, kỳ hạn
                     vay không nằm trong danh sách cho phép.
    CẢNH BÁO (warn)  Dữ liệu hợp lệ nhưng đáng ngờ → vẫn chạy, nhưng cờ phải
                     nổi lên tới tầng `llm` để nó nói ra. Chi > thu chính là
                     nhóm EMERGENCY của ML01 — chặn nó là chặn đúng đối tượng
                     cần tư vấn nhất (§4.2).

Chuẩn hoá đơn vị
-----------------
Form nhận tiền theo THÁNG và kỳ hạn theo THÁNG. Rule dùng đơn vị tháng; ML02
được train trên Home Credit vốn tính theo NĂM. Việc quy đổi nằm ở
`hfml.pipeline.adapters`, không ở đây — tầng này chỉ đưa mọi thứ về `Decimal`
và kiểu chuẩn, giữ nguyên kỳ.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any, Final

from pydantic import ValidationError

from hfml.data.schema import (
    DataQualityFlag,
    HouseholdProfile,
    LoanApplication,
)
from hfml.logger import get_logger

log = get_logger(__name__)

#: Trường tiền có điều kiện: để trống = 0 ĐÃ BIẾT CHẮC, không phải "chưa biết".
#: Cờ `has_savings` / `has_debt` đã mang thông tin có/không, nên điền trung vị
#: vào đây là bịa cho hộ không tiết kiệm một khoản tiết kiệm bằng nửa dân số.
ZERO_WHEN_ABSENT: Final[tuple[str, ...]] = (
    "savings_amount", "total_current_debt", "monthly_debt_payment",
)


@dataclass
class InputIssue:
    """Một vấn đề của dữ liệu đầu vào."""

    field: str
    code: str
    message: str
    severity: str = "error"      # "error" | "warning"

    def to_dict(self) -> dict:
        return {"field": self.field, "code": self.code,
                "message": self.message, "severity": self.severity}


@dataclass
class NormalizedInput:
    """Đầu vào đã kiểm và chuẩn hoá.

    `is_valid` False nghĩa là KHÔNG được chạy rule hay ML — `profile` khi đó
    là `None`, nên không có cách nào lỡ tay dùng tiếp.
    """

    profile: HouseholdProfile | None = None
    loan: LoanApplication | None = None
    issues: list[InputIssue] = field(default_factory=list)
    quality_flags: list[str] = field(default_factory=list)

    @property
    def errors(self) -> list[InputIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[InputIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def is_valid(self) -> bool:
        return self.profile is not None and not self.errors

    @property
    def has_loan_data(self) -> bool:
        """ML02 chỉ chạy khi có đủ dữ liệu khoản vay."""
        return self.loan is not None

    def summary(self) -> dict:
        """Tóm tắt đầu vào cho structured result (task 6)."""
        if self.profile is None:
            return {"valid": False, "n_errors": len(self.errors)}

        p = self.profile
        return {
            "valid": True,
            "representative_name": p.representative_name,
            "household_size": p.household_size,
            "children_count": p.children_count,
            "age": _age_from(p),
            "monthly_income": float(p.average_monthly_income),
            "monthly_expense": (float(p.average_monthly_expense)
                                if p.average_monthly_expense is not None else None),
            "has_debt": p.has_debt,
            "has_savings": p.has_savings,
            "financial_needs": [n.value for n in p.financial_needs],
            "has_loan_application": self.has_loan_data,
            "quality_flags": list(self.quality_flags),
            "n_warnings": len(self.warnings),
        }


def _age_from(profile: HouseholdProfile) -> int | None:
    """Tuổi suy từ năm sinh. `None` khi thiếu — KHÔNG điền tuổi mặc định.

    Điền bừa thì ML01 vẫn trả về một nhóm trông hợp lý và không ai biết nó dựa
    trên tuổi bịa.
    """
    from datetime import date

    return (date.today().year - profile.birth_year
            if profile.birth_year is not None else None)


def _to_decimal(value: Any) -> Decimal | None:
    """Ép về `Decimal`. Chuỗi rỗng và `None` đều thành `None`, không thành 0.

    Phân biệt "không khai" với "khai là 0" là việc của tầng này; gộp chúng lại
    thì mất thông tin mà `has_debt` / `has_savings` đang mang.
    """
    if value is None or value == "":
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None


def _apply_zero_convention(payload: dict) -> dict:
    """Áp quy ước `ZERO_WHEN_ABSENT` — chỉ khi cờ tương ứng nói là KHÔNG có."""
    out = dict(payload)
    for name, flag in (("savings_amount", "has_savings"),
                       ("total_current_debt", "has_debt"),
                       ("monthly_debt_payment", "has_debt")):
        if out.get(name) in (None, "") and out.get(flag) is False:
            out[name] = Decimal(0)
    return out


def _issues_from_pydantic(error: ValidationError, prefix: str = "") -> list[InputIssue]:
    """Đổi lỗi pydantic thành `InputIssue` có tên trường đọc được."""
    issues = []
    for item in error.errors():
        location = ".".join(str(part) for part in item["loc"]) or "(toàn hồ sơ)"
        issues.append(InputIssue(
            field=f"{prefix}{location}",
            code=str(item["type"]),
            message=str(item["msg"]),
        ))
    return issues


#: Trường nào của KHỐI VAY trên `HouseholdProfile` lấy được từ đâu trong
#: `LoanApplication`.
#:
#: Hai chỗ này chứa cùng một dữ liệu vì chúng ra đời cách nhau: khối vay của
#: `HouseholdProfile` là thiết kế F01 (5 ô nhét vào form "Nhập thông tin"), còn
#: `LoanApplication` là màn "Thông tin khoản vay" tách riêng (15/08/2026).
#:
#: `HouseholdProfile` bắt buộc khối vay khi người dùng chọn `home_loan`, nên
#: một payload để dữ liệu ở `loan_application` sẽ trượt validate dù **không
#: thiếu gì**. Chép sang là dời chỗ, KHÔNG phải suy diễn dữ liệu thiếu.
_LOAN_BLOCK_SOURCE: Final[dict[str, str]] = {
    "occupation": "occupation",
    "employment_years": "employment_years",
    "asset_price": "asset_price",
    "loan_amount": "loan_amount",
    "loan_term_months": "loan_term_months",
}


def _backfill_loan_block(household: dict, loan: dict | None) -> dict:
    """Điền khối vay của hồ sơ từ `loan_application` nếu hồ sơ chưa có.

    Chỉ điền ô nào đang TRỐNG — giá trị người dùng đã khai ở form hộ gia đình
    luôn thắng, vì ghi đè nó là âm thầm đổi dữ liệu người dùng nhập.
    """
    if not loan:
        return household

    out = dict(household)
    for target, source in _LOAN_BLOCK_SOURCE.items():
        if out.get(target) in (None, "") and loan.get(source) not in (None, ""):
            out[target] = loan[source]
    return out


def normalize_input(payload: dict[str, Any]) -> NormalizedInput:
    """Kiểm và chuẩn hoá một payload inference.

    `payload` gồm phần hồ sơ hộ gia đình (phẳng, đúng tên field của API) và
    tuỳ chọn khoá `loan_application` cho màn "Thông tin khoản vay".

    Không ném ngoại lệ — trả về `NormalizedInput` với danh sách lỗi. Ở biên
    của một service, ngoại lệ nghĩa là 500 và người dùng không biết mình sai
    ở ô nào.
    """
    result = NormalizedInput()

    raw_loan = payload.get("loan_application")
    household_payload = {k: v for k, v in payload.items() if k != "loan_application"}

    # -- Hồ sơ hộ gia đình --------------------------------------------------
    try:
        result.profile = HouseholdProfile(
            **_apply_zero_convention(
                _backfill_loan_block(household_payload, raw_loan)))
    except ValidationError as exc:
        result.issues.extend(_issues_from_pydantic(exc))
        log.warning("Hồ sơ không hợp lệ: %d lỗi", len(result.errors))
        return result

    # -- Cảnh báo chất lượng — KHÔNG chặn ------------------------------------
    result.quality_flags = [
        flag.value if isinstance(flag, DataQualityFlag) else str(flag)
        for flag in result.profile.data_quality_flags()
    ]
    for flag in result.quality_flags:
        result.issues.append(InputIssue(
            field="(hồ sơ)", code=flag, severity="warning",
            message=_FLAG_MESSAGES.get(flag, f"Dữ liệu đáng ngờ: {flag}")))

    # -- Khoản vay (tuỳ chọn) ------------------------------------------------
    if raw_loan:
        try:
            result.loan = LoanApplication(**raw_loan)
        except ValidationError as exc:
            # Khoản vay hỏng KHÔNG làm hỏng cả request: hồ sơ vẫn chạy được
            # rule + ML01, chỉ mất phần ML02. Đánh dấu là cảnh báo và nói rõ.
            result.issues.extend(
                [InputIssue(i.field, i.code, i.message, severity="warning")
                 for i in _issues_from_pydantic(exc, prefix="loan_application.")])
            log.warning("Thông tin khoản vay không hợp lệ — bỏ qua ML02.")

    return result


#: Lời giải thích cho từng cờ chất lượng, để tầng `llm` nói ra thay vì in mã.
_FLAG_MESSAGES: Final[dict[str, str]] = {
    DataQualityFlag.EXPENSE_EXCEEDS_INCOME.value:
        "Chi tiêu đang lớn hơn thu nhập — dòng tiền âm.",
    DataQualityFlag.SAVINGS_RATE_TOO_HIGH.value:
        "Tỉ lệ tiết kiệm trên 60% thu nhập, cao bất thường — kiểm lại số liệu.",
    DataQualityFlag.DEBT_PAYMENT_EXCEEDS_INCOME.value:
        "Khoản trả nợ hàng tháng vượt thu nhập.",
    DataQualityFlag.LOAN_EXCEEDS_ASSET_PRICE.value:
        "Số tiền vay lớn hơn giá trị tài sản.",
    DataQualityFlag.MISSING_EXPENSE.value:
        "Chưa khai chi tiêu hàng tháng — một số phân tích sẽ không đầy đủ.",
    DataQualityFlag.MISSING_DEBT_PAYMENT.value:
        "Có nợ nhưng chưa khai số tiền trả hàng tháng.",
}
