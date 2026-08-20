"""AI-03 task 1 — Lớp `input`: đưa payload ngoài về đúng một hình dạng (F05 · M07).

Cùng một hộ gia đình đang được gọi bằng hai bộ tên khác nhau:

    Laravel / màn Chatbot          Schema chuẩn (`HouseholdProfile`)
    ---------------------------    ---------------------------------
    monthly_income                 average_monthly_income
    monthly_living_cost            average_monthly_expense
    total_debt                     total_current_debt
    current_savings                savings_amount
    supports_elderly               has_dependents

Không có gì sai với việc backend dùng tên của nó. Sai là ở chỗ **để hai bộ tên
cùng đi sâu vào trong hệ thống**: khi đó mỗi hàm phía trong phải tự đoán mình
đang nhận bộ nào, và cái đoán đó lặp lại ở mọi nơi.

Nên quy đổi được làm đúng MỘT lần, ngay ở cửa vào. Từ sau hàm này trở đi, cả
pipeline chỉ còn biết một bộ tên duy nhất.

Vì sao quy đổi nằm ở đây chứ không nằm trong `normalizer`
----------------------------------------------------------
`normalizer` là schema chuẩn — nó định nghĩa hộ gia đình TRONG hệ thống trông
như thế nào. Nhét tên riêng của một client cụ thể vào đó là để hình dạng bên
ngoài quyết định hình dạng bên trong; thêm một client thứ ba là schema chuẩn
lại phình ra.

Ở đây thì ngược lại: mỗi client cần một bảng quy đổi, còn lõi không đổi.

Không quy đổi thì hỏng thế nào
-------------------------------
Đã thấy đúng lỗi đó khi nối nhánh ML02: payload Laravel đi thẳng vào
`normalize_input` cho ra bảy lỗi `Field required` / `Extra inputs are not
permitted`, và người dùng nhận được câu "hồ sơ còn thiếu dữ liệu" trong khi họ
đã điền đủ mọi thứ.
"""
from __future__ import annotations

from typing import Any, Final

from hfml.logger import get_logger

log = get_logger(__name__)

#: Tên bên ngoài → tên trong schema chuẩn.
#:
#: Chỉ liệt kê những trường THẬT SỰ khác tên. Trường trùng tên
#: (`representative_name`, `birth_year`, `household_size`, …) đi thẳng qua.
FIELD_ALIASES: Final[dict[str, str]] = {
    "monthly_income": "average_monthly_income",
    "monthly_expense": "average_monthly_expense",
    "monthly_living_cost": "average_monthly_expense",
    "total_debt": "total_current_debt",
    "current_savings": "savings_amount",
    "supports_elderly": "has_dependents",
}

#: Trường của schema chuẩn — mọi khoá lạ ngoài danh sách này bị bỏ.
#:
#: Bỏ chứ không chuyển tiếp: `HouseholdProfile` từ chối khoá lạ, nên một trường
#: phụ mà backend gửi kèm (id, timestamp, cờ nội bộ) sẽ làm hỏng cả hồ sơ hợp
#: lệ. Người dùng khi đó bị báo "thiếu dữ liệu" vì một thứ họ không hề nhập.
_KNOWN: Final[frozenset[str]] = frozenset({
    "representative_name", "birth_year", "residence", "household_size",
    "children_count", "has_dependents", "average_monthly_income",
    "average_monthly_expense", "has_debt", "total_current_debt",
    "monthly_debt_payment", "has_savings", "savings_amount", "assets",
    "financial_needs", "occupation", "employment_years", "asset_price",
    "loan_amount", "loan_term_months", "guest_session_id", "loan_application",
})


def normalize_payload(
    household: dict[str, Any],
    loan_application: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Đưa payload của client về đúng hình dạng `normalize_input` nhận.

    Quy đổi tên, bỏ khoá lạ, và gắn khối khoản vay vào đúng chỗ. KHÔNG kiểm
    tra giá trị — đó là việc của `normalize_input`, và làm hai lần ở hai nơi
    là tạo hai bộ luật có thể lệch nhau.
    """
    if not isinstance(household, dict):
        return {}

    payload: dict[str, Any] = {}
    dropped: list[str] = []

    for key, value in household.items():
        name = FIELD_ALIASES.get(key, key)
        if name not in _KNOWN:
            dropped.append(key)
            continue
        # Tên chuẩn đã có giá trị thì giữ nguyên: client gửi cả hai biến thể
        # thì bản đúng tên là bản đáng tin hơn.
        if payload.get(name) is None:
            payload[name] = value

    if loan_application is not None:
        payload["loan_application"] = loan_application

    if dropped:
        log.debug("Bỏ %d khoá không thuộc schema: %s",
                  len(dropped), ", ".join(sorted(dropped)))
    return payload
