"""AI-01 task 2 — Áp preprocessing khi inference (F05 · M05).

Dịch hồ sơ form sang ĐÚNG bộ cột mà từng model được huấn luyện trên đó, rồi
để chính Pipeline đã `joblib.dump` lo phần biến đổi.

Nguyên tắc: KHÔNG dựng lại preprocessing bằng tay
---------------------------------------------------
Tầng này chỉ ĐỔI TÊN CỘT và QUY ĐỔI ĐƠN VỊ. Mọi phép biến đổi có học — điền
thiếu, kẹp biên, mã hoá, trung vị thu nhập đầu người — đều do transformer đã
`fit` trên tập train thực hiện. Dựng lại bằng tay là lệch so với training, và
lệch theo kiểu không báo lỗi.

Hai model, hai hợp đồng khác hẳn nhau
--------------------------------------
    ML01   17 biến THÔ của form (`RAW_FEATURES`). Ánh xạ gần như 1-1, chỉ có
           `age` phải suy từ `birth_year` và `assets` phải trải thành 6 cột.
    ML02   Cột tên Home Credit + phần tổng hợp bureau. Đây là chỗ khó.

⚠️ Vì sao ML02 cần một thao tác đặc biệt lúc nạp model
-------------------------------------------------------
Artifact `ml02_xgboost_reduced_vfinal` có `BureauJoiner` ở bước đầu, và bước
đó **tra cứu lịch sử tín dụng theo `SK_ID_CURR`** từ bảng gộp 305.810 khách
hàng Home Credit nướng sẵn trong file. Người dùng Việt Nam không có
`SK_ID_CURR` nào khớp, nên nếu để nguyên thì:

    · mọi feature bureau về 0 và `bureau_no_record = 1`
    · **kể cả khi người dùng đã khai đủ mục C trên form**

Mà task 13 đo được ba trong năm feature mạnh nhất của model triển khai chính
là nhóm bureau. Để nguyên là vứt bỏ phần lớn tín hiệu mà form vừa thu được.

Cách xử lý: `neutralise_bureau_lookup()` đặt `aggregates = None` MỘT LẦN lúc
nạp model. `BureauJoiner.transform` khi đó trả nguyên khung đầu vào (nhánh
no-op có sẵn của nó), và tầng này cấp thẳng các cột `BUREAU_*` từ form. Không
sửa gì trong training, không train lại — chỉ tắt một bước tra cứu vô nghĩa ở
inference, và tiện thể bỏ 22 MB bảng tra khỏi bộ nhớ.

Quy đổi kỳ: Home Credit tính theo NĂM
--------------------------------------
`AMT_INCOME_TOTAL` và `AMT_ANNUITY` của Home Credit **cùng kỳ**, và §2.1b đã
chứng minh đó là kỳ NĂM: `credit_income_ratio` trung vị 3,27 chỉ khớp khi thu
nhập tính theo năm. Form thì hỏi theo THÁNG. Vì vậy:

    AMT_INCOME_TOTAL = thu nhập tháng × 12
    AMT_ANNUITY      = khoản trả tháng × 12     ← phải nhân, để cùng kỳ
    AMT_CREDIT       = số tiền vay (không có kỳ)

Nhân cả hai giữ nguyên `dti`, và làm `credit_income_ratio` rơi đúng dải mà
model từng thấy. Quên nhân `AMT_ANNUITY` thì `dti` bị chia 12 — hồ sơ nào
cũng thành "gánh nặng trả nợ rất nhẹ".
"""
from __future__ import annotations

from decimal import Decimal
from typing import Any, Final

import pandas as pd

from hfml.data.schema import (
    ASSET_COLUMNS,
    AssetType,
    HouseholdProfile,
    LoanApplication,
    asset_column,
)
from hfml.logger import get_logger
from hfml.ml.ml01_recommendation.labeler import RAW_FEATURES

log = get_logger(__name__)

#: Số tháng một năm — quy đổi kỳ khi sang cột Home Credit.
MONTHS_PER_YEAR: Final[int] = 12
#: Home Credit lưu ngày dưới dạng số ÂM (số ngày trước ngày nộp đơn).
DAYS_PER_YEAR: Final[float] = 365.25

#: `SK_ID_CURR` giả cho hồ sơ inference.
#:
#: Cột này bị `ColumnTransformer` loại tường minh (`remainder="drop"`) nên giá
#: trị của nó không bao giờ vào model. Dùng số âm để nếu có ai lỡ đưa nó vào
#: feature set thì con số vô lý đó lộ ra ngay, thay vì lẫn vào dải id thật.
SYNTHETIC_ID: Final[int] = -1


def _f(value: Decimal | float | None, default: float = 0.0) -> float:
    return default if value is None else float(value)


def _unknown_unless_never_borrowed(never_borrowed: bool) -> float:
    """`0.0` nếu chưa từng vay, ngược lại `NaN`.

    Hai trạng thái này KHÁC nhau và không được gộp:

        chưa từng vay   → đại lượng bằng 0, và ta BIẾT chắc điều đó
        đã từng vay     → đại lượng có tồn tại nhưng form không hỏi → không biết

    Điền 0 cho nhóm thứ hai là khẳng định họ không còn khoản nào hiệu lực và
    chưa từng được cấp hạn mức nào — sai. Điền một cận trên/cận dưới "an toàn"
    cũng sai theo kiểu khó thấy hơn: nó đúng về bất đẳng thức nhưng đặt hồ sơ
    vào vùng cực đoan của phân phối huấn luyện.

    `NaN` để bước `SimpleImputer(median)` trong Pipeline điền bằng trung vị
    học từ tập train — tức đặt hồ sơ vào chỗ TRUNG TÍNH khi không có thông
    tin, đúng nghĩa "không biết". Cùng quy ước với `BUREAU_HISTORY_YEARS`.
    """
    return 0.0 if never_borrowed else float("nan")


# --------------------------------------------------------------------------
# ML01 — 17 biến thô của form
# --------------------------------------------------------------------------
def to_ml01_frame(profile: HouseholdProfile, age: int) -> pd.DataFrame:
    """Dựng khung một dòng cho ML01, đúng thứ tự `RAW_FEATURES`.

    `age` truyền vào chứ không suy ở đây: thiếu năm sinh là LỖI phải báo ở
    tầng gọi, không phải chỗ để điền một tuổi mặc định (§6.1c). Bắt tham số
    này thành bắt buộc khiến chỗ gọi phải xử lý trường hợp thiếu.

    Ba trường tiền có điều kiện đã được `normalizer` đưa về 0 theo quy ước
    `ZERO_WHEN_ABSENT`, nên `_f(..., 0.0)` ở đây không phải là suy diễn thêm.
    """
    owned = {a.value for a in profile.assets}
    row: dict[str, Any] = {
        "average_monthly_income": _f(profile.average_monthly_income),
        "average_monthly_expense": _f(profile.average_monthly_expense),
        "savings_amount": _f(profile.savings_amount),
        "total_current_debt": _f(profile.total_current_debt),
        "monthly_debt_payment": _f(profile.monthly_debt_payment),
        "household_size": int(profile.household_size),
        "children_count": int(profile.children_count),
        "age": int(age),
        "has_debt": bool(profile.has_debt),
        "has_savings": bool(profile.has_savings),
        "has_dependents": bool(profile.has_dependents),
    }
    # Sáu cột multi-hot: `assets` là nhiều lựa chọn nên không one-hot được.
    for asset in AssetType:
        row[asset_column(asset)] = asset.value in owned

    return pd.DataFrame([[row[name] for name in RAW_FEATURES]],
                        columns=list(RAW_FEATURES))


# --------------------------------------------------------------------------
# ML02 — cột tên Home Credit + tổng hợp bureau từ form
# --------------------------------------------------------------------------
#: Cột mà đường RÚT GỌN của `HomeCreditFeatureBuilder` thực sự đọc.
#:
#: Artifact khai `feature_names_in_` là 137 cột vì nó được `fit` trên khung đã
#: làm sạch đầy đủ, nhưng đường rút gọn chỉ chạm đúng những cột dưới đây. Liệt
#: kê ra để `to_ml02_frame` biết mình phải cấp gì, và để test canh được.
ML02_APPLICATION_COLUMNS: Final[tuple[str, ...]] = (
    "SK_ID_CURR", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY",
    "CNT_CHILDREN", "CNT_FAM_MEMBERS", "DAYS_BIRTH", "DAYS_EMPLOYED",
)

ML02_BUREAU_COLUMNS: Final[tuple[str, ...]] = (
    "BUREAU_LOAN_COUNT", "BUREAU_ACTIVE_LOAN_COUNT", "BUREAU_OVERDUE_LOAN_COUNT",
    "BUREAU_HAS_OVERDUE", "BUREAU_TOTAL_OVERDUE", "BUREAU_TOTAL_DEBT",
    "BUREAU_TOTAL_CREDIT", "BUREAU_HISTORY_YEARS", "BUREAU_NO_RECORD",
)


def to_ml02_frame(
    profile: HouseholdProfile,
    loan: LoanApplication,
) -> pd.DataFrame:
    """Dựng khung một dòng cho ML02, dùng tên cột của Home Credit.

    Ba nhóm quy đổi, mỗi nhóm một cái bẫy riêng:

    **Kỳ tiền** — nhân 12 cho cả thu nhập lẫn khoản trả để cùng kỳ NĂM như
    Home Credit. Quên nhân `AMT_ANNUITY` thì `dti` bị chia 12.

    **Dấu của `DAYS_*`** — Home Credit lưu số ngày TRƯỚC ngày nộp đơn nên giá
    trị luôn ÂM. Để dương thì `employment_ratio = DAYS_EMPLOYED / DAYS_BIRTH`
    vẫn ra số dương hợp lý trông y hệt, nhưng `age_years` thành âm.

    **Mục C → bureau** — form hỏi "số lần trả chậm", bureau đếm "số KHOẢN đang
    quá hạn". Hai đại lượng gần nhau nhưng không bằng nhau; đây là chỗ lệch
    định nghĩa đã ghi vào phần giới hạn của model. Ánh xạ thẳng vì đó là xấp
    xỉ tốt nhất có được, không phải vì chúng đồng nhất.

    Quy tắc cho cột form KHÔNG hỏi: `NaN`, đừng thay bằng cận trên/cận dưới
    ------------------------------------------------------------------------
    Một xấp xỉ "an toàn về bất đẳng thức" vẫn có thể sai nặng về PHÂN PHỐI, và
    kiểu sai đó không báo lỗi. Hai cột dưới đây từng được gán cận, và hệ quả
    đo được trên 46.127 hồ sơ validation là:

        BUREAU_TOTAL_CREDIT  := dư nợ hiện tại   ⟹ ép `debt == credit`
                                                   (chỉ 0,63% hồ sơ train có)
        BUREAU_ACTIVE_LOAN_COUNT := tổng số khoản ⟹ ép `active == total`
                                                   (chỉ 12,47% hồ sơ train có)

        P trung bình   0,08073 → 0,13118   (+63%)
        tỉ lệ cảnh báo  17,78% → 42,79%
        PR-AUC          0,1709 → 0,1362    (−20%)
        25,36% hồ sơ đổi nhãn LOW_RISK/HIGH_RISK

    Xác suất trung bình 0,08073 trùng khít tỉ lệ nền 8,07% của Home Credit —
    tức lớp hiệu chuẩn vốn rất chuẩn, và chính hai phép gán này phá nó.
    """
    annual_income = _f(profile.average_monthly_income) * MONTHS_PER_YEAR
    chua_tung_vay = loan.previous_loan_count == 0

    row = {
        # -- Cột application ------------------------------------------------
        "SK_ID_CURR": SYNTHETIC_ID,
        "AMT_INCOME_TOTAL": annual_income,
        "AMT_CREDIT": _f(loan.loan_amount),
        "AMT_ANNUITY": _f(loan.monthly_payment) * MONTHS_PER_YEAR,
        "CNT_CHILDREN": int(loan.children_count),
        "CNT_FAM_MEMBERS": float(profile.household_size),
        "DAYS_BIRTH": -loan.borrower_age * DAYS_PER_YEAR,
        "DAYS_EMPLOYED": -float(loan.employment_years) * DAYS_PER_YEAR,

        # -- Mục C của form → phần tổng hợp bureau ---------------------------
        "BUREAU_LOAN_COUNT": float(loan.previous_loan_count),
        # Số khoản CÒN HIỆU LỰC: form không hỏi → `NaN`, trừ khi chưa từng vay.
        #
        # Trước đây gán bằng `previous_loan_count` với lý do "dùng cận trên".
        # Đo lại mới thấy cái giá: đẳng thức `active == total` chỉ đúng ở
        # **12,47%** hồ sơ train (tỉ lệ thật trung vị 0,375), nên mọi hồ sơ
        # inference bị đẩy vào một góc phân phối mà model gần như chưa thấy.
        # Cận trên nghe hợp lý nhưng nó KHÔNG phải giá trị trung tính — nó là
        # giá trị cực đoan.
        "BUREAU_ACTIVE_LOAN_COUNT": _unknown_unless_never_borrowed(chua_tung_vay),
        # "số lần trả chậm" (form) ↔ "số khoản đang quá hạn" (bureau) — kẹp lại
        # không vượt tổng số khoản, vì bureau không thể đếm nhiều hơn thế.
        "BUREAU_OVERDUE_LOAN_COUNT": float(
            min(loan.late_payment_count, loan.previous_loan_count)),
        "BUREAU_HAS_OVERDUE": float(loan.has_overdue_loan),
        "BUREAU_TOTAL_OVERDUE": _f(loan.total_overdue_amount),
        # Dư nợ HIỆN TẠI: form có hỏi → ánh xạ thẳng, đây là số thật.
        "BUREAU_TOTAL_DEBT": _f(profile.total_current_debt),
        # Tổng hạn mức TỪNG ĐƯỢC CẤP: form không hỏi → `NaN`.
        #
        # Trước đây gán bằng dư nợ hiện tại với lý do "cận dưới đúng nghĩa".
        # Đúng về mặt bất đẳng thức, sai về mặt phân phối: nó ép
        # `debt == credit`, một đẳng thức chỉ xuất hiện ở **0,63%** hồ sơ train
        # (tỉ lệ thật trung vị 0,2091). Hệ quả đo được: xác suất trung bình
        # 0,0807 → 0,1312 và tỉ lệ cảnh báo 17,78% → 42,79%.
        "BUREAU_TOTAL_CREDIT": _unknown_unless_never_borrowed(chua_tung_vay),
        # Số năm có quan hệ tín dụng: form KHÔNG hỏi. Để `NaN` chứ không đoán —
        # người chưa từng vay thì đại lượng này không tồn tại, còn người từng
        # vay thì ta không biết. Bước điền thiếu trong Pipeline lo tiếp.
        "BUREAU_HISTORY_YEARS": float("nan"),
        "BUREAU_NO_RECORD": float(chua_tung_vay),
    }
    return pd.DataFrame([row])


def neutralise_bureau_lookup(model) -> bool:
    """Tắt bước tra cứu bureau theo `SK_ID_CURR` trong artifact ML02.

    Gọi MỘT LẦN lúc nạp model. Trả về `True` nếu đã tắt được.

    Vì sao cần: xem docstring đầu file. Tóm tắt — bảng tra trong artifact là
    của 305.810 khách hàng Home Credit, hoàn toàn vô nghĩa với một người dùng
    Việt Nam, và nếu để nguyên thì nó GHI ĐÈ mục C mà form vừa thu được.

    Đây KHÔNG phải sửa training: `BureauJoiner` có sẵn nhánh no-op khi
    `aggregates is None`, và ta chỉ dùng đúng nhánh đó.
    """
    try:
        joiner = (model.calibrated.estimator.estimator
                  .named_steps["features"].named_steps["bureau"])
    except AttributeError:
        log.warning("Không tìm thấy bước bureau trong artifact — bỏ qua.")
        return False

    if joiner.aggregates is None:
        return True

    n_rows = len(joiner.aggregates)
    joiner.aggregates = None
    log.info("Đã tắt tra cứu bureau (%d dòng) — mục C lấy thẳng từ form.", n_rows)
    return True
