"""ML02 task 3 — Feature Engineering (F04 · M04 · Tuần 4).

Nối tiếp task 2 ([clean.py](clean.py)). Đầu vào là dữ liệu đã hết bẩn ở mức
từng dòng; đầu ra là hai bộ feature cùng một Pipeline dùng được y hệt lúc
inference.

Ba ràng buộc chi phối mọi quyết định trong file này
---------------------------------------------------
**1. Không giá trị tiền tuyệt đối nào được vào feature set (PLAN.md §2.1).**
`AMT_INCOME_TOTAL` trung vị của Home Credit là 147.150; người dùng Việt Nam
nhập 50.000.000 — lệch ~340 lần. Model gặp giá trị ngoài phân phối huấn luyện
sẽ trả về số vô nghĩa mà **không báo lỗi**. Mọi feature tiền tệ vì vậy đều là
TỈ LỆ giữa hai đại lượng cùng đơn vị, và `RatioFeature.currency_invariant`
kiểm điều đó bằng mã chứ không bằng lời hứa.

**2. Bước nào HỌC từ quần thể thì phải nằm trong Pipeline và `fit` chỉ trên
train.** Ở task này đúng một bước như vậy: `income_per_capita_ratio` cần trung
vị thu nhập đầu người làm mẫu số. Nó được cài thành transformer có `fit()`
(`HomeCreditFeatureBuilder`), không phải một phép tính chạy sẵn trên toàn bộ
dữ liệu rồi lưu lại.

Phần còn lại — chia hai cột, gộp bureau theo từng khách — đều **theo từng
dòng**: kết quả của một hồ sơ không phụ thuộc hồ sơ nào khác, nên chạy trước
hay sau khi chia tập cũng cho cùng một con số.

**3. Cùng một đoạn mã chạy cho cả train lẫn inference.** Hai đường tính riêng
cho train và cho form là chỗ sinh ra sai lệch âm thầm nhất: model vẫn trả xác
suất, chỉ là trên một bộ feature khác với bộ nó được huấn luyện.
`build_feature_pipeline()` trả về đúng một đối tượng, `joblib.dump` cùng model.

Hai bộ feature (PLAN.md §7.2)
------------------------------
    FULL      126 cột đã làm sạch + bureau + tỉ lệ. Có `EXT_SOURCE_1/2/3` —
              nhóm mạnh nhất (IV 0,15–0,33) nhưng form KHÔNG thu được.
              Dùng để chứng minh năng lực kỹ thuật.
    REDUCED   Chỉ feature mà form người dùng cũng sinh ra được. Đây là model
              THỰC SỰ DEPLOY.

Chênh lệch PR-AUC giữa hai bộ chính là mục "phân tích tính khả thi triển
khai" của báo cáo. Không train ở task này.
"""
from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from hfml.data.features.builder import (
    DAYS_PER_YEAR,
    SHARED_FEATURES,
    safe_divide,
)
from hfml.data.preprocessing.pipeline import build_preprocessing_pipeline
from hfml.logger import get_logger
from hfml.ml.ml02_credit_risk.clean import ID_COLUMN, NON_FEATURE_COLUMNS

log = get_logger(__name__)

# --------------------------------------------------------------------------
# Gộp bureau.csv — nguồn của mục C trên form
# --------------------------------------------------------------------------
#: Cột cần đọc từ `bureau.csv`. Đọc đủ 17 cột tốn RAM vô ích.
BUREAU_COLUMNS: Final[tuple[str, ...]] = (
    "SK_ID_CURR", "SK_ID_BUREAU", "CREDIT_ACTIVE", "CREDIT_DAY_OVERDUE",
    "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_SUM_OVERDUE",
    "DAYS_CREDIT",
)

#: Cột sau khi gộp mà mang đơn vị TIỀN. Chúng KHÔNG được vào feature set
#: trực tiếp — chỉ dùng làm tử số cho các tỉ lệ ở `_bureau_ratios()`.
#: Tách hằng số này ra để `assert_no_absolute_money()` có cái mà đối chiếu.
BUREAU_MONEY_COLUMNS: Final[tuple[str, ...]] = (
    "BUREAU_TOTAL_CREDIT", "BUREAU_TOTAL_DEBT", "BUREAU_TOTAL_OVERDUE",
)


def aggregate_bureau(bureau: pd.DataFrame) -> pd.DataFrame:
    """Gộp `bureau.csv` về MỘT DÒNG mỗi khách hàng.

    `bureau.csv` là quan hệ một-nhiều (1,72 triệu khoản vay / 305.811 khách),
    còn form chỉ hỏi bốn con số tổng hợp. Hàm này là bản dịch giữa hai bên và
    phải khớp ĐÚNG định nghĩa form hỏi — lệch định nghĩa thì model học một
    thứ còn người dùng khai một thứ khác.

    Gộp theo từng khách nên KHÔNG học gì từ quần thể: chạy trước hay sau khi
    chia tập đều cho cùng kết quả.
    """
    grouped = bureau.groupby("SK_ID_CURR")
    return pd.DataFrame({
        # ← form: "Số khoản vay trước đây"
        "BUREAU_LOAN_COUNT": grouped["SK_ID_BUREAU"].size(),
        # ← form: "Số lần trả chậm".
        #
        # ⚠️ Lệch định nghĩa, phải ghi vào model_card: bureau chỉ ghi trạng
        # thái quá hạn HIỆN TẠI của mỗi khoản (`CREDIT_DAY_OVERDUE`), không
        # ghi lịch sử từng kỳ. Đây là số KHOẢN đang quá hạn, không phải số
        # LẦN trả chậm. Gần nhau nhưng không bằng nhau.
        "BUREAU_OVERDUE_LOAN_COUNT": grouped["CREDIT_DAY_OVERDUE"]
            .apply(lambda s: int((s > 0).sum())),
        # ← form: "Có khoản vay quá hạn"
        "BUREAU_HAS_OVERDUE": (grouped["CREDIT_DAY_OVERDUE"].max() > 0).astype(int),
        # ← form: "Tổng nợ quá hạn"
        "BUREAU_TOTAL_OVERDUE": grouped["AMT_CREDIT_SUM_OVERDUE"].sum(),
        "BUREAU_TOTAL_DEBT": grouped["AMT_CREDIT_SUM_DEBT"].sum(),
        "BUREAU_TOTAL_CREDIT": grouped["AMT_CREDIT_SUM"].sum(),
        "BUREAU_ACTIVE_LOAN_COUNT": grouped["CREDIT_ACTIVE"]
            .apply(lambda s: int((s == "Active").sum())),
        "BUREAU_MAX_DAYS_OVERDUE": grouped["CREDIT_DAY_OVERDUE"].max(),
        "BUREAU_HISTORY_YEARS": -grouped["DAYS_CREDIT"].min() / DAYS_PER_YEAR,
    })


#: Cột gộp được điền 0 khi khách không có bản ghi bureau nào.
#:
#: Điền **0 chứ không phải NaN**: không tìm thấy gì ở trung tâm tín dụng
#: nghĩa là *chưa từng vay*, đúng bằng câu trả lời `previous_loan_count = 0`
#: mà form cho phép chọn. Để NaN rồi impute trung vị sẽ gán cho người chưa
#: từng vay một lịch sử tín dụng trung bình mà họ không hề có.
_BUREAU_ZERO_FILLED: Final[tuple[str, ...]] = (
    "BUREAU_LOAN_COUNT", "BUREAU_OVERDUE_LOAN_COUNT", "BUREAU_HAS_OVERDUE",
    "BUREAU_TOTAL_OVERDUE", "BUREAU_TOTAL_DEBT", "BUREAU_TOTAL_CREDIT",
    "BUREAU_ACTIVE_LOAN_COUNT", "BUREAU_MAX_DAYS_OVERDUE",
)


class BureauJoiner(BaseEstimator, TransformerMixin):
    """Nối phần tổng hợp bureau vào hồ sơ, theo `SK_ID_CURR`.

    `fit()` rỗng — bảng gộp được truyền vào lúc dựng, không học từ `X`. Nhờ
    vậy transformer này chạy được cả lúc train lẫn lúc inference mà không
    mang theo thống kê nào của tập train.

    `BUREAU_HISTORY_YEARS` giữ `NaN` cho người không có bản ghi, khác với
    nhóm cột được điền 0: "số năm có lịch sử tín dụng" của người chưa từng
    vay không phải 0 năm, nó **không tồn tại**. Điền 0 là khẳng định họ vừa
    mở quan hệ tín dụng hôm nay — một điều sai và model sẽ học theo.
    """

    def __init__(self, aggregates: pd.DataFrame | None = None):
        self.aggregates = aggregates

    def fit(self, X: pd.DataFrame, y=None) -> "BureauJoiner":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.aggregates is None or self.aggregates.empty:
            return X.copy()

        merged = X.merge(self.aggregates, how="left",
                         left_on=ID_COLUMN, right_index=True)
        zero_filled = [c for c in _BUREAU_ZERO_FILLED if c in merged.columns]
        merged[zero_filled] = merged[zero_filled].fillna(0)

        # `concat` chứ không `assign`: sau `merge`, khung 137 cột đã phân mảnh
        # nên thêm một cột lẻ làm pandas cảnh báo hiệu năng ở mọi lần gọi.
        no_record = merged["BUREAU_HISTORY_YEARS"].isna().astype(int).rename(
            "BUREAU_NO_RECORD")
        return pd.concat([merged, no_record], axis=1)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        extra = list(self.aggregates.columns) + ["BUREAU_NO_RECORD"] \
            if self.aggregates is not None else []
        return np.asarray(list(input_features or []) + extra, dtype=object)


# --------------------------------------------------------------------------
# Feature tỉ lệ
# --------------------------------------------------------------------------
#: Feature sinh thêm ở task này, ngoài bảy `SHARED_FEATURES` của
#: `hfml.data.features.builder`. Mỗi dòng: (tên, mô tả, công thức, có ở bộ
#: rút gọn không).
#:
#: Cột "rút gọn" trả lời câu hỏi duy nhất đáng quan tâm khi thêm feature cho
#: ML02: *form người dùng có thu được nó không*. Không thu được thì feature đó
#: chỉ làm đẹp bộ FULL, không giúp gì cho model deploy thật.
ENGINEERED_FEATURES: Final[tuple[tuple[str, str, str, bool], ...]] = (
    ("credit_term_implied",
     "Số kỳ trả ngụ ý của khoản vay",
     "AMT_CREDIT / AMT_ANNUITY", True),
    ("bureau_loan_count",
     "Số khoản vay từng có ở trung tâm tín dụng",
     "COUNT(bureau)", True),
    ("bureau_active_loan_count",
     "Số khoản vay còn hiệu lực",
     "COUNT(CREDIT_ACTIVE = 'Active')", True),
    ("bureau_overdue_loan_count",
     "Số khoản đang quá hạn",
     "COUNT(CREDIT_DAY_OVERDUE > 0)", True),
    ("bureau_has_overdue",
     "Có khoản vay quá hạn hay không",
     "MAX(CREDIT_DAY_OVERDUE) > 0", True),
    ("bureau_overdue_loan_share",
     "Tỉ lệ khoản vay đang quá hạn trên tổng số khoản",
     "overdue_count / loan_count", True),
    ("bureau_debt_income_ratio",
     "Dư nợ ở trung tâm tín dụng trên thu nhập năm",
     "SUM(AMT_CREDIT_SUM_DEBT) / AMT_INCOME_TOTAL", True),
    ("bureau_overdue_income_ratio",
     "Nợ quá hạn trên thu nhập năm",
     "SUM(AMT_CREDIT_SUM_OVERDUE) / AMT_INCOME_TOTAL", True),
    ("bureau_credit_income_ratio",
     "Tổng hạn mức tín dụng từng được cấp trên thu nhập năm",
     "SUM(AMT_CREDIT_SUM) / AMT_INCOME_TOTAL", True),
    ("bureau_history_years",
     "Số năm có quan hệ tín dụng",
     "-MIN(DAYS_CREDIT) / 365.25", False),
    ("bureau_no_record",
     "Không có bản ghi nào ở trung tâm tín dụng",
     "bureau join rỗng", True),
    ("credit_goods_markup",
     "Mức đội giá của khoản vay so với giá hàng",
     "AMT_CREDIT / AMT_GOODS_PRICE", False),
)

#: Feature nằm trong `SHARED_FEATURES` nhưng bị loại khỏi bộ RÚT GỌN, kèm lý do.
#:
#: Khác `FULL_ONLY_FEATURES` (form không sinh ra được): chỗ này form **có** sinh
#: ra, nhưng giá trị sinh ra lại vô nghĩa với model đã train trên Home Credit.
REDUCED_EXCLUDED: Final[dict[str, str]] = {
    "income_per_capita_ratio":
        "Mẫu số là TRUNG VỊ THU NHẬP ĐẦU NGƯỜI CỦA HOME CREDIT, học lúc `fit`. "
        "Tử số lúc inference lại là thu nhập đầu người của hộ Việt Nam — lệch "
        "khoảng ba bậc. Đo thật: hộ 30 triệu/tháng cho tỉ lệ 1.200, hộ 400 "
        "triệu cho 16.000, trong khi biên kẹp trên chỉ 9,00. MỌI hồ sơ Việt "
        "Nam vì vậy nhận đúng một giá trị 9,00 — feature thành HẰNG SỐ lúc "
        "chạy thật.\n"
        "Tác hại không dừng ở 'phí một cột': lúc train nó CÓ phương sai nên "
        "model đã học các nhánh chẻ trên nó, và lúc inference mọi nhánh đó rẽ "
        "cố định một phía. Model chạy trên một cây đã bị cắt nhánh âm thầm.\n"
        "PLAN §2.1b đã chốt nguyên tắc: chưa có mức tham chiếu của chính quần "
        "thể đó thì để `NaN`, KHÔNG bịa số. Mà một cột `NaN` toàn phần thì "
        "không phải feature — nên loại thẳng khỏi bộ deploy. Có số liệu GSO "
        "cho Việt Nam thì đưa lại vào, đó mới là cách khôi phục đúng.\n"
        "Bộ FULL vẫn giữ: ở đó cả tử lẫn mẫu đều là Home Credit nên tỉ lệ có "
        "nghĩa, và bộ FULL không dùng để deploy.",
}

#: Bộ RÚT GỌN = feature dùng chung (trừ `REDUCED_EXCLUDED`) + phần sinh thêm
#: mà form thu được. Thứ tự cố định — thứ tự cột sai là lỗi im lặng
#: (`hfml.ml.registry`).
REDUCED_FEATURES: Final[tuple[str, ...]] = tuple(
    name for name in SHARED_FEATURES if name not in REDUCED_EXCLUDED
) + tuple(
    name for name, _, _, in_reduced in ENGINEERED_FEATURES if in_reduced
)

#: Feature chỉ có ở bộ FULL.
FULL_ONLY_FEATURES: Final[tuple[str, ...]] = tuple(
    name for name, _, _, in_reduced in ENGINEERED_FEATURES if not in_reduced
)


#: Tên các nhóm feature dẫn xuất, khai báo một lần để `transform()` và
#: `get_feature_names_out()` không thể nói khác nhau.
#:
#: Đã sập đúng lỗi đó lúc chạy thật: `get_feature_names_out` tự liệt kê lại
#: danh sách tên theo trí nhớ, sót 6 tỉ lệ dùng chung, và sklearn ném lỗi độ
#: dài không khớp khi gán tên cột cho bộ FULL. Nay cả hai chỗ cùng đọc ba
#: hằng số dưới đây.
_APPLICATION_RATIO_NAMES: Final[tuple[str, ...]] = (
    "dti", "credit_income_ratio", "children_ratio", "age_years",
    "employment_years", "employment_ratio", "credit_term_implied",
    "income_per_capita_ratio",
)

_BUREAU_RATIO_NAMES: Final[tuple[str, ...]] = (
    "bureau_loan_count", "bureau_active_loan_count", "bureau_overdue_loan_count",
    "bureau_has_overdue", "bureau_overdue_loan_share", "bureau_debt_income_ratio",
    "bureau_overdue_income_ratio", "bureau_credit_income_ratio",
    "bureau_history_years", "bureau_no_record",
)

_GOODS_RATIO_NAMES: Final[tuple[str, ...]] = ("credit_goods_markup",)


def engineered_names_for(columns: list[str]) -> list[str]:
    """Tên feature dẫn xuất sẽ sinh ra từ một bộ cột đầu vào, ĐÚNG THỨ TỰ.

    Dùng chung cho cả `transform()` lẫn `get_feature_names_out()`. Hai nơi tự
    liệt kê riêng là lỗi im lặng dạng nặng nhất: model vẫn chạy, vẫn trả xác
    suất, chỉ có điều tên cột lệch khỏi nội dung cột.
    """
    names = list(_APPLICATION_RATIO_NAMES)
    if "BUREAU_LOAN_COUNT" in columns:
        names += list(_BUREAU_RATIO_NAMES)
    if "AMT_GOODS_PRICE" in columns:
        names += list(_GOODS_RATIO_NAMES)
    return names


def _application_ratios(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Bảy tỉ lệ dùng chung + số kỳ trả ngụ ý, từ `application_train`."""
    return {
        # Annuity / Income — phần thu nhập dành trả nợ.
        "dti": safe_divide(df["AMT_ANNUITY"], df["AMT_INCOME_TOTAL"]),
        # Credit / Income — số năm thu nhập để trả hết khoản vay.
        "credit_income_ratio": safe_divide(df["AMT_CREDIT"], df["AMT_INCOME_TOTAL"]),
        "children_ratio": safe_divide(df["CNT_CHILDREN"], df["CNT_FAM_MEMBERS"]),
        "age_years": -df["DAYS_BIRTH"] / DAYS_PER_YEAR,
        "employment_years": -df["DAYS_EMPLOYED"] / DAYS_PER_YEAR,
        "employment_ratio": safe_divide(-df["DAYS_EMPLOYED"], -df["DAYS_BIRTH"]),
        # Credit / Annuity — số kỳ trả ngụ ý. Không mang đơn vị tiền (tiền
        # chia tiền), và nó bù cho việc kỳ hạn vay không có sẵn cột riêng.
        "credit_term_implied": safe_divide(df["AMT_CREDIT"], df["AMT_ANNUITY"]),
    }


def _bureau_ratios(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Nhóm lịch sử tín dụng — đếm giữ nguyên, tiền quy về tỉ lệ.

    Số ĐẾM (số khoản vay, số khoản quá hạn) không mang đơn vị tiền tệ nên
    giữ nguyên được: "3 khoản vay" ở Việt Nam và ở Home Credit là cùng một
    thứ. Chỉ ba cột TIỀN mới phải quy về tỉ lệ trên thu nhập.
    """
    income = df["AMT_INCOME_TOTAL"]
    return {
        "bureau_loan_count": df["BUREAU_LOAN_COUNT"],
        "bureau_active_loan_count": df["BUREAU_ACTIVE_LOAN_COUNT"],
        "bureau_overdue_loan_count": df["BUREAU_OVERDUE_LOAN_COUNT"],
        "bureau_has_overdue": df["BUREAU_HAS_OVERDUE"],
        # Mẫu số 0 (chưa từng vay) → NaN, không phải inf. Cờ
        # `bureau_no_record` đã mang đúng thông tin "chưa từng vay" nên ở đây
        # NaN là câu trả lời đúng: tỉ lệ này không định nghĩa được.
        "bureau_overdue_loan_share": safe_divide(
            df["BUREAU_OVERDUE_LOAN_COUNT"], df["BUREAU_LOAN_COUNT"]),
        "bureau_debt_income_ratio": safe_divide(df["BUREAU_TOTAL_DEBT"], income),
        "bureau_overdue_income_ratio": safe_divide(df["BUREAU_TOTAL_OVERDUE"], income),
        "bureau_credit_income_ratio": safe_divide(df["BUREAU_TOTAL_CREDIT"], income),
        "bureau_history_years": df["BUREAU_HISTORY_YEARS"],
        "bureau_no_record": df["BUREAU_NO_RECORD"],
    }


class HomeCreditFeatureBuilder(BaseEstimator, TransformerMixin):
    """Sinh feature tỉ lệ từ dữ liệu Home Credit đã làm sạch.

    Đây là bước DUY NHẤT của task 3 có `fit()` không rỗng, và nó học đúng một
    con số: trung vị thu nhập đầu người, dùng làm mẫu số cho
    `income_per_capita_ratio`.

    Vì sao con số đó bắt buộc phải học trong Pipeline chứ không tính sẵn:
    thu nhập đầu người là TIỀN, đưa thẳng vào feature set là tái tạo đúng
    domain gap mà §2.1 muốn diệt (hộ VN 50.000.000 ÷ 4 = 12.500.000 so với
    Home Credit 147.150 ÷ 2 = 73.575 — vẫn lệch 170 lần). Chia cho mức tham
    chiếu của **chính quần thể đó** mới làm nó bất biến. Mà "mức tham chiếu
    của quần thể" là một thống kê, nên tính trên toàn bộ dữ liệu rồi mới chia
    tập là rò rỉ.

    `feature_set`:
        "full"     giữ nguyên mọi cột đầu vào và THÊM feature tỉ lệ
        "reduced"  chỉ trả về `REDUCED_FEATURES`, đúng thứ tự
    """

    def __init__(self, feature_set: str = "full"):
        self.feature_set = feature_set

    def fit(self, X: pd.DataFrame, y=None) -> "HomeCreditFeatureBuilder":
        if self.feature_set not in ("full", "reduced"):
            raise ValueError(
                f"feature_set không hợp lệ: {self.feature_set!r} "
                "(chỉ nhận 'full' hoặc 'reduced')")

        per_capita = safe_divide(X["AMT_INCOME_TOTAL"], X["CNT_FAM_MEMBERS"])
        median = float(per_capita.median())
        # Trung vị ≤ 0 hoặc NaN thì không dùng làm mẫu số được. Để `None` và
        # cho feature ra NaN, thay vì chia cho một số vô nghĩa.
        self.reference_income_per_capita_ = median if median > 0 else None
        if self.reference_income_per_capita_ is None:
            log.warning("Không tính được trung vị thu nhập đầu người "
                        "→ income_per_capita_ratio = NaN")

        self.feature_names_in_ = list(X.columns)
        self.n_features_in_ = X.shape[1]
        # Chốt danh sách tên ngay lúc fit, từ chính bộ cột đầu vào — cùng một
        # hàm mà `transform()` dùng, nên hai bên không thể lệch.
        self.engineered_names_ = engineered_names_for(self.feature_names_in_)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        engineered = pd.DataFrame(index=X.index)
        for name, series in _application_ratios(X).items():
            engineered[name] = np.asarray(series)

        if self.reference_income_per_capita_:
            per_capita = safe_divide(X["AMT_INCOME_TOTAL"], X["CNT_FAM_MEMBERS"])
            engineered["income_per_capita_ratio"] = (
                np.asarray(per_capita) / self.reference_income_per_capita_)
        else:
            engineered["income_per_capita_ratio"] = np.nan

        if "BUREAU_LOAN_COUNT" in X.columns:
            for name, series in _bureau_ratios(X).items():
                engineered[name] = np.asarray(series)

        if "AMT_GOODS_PRICE" in X.columns:
            # KHÔNG đặt tên `ltv`: đại lượng này LUÔN ≥ 1,0 vì AMT_CREDIT đã
            # cộng phí và bảo hiểm lên giá hàng (p1 = 1,000 · trung vị 1,119).
            # Nó đo MỨC ĐỘI GIÁ, không đo tỉ lệ vay trên tài sản. Gộp nó với
            # `ltv` của form là đẩy hồ sơ VN ra ngoài phân phối huấn luyện.
            engineered["credit_goods_markup"] = np.asarray(
                safe_divide(X["AMT_CREDIT"], X["AMT_GOODS_PRICE"]))

        if self.feature_set == "reduced":
            missing = [c for c in REDUCED_FEATURES if c not in engineered.columns]
            if missing:
                raise ValueError(
                    f"Bộ rút gọn thiếu feature: {missing}. Nhiều khả năng chưa "
                    "nối bureau — xem `BureauJoiner`.")
            return engineered[list(REDUCED_FEATURES)]

        # Bộ FULL: giữ nguyên cột gốc rồi thêm feature dẫn xuất. Loại các cột
        # tiền tuyệt đối đã dùng làm tử số? KHÔNG — bộ Full có chủ đích dùng
        # mọi thứ Home Credit có, kể cả cột không deploy được. Đó chính là
        # điều nó tồn tại để đo (§7.2).
        kept = [c for c in X.columns if c not in engineered.columns]
        return pd.concat([X[kept], engineered], axis=1)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        if self.feature_set == "reduced":
            return np.asarray(REDUCED_FEATURES, dtype=object)

        base = list(input_features if input_features is not None
                    else self.feature_names_in_)
        engineered = engineered_names_for(base)
        return np.asarray(
            [c for c in base if c not in engineered] + engineered, dtype=object)


# --------------------------------------------------------------------------
# Kiểm tra bất biến đơn vị tiền tệ
# --------------------------------------------------------------------------
#: Tiền tố của cột mang giá trị tiền TUYỆT ĐỐI ở Home Credit.
_MONEY_PREFIXES: Final[tuple[str, ...]] = ("AMT_",)


def absolute_money_columns(columns: list[str]) -> list[str]:
    """Các cột mang giá trị tiền tuyệt đối trong một danh sách feature.

    Bộ RÚT GỌN phải không có cột nào ở đây. Bộ FULL thì có, và đó là chủ ý —
    nó cố tình dùng mọi thứ Home Credit có để đo xem việc không deploy được
    đổi lại bao nhiêu năng lực dự báo.
    """
    return [
        c for c in columns
        if any(c.startswith(p) for p in _MONEY_PREFIXES)
        or c in BUREAU_MONEY_COLUMNS
    ]


# --------------------------------------------------------------------------
# Pipeline dùng chung train ↔ inference
# --------------------------------------------------------------------------
#: Feature MIỄN kẹp biên — đuôi phân phối của chúng là TÍN HIỆU, không phải
#: nhiễu đo đạc. Đo trên 307.511 hồ sơ `application_train.csv`:
#:
#:     bureau_overdue_loan_count    98,90% bằng 0 → p99,9 = 1, nên biên kẹp là
#:         [0 · 1]. Ba mức 1 / 2 / ≥3 có tỉ lệ vỡ nợ 14,48% / 28,83% / 57,89%
#:         (lift 1,79 / 3,57 / 7,17) bị ép hết về một. Tệ hơn: sau khi kẹp cột
#:         này TRÙNG KHÍT 100% với `bureau_has_overdue`, mà bộ rút gọn lại TẮT
#:         khử tương quan — hai cột y hệt nhau cùng vào model và chia đôi
#:         importance của cùng một tín hiệu.
#:     bureau_overdue_loan_share    Đã bị chặn trong [0 · 1] bởi chính công
#:         thức (count ÷ loan_count) nên KHÔNG thể có ngoại lai. Biên [0 · 0,5]
#:         cắt mất 173 hồ sơ có tỉ lệ vỡ nợ 26,59%.
#:     bureau_overdue_income_ratio  Biên [0 · 0,1147] cắt mất 342 hồ sơ có tỉ
#:         lệ vỡ nợ 39,47%.
#:     bureau_has_overdue           Nhị phân sẵn, kẹp là phép đồng nhất. Để
#:         vào đây cho đủ nhóm và để không ai tưởng nó bị bỏ sót.
#:
#: Vì sao miễn kẹp là AN TOÀN ở đây: bốn thuật toán của ML02 đều là cây và
#: `scaling="none"`. Cây bất biến với biến đổi đơn điệu, nên một giá trị lớn
#: chỉ rơi vào lá ngoài cùng — nó không kéo lệch scaler như
#: `AMT_INCOME_TOTAL = 117.000.000` từng làm ở §4.3c. Cái giá của việc kẹp thì
#: có thật và không khôi phục được, còn cái lợi thì bằng không.
#:
#: Vì sao KHÔNG dùng `log1p`: cây bất biến với mọi biến đổi ĐƠN ĐIỆU, nên
#: `log1p(count)` cho ra đúng cùng một phép phân hoạch với `count`. Nó không
#: sửa được gì — thứ phá tín hiệu là phép kẹp NHIỀU-VỀ-MỘT, không phải thang đo.
#:     credit_term_implied          Biên `[8,33 · 37,92]`. Đây là feature MẠNH
#:         NHẤT của bộ rút gọn (SHAP 0,373 — gấp 4 lần `dti`), và biên DƯỚI của
#:         nó cắt đúng phần mang tin tốt: `AMT_CREDIT / AMT_ANNUITY` nhỏ nghĩa
#:         là trả xong nhanh, tín hiệu rủi ro thấp rõ rệt. Mọi hồ sơ trả nhanh
#:         hơn 8,33 năm bị gộp thành một giá trị, nên model không phân biệt nổi
#:         người trả trong 3 năm với người trả trong 8 năm. Cùng lỗi với nhóm
#:         overdue, chỉ khác là ở đầu dưới thay vì đầu trên.
NO_CLIP_FEATURES: Final[tuple[str, ...]] = (
    "bureau_overdue_loan_count",
    "bureau_has_overdue",
    "bureau_overdue_loan_share",
    "bureau_overdue_income_ratio",
    "credit_term_implied",
)


def build_feature_pipeline(
    *,
    feature_set: str = "reduced",
    bureau_aggregates: pd.DataFrame | None = None,
    protect: tuple[str, ...] = (),
    **preprocessing_kwargs,
) -> Pipeline:
    """Pipeline đầy đủ: nối bureau → sinh feature → tiền xử lý.

    Trả về MỘT đối tượng `joblib.dump` được cùng model. Lúc inference nạp lại
    và gọi `transform()` — cùng đoạn mã, cùng thống kê đã học, nên không có
    chỗ nào để train và inference lệch nhau.

    Bộ RÚT GỌN tắt bước khử tương quan (`correlation_threshold=None`): 18 cột
    đã chọn tay theo tiêu chí "form có thu được không", lọc thêm bằng tương
    quan là bỏ mất feature mà chính hệ thống đang hỏi người dùng.
    """
    if feature_set not in ("full", "reduced"):
        raise ValueError(f"feature_set không hợp lệ: {feature_set!r}")

    defaults: dict = {"encoding": "ordinal", "scaling": "none",
                      "clip_exclude": NO_CLIP_FEATURES}
    if feature_set == "reduced":
        defaults["correlation_threshold"] = None
    defaults.update(preprocessing_kwargs)

    return Pipeline([
        ("bureau", BureauJoiner(aggregates=bureau_aggregates)),
        ("features", HomeCreditFeatureBuilder(feature_set=feature_set)),
        ("preprocess", build_preprocessing_pipeline(protect=protect, **defaults)),
    ])


def merge_bureau(app: pd.DataFrame, aggregates: pd.DataFrame) -> pd.DataFrame:
    """Nối bureau vào hồ sơ — dạng hàm, cho việc khảo sát ngoài Pipeline.

    Cùng một phép biến đổi với `BureauJoiner.transform()`, không phải bản
    sao: hàm này gọi thẳng transformer đó. Hai định nghĩa cho cùng một phép
    gộp là chỗ sớm muộn cho ra hai con số khác nhau về cùng một khách hàng.
    """
    return BureauJoiner(aggregates=aggregates).transform(app)


def split_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Tách `X` và `y`, loại đúng ba cột không bao giờ là feature.

    `SK_ID_CURR` được GIỮ trong `X` vì `BureauJoiner` cần nó để nối; chính
    `ColumnTransformer` ở bước cuối mới loại nó (`remainder="drop"`). Không
    loại sớm ở đây, và cũng không để nó lọt vào feature set — hai việc đó
    được `build_preprocessing_pipeline` bảo đảm.
    """
    from hfml.ml.ml02_credit_risk.clean import TARGET_COLUMN

    drop = [c for c in NON_FEATURE_COLUMNS if c in df.columns and c != ID_COLUMN]
    return df.drop(columns=drop), df[TARGET_COLUMN]
