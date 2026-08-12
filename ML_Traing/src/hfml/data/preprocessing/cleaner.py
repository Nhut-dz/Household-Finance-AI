"""Xử lý missing values (F01 task 8).

Ranh giới quan trọng nhất của file này
--------------------------------------
Không phải mọi việc "xử lý missing" đều thuộc về đây. Chia theo tiêu chí
**có học gì từ dữ liệu hay không**:

    Ở ĐÂY — biến đổi xác định, theo từng dòng, không học gì:
        sentinel → NaN, chuỗi giả (XNA/Unknown) → NaN, sinh cờ `_MISSING`.
        Chạy trước khi split cũng KHÔNG rò rỉ, vì kết quả của một dòng không
        phụ thuộc dòng nào khác.

    TRONG PIPELINE (task 10, 11, 14) — có học từ dữ liệu:
        điền trung vị / mode, chọn cột theo tỉ lệ thiếu.
        Bắt buộc `fit` chỉ trên tập train, nếu không là rò rỉ (PLAN.md §4.4).

`MissingNormalizer` cố tình để `fit()` rỗng — đó là bằng chứng nó không học
gì. `HighMissingDropper` thì ngược lại, có học, nên phải nằm trong Pipeline.

Vì sao giữ cờ `_MISSING` thay vì điền rồi quên
----------------------------------------------
Ở Home Credit, việc THIẾU dữ liệu tự nó dự báo được vỡ nợ. Đo trên toàn bộ
307.511 hồ sơ (tỉ lệ vỡ nợ chung 8,07%):

    cột                        thiếu    vỡ nợ khi thiếu / khi có   lift
    DAYS_EMPLOYED             18,01%          5,40% / 8,66%       0,624
    OCCUPATION_TYPE           31,35%          6,51% / 8,79%       0,741
    AMT_REQ_CREDIT_BUREAU_*   13,50%         10,34% / 7,72%       1,339
    TOTALAREA_MODE (nhà ở)    48,27%          9,23% / 6,99%       1,321
    EXT_SOURCE_3              19,83%          9,31% / 7,77%       1,199
    EXT_SOURCE_1              56,38%          8,52% / 7,50%       1,137

Điền trung vị rồi bỏ cờ đi là xóa hẳn tín hiệu này. Giữ cờ thì model vẫn
dùng được giá trị đã điền mà không mất thông tin "vốn dĩ không có".

Vì sao 6 cờ chứ không phải 63
-----------------------------
63 cột có missing chỉ tạo ra 28 mẫu thiếu phân biệt, và 21 trong số đó là
cùng một khối "không có thông tin nhà ở" với lift gần như nhau (1,14–1,32).
Tạo cờ cho từng cột là nhân bản một thông tin hàng chục lần, làm loãng
feature importance và khiến bảng giải thích không đọc được.

Ba nhóm bị loại có lý do rõ ràng:
    EXT_SOURCE_2      lift 0,976 — thiếu hay không cũng vỡ nợ như nhau
    SOCIAL_CIRCLE     lift 0,436 nhưng chỉ 0,33% (≈1.000 dòng) — n quá nhỏ
    NAME_TYPE_SUITE   lift 0,670 nhưng chỉ 0,42% — n quá nhỏ
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from hfml.data.quality import KNOWN_SENTINELS, PLACEHOLDER_VALUES
from hfml.logger import get_logger

log = get_logger(__name__)

FLAG_SUFFIX: Final[str] = "_MISSING"


@dataclass(frozen=True)
class MissingFlag:
    """Một cờ nhị phân, kèm số đo biện minh cho sự tồn tại của nó."""

    name: str
    #: Cột đại diện — cờ bật khi cột này thiếu.
    source: str
    #: Các cột khác cùng mẫu thiếu, cờ này đại diện cho cả nhóm.
    covers: tuple[str, ...] = ()
    #: Tỉ lệ thiếu đo trên application_train.csv.
    missing_rate: float = 0.0
    #: Tỉ lệ vỡ nợ khi thiếu ÷ khi có. Xa 1,0 = tín hiệu mạnh.
    lift: float = 1.0
    note: str = ""


#: Sáu cờ được giữ. Thứ tự cố định để `feature_names` tái lập được.
MISSING_FLAGS: Final[tuple[MissingFlag, ...]] = (
    MissingFlag(
        name="DAYS_EMPLOYED" + FLAG_SUFFIX,
        source="DAYS_EMPLOYED",
        missing_rate=0.1801, lift=0.624,
        note="Nghỉ hưu (55.352) hoặc thất nghiệp (22). Trùng khít "
             "ORGANIZATION_TYPE='XNA' và OCCUPATION_TYPE rỗng. Vỡ nợ ÍT hơn hẳn.",
    ),
    MissingFlag(
        name="OCCUPATION_TYPE" + FLAG_SUFFIX,
        source="OCCUPATION_TYPE",
        missing_rate=0.3135, lift=0.741,
        note="Bao gồm cả nhóm nghỉ hưu ở trên, cộng người đi làm không khai nghề.",
    ),
    MissingFlag(
        name="CREDIT_BUREAU_REQ" + FLAG_SUFFIX,
        source="AMT_REQ_CREDIT_BUREAU_HOUR",
        covers=(
            "AMT_REQ_CREDIT_BUREAU_HOUR", "AMT_REQ_CREDIT_BUREAU_DAY",
            "AMT_REQ_CREDIT_BUREAU_WEEK", "AMT_REQ_CREDIT_BUREAU_MON",
            "AMT_REQ_CREDIT_BUREAU_QRT", "AMT_REQ_CREDIT_BUREAU_YEAR",
        ),
        missing_rate=0.1350, lift=1.339,
        note="Không có bản ghi truy vấn tín dụng — tín hiệu RỦI RO mạnh nhất "
             "trong nhóm cờ.",
    ),
    MissingFlag(
        name="BUILDING_INFO" + FLAG_SUFFIX,
        source="TOTALAREA_MODE",
        covers=("COMMONAREA_AVG", "LIVINGAPARTMENTS_AVG", "FLOORSMAX_AVG",
                "ENTRANCES_AVG", "APARTMENTS_AVG", "YEARS_BUILD_AVG",
                "EMERGENCYSTATE_MODE", "HOUSETYPE_MODE", "WALLSMATERIAL_MODE"),
        missing_rate=0.4827, lift=1.321,
        note="Đại diện cho khối ~21 cột thông tin nhà ở có cùng mẫu thiếu.",
    ),
    MissingFlag(
        name="EXT_SOURCE_3" + FLAG_SUFFIX,
        source="EXT_SOURCE_3",
        missing_rate=0.1983, lift=1.199,
    ),
    MissingFlag(
        name="EXT_SOURCE_1" + FLAG_SUFFIX,
        source="EXT_SOURCE_1",
        missing_rate=0.5638, lift=1.137,
    ),
)

#: Cờ bị loại, kèm lý do — để trả lời "sao không làm cho cột này?".
REJECTED_FLAGS: Final[dict[str, str]] = {
    "EXT_SOURCE_2": "lift 0,976 — thiếu hay không cũng vỡ nợ như nhau",
    "OBS_30_CNT_SOCIAL_CIRCLE": "lift 0,436 nhưng chỉ 0,33% (~1.000 dòng), n quá nhỏ",
    "NAME_TYPE_SUITE": "lift 0,670 nhưng chỉ 0,42%, n quá nhỏ",
}


def _text_columns(df: pd.DataFrame) -> list[str]:
    """Cột dạng chữ. pandas 2 dùng `object`, pandas 3 dùng `str` — nhận cả hai."""
    return [
        col for col in df.columns
        if df[col].dtype == object or pd.api.types.is_string_dtype(df[col])
    ]


def replace_sentinels(df: pd.DataFrame) -> pd.DataFrame:
    """Giá trị canh gác → `NaN`. Xem `quality.KNOWN_SENTINELS`."""
    out = df.copy()
    for col, value in KNOWN_SENTINELS.items():
        if col not in out.columns:
            continue
        hits = int((out[col] == value).sum())
        if hits:
            out.loc[out[col] == value, col] = np.nan
            log.info("%s: %d giá trị canh gác %s → NaN", col, hits, f"{value:,.0f}")
    return out


def replace_placeholders(df: pd.DataFrame) -> pd.DataFrame:
    """Chuỗi `XNA` / `XAP` / `Unknown` → `NaN`.

    Để nguyên thì encoder biến chúng thành một hạng mục thật, và model học
    "XNA" như thể đó là một loại tổ chức có thật.
    """
    out = df.copy()
    for col in _text_columns(out):
        mask = out[col].isin(PLACEHOLDER_VALUES)
        hits = int(mask.sum())
        if hits:
            out.loc[mask, col] = np.nan
            log.info("%s: %d chuỗi giả → NaN", col, hits)
    return out


def add_missing_flags(
    df: pd.DataFrame,
    flags: tuple[MissingFlag, ...] = MISSING_FLAGS,
) -> pd.DataFrame:
    """Thêm cờ nhị phân `1 = thiếu`, kiểu `int8`.

    Bỏ qua cờ nào không có cột nguồn — bộ feature rút gọn (PLAN.md §7.2)
    không có `EXT_SOURCE_*` nên chỉ nhận được các cờ tương ứng.
    """
    out = df.copy()
    for flag in flags:
        if flag.source not in out.columns:
            continue
        mask = out[flag.source].isna()
        # Gọi trước replace_sentinels thì sentinel vẫn là số, phải tính thêm.
        if flag.source in KNOWN_SENTINELS:
            mask = mask | (out[flag.source] == KNOWN_SENTINELS[flag.source])
        out[flag.name] = mask.astype("int8")
    return out


def normalize_missing(
    df: pd.DataFrame,
    add_flags: bool = True,
    flags: tuple[MissingFlag, ...] = MISSING_FLAGS,
) -> pd.DataFrame:
    """Chuẩn hóa toàn bộ dạng thiếu về `NaN`, kèm cờ.

    Thứ tự bắt buộc: sinh cờ TRƯỚC khi thay sentinel, vì sau khi thay thì
    `DAYS_EMPLOYED = 365243` đã thành `NaN` và không phân biệt được với
    missing thật sự.
    """
    out = add_missing_flags(df, flags) if add_flags else df
    out = replace_sentinels(out)
    return replace_placeholders(out)


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Bảng tỉ lệ thiếu còn lại sau khi chuẩn hóa, giảm dần."""
    rate = df.isna().mean().sort_values(ascending=False)
    return pd.DataFrame({
        "column": rate.index,
        "missing": df.isna().sum().reindex(rate.index).to_numpy(),
        "missing_rate": rate.to_numpy(),
    }).reset_index(drop=True)


class MissingNormalizer(BaseEstimator, TransformerMixin):
    """Bước Pipeline không trạng thái: sentinel/chuỗi giả → NaN, thêm cờ.

    `fit` không làm gì — đó là điểm mấu chốt. Vì không học gì từ dữ liệu,
    bước này chạy trước hay sau khi split đều cho kết quả hệt nhau, nên
    không thể gây rò rỉ.
    """

    def __init__(self, add_flags: bool = True,
                 flags: tuple[MissingFlag, ...] = MISSING_FLAGS):
        self.add_flags = add_flags
        self.flags = flags

    def fit(self, X: pd.DataFrame, y=None) -> "MissingNormalizer":
        self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return normalize_missing(X, add_flags=self.add_flags, flags=self.flags)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        names = list(input_features if input_features is not None
                     else getattr(self, "feature_names_in_", []))
        if self.add_flags:
            names += [f.name for f in self.flags if f.source in names]
        return np.asarray(names, dtype=object)


class HighMissingDropper(BaseEstimator, TransformerMixin):
    """Bỏ cột thiếu quá ngưỡng. CÓ học — bắt buộc nằm trong Pipeline.

    Danh sách cột bỏ được xác định lúc `fit`, tức chỉ từ tập train. Nếu tính
    ngưỡng trên toàn bộ dữ liệu rồi mới split thì tập test đã góp phần quyết
    định feature set — rò rỉ, và metric CV sẽ cao bất thường (PLAN.md §14).

    Cờ `_MISSING` không bao giờ bị bỏ: chúng không có giá trị thiếu, và
    chúng chính là thứ giữ lại thông tin của các cột bị bỏ.
    """

    def __init__(self, threshold: float = 0.60):
        self.threshold = threshold

    def fit(self, X: pd.DataFrame, y=None) -> "HighMissingDropper":
        rate = X.isna().mean()
        self.columns_to_drop_ = [
            col for col in X.columns
            if rate[col] > self.threshold and not col.endswith(FLAG_SUFFIX)
        ]
        self.feature_names_in_ = list(X.columns)
        log.info("Bỏ %d/%d cột thiếu trên %.0f%%",
                 len(self.columns_to_drop_), X.shape[1], self.threshold * 100)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop(columns=[c for c in self.columns_to_drop_ if c in X.columns])

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        names = list(input_features if input_features is not None
                     else getattr(self, "feature_names_in_", []))
        return np.asarray([n for n in names if n not in self.columns_to_drop_],
                          dtype=object)
