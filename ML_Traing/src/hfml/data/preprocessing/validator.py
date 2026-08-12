"""Xử lý duplicate và dữ liệu bất hợp lệ (F01 task 9).

Tách khỏi `cleaner.py` vì hai việc khác bản chất: task 8 xử lý dữ liệu
KHÔNG CÓ, task 9 xử lý dữ liệu CÓ NHƯNG SAI.

Bất đối xứng giữa train và inference — điểm thiết kế chính
----------------------------------------------------------
Lúc train, gặp dòng hỏng thì bỏ đi: 307.511 hồ sơ, bỏ vài chục dòng không
ảnh hưởng gì.

Lúc inference thì KHÔNG ĐƯỢC BỎ. Người dùng vừa điền xong form và đang chờ
kết quả; trả về "hồ sơ của bạn bị loại" là hệ thống hỏng, không phải dữ liệu
hỏng. Nên inference chỉ được **kẹp giá trị về biên** và **gắn cờ cảnh báo**
để tầng `llm` nói ra.

    train      → drop_duplicates + drop_invalid_rows + OutlierClipper
    inference  → CHỈ OutlierClipper (biên đã học từ train) + cờ cảnh báo

Vì thế `clean_for_training()` không có bản song sinh `clean_for_inference()`
làm cùng việc — cố tình vậy, để không ai vô tình gọi nhầm hàm bỏ dòng vào
đường inference.

Rò rỉ dữ liệu
-------------
`OutlierClipper` học phân vị lúc `fit`, nên bắt buộc `fit` chỉ trên tập
train và nằm trong Pipeline (PLAN.md §4.4). Tính phân vị trên toàn bộ dữ
liệu rồi mới split là tập test đã góp phần định nghĩa biên — rò rỉ.

Kiểm chứng trên Home Credit (307.511 hồ sơ) — dataset SẠCH
----------------------------------------------------------
    dòng trùng hoàn toàn            0
    `SK_ID_CURR` trùng              0
    cột trùng nội dung              0
    đi làm trước khi sinh           0
    số con ≥ số nhân khẩu           0

Phần bất hợp lệ còn lại rất nhỏ và đều là ngoại lai chứ không phải sai
logic: `AMT_INCOME_TOTAL` cao nhất 117.000.000 (**247× phân vị 99**),
`CNT_CHILDREN` tới 19, `OBS_30_CNT_SOCIAL_CIRCLE` tới 348 (35× p99).
Những giá trị này không nên bỏ dòng — kẹp về biên là đủ và giữ được hồ sơ.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from hfml.data.preprocessing.cleaner import FLAG_SUFFIX
from hfml.logger import get_logger

log = get_logger(__name__)


# --------------------------------------------------------------------------
# Duplicate
# --------------------------------------------------------------------------
def duplicate_id_mask(df: pd.DataFrame, id_column: str = "SK_ID_CURR") -> pd.Series:
    """`True` ở các lần xuất hiện SAU của một ID — giữ lần đầu."""
    if id_column not in df.columns:
        return pd.Series(False, index=df.index)
    return df[id_column].duplicated(keep="first")


def duplicate_row_mask(df: pd.DataFrame, id_column: str = "SK_ID_CURR") -> pd.Series:
    """Dòng giống hệt nhau ở MỌI cột trừ ID.

    ⚠️ Chỉ có nghĩa khi `df` là bản ghi ĐẦY ĐỦ. Trên một bộ feature rút gọn,
    hai khách hàng hoàn toàn khác nhau vẫn dễ trùng nhau ở vài cột — đo trên
    Home Credit: đủ 122 cột thì 0 dòng trùng, nhưng chỉ lấy 7 cột thì đã có
    dòng "trùng". Gọi hàm này trên bộ rút gọn rồi bỏ dòng là xóa dữ liệu
    huấn luyện hợp lệ. Vì vậy `drop_duplicates` mặc định KHÔNG dùng nó.
    """
    subset = df.drop(columns=[id_column], errors="ignore")
    return subset.duplicated(keep="first")


def drop_duplicates(
    df: pd.DataFrame,
    id_column: str = "SK_ID_CURR",
    full_row: bool = False,
) -> pd.DataFrame:
    """Bỏ dòng trùng. CHỈ dùng khi train.

    Mặc định chỉ khử trùng theo `id_column` — luôn an toàn, vì cùng một ID
    thì đúng là cùng một hồ sơ.

    `full_row=True` khử thêm dòng trùng toàn bộ đặc trưng. Chỉ bật khi `df`
    là bản ghi đầy đủ; xem cảnh báo ở `duplicate_row_mask`.
    """
    mask = duplicate_id_mask(df, id_column)
    if full_row:
        mask = mask | duplicate_row_mask(df, id_column)
    if mask.any():
        log.warning("Bỏ %d dòng trùng lặp (full_row=%s)", int(mask.sum()), full_row)
    return df.loc[~mask].reset_index(drop=True)


# --------------------------------------------------------------------------
# Quy tắc hợp lệ
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ValidationRule:
    """Một quy tắc. `mask` trả về `True` ở những dòng VI PHẠM."""

    code: str
    description: str
    mask: Callable[[pd.DataFrame], pd.Series]
    #: `True` → dòng vi phạm bị bỏ khi train. `False` → chỉ ghi nhận.
    drop_on_train: bool = True


def _safe(fn: Callable[[pd.DataFrame], pd.Series]) -> Callable[[pd.DataFrame], pd.Series]:
    """Quy tắc thiếu cột thì coi như không vi phạm, không làm sập cả pipeline.

    Bộ feature rút gọn (PLAN.md §7.2) không có mọi cột, và quy tắc không áp
    dụng được thì khác với quy tắc bị vi phạm.
    """
    def wrapped(df: pd.DataFrame) -> pd.Series:
        try:
            result = fn(df)
        except KeyError:
            return pd.Series(False, index=df.index)
        return result.fillna(False).astype(bool)
    return wrapped


def _prefixed(df: pd.DataFrame, prefix: str) -> list[str]:
    """Cột theo tiền tố, LOẠI TRỪ cờ `_MISSING` do task 8 sinh ra.

    `DAYS_EMPLOYED_MISSING` cũng bắt đầu bằng `DAYS_` nhưng là cờ nhị phân,
    giá trị 1 hoàn toàn hợp lệ. Không loại ra thì quy tắc "DAYS_* phải ≤ 0"
    sẽ bắt nhầm đúng 55.374 dòng — bằng số cờ đang bật.
    """
    return [
        c for c in df.columns
        if c.startswith(prefix) and not c.endswith(FLAG_SUFFIX)
    ]


def _days_positive(df: pd.DataFrame) -> pd.Series:
    """`DAYS_*` là số ngày TRƯỚC ngày nộp đơn nên luôn ≤ 0."""
    cols = _prefixed(df, "DAYS_")
    if not cols:
        return pd.Series(False, index=df.index)
    return (df[cols] > 0).any(axis=1)


def _negative_amount(df: pd.DataFrame) -> pd.Series:
    cols = _prefixed(df, "AMT_")
    if not cols:
        return pd.Series(False, index=df.index)
    return (df[cols] < 0).any(axis=1)


def _negative_count(df: pd.DataFrame) -> pd.Series:
    cols = _prefixed(df, "CNT_")
    if not cols:
        return pd.Series(False, index=df.index)
    return (df[cols] < 0).any(axis=1)


#: Quy tắc áp cho `application_train.csv`. Thứ tự cố định để báo cáo ổn định.
INVALID_RULES: Final[tuple[ValidationRule, ...]] = (
    ValidationRule(
        "nonpositive_income",
        "AMT_INCOME_TOTAL ≤ 0 — không tính được mọi tỉ lệ có thu nhập ở mẫu số",
        _safe(lambda df: df["AMT_INCOME_TOTAL"] <= 0),
    ),
    ValidationRule(
        "negative_amount",
        "Cột AMT_* có giá trị âm",
        _safe(_negative_amount),
    ),
    ValidationRule(
        "days_positive",
        "Cột DAYS_* dương — phải là số ngày trước khi nộp đơn (đã trừ sentinel)",
        _safe(_days_positive),
    ),
    ValidationRule(
        "employed_before_birth",
        "DAYS_EMPLOYED < DAYS_BIRTH — đi làm trước khi sinh ra",
        _safe(lambda df: df["DAYS_EMPLOYED"] < df["DAYS_BIRTH"]),
    ),
    ValidationRule(
        "children_exceed_family",
        "CNT_CHILDREN ≥ CNT_FAM_MEMBERS — số con không thể bằng cả hộ",
        _safe(lambda df: df["CNT_CHILDREN"] >= df["CNT_FAM_MEMBERS"]),
    ),
    ValidationRule(
        "negative_count",
        "Cột CNT_* âm — số con và số nhân khẩu không thể âm",
        _safe(_negative_count),
    ),
)


def validation_report(
    df: pd.DataFrame,
    rules: tuple[ValidationRule, ...] = INVALID_RULES,
) -> pd.DataFrame:
    """Đếm số dòng vi phạm từng quy tắc. Không sửa gì."""
    rows = []
    for rule in rules:
        hits = int(rule.mask(df).sum())
        rows.append({
            "code": rule.code,
            "n_violations": hits,
            "rate": hits / len(df) if len(df) else 0.0,
            "drop_on_train": rule.drop_on_train,
            "description": rule.description,
        })
    return pd.DataFrame(rows)


def invalid_mask(
    df: pd.DataFrame,
    rules: tuple[ValidationRule, ...] = INVALID_RULES,
) -> pd.Series:
    """`True` ở các dòng vi phạm ít nhất một quy tắc có `drop_on_train`."""
    mask = pd.Series(False, index=df.index)
    for rule in rules:
        if rule.drop_on_train:
            mask |= rule.mask(df)
    return mask


def drop_invalid_rows(
    df: pd.DataFrame,
    rules: tuple[ValidationRule, ...] = INVALID_RULES,
) -> pd.DataFrame:
    """Bỏ dòng vi phạm. CHỈ dùng khi train — xem docstring module."""
    mask = invalid_mask(df, rules)
    if mask.any():
        report = validation_report(df, rules)
        for _, row in report[report["n_violations"] > 0].iterrows():
            log.warning("%s: %d dòng vi phạm", row["code"], row["n_violations"])
        log.warning("Bỏ tổng cộng %d/%d dòng bất hợp lệ", int(mask.sum()), len(df))
    return df.loc[~mask].reset_index(drop=True)


# --------------------------------------------------------------------------
# Ngoại lai
# --------------------------------------------------------------------------
class OutlierClipper(BaseEstimator, TransformerMixin):
    """Kẹp giá trị số về biên phân vị học từ tập train (winsorize).

    Vì sao kẹp chứ không bỏ dòng: `AMT_INCOME_TOTAL` cao nhất của Home Credit
    là 117.000.000, gấp **247 lần phân vị 99**. Một giá trị như vậy kéo lệch
    scaler và làm cây chẻ nhánh vô nghĩa, nhưng bản thân hồ sơ vẫn có ích.

    Và quan trọng hơn: đây là bước DÙNG CHUNG cho cả train lẫn inference.
    Biên học một lần trên train, rồi áp y hệt lúc dự đoán — nhờ vậy một người
    dùng khai thu nhập 900 triệu/tháng không đẩy model ra ngoài phân phối
    huấn luyện mà chỉ bị kẹp về biên, kèm cờ cảnh báo ở tầng `pipeline`.

    Không đụng tới: cột cờ `_MISSING`, cột ID, cột nhãn — kẹp chúng là vô
    nghĩa hoặc làm hỏng nhãn.
    """

    def __init__(
        self,
        lower_quantile: float = 0.001,
        upper_quantile: float = 0.999,
        exclude: tuple[str, ...] = ("SK_ID_CURR", "TARGET"),
    ):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.exclude = exclude

    def _clippable(self, X: pd.DataFrame) -> list[str]:
        return [
            col for col in X.select_dtypes(include="number").columns
            if col not in self.exclude and not col.endswith(FLAG_SUFFIX)
        ]

    def fit(self, X: pd.DataFrame, y=None) -> "OutlierClipper":
        if not 0 <= self.lower_quantile < self.upper_quantile <= 1:
            raise ValueError(
                f"Phân vị không hợp lệ: [{self.lower_quantile}, {self.upper_quantile}]")
        cols = self._clippable(X)
        self.bounds_ = {
            col: (float(X[col].quantile(self.lower_quantile)),
                  float(X[col].quantile(self.upper_quantile)))
            for col in cols
        }
        self.feature_names_in_ = list(X.columns)
        log.info("Học biên kẹp cho %d cột số (phân vị %.3f–%.3f)",
                 len(cols), self.lower_quantile, self.upper_quantile)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for col, (low, high) in self.bounds_.items():
            if col in out.columns:
                out[col] = out[col].clip(low, high)
        return out

    def clipped_mask(self, X: pd.DataFrame) -> pd.Series:
        """`True` ở dòng có ít nhất một giá trị bị kẹp.

        Dùng ở tầng `pipeline` để gắn cờ cảnh báo cho hồ sơ người dùng, thay
        vì im lặng sửa số của họ.
        """
        mask = pd.Series(False, index=X.index)
        for col, (low, high) in self.bounds_.items():
            if col in X.columns:
                mask |= (X[col] < low) | (X[col] > high)
        return mask.fillna(False)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        names = input_features if input_features is not None \
            else getattr(self, "feature_names_in_", [])
        return np.asarray(list(names), dtype=object)


# --------------------------------------------------------------------------
# Điểm vào
# --------------------------------------------------------------------------
def clean_for_training(
    df: pd.DataFrame,
    id_column: str = "SK_ID_CURR",
    rules: tuple[ValidationRule, ...] = INVALID_RULES,
    full_row_duplicates: bool = False,
) -> pd.DataFrame:
    """Bỏ trùng lặp + bỏ dòng bất hợp lệ. **Chỉ dùng cho tập train.**

    `full_row_duplicates` mặc định tắt: bật lên trên một bộ feature rút gọn
    sẽ xóa nhầm khách hàng hợp lệ (xem `duplicate_row_mask`).

    Cố ý KHÔNG có hàm tương ứng cho inference: lúc inference không được bỏ
    dòng nào, chỉ kẹp biên bằng `OutlierClipper` và gắn cờ.
    """
    before = len(df)
    out = drop_invalid_rows(
        drop_duplicates(df, id_column, full_row=full_row_duplicates), rules)
    removed = before - len(out)
    log.info("Làm sạch train: %d → %d dòng (bỏ %d, %.3f%%)",
             before, len(out), removed, removed / before * 100 if before else 0)
    return out
