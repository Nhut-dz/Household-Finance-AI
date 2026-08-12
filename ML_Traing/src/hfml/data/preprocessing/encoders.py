"""Encoding categorical và scaling numerical (F01 task 10, 11).

Cả hai lựa chọn ở file này đều bị chi phối bởi một sự thật: **bốn thuật toán
của đồ án đều là cây** — Decision Tree, Bagging, Random Forest, XGBoost
(PLAN.md §6.3, §7.4). Cây có hai tính chất quyết định:

    1. Bất biến với mọi phép biến đổi đơn điệu → KHÔNG cần scaling.
    2. Chẻ nhánh theo ngưỡng trên từng cột → ordinal encoding dùng được,
       không bắt buộc one-hot.

Task 10 — vì sao ordinal chứ không one-hot
------------------------------------------
Đo trên `application_train.csv`: 16 cột categorical, one-hot làm số cột nhảy
từ **128 lên 249**. Riêng `ORGANIZATION_TYPE` có 57 hạng mục → 57 cột nhị
phân cực kỳ thưa. Với cây, điều đó làm loãng độ quan trọng của biến gốc
(một biến bị xé thành 57 mảnh) và đẩy cây sâu ra vô ích.

One-hot vẫn giữ được, dùng `strategy="onehot"`, cho trường hợp muốn so với
một baseline tuyến tính. Khi đó chỉ one-hot các cột cardinality thấp.

Task 11 — vì sao mặc định KHÔNG scale
-------------------------------------
Dải giá trị giữa các cột lệch nhau tới 10⁹ lần (`AMT_INCOME_TOTAL` tới
117.000.000 so với `REGION_POPULATION_RELATIVE` 0,0003–0,07). Với hồi quy
tuyến tính hay SVM thì đó là tai họa; với cây thì hoàn toàn vô hại — ngưỡng
chẻ nhánh `x ≤ 147.150` và `x_scaled ≤ 0,31` cho ra ĐÚNG một phép phân
hoạch. `test_scaling_does_not_change_tree_predictions` chứng minh điều này
bằng thực nghiệm, và đó là câu trả lời cho *"sao không chuẩn hóa dữ liệu?"*.

Scaling vẫn được cài đầy đủ (`standard` / `minmax` / `robust`) để bật khi
cần baseline tuyến tính. Nếu bật, nên dùng `robust`: dữ liệu có ngoại lai
247× phân vị 99, `StandardScaler` sẽ bị chính ngoại lai đó kéo lệch.

Missing value của categorical
-----------------------------
KHÔNG điền bằng mode. Điền mode là gán cho một người nghề "Laborers" mà họ
không làm. Thay vào đó mã hóa "thiếu" thành một mức riêng (`MISSING_CODE`)
để cây tự học — hợp lý vì task 8 đã đo được rằng chính việc thiếu dữ liệu
mới là tín hiệu dự báo (`OCCUPATION_TYPE` thiếu → vỡ nợ 6,51% so với 8,79%).

Giá trị lạ lúc inference
------------------------
Người dùng có thể gửi lên một nghề nghiệp chưa từng xuất hiện trong tập
train. Encoder phải trả về `UNKNOWN_CODE` chứ không được ném lỗi — F06 task 1
yêu cầu rõ "encode giá trị lạ không crash". Mã của "lạ" và "thiếu" khác nhau
để cây phân biệt được hai tình huống.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Literal

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)

from hfml.data.preprocessing.cleaner import FLAG_SUFFIX
from hfml.logger import get_logger

log = get_logger(__name__)

#: Hạng mục chưa từng thấy lúc train (người dùng gửi giá trị lạ).
UNKNOWN_CODE: Final[int] = -1
#: Hạng mục thiếu (NaN). Khác `UNKNOWN_CODE` để cây phân biệt được.
MISSING_CODE: Final[int] = -2

#: Trên ngưỡng này thì one-hot làm nổ chiều — `ORGANIZATION_TYPE` có 57.
MAX_ONEHOT_CARDINALITY: Final[int] = 10

EncodingStrategy = Literal["ordinal", "onehot"]
ScalingKind = Literal["none", "standard", "minmax", "robust"]


@dataclass
class ColumnGroups:
    """Phân loại cột để `ColumnTransformer` biết áp bước nào lên đâu."""

    numeric: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)
    #: Cờ `_MISSING` — đã là 0/1, không encode, không scale.
    flags: list[str] = field(default_factory=list)
    #: ID và nhãn — không được đưa vào feature set.
    excluded: list[str] = field(default_factory=list)

    @property
    def features(self) -> list[str]:
        return self.numeric + self.categorical + self.flags

    def summary(self) -> str:
        return (f"{len(self.numeric)} số · {len(self.categorical)} categorical · "
                f"{len(self.flags)} cờ · {len(self.excluded)} loại trừ")


def classify_columns(
    df: pd.DataFrame,
    exclude: tuple[str, ...] = ("SK_ID_CURR", "TARGET"),
) -> ColumnGroups:
    """Chia cột thành 4 nhóm theo kiểu dữ liệu và vai trò.

    Cờ `_MISSING` tách riêng: chúng đã là 0/1 nên encode lại là vô nghĩa, và
    scale chúng làm mất ý nghĩa nhị phân.
    """
    groups = ColumnGroups()
    for col in df.columns:
        if col in exclude:
            groups.excluded.append(col)
        elif col.endswith(FLAG_SUFFIX):
            groups.flags.append(col)
        elif df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
            groups.categorical.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            groups.numeric.append(col)
        else:
            groups.excluded.append(col)
    log.info("Phân loại cột: %s", groups.summary())
    return groups


def build_categorical_encoder(
    strategy: EncodingStrategy = "ordinal",
    max_onehot_cardinality: int = MAX_ONEHOT_CARDINALITY,
) -> BaseEstimator:
    """Encoder cho cột categorical (task 10).

    `ordinal` (mặc định, hợp với cây):
        Hạng mục → số nguyên. Thiếu → `MISSING_CODE`, lạ → `UNKNOWN_CODE`.
        Không nở chiều, không cần imputer riêng.

    `onehot`:
        Cho baseline tuyến tính. Gộp hạng mục hiếm để không nở chiều:
        `min_frequency` cắt đuôi dài của `ORGANIZATION_TYPE`.
    """
    if strategy == "ordinal":
        return OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=UNKNOWN_CODE,
            encoded_missing_value=MISSING_CODE,
        )
    if strategy == "onehot":
        return OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=0.01,
            max_categories=max_onehot_cardinality,
            sparse_output=False,
        )
    raise ValueError(f"strategy không hợp lệ: {strategy!r}")


def build_scaler(kind: ScalingKind = "none") -> BaseEstimator | None:
    """Scaler cho cột số (task 11). `None` = không scale.

    Mặc định `none` vì bốn thuật toán đều là cây — xem docstring module.
    Nếu bật, ưu tiên `robust`: dữ liệu có ngoại lai 247× phân vị 99 và
    `StandardScaler` sẽ bị chính ngoại lai đó kéo lệch trung bình/độ lệch.
    """
    if kind == "none":
        return None
    if kind == "standard":
        return StandardScaler()
    if kind == "minmax":
        return MinMaxScaler()
    if kind == "robust":
        return RobustScaler()
    raise ValueError(f"kind không hợp lệ: {kind!r}")


def build_numeric_transformer(
    impute_strategy: str = "median",
    scaling: ScalingKind = "none",
) -> Pipeline:
    """Nhánh xử lý cột số: điền thiếu → (tùy chọn) scale.

    Điền bằng **trung vị** chứ không phải trung bình: cùng lý do với
    `RobustScaler` — một hồ sơ thu nhập 117.000.000 kéo trung bình đi rất xa
    nhưng không ảnh hưởng trung vị.

    `SimpleImputer` HỌC từ dữ liệu (trung vị của tập train), nên bước này bắt
    buộc nằm trong Pipeline và `fit` chỉ trên train (PLAN.md §4.4). Đây chính
    là mắt xích cuối của task 8 mà `cleaner.py` cố tình không làm.
    """
    steps: list[tuple[str, BaseEstimator]] = [
        ("impute", SimpleImputer(strategy=impute_strategy)),
    ]
    scaler = build_scaler(scaling)
    if scaler is not None:
        steps.append(("scale", scaler))
    return Pipeline(steps)


def build_categorical_transformer(
    strategy: EncodingStrategy = "ordinal",
    max_onehot_cardinality: int = MAX_ONEHOT_CARDINALITY,
) -> Pipeline:
    """Nhánh xử lý cột categorical.

    Với `ordinal` KHÔNG có bước impute: `encoded_missing_value` đã biến NaN
    thành một mức riêng, và đó là cách đúng — điền mode là gán cho người ta
    một nghề họ không làm.

    Với `onehot` thì phải điền trước, vì `OneHotEncoder` không nhận NaN.
    Điền bằng hằng `"__MISSING__"` để "thiếu" vẫn là một cột riêng, không
    hòa lẫn vào hạng mục phổ biến nhất.
    """
    encoder = build_categorical_encoder(strategy, max_onehot_cardinality)
    if strategy == "ordinal":
        return Pipeline([("encode", encoder)])
    return Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
        ("encode", encoder),
    ])


class PassthroughFlags(BaseEstimator, TransformerMixin):
    """Giữ nguyên cờ `_MISSING`. Không học gì, không biến đổi gì.

    Tồn tại để `ColumnTransformer` có một nhánh tường minh cho cờ, thay vì
    dựa vào `remainder="passthrough"` — nhánh ngầm định khiến thứ tự cột khó
    truy vết, mà thứ tự cột sai là lỗi im lặng (xem `ml/registry.py`).
    """

    def fit(self, X: pd.DataFrame, y=None) -> "PassthroughFlags":
        self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X

    def get_feature_names_out(self, input_features=None):
        import numpy as np

        names = input_features if input_features is not None \
            else getattr(self, "feature_names_in_", [])
        return np.asarray(list(names), dtype=object)
