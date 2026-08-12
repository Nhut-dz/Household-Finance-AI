"""Đóng gói toàn bộ tiền xử lý thành MỘT Pipeline (F01 task 14).

Đây là mục PLAN.md §13 ghi "không được cắt", và lý do rất cụ thể: nếu các
bước impute/encode/scale chạy rời trước khi chia train/test thì

    1. thống kê của tập test đã tham gia vào phép biến đổi → **rò rỉ dữ liệu**,
       metric CV đẹp hơn thực tế mà không có dấu hiệu gì;
    2. lúc inference phải dựng lại y hệt chuỗi biến đổi đó bằng tay → sớm muộn
       cũng lệch, và lệch im lặng.

Gói tất cả vào một object rồi `joblib.dump` cùng model thì cả hai vấn đề biến
mất: `fit` một lần trên train, `transform` dùng lại đúng object đó mãi mãi.

Thứ tự các bước và lý do
------------------------
    1. MissingNormalizer        sentinel/chuỗi giả → NaN, sinh cờ `_MISSING`
                                (không trạng thái — chạy trước split cũng an toàn)
    2. HighMissingDropper       bỏ cột thiếu quá ngưỡng      (học từ train)
    3. OutlierClipper           kẹp biên phân vị             (học từ train)
    4. ColumnTransformer        số: impute trung vị (+ scale tùy chọn)
                                categorical: ordinal, NaN và giá trị lạ có mã riêng
                                cờ: giữ nguyên
    5. NearZeroVarianceRemover  bỏ cột gần hằng số           (không giám sát)
    6. CorrelatedFeatureRemover bỏ cột trùng thông tin        (không giám sát)
    7. SupervisedFeatureSelector  (tùy chọn) mutual information — CÓ nhìn nhãn

Vì sao bước 5–6 đặt SAU ColumnTransformer chứ không phải trước: trước khi
encode, cột categorical còn là chuỗi nên không tính được tương quan. Và vì
sao bước 7 nằm TRONG pipeline chứ không chạy rời: nó nhìn nhãn, chạy rời
trước cross-validation là rò rỉ (xem `features/selection.py`).

Bước 3 kẹp biên trước bước 4 điền thiếu — làm ngược lại thì giá trị vừa điền
(trung vị) sẽ tham gia tính phân vị, và ngoại lai đã kịp kéo lệch trung vị.

Chọn cột động
-------------
Bước 2 và 5–6 làm thay đổi tập cột, nên `ColumnTransformer` không thể biết
trước tên cột lúc dựng. Vì thế ba nhánh dùng **hàm chọn cột** thay cho danh
sách cứng — sklearn gọi chúng lúc `fit`. Ba hàm đó phải ở cấp module (không
phải lambda) để `joblib` pickle được.
"""
from __future__ import annotations

from typing import Final

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from hfml.data.features.selection import (
    DEFAULT_CORRELATION_THRESHOLD,
    DEFAULT_NZV_THRESHOLD,
    CorrelatedFeatureRemover,
    NearZeroVarianceRemover,
    SupervisedFeatureSelector,
)
from hfml.data.preprocessing.cleaner import HighMissingDropper, MissingNormalizer
from hfml.data.preprocessing.encoders import (
    EncodingStrategy,
    PassthroughFlags,
    ScalingKind,
    build_categorical_transformer,
    build_numeric_transformer,
    classify_columns,
)
from hfml.data.preprocessing.validator import OutlierClipper
from hfml.logger import get_logger

log = get_logger(__name__)

#: Cột không bao giờ được vào feature set.
DEFAULT_EXCLUDE: Final[tuple[str, ...]] = ("SK_ID_CURR", "TARGET")

#: Cờ nên truyền vào `protect=` khi train bộ FULL của ML02.
#:
#: Đo trên toàn bộ dữ liệu: bước khử tương quan bỏ hai cờ này vì Home Credit
#: đã có sẵn cột mang đúng thông tin đó —
#:     DAYS_EMPLOYED_MISSING  |r| = 0,9999 với FLAG_EMP_PHONE
#:     BUILDING_INFO_MISSING  |r| = 0,9783 với EMERGENCYSTATE_MODE
#: Nên bỏ chúng KHÔNG mất thông tin. Nhưng giữ lại thì bảng feature importance
#: và SHAP đọc được bằng tiếng Việt ("không có dữ liệu việc làm") thay vì một
#: cái tên không giải thích nổi ("FLAG_EMP_PHONE") — mà SHAP top-5 chính là
#: thứ đưa sang tầng `llm` để diễn đạt (PLAN.md §7.4).
#:
#: Với bộ RÚT GỌN thì không cần `protect`: hai cột gốc kia không lấy được từ
#: form nên không có gì để trùng.
INTERPRETABLE_FLAGS: Final[tuple[str, ...]] = (
    "DAYS_EMPLOYED_MISSING",
    "BUILDING_INFO_MISSING",
)


# --- Hàm chọn cột: phải ở cấp module để joblib pickle được ------------------
def select_numeric(X: pd.DataFrame) -> list[str]:
    return classify_columns(X, exclude=DEFAULT_EXCLUDE).numeric


def select_categorical(X: pd.DataFrame) -> list[str]:
    return classify_columns(X, exclude=DEFAULT_EXCLUDE).categorical


def select_flags(X: pd.DataFrame) -> list[str]:
    return classify_columns(X, exclude=DEFAULT_EXCLUDE).flags


def build_preprocessing_pipeline(
    *,
    encoding: EncodingStrategy = "ordinal",
    scaling: ScalingKind = "none",
    impute_strategy: str = "median",
    high_missing_threshold: float = 0.60,
    clip_quantiles: tuple[float, float] = (0.001, 0.999),
    nzv_threshold: float = DEFAULT_NZV_THRESHOLD,
    correlation_threshold: float | None = DEFAULT_CORRELATION_THRESHOLD,
    select_k: int | None = None,
    protect: tuple[str, ...] = (),
    random_state: int = 42,
) -> Pipeline:
    """Dựng Pipeline tiền xử lý đầy đủ.

    Mặc định khớp với lựa chọn đã biện minh ở task 10–13: ordinal encoding và
    KHÔNG scaling, vì bốn thuật toán của đồ án đều là cây.

    `correlation_threshold=None` bỏ bước khử tương quan — dùng cho bộ feature
    rút gọn của ML02 (7 cột đã chọn tay, không cần lọc thêm).

    `select_k=None` bỏ bước chọn feature có giám sát. Chỉ bật khi thật sự cần,
    và khi bật thì nó vẫn nằm TRONG pipeline nên mỗi CV fold fit lại.

    `protect` giữ lại các cột dù chúng gần hằng số hay trùng tương quan — ví
    dụ khi muốn giữ `DAYS_EMPLOYED_MISSING` thay vì `FLAG_EMP_PHONE`.
    """
    lower, upper = clip_quantiles

    column_transformer = ColumnTransformer(
        transformers=[
            ("num", build_numeric_transformer(impute_strategy, scaling), select_numeric),
            ("cat", build_categorical_transformer(encoding), select_categorical),
            ("flag", PassthroughFlags(), select_flags),
        ],
        remainder="drop",           # ID và nhãn bị loại tường minh
        verbose_feature_names_out=False,   # giữ tên gốc, không thêm tiền tố "num__"
    )

    steps: list[tuple[str, object]] = [
        ("missing", MissingNormalizer()),
        ("drop_high_missing", HighMissingDropper(threshold=high_missing_threshold)),
        ("clip", OutlierClipper(lower_quantile=lower, upper_quantile=upper)),
        ("encode", column_transformer),
        ("nzv", NearZeroVarianceRemover(threshold=nzv_threshold, protect=protect)),
    ]
    if correlation_threshold is not None:
        steps.append(("decorrelate", CorrelatedFeatureRemover(
            threshold=correlation_threshold, protect=protect)))
    if select_k is not None:
        steps.append(("select", SupervisedFeatureSelector(
            k=select_k, random_state=random_state)))

    pipeline = Pipeline(steps)
    # Buộc mọi bước trả về DataFrame — các bước tự viết cần tên cột, và tên
    # cột là thứ `metadata.json` lưu để F06 task 3 đối chiếu.
    pipeline.set_output(transform="pandas")
    log.info("Dựng Pipeline tiền xử lý: %s", " → ".join(name for name, _ in steps))
    return pipeline


def feature_names(pipeline: Pipeline) -> list[str]:
    """Tên feature đầu ra, ĐÚNG THỨ TỰ. Ghi vào `metadata.json`.

    Thứ tự cột sai là lỗi im lặng: model vẫn chạy, vẫn trả xác suất, chỉ có
    điều xác suất đó vô nghĩa. F06 task 3 đối chiếu danh sách này lúc
    inference (xem `hfml.ml.registry`).
    """
    return list(pipeline.get_feature_names_out())


def fit_preprocessing(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series | None = None,
) -> tuple[Pipeline, pd.DataFrame]:
    """`fit` **chỉ trên tập train**, trả về pipeline đã fit và train đã biến đổi.

    Hàm này tồn tại để chỗ gọi phải viết ra chữ `X_train` — nếu ai đó truyền
    vào toàn bộ dữ liệu thì lỗi nằm ngay trên mặt câu lệnh, không ẩn trong
    một chuỗi biến đổi rời rạc.
    """
    transformed = pipeline.fit_transform(X_train, y_train)
    log.info("Đã fit trên %d dòng train → %d feature", len(X_train), transformed.shape[1])
    return pipeline, transformed
