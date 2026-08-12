"""Feature selection (F01 task 13).

Hai loại chọn feature, khác nhau hoàn toàn về mức nguy hiểm
-----------------------------------------------------------
**Không giám sát** — chỉ nhìn X, không nhìn nhãn:

    NearZeroVarianceRemover   bỏ cột gần như một giá trị
    CorrelatedFeatureRemover  bỏ cột trùng thông tin với cột khác

    Vẫn phải `fit` trên train (ngưỡng tính từ dữ liệu), nhưng không thể rò rỉ
    nhãn. Đặt ở đầu Pipeline được.

**Có giám sát** — dùng nhãn để chấm điểm feature:

    SupervisedFeatureSelector   mutual information với TARGET

    ⚠️ ĐÂY LÀ CÁI BẪY. Chọn feature bằng nhãn TRÊN TOÀN BỘ tập train rồi mới
    chạy cross-validation thì phần validation của mỗi fold đã góp phần quyết
    định feature nào tồn tại → metric CV lạc quan giả. Bước này **bắt buộc
    nằm trong Pipeline** để mỗi fold `fit` lại từ đầu (PLAN.md §14, dòng
    "Rò rỉ do preprocessing").

    `test_supervised_selection_is_data_dependent` chứng minh vì sao: fit trên
    hai nửa dữ liệu khác nhau cho ra hai tập feature khác nhau. Cái gì phụ
    thuộc dữ liệu thì phải nằm trong fold.

Đo trên `application_train.csv` (110 cột số sau task 8)
-------------------------------------------------------
    Cột gần hằng số (một giá trị > 99%)      18
        FLAG_MOBIL 100% = 1 · FLAG_DOCUMENT_2/10/12 100% = 0
        9/20 cột FLAG_DOCUMENT_* có tỉ lệ bật < 0,1%
    Cặp |r| > 0,95                            46   (bỏ tham lam → mất 32 cột)
    Cặp |r| > 0,99                            16
        Chủ yếu là bộ ba AVG/MODE/MEDI của khối thông tin nhà ở.

Một cặp đáng chú ý: `FLAG_EMP_PHONE` ~ `DAYS_EMPLOYED_MISSING` có **r = 0,9999**.
Cờ sinh ở task 8 gần như trùng khít một cột có sẵn — hợp lý, vì người nghỉ hưu
không có điện thoại cơ quan. Đây vừa là bằng chứng cờ đó đúng ý nghĩa, vừa là
lý do phải khử trùng lặp.

Tính tái lập
------------
Mọi bước ở đây phải cho ra CÙNG một danh sách cột khi chạy lại — F06 task 6
yêu cầu metric trùng đến 4 chữ số thập phân. Vì thế thứ tự duyệt cặp tương
quan được sắp xếp tường minh, không dựa vào thứ tự ngẫu nhiên của `set`.
"""
from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, mutual_info_classif

from hfml.logger import get_logger

log = get_logger(__name__)

#: Một giá trị chiếm hơn ngưỡng này thì cột gần như là hằng số.
DEFAULT_NZV_THRESHOLD: Final[float] = 0.99
#: |r| trên ngưỡng này thì hai cột coi như trùng thông tin.
DEFAULT_CORRELATION_THRESHOLD: Final[float] = 0.95


class NearZeroVarianceRemover(BaseEstimator, TransformerMixin):
    """Bỏ cột mà một giá trị chiếm gần hết.

    Khác với "hằng số tuyệt đối": `FLAG_MOBIL` có 307.510/307.511 dòng bằng 1
    nên `nunique() == 2`, kiểm tra hằng số thông thường bỏ sót. Một cột như
    vậy không tách được hồ sơ nào, chỉ làm loãng feature importance.
    """

    def __init__(self, threshold: float = DEFAULT_NZV_THRESHOLD,
                 protect: tuple[str, ...] = ()):
        self.threshold = threshold
        self.protect = protect

    def fit(self, X: pd.DataFrame, y=None) -> "NearZeroVarianceRemover":
        if not 0 < self.threshold <= 1:
            raise ValueError(f"threshold phải trong (0, 1]: {self.threshold}")
        dropped = []
        for col in X.columns:
            if col in self.protect:
                continue
            counts = X[col].value_counts(dropna=False, normalize=True)
            if len(counts) and counts.iloc[0] > self.threshold:
                dropped.append(col)
        self.columns_to_drop_ = sorted(dropped)
        self.feature_names_in_ = list(X.columns)
        log.info("Gần hằng số: bỏ %d/%d cột", len(dropped), X.shape[1])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop(columns=[c for c in self.columns_to_drop_ if c in X.columns])

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        names = list(input_features if input_features is not None
                     else getattr(self, "feature_names_in_", []))
        return np.asarray([n for n in names if n not in self.columns_to_drop_],
                          dtype=object)


class CorrelatedFeatureRemover(BaseEstimator, TransformerMixin):
    """Trong mỗi cặp cột tương quan cao, giữ một và bỏ một.

    Quy tắc chọn giữ, theo thứ tự ưu tiên — cố định để tái lập được:
        1. Cột nằm trong `protect` luôn được giữ.
        2. Cột thiếu ít dữ liệu hơn.
        3. Cột xuất hiện trước trong khung dữ liệu.

    Ưu tiên 2 quan trọng: trong bộ ba `AVG`/`MODE`/`MEDI` của Home Credit, ba
    cột gần như trùng nhau nhưng tỉ lệ thiếu chênh nhau; giữ cột đầy đủ nhất
    thì phần impute phải bịa ít nhất.
    """

    def __init__(self, threshold: float = DEFAULT_CORRELATION_THRESHOLD,
                 protect: tuple[str, ...] = ()):
        self.threshold = threshold
        self.protect = protect

    def fit(self, X: pd.DataFrame, y=None) -> "CorrelatedFeatureRemover":
        if not 0 < self.threshold <= 1:
            raise ValueError(f"threshold phải trong (0, 1]: {self.threshold}")

        numeric = X.select_dtypes(include="number")
        order = {col: i for i, col in enumerate(numeric.columns)}
        missing = numeric.isna().mean()

        corr = numeric.corr(numeric_only=True).abs()
        # Chỉ lấy nửa trên để mỗi cặp xét đúng một lần.
        upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))

        pairs = upper.stack()
        pairs = pairs[pairs > self.threshold]
        # Sắp xếp tường minh: tương quan giảm dần, rồi theo tên — để chạy lại
        # ra đúng cùng kết quả (F06 task 6).
        pairs = pairs.sort_values(ascending=False, kind="mergesort")

        dropped: set[str] = set()
        self.dropped_because_: dict[str, str] = {}
        for (a, b), r in pairs.items():
            if a in dropped or b in dropped:
                continue
            keep, drop = self._choose(a, b, missing, order)
            if drop is None:
                continue
            dropped.add(drop)
            self.dropped_because_[drop] = f"|r|={r:.4f} với {keep}"

        self.columns_to_drop_ = sorted(dropped)
        self.feature_names_in_ = list(X.columns)
        log.info("Tương quan > %.2f: bỏ %d/%d cột",
                 self.threshold, len(dropped), X.shape[1])
        return self

    def _choose(self, a: str, b: str, missing: pd.Series,
                order: dict[str, int]) -> tuple[str, str | None]:
        """Trả về `(giữ, bỏ)`. `bỏ = None` khi cả hai đều được bảo vệ."""
        a_protected, b_protected = a in self.protect, b in self.protect
        if a_protected and b_protected:
            return a, None
        if a_protected:
            return a, b
        if b_protected:
            return b, a
        if missing[a] != missing[b]:
            return (a, b) if missing[a] < missing[b] else (b, a)
        return (a, b) if order[a] < order[b] else (b, a)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop(columns=[c for c in self.columns_to_drop_ if c in X.columns])

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        names = list(input_features if input_features is not None
                     else getattr(self, "feature_names_in_", []))
        return np.asarray([n for n in names if n not in self.columns_to_drop_],
                          dtype=object)


class SupervisedFeatureSelector(BaseEstimator, TransformerMixin):
    """Chọn `k` feature theo mutual information với nhãn.

    ⚠️ **Bắt buộc nằm trong Pipeline, không được chạy rời trước khi split.**
    Bước này nhìn nhãn, nên chọn feature trên toàn bộ tập train rồi mới chạy
    cross-validation là để phần validation của mỗi fold tham gia quyết định
    feature — metric CV sẽ đẹp hơn thực tế.

    Dùng mutual information chứ không dùng tương quan Pearson: quan hệ giữa
    feature và rủi ro vỡ nợ thường phi tuyến (ví dụ tuổi), và Pearson chỉ bắt
    được quan hệ tuyến tính. Trên `application_train.csv`, tương quan tuyến
    tính tuyệt đối cao nhất chỉ là 0,179 (`EXT_SOURCE_3`) — nếu tin vào
    Pearson thì sẽ kết luận nhầm là "chẳng feature nào có ích".
    """

    def __init__(self, k: int = 30, random_state: int = 42):
        self.k = k
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y=None) -> "SupervisedFeatureSelector":
        if y is None:
            raise ValueError(
                "SupervisedFeatureSelector cần nhãn `y`. Nếu không có nhãn thì "
                "dùng NearZeroVarianceRemover / CorrelatedFeatureRemover.")
        if X.isna().to_numpy().any():
            raise ValueError(
                "Còn NaN — đặt bước này SAU imputer trong Pipeline.")

        k = min(self.k, X.shape[1])

        def score(a, b):
            return mutual_info_classif(a, b, random_state=self.random_state)

        self.selector_ = SelectKBest(score_func=score, k=k).fit(X, y)
        mask = self.selector_.get_support()
        self.selected_ = [c for c, keep in zip(X.columns, mask) if keep]
        self.scores_ = pd.Series(self.selector_.scores_, index=X.columns
                                 ).sort_values(ascending=False)
        self.feature_names_in_ = list(X.columns)
        log.info("Mutual information: giữ %d/%d feature", k, X.shape[1])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[[c for c in self.selected_ if c in X.columns]]

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.asarray(list(self.selected_), dtype=object)


def selection_report(
    X: pd.DataFrame,
    nzv_threshold: float = DEFAULT_NZV_THRESHOLD,
    correlation_threshold: float = DEFAULT_CORRELATION_THRESHOLD,
    protect: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Bảng "bỏ cột nào, vì sao" — đưa thẳng vào báo cáo.

    Chỉ dùng các bước KHÔNG giám sát, nên gọi được trên tập train mà không
    lo rò rỉ nhãn.
    """
    nzv = NearZeroVarianceRemover(nzv_threshold, protect).fit(X)
    after_nzv = nzv.transform(X)
    corr = CorrelatedFeatureRemover(correlation_threshold, protect).fit(after_nzv)

    rows = [{"column": c, "reason": "near_zero_variance",
             "detail": f"một giá trị chiếm > {nzv_threshold:.0%}"}
            for c in nzv.columns_to_drop_]
    rows += [{"column": c, "reason": "correlated",
              "detail": corr.dropped_because_[c]}
             for c in corr.columns_to_drop_]

    report = pd.DataFrame(rows, columns=["column", "reason", "detail"])
    log.info("Chọn feature: %d → %d cột (bỏ %d)",
             X.shape[1], X.shape[1] - len(report), len(report))
    return report
