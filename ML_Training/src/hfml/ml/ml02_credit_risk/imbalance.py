"""ML02 task 4 — Xử lý mất cân bằng lớp (F04 · M04 · Tuần 4).

Nối tiếp task 3. Task này KHÔNG train model — nó chốt **cách** bốn thuật toán
sẽ đối xử với việc chỉ 8,07% hồ sơ là dương, và chốt sao cho cách đó không tạo
ra rò rỉ dữ liệu.

Con số gốc (đo trên 307.511 hồ sơ đã làm sạch)
----------------------------------------------
    dương              24.825 / 307.511  =  8,0729%
    scale_pos_weight   282.686 / 24.825  =  11,3872
    accuracy nếu đoán TOÀN BỘ là 0       =  91,9271%

Dòng cuối là lý do **không dùng accuracy để chọn model** (PLAN.md §7.3): một
model không học gì đã hơn 91%. Chỉ số chọn model là **PR-AUC**.

Phương án đã chọn: HỌC CÓ TRỌNG SỐ, không lấy mẫu lại
------------------------------------------------------
Mọi thuật toán nhận cùng một tỉ số phạt **11,3872** — sai một hồ sơ dương bị
phạt gấp 11,39 lần sai một hồ sơ âm. Không sinh thêm dòng, không bỏ bớt dòng.

Vì sao KHÔNG dùng SMOTE / oversample / undersample — bốn lý do, xếp theo mức
quan trọng với chính đồ án này:

**1. Lấy mẫu lại phá hỏng hiệu chuẩn xác suất, mà hiệu chuẩn là yêu cầu bắt
buộc của ML02.** Cân bằng dữ liệu về 50/50 nghĩa là model học trên một quần
thể có tỉ lệ vỡ nợ 50%, nên xác suất nó trả về không còn là ước lượng của
P(vỡ nợ) trong quần thể thật — nó bị đẩy lên cao một cách hệ thống. Mà PLAN.md
§7.4 yêu cầu `CalibratedClassifierCV` + calibration curve + Brier score, và
§8.1 yêu cầu ra quyết định theo NGƯỠNG xác suất. Xác suất chưa hiệu chuẩn thì
ngưỡng vô nghĩa. Học có trọng số giữ nguyên tỉ lệ nền nên không gặp vấn đề này.

**2. Lấy mẫu lại là một cửa rò rỉ rất dễ mở nhầm.** Nó phải nằm TRONG fold,
chỉ áp lên phần train. Chạy nó trước khi chia tập thì SMOTE nội suy giữa các
dòng mà sau đó có dòng rơi vào validation — tức tập validation góp phần tạo ra
chính dữ liệu huấn luyện. Chỉ số thu được sẽ đẹp mà không có dấu hiệu gì.
Trọng số không mở cửa đó: nó chỉ là một con số nhân vào hàm mất mát.

**3. SMOTE vô nghĩa trên dữ liệu của bài toán này.** Task 3 mã hoá categorical
bằng **ordinal**, nên `ORGANIZATION_TYPE` mã 37 và mã 38 là hai tổ chức chẳng
liên quan gì nhau. SMOTE nội suy theo khoảng cách Euclid sẽ sinh ra "mã 37,5"
— một tổ chức không tồn tại. Với 16 cột categorical trong bộ FULL thì đó không
phải chi tiết nhỏ.

**4. Bốn thuật toán phải được so công bằng.** Trọng số cho cả bốn cùng một tỉ
số trên cùng một tập dữ liệu; lấy mẫu lại thì mỗi model nhìn thấy một tập khác
nhau và bảng so sánh mất ý nghĩa.

Đã kiểm bằng số, không phải suy luận: `class_weight="balanced"` của sklearn cho
trọng số lớp 0 = 0,543909 và lớp 1 = 6,193575, **tỉ số 11,387150** — trùng khít
`scale_pos_weight` của XGBoost tới sáu chữ số. Nghĩa là bốn thuật toán thật sự
nhận cùng một mức phạt dù dùng hai cơ chế khác tên.

Điểm rò rỉ phải canh
--------------------
Trọng số suy từ tỉ lệ dương, mà tỉ lệ dương là một **thống kê của dữ liệu**.
Tính nó trên toàn bộ dataset rồi mới chia tập là để tập test góp phần định
hình hàm mất mát của quá trình huấn luyện. Con số 11,3872 ghi ở đầu file này
là **số tham chiếu để báo cáo**, không phải số đem đi train:
`scale_pos_weight_from(y_train)` bắt chỗ gọi phải viết ra chữ `y_train`.

Với `class_weight="balanced"` thì sklearn tự tính từ đúng `y` được truyền vào
`fit()` — nên miễn là `fit` chỉ nhận train, không có đường nào rò rỉ.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Final

import numpy as np
import pandas as pd

from hfml.logger import get_logger

log = get_logger(__name__)

# --------------------------------------------------------------------------
# Đo mức mất cân bằng
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ImbalanceReport:
    """Mức mất cân bằng của một tập nhãn."""

    n_rows: int
    n_positive: int
    n_negative: int
    positive_rate: float
    #: Tỉ số phạt: sai một hồ sơ dương bị phạt gấp ngần này lần.
    scale_pos_weight: float
    #: Accuracy của model đoán toàn lớp đa số — mốc cho thấy accuracy vô dụng.
    majority_class_accuracy: float

    def to_dict(self) -> dict:
        return asdict(self)


def measure_imbalance(y: pd.Series | np.ndarray) -> ImbalanceReport:
    """Đo mức mất cân bằng của một tập nhãn nhị phân.

    Ném lỗi khi nhãn chỉ có một lớp: `scale_pos_weight` khi đó là chia cho 0,
    và trả về `inf` sẽ để một tập dữ liệu hỏng chảy tiếp xuống bước train mà
    không ai biết.
    """
    labels = pd.Series(y).astype(int)
    n = len(labels)
    if n == 0:
        raise ValueError("Tập nhãn rỗng — không đo được mức mất cân bằng.")

    positive = int(labels.sum())
    negative = n - positive
    if positive == 0 or negative == 0:
        raise ValueError(
            f"Nhãn chỉ có một lớp ({positive} dương / {negative} âm) — "
            "không tính được scale_pos_weight.")

    return ImbalanceReport(
        n_rows=n,
        n_positive=positive,
        n_negative=negative,
        positive_rate=positive / n,
        scale_pos_weight=negative / positive,
        majority_class_accuracy=max(positive, negative) / n,
    )


def scale_pos_weight_from(y_train: pd.Series | np.ndarray) -> float:
    """`scale_pos_weight` cho XGBoost, tính từ **riêng tập train**.

    Tên tham số cố tình là `y_train` chứ không phải `y`: chỗ gọi phải viết ra
    chữ đó, nên truyền nhầm toàn bộ nhãn vào đây là lỗi nhìn thấy ngay trên
    mặt câu lệnh chứ không ẩn trong một chuỗi biến đổi.
    """
    return measure_imbalance(y_train).scale_pos_weight


# --------------------------------------------------------------------------
# Áp dụng cho từng thuật toán
# --------------------------------------------------------------------------
#: Bốn thuật toán của F04, đúng thứ tự task 7 → 10.
ALGORITHMS: Final[tuple[str, ...]] = (
    "decision_tree", "bagging", "random_forest", "xgboost",
)

#: Cơ chế mỗi thuật toán dùng để nhận cùng một tỉ số phạt.
#:
#: `bagging` là chỗ duy nhất phải làm khác: `BaggingClassifier` KHÔNG có tham
#: số `class_weight`. Truyền vào sẽ `TypeError`; tệ hơn là nếu ai đó bọc nó
#: trong `**kwargs` thì tham số bị nuốt im lặng và model train mất cân bằng
#: trong khi bảng cấu hình vẫn ghi là đã cân bằng. Trọng số phải đặt trên
#: `estimator` — tức cây con bên trong.
IMBALANCE_MECHANISM: Final[dict[str, str]] = {
    "decision_tree": "class_weight='balanced' trên chính model",
    "bagging": "class_weight='balanced' trên ESTIMATOR CON, "
               "vì BaggingClassifier không có tham số này",
    "random_forest": "class_weight='balanced' trên chính model",
    "xgboost": "scale_pos_weight = n_âm / n_dương của tập train",
}

#: Giá trị `class_weight` dùng chung cho ba thuật toán họ cây của sklearn.
BALANCED: Final[str] = "balanced"


def imbalance_params(algo: str, y_train: pd.Series | np.ndarray) -> dict:
    """Tham số cân bằng lớp cho một thuật toán, tính từ **riêng tập train**.

    Trả về dict để chỗ gọi bung thẳng vào constructor. `bagging` trả về dict
    RỖNG một cách có chủ ý — không phải quên: `BaggingClassifier` không nhận
    `class_weight`, nên tham số đó thuộc về estimator con và task 8 phải đặt
    ở đó. Xem `estimator_params()`.
    """
    if algo not in ALGORITHMS:
        raise ValueError(f"Thuật toán không nằm trong F04: {algo!r}. "
                         f"Chọn một trong {ALGORITHMS}.")

    if algo == "xgboost":
        return {"scale_pos_weight": scale_pos_weight_from(y_train)}
    if algo == "bagging":
        return {}
    return {"class_weight": BALANCED}


def estimator_params(algo: str) -> dict:
    """Tham số đặt trên ESTIMATOR CON. Chỉ `bagging` dùng tới."""
    return {"class_weight": BALANCED} if algo == "bagging" else {}


# --------------------------------------------------------------------------
# Phương án đã cân nhắc rồi loại
# --------------------------------------------------------------------------
#: Loại bỏ có lý do, ghi lại để trả lời câu "sao không dùng SMOTE?" bằng một
#: câu trả lời cụ thể thay vì "bọn em không dùng".
REJECTED_STRATEGIES: Final[tuple[tuple[str, str], ...]] = (
    ("SMOTE",
     "Phá hiệu chuẩn xác suất (§7.4 bắt buộc có calibration curve + Brier); "
     "và nội suy Euclid vô nghĩa trên 16 cột mã hoá ordinal — 'mã 37,5' không "
     "phải tổ chức nào cả."),
    ("Random oversampling",
     "Nhân bản y hệt dòng dương → cây khớp thuộc chính các dòng đó, và tỉ lệ "
     "nền vẫn bị đổi nên xác suất vẫn lệch."),
    ("Random undersampling",
     "Vứt bỏ ~258.000 hồ sơ âm, tức phần lớn dữ liệu. Với 8,07% dương thì "
     "cân về 50/50 nghĩa là chỉ giữ lại 16% dataset."),
    ("Đổi ngưỡng quyết định thay cho trọng số",
     "Không mâu thuẫn với trọng số mà là việc của bước SAU (task 14): chọn "
     "ngưỡng LOW_RISK/HIGH_RISK. Nó không thay thế được việc làm cho model "
     "chú ý tới lớp thiểu số ngay từ lúc học."),
)


def strategy_table(y_train: pd.Series | np.ndarray) -> pd.DataFrame:
    """Bảng "thuật toán → cơ chế → tham số thật", đưa thẳng vào báo cáo được."""
    rows = []
    for algo in ALGORITHMS:
        params = imbalance_params(algo, y_train)
        child = estimator_params(algo)
        rows.append({
            "algorithm": algo,
            "mechanism": IMBALANCE_MECHANISM[algo],
            "model_params": params or "—",
            "estimator_params": child or "—",
            "effective_ratio": scale_pos_weight_from(y_train),
        })
    return pd.DataFrame(rows)


def rejected_table() -> pd.DataFrame:
    return pd.DataFrame(
        [{"strategy": name, "reason": reason} for name, reason in REJECTED_STRATEGIES])
