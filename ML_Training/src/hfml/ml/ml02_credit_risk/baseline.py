"""ML02 task 6 — Xây dựng baseline (F04 · M04 · Tuần 4).

Nối tiếp task 5. Task này KHÔNG train model chính — nó dựng cái **mốc** mà
bốn thuật toán ở task 7–10 phải vượt qua, và dựng sao cho con số đó đọc đúng.

Baseline chính thức: `DummyClassifier(strategy="stratified")`
--------------------------------------------------------------
Đoán ngẫu nhiên theo đúng tỉ lệ lớp của tập train (8,07% dương). Nó không nhìn
feature nào cả — và đó chính là điều làm nó thành mốc: mọi thứ một model thật
đạt được **trên mức này** là phần do feature đóng góp.

Vì sao con số PR-AUC của baseline lại quan trọng đến thế
--------------------------------------------------------
**Sàn của PR-AUC là TỈ LỆ DƯƠNG, không phải 0,5.** Đây là chỗ đọc nhầm nhiều
nhất ở bài toán mất cân bằng:

    ROC-AUC   đoán bừa → 0,50.  Sàn là 0,5, ai cũng biết.
    PR-AUC    đoán bừa → 0,0807 (bằng tỉ lệ dương). Sàn KHÔNG phải 0,5.

Nên một model đạt PR-AUC 0,20 không hề "tệ hơn ngẫu nhiên" — nó gấp **2,5 lần**
mức ngẫu nhiên. Thiếu hàng baseline trong bảng thì cả hội đồng lẫn người viết
báo cáo đều dễ kết luận ngược. `pr_auc_lift` tính sẵn tỉ số đó.

Hàng tham chiếu: `DummyClassifier(strategy="most_frequent")`
------------------------------------------------------------
Đoán TOÀN BỘ là 0. Nó **không phải baseline** — nó là bằng chứng cho §7.3:
model này đạt accuracy **91,93%** mà không bắt được một ca vỡ nợ nào
(recall lớp dương = 0). Có nó trong bảng thì câu "accuracy 92%" không còn nghe
như một kết quả tốt.

Không có gì để rò rỉ ở task này
-------------------------------
`DummyClassifier` không đọc `X`. Nó chỉ cần phân bố lớp của `y_train`, mà
`fit()` chỉ nhận tập train. Chấm trên **validation**; tập test vẫn khoá,
chỉ mở ở task 14.

Cũng vì không đọc `X`, baseline **giống hệt nhau ở bộ FULL và bộ RÚT GỌN** —
nên nó được đo một lần và dùng chung cho cả hai bảng so sánh.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

from hfml.config import CONFIG
from hfml.logger import get_logger
from hfml.ml.evaluation.metrics import binary_confusion, binary_metrics
from hfml.ml.ml02_credit_risk.clean import TARGET_COLUMN

log = get_logger(__name__)

#: Tên baseline chính thức, dùng làm khoá trong `results.csv`.
BASELINE_NAME: Final[str] = "baseline_stratified"

#: Hàng tham chiếu — KHÔNG phải baseline. Xem docstring.
MAJORITY_NAME: Final[str] = "reference_most_frequent"

#: Chiến lược `DummyClassifier` và vai trò của từng cái.
STRATEGIES: Final[tuple[tuple[str, str, str], ...]] = (
    (BASELINE_NAME, "stratified",
     "BASELINE chính thức — đoán ngẫu nhiên theo tỉ lệ lớp của tập train"),
    (MAJORITY_NAME, "most_frequent",
     "Tham chiếu — đoán toàn 0, cho thấy accuracy 91,93% là vô nghĩa"),
)

#: Mức PR-AUC tối thiểu mà một model thật phải vượt baseline để được coi là có
#: học được gì. Không phải ngưỡng đạt/trượt của đồ án — chỉ là mốc đọc bảng.
MEANINGFUL_LIFT: Final[float] = 1.5


@dataclass
class BaselineResult:
    """Kết quả một baseline, chấm trên tập validation."""

    name: str
    strategy: str
    role: str
    metrics: dict[str, float] = field(default_factory=dict)
    confusion: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def pr_auc(self) -> float:
        return self.metrics.get("pr_auc", float("nan"))


def fit_baseline(
    y_train: pd.Series,
    y_validation: pd.Series,
    strategy: str = "stratified",
    seed: int | None = None,
) -> tuple[DummyClassifier, np.ndarray]:
    """Fit một `DummyClassifier` trên train, trả xác suất trên validation.

    `X` chỉ cần đúng số dòng — `DummyClassifier` không đọc giá trị nào của nó.
    Truyền một cột số 0 thay vì bộ feature thật là có chủ ý: nó làm rõ ngay
    trên mặt câu lệnh rằng baseline không dùng feature, nên mọi phần một model
    thật hơn được baseline đều là do feature đóng góp.
    """
    seed = CONFIG.random_seed if seed is None else seed
    model = DummyClassifier(strategy=strategy, random_state=seed)
    model.fit(np.zeros((len(y_train), 1)), y_train)

    proba = model.predict_proba(np.zeros((len(y_validation), 1)))
    # Cột của lớp DƯƠNG theo đúng thứ tự `classes_`, không đoán vị trí.
    positive_column = list(model.classes_).index(1)
    return model, proba[:, positive_column]


def evaluate_baselines(
    y_train: pd.Series,
    y_validation: pd.Series,
    seed: int | None = None,
) -> list[BaselineResult]:
    """Chấm cả baseline chính thức lẫn hàng tham chiếu, trên validation."""
    results: list[BaselineResult] = []
    for name, strategy, role in STRATEGIES:
        _, proba = fit_baseline(y_train, y_validation, strategy=strategy, seed=seed)
        results.append(BaselineResult(
            name=name,
            strategy=strategy,
            role=role,
            metrics=binary_metrics(y_validation, proba),
            confusion=binary_confusion(y_validation, proba),
        ))
        log.info("%s: PR-AUC %.4f · ROC-AUC %.4f · recall lớp dương %.4f",
                 name, results[-1].metrics["pr_auc"],
                 results[-1].metrics["roc_auc"],
                 results[-1].metrics["recall_positive"])
    return results


def metrics_frame(results: list[BaselineResult]) -> pd.DataFrame:
    """Bảng chỉ số, một dòng mỗi baseline — đưa thẳng vào báo cáo được."""
    return pd.DataFrame([
        {"name": r.name, "strategy": r.strategy, "role": r.role, **r.metrics}
        for r in results
    ])


def baseline_of(results: list[BaselineResult]) -> BaselineResult:
    """Baseline CHÍNH THỨC. Hàng `most_frequent` chỉ để tham chiếu."""
    for result in results:
        if result.name == BASELINE_NAME:
            return result
    raise ValueError(f"Không tìm thấy baseline {BASELINE_NAME!r} trong kết quả.")


def expected_random_pr_auc(y: pd.Series) -> float:
    """PR-AUC lý thuyết của một model đoán bừa = tỉ lệ dương.

    Có hàm này để đối chiếu với PR-AUC ĐO ĐƯỢC của baseline. Hai số lệch nhau
    nhiều nghĩa là baseline đang làm điều gì đó ngoài dự kiến — hoặc tập
    validation không còn giữ đúng tỉ lệ nền.
    """
    return float(pd.Series(y).astype(int).mean())
