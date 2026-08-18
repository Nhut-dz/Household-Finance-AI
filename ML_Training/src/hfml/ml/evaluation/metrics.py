"""Chỉ số đánh giá — dùng chung cho ML01 và ML02 (PLAN.md §11).

Một hàm cho cả hai bài toán, nhưng **chỉ số chọn model thì khác nhau**:

    ML01   4 lớp không cân bằng   →  Macro-F1
    ML02   2 lớp, 8,07% dương     →  PR-AUC

`SELECTION_METRIC` giữ ánh xạ đó. Đừng chọn model bằng accuracy ở ML02: đoán
"không ai vỡ nợ" đã đạt 91,93%, cao hơn phần lớn model thật. Accuracy vẫn
được tính và báo cáo đầy đủ — nó nằm trong bảng, chỉ không được cầm lái.

Vì sao trả về dict phẳng toàn số vô hướng
-----------------------------------------
Mỗi lần chạy là một dòng trong `experiments/results.csv` (F07 task 1) và một
dòng trong bảng so sánh 4 thuật toán. Dict lồng nhau thì mỗi chỗ dùng lại
phải tự trải phẳng theo một kiểu riêng, rồi tên cột giữa các bảng lệch nhau.
Bảng per-class và confusion matrix vốn là ma trận nên tách ra hàm riêng.
"""
from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

#: Chỉ số quyết định model nào được chọn, theo bài toán (PLAN.md §11).
SELECTION_METRIC: Final[dict[str, str]] = {
    "ml01": "macro_f1",
    "ml02": "pr_auc",
}


def classification_metrics(
    y_true,
    y_pred,
    y_proba: np.ndarray | None = None,
    labels: list[str] | None = None,
) -> dict[str, float]:
    """Chỉ số vô hướng cho một lần đánh giá.

    `y_proba` là ma trận `(n_samples, n_classes)` theo đúng thứ tự `labels`.
    Thiếu nó thì bỏ qua các chỉ số cần xác suất — `DummyClassifier` vẫn có
    `predict_proba`, nên baseline không vì thế mà thiếu chỉ số.

    Ba chỉ số nhị phân (`pr_auc`, `roc_auc`, `brier`) chỉ tính khi bài toán
    đúng 2 lớp. Với 4 lớp của ML01 chúng không có định nghĩa dùng được ngay,
    và điền một giá trị đại khái vào đó thì sớm muộn có người đọc bảng và
    tưởng đó là số thật.
    """
    labels = list(labels) if labels is not None else sorted(set(map(str, y_true)))
    y_true = np.asarray(y_true, dtype=object).astype(str)
    y_pred = np.asarray(y_pred, dtype=object).astype(str)

    # Precision/recall macro: trung bình KHÔNG trọng số qua 4 lớp, cùng cách
    # gộp với `macro_f1` để ba con số đọc cạnh nhau được. Trọng số theo
    # support thì lớp DEBT_FOCUS (~15%) gần như biến mất khỏi chỉ số.
    macro_precision, macro_recall, _, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0)

    out: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels,
                                   average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels,
                                      average="weighted", zero_division=0)),
    }

    if y_proba is not None and len(labels) == 2:
        # Lớp DƯƠNG là lớp thứ hai theo thứ tự `labels` — cùng quy ước với
        # `predict_proba` của sklearn, nên không cần đoán.
        positive = np.asarray(y_proba)[:, 1]
        truth = (y_true == labels[1]).astype(int)
        out["pr_auc"] = float(average_precision_score(truth, positive))
        out["roc_auc"] = float(roc_auc_score(truth, positive))
        out["brier"] = float(brier_score_loss(truth, positive))

    return out


#: Sàn thật của PR-AUC là TỈ LỆ DƯƠNG, không phải 0,5.
#:
#: Đây là chỗ đọc nhầm nhiều nhất khi báo cáo bài toán mất cân bằng. ROC-AUC
#: có sàn 0,5 vì đoán bừa cho ra đúng 0,5. PR-AUC thì không: một model đoán
#: bừa cho PR-AUC ≈ tỉ lệ dương, tức **0,0807** ở ML02. Nên "PR-AUC = 0,15"
#: không phải "kém hơn ngẫu nhiên", nó là **gần gấp đôi** mức ngẫu nhiên.
#: `pr_auc_lift` dưới đây tính sẵn tỉ số đó để khỏi ai phải nhẩm.
PR_AUC_FLOOR_IS_BASE_RATE: Final[bool] = True


def binary_metrics(
    y_true,
    y_proba_positive,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Bộ chỉ số cho bài toán nhị phân mất cân bằng (ML02).

    Khác `classification_metrics` ở chỗ báo cáo **riêng lớp dương** thay vì
    trung bình macro. Với 8,07% dương, macro-recall gộp cả lớp âm vào nên một
    model bỏ rơi hoàn toàn lớp dương vẫn có macro-recall ~0,5 — nghe không tệ,
    trong khi nó không bắt được ca vỡ nợ nào.

    `threshold` mặc định 0,5 chỉ để tính các chỉ số cần NHÃN cứng (F1, recall,
    confusion matrix). Nó **không phải** ngưỡng nghiệp vụ `LOW_RISK/HIGH_RISK`
    — ngưỡng đó chốt ở task 14 sau khi hiệu chuẩn, và với tỉ lệ nền 8% thì 0,5
    chắc chắn không phải giá trị đúng. Hai chỉ số quan trọng nhất (`pr_auc`,
    `roc_auc`) không phụ thuộc ngưỡng nên không bị ảnh hưởng.
    """
    truth = np.asarray(y_true).astype(int)
    proba = np.asarray(y_proba_positive, dtype=float)
    predicted = (proba >= threshold).astype(int)

    base_rate = float(truth.mean())
    pr_auc = float(average_precision_score(truth, proba))
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, predicted, labels=[1], average="binary", zero_division=0)

    return {
        # Chỉ số CHỌN MODEL của ML02.
        "pr_auc": pr_auc,
        # PR-AUC hơn mức đoán bừa bao nhiêu lần. 1,0 = không hơn gì.
        "pr_auc_lift": pr_auc / base_rate if base_rate else float("nan"),
        "roc_auc": float(roc_auc_score(truth, proba)),
        # Ba chỉ số của RIÊNG lớp dương — thứ macro làm mờ mất.
        "precision_positive": float(precision),
        "recall_positive": float(recall),
        "f1_positive": float(f1),
        # Có trong bảng nhưng KHÔNG được cầm lái: đoán toàn 0 đã đạt 91,93%.
        "accuracy": float(accuracy_score(truth, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        # Hiệu chuẩn xác suất — càng nhỏ càng tốt (PLAN.md §7.4).
        "brier": float(brier_score_loss(truth, proba)),
        "base_rate": base_rate,
        "threshold": float(threshold),
    }


def binary_confusion(y_true, y_proba_positive, threshold: float = 0.5) -> pd.DataFrame:
    """Confusion matrix nhị phân có tên hàng/cột đọc được bằng tiếng Việt."""
    truth = np.asarray(y_true).astype(int)
    predicted = (np.asarray(y_proba_positive, dtype=float) >= threshold).astype(int)
    matrix = confusion_matrix(truth, predicted, labels=[0, 1])
    names = ["0 · trả nợ bình thường", "1 · khó khăn trả nợ"]
    return pd.DataFrame(matrix,
                        index=pd.Index(names, name="thật"),
                        columns=pd.Index(names, name="dự đoán"))


def per_class_table(y_true, y_pred, labels: list[str]) -> pd.DataFrame:
    """Precision / recall / F1 / support từng lớp.

    Bắt buộc có trong báo cáo ML01: macro-F1 gộp 4 lớp thành một số, nên chỉ
    nhìn nó thì không thấy được model đang bỏ rơi lớp nào.
    """
    y_true = np.asarray(y_true, dtype=object).astype(str)
    y_pred = np.asarray(y_pred, dtype=object).astype(str)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    return pd.DataFrame({
        "label": labels,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    })


def confusion_table(y_true, y_pred, labels: list[str]) -> pd.DataFrame:
    """Confusion matrix có tên hàng/cột — hàng là THẬT, cột là DỰ ĐOÁN."""
    y_true = np.asarray(y_true, dtype=object).astype(str)
    y_pred = np.asarray(y_pred, dtype=object).astype(str)
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(matrix,
                        index=pd.Index(labels, name="thật"),
                        columns=pd.Index(labels, name="dự đoán"))


def aggregate_folds(fold_metrics: list[dict[str, float]]) -> dict[str, float]:
    """Gộp chỉ số các fold thành `<tên>` (trung bình) và `<tên>_std`.

    Giữ độ lệch chuẩn chứ không chỉ trung bình: chênh lệch macro-F1 0,002
    giữa hai thuật toán là vô nghĩa nếu std giữa các fold là 0,03. Không có
    std thì bảng so sánh dẫn người đọc tới kết luận sai về "model tốt nhất".
    """
    if not fold_metrics:
        return {}
    keys = [k for k in fold_metrics[0] if all(k in m for m in fold_metrics)]
    out: dict[str, float] = {}
    for key in keys:
        values = np.array([m[key] for m in fold_metrics], dtype=float)
        out[key] = float(values.mean())
        out[f"{key}_std"] = float(values.std(ddof=0))
    return out
