"""ML02 task 11 — Đánh giá model (F04 · M04 · Tuần 4).

Nối tiếp task 7–10. Task này **đánh giá**, chưa xếp hạng và chưa chọn:

    task 11 (đây)   đo đầy đủ 8 model đã train, trên validation
    task 12         so sánh và xác định model tốt nhất
    task 14         chốt model + kiểm trên test

Vì vậy file này KHÔNG có hàm nào sắp xếp theo chỉ số hay trả về "model tốt
nhất". Trộn hai việc lại thì phần đánh giá dễ bị rút gọn thành "cái nào PR-AUC
cao nhất", mà đó chính là chỗ bỏ sót những thứ chỉ lộ ra khi nhìn kỹ từng model
— học thuộc, hiệu chuẩn lệch, hay recall cao đổi bằng precision thấp.

Nạp artifact, KHÔNG train lại
------------------------------
Tám model đã được `joblib.dump` ở task 7–10. Train lại ở đây tốn ~12 phút và
tệ hơn là **có nguy cơ ra số khác** nếu một tham số nào đó lệch — khi đó bảng
đánh giá không còn mô tả đúng những model đã báo cáo.

Bốn nhóm chỉ số, và vì sao cần cả bốn
--------------------------------------
1. **Không phụ thuộc ngưỡng** — PR-AUC (chỉ số chọn model), ROC-AUC. Đo khả
   năng XẾP HẠNG rủi ro, độc lập với việc cắt ở đâu.
2. **Tại một ngưỡng** — F1 / precision / recall của lớp dương, confusion
   matrix. Đây là thứ người dùng thật sự gặp, nhưng phụ thuộc ngưỡng nên phải
   đọc kèm nhóm 1.
3. **Hiệu chuẩn** — Brier + đường tin cậy. ML02 ra quyết định theo NGƯỠNG xác
   suất (§8.1), nên "xác suất 30%" phải thật sự nghĩa là 30% ca như vậy vỡ nợ.
   Một model xếp hạng giỏi mà hiệu chuẩn lệch vẫn hỏng ở khâu đặt ngưỡng.
4. **Theo tỉ lệ cảnh báo** — bắt được bao nhiêu phần ca vỡ nợ nếu chỉ soi k%
   hồ sơ rủi ro nhất. Đây là cách đọc gần với vận hành nhất, và nó **không
   cần chọn ngưỡng** — chỉ cần một ngân sách rà soát.

Quét ngưỡng KHÔNG phải chọn ngưỡng
-----------------------------------
`threshold_sweep()` cho thấy các chỉ số biến thiên thế nào theo ngưỡng. Đó là
NGUYÊN LIỆU cho task 14, không phải quyết định — hàm này không trả về ngưỡng
nào cả. Ngưỡng `LOW_RISK/HIGH_RISK` chốt ở task 14 sau khi hiệu chuẩn, và
chắc chắn không phải 0,5 (tỉ lệ nền chỉ 8,07%).

Tập test không bị chạm ở task này.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve

from hfml.config import CONFIG
from hfml.logger import get_logger
from hfml.ml.evaluation.metrics import binary_confusion, binary_metrics
from hfml.ml.ml02_credit_risk.train import (
    ALGORITHMS,
    FEATURE_SETS,
    TrainingData,
    load_training_data,
    models_dir,
)

log = get_logger(__name__)

#: Ngưỡng dùng để tính các chỉ số cần NHÃN cứng trong bảng chính.
#:
#: 0,5 ở đây là quy ước để bảng có một cột so được, KHÔNG phải ngưỡng nghiệp
#: vụ. Với tỉ lệ nền 8,07% thì 0,5 gần như chắc chắn sai — nhưng chọn số khác
#: bây giờ là làm thay việc của task 14.
REPORTING_THRESHOLD: Final[float] = 0.5

#: Các mức ngưỡng đem quét. Dày ở vùng thấp vì đó là nơi ngưỡng thật sẽ nằm
#: với một bài toán 8% dương.
SWEEP_THRESHOLDS: Final[tuple[float, ...]] = (
    0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
    0.45, 0.50, 0.55, 0.60, 0.70, 0.80, 0.90,
)

#: Tỉ lệ hồ sơ được đưa vào diện rà soát, cho bảng "theo ngân sách".
ALERT_RATES: Final[tuple[float, ...]] = (0.05, 0.10, 0.20, 0.30, 0.50)

#: Số khoảng của đường tin cậy (hiệu chuẩn).
CALIBRATION_BINS: Final[int] = 10

EVAL_SUBDIR: Final[str] = "ml02_evaluation"


@dataclass
class ModelEvaluation:
    """Kết quả đánh giá đầy đủ của MỘT model."""

    algo: str
    feature_set: str
    y_true: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    y_proba: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    metrics: dict[str, float] = field(default_factory=dict)
    confusion: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def slug(self) -> str:
        return f"ml02_{self.algo}_{self.feature_set}"


# --------------------------------------------------------------------------
# Nạp artifact và tính lại xác suất trên validation
# --------------------------------------------------------------------------
def artifact_path(algo: str, feature_set: str) -> Path:
    return models_dir() / f"ml02_{algo}_{feature_set}.joblib"


def available_models() -> list[tuple[str, str]]:
    """Các cặp (thuật toán, bộ feature) có artifact trên đĩa."""
    return [(algo, fs) for algo in ALGORITHMS for fs in FEATURE_SETS
            if artifact_path(algo, fs).exists()]


def evaluate_model(
    algo: str,
    feature_set: str,
    data: TrainingData,
    threshold: float = REPORTING_THRESHOLD,
) -> ModelEvaluation:
    """Nạp một artifact và chấm lại trên tập validation."""
    path = artifact_path(algo, feature_set)
    if not path.exists():
        raise FileNotFoundError(
            f"Chưa có artifact {path.name}. Chạy task 7–10 trước.")

    pipeline = joblib.load(path)
    proba = pipeline.predict_proba(data.X_validation)[:, 1]
    truth = np.asarray(data.y_validation).astype(int)

    return ModelEvaluation(
        algo=algo,
        feature_set=feature_set,
        y_true=truth,
        y_proba=proba,
        metrics=binary_metrics(truth, proba, threshold=threshold),
        confusion=binary_confusion(truth, proba, threshold=threshold),
    )


def evaluate_all(
    data: TrainingData | None = None,
    threshold: float = REPORTING_THRESHOLD,
) -> list[ModelEvaluation]:
    """Đánh giá mọi model có artifact. KHÔNG xếp hạng — đó là task 12."""
    data = data if data is not None else load_training_data()
    pairs = available_models()
    if not pairs:
        raise FileNotFoundError(
            f"Không tìm thấy artifact nào ở {models_dir()}. Chạy task 7–10 trước.")

    results = []
    for algo, feature_set in pairs:
        log.info("Đánh giá %s · bộ %s", algo, feature_set)
        results.append(evaluate_model(algo, feature_set, data, threshold))
    return results


# --------------------------------------------------------------------------
# Nhóm 1 + 2 — bảng chỉ số chính
# --------------------------------------------------------------------------
#: Thứ tự cột của bảng chính: chỉ số không phụ thuộc ngưỡng trước, rồi tới
#: chỉ số tại ngưỡng, accuracy đứng CUỐI vì nó không được cầm lái (§7.3).
METRIC_ORDER: Final[tuple[str, ...]] = (
    "pr_auc", "pr_auc_lift", "roc_auc",
    "f1_positive", "recall_positive", "precision_positive",
    "brier", "balanced_accuracy", "accuracy",
)


def metrics_table(evaluations: list[ModelEvaluation]) -> pd.DataFrame:
    """Bảng chỉ số chính, một dòng mỗi model.

    Giữ nguyên thứ tự nạp vào — KHÔNG sắp xếp theo chỉ số nào. Sắp xếp là
    hành vi của bước so sánh (task 12); làm ở đây thì bảng đánh giá ngầm biến
    thành bảng xếp hạng.
    """
    rows = []
    for evaluation in evaluations:
        rows.append({
            "algo": evaluation.algo,
            "feature_set": evaluation.feature_set,
            **{k: evaluation.metrics[k] for k in METRIC_ORDER
               if k in evaluation.metrics},
        })
    return pd.DataFrame(rows)


def confusion_long(evaluations: list[ModelEvaluation]) -> pd.DataFrame:
    """Confusion matrix của mọi model, dạng bảng dài để ghi một file.

    Kèm bốn ô đếm được đặt tên rõ ràng: nhìn `false_negative` biết ngay đó là
    số ca vỡ nợ bị bỏ lọt — thứ đắt nhất trong bài toán này.
    """
    rows = []
    for evaluation in evaluations:
        matrix = evaluation.confusion.to_numpy()
        rows.append({
            "algo": evaluation.algo,
            "feature_set": evaluation.feature_set,
            "true_negative": int(matrix[0, 0]),
            "false_positive": int(matrix[0, 1]),
            "false_negative": int(matrix[1, 0]),
            "true_positive": int(matrix[1, 1]),
            "threshold": evaluation.metrics["threshold"],
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Nhóm 3 — hiệu chuẩn
# --------------------------------------------------------------------------
def calibration_table(
    evaluations: list[ModelEvaluation],
    n_bins: int = CALIBRATION_BINS,
) -> pd.DataFrame:
    """Đường tin cậy: xác suất model nói so với tần suất thật sự xảy ra.

    Cột `gap` = dự đoán − thực tế. Dương nghĩa là model **nói quá** — và với
    `class_weight='balanced'` / `scale_pos_weight`, nói quá là điều PHẢI xảy
    ra: trọng số đẩy xác suất lớp dương lên trên tỉ lệ nền thật.
    Đó là lý do §7.4 yêu cầu hiệu chuẩn trước khi đặt ngưỡng.
    """
    rows = []
    for evaluation in evaluations:
        observed, predicted = calibration_curve(
            evaluation.y_true, evaluation.y_proba,
            n_bins=n_bins, strategy="quantile")
        for index, (obs, pred) in enumerate(zip(observed, predicted)):
            rows.append({
                "algo": evaluation.algo,
                "feature_set": evaluation.feature_set,
                "bin": index,
                "mean_predicted": float(pred),
                "observed_rate": float(obs),
                "gap": float(pred - obs),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Nhóm 4 — theo ngân sách rà soát
# --------------------------------------------------------------------------
def capture_table(
    evaluations: list[ModelEvaluation],
    alert_rates: tuple[float, ...] = ALERT_RATES,
) -> pd.DataFrame:
    """Soi k% hồ sơ rủi ro nhất thì bắt được bao nhiêu phần ca vỡ nợ.

    Cách đọc gần với vận hành nhất, và **không cần chọn ngưỡng** — chỉ cần một
    ngân sách rà soát. `lift` = tỉ lệ bắt được ÷ tỉ lệ soi: soi 10% mà bắt được
    30% số ca thì lift = 3,0, tức gấp ba lần soi ngẫu nhiên.
    """
    rows = []
    for evaluation in evaluations:
        order = np.argsort(-evaluation.y_proba)     # rủi ro cao xuống thấp
        sorted_truth = evaluation.y_true[order]
        total_positive = int(sorted_truth.sum())

        for rate in alert_rates:
            k = max(1, int(round(len(sorted_truth) * rate)))
            caught = int(sorted_truth[:k].sum())
            rows.append({
                "algo": evaluation.algo,
                "feature_set": evaluation.feature_set,
                "alert_rate": rate,
                "n_reviewed": k,
                "n_caught": caught,
                "capture_rate": caught / total_positive if total_positive else 0.0,
                "precision_at_k": caught / k,
                "lift": (caught / total_positive / rate) if total_positive else 0.0,
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Quét ngưỡng — nguyên liệu cho task 14, KHÔNG chọn gì
# --------------------------------------------------------------------------
def threshold_sweep(
    evaluations: list[ModelEvaluation],
    thresholds: tuple[float, ...] = SWEEP_THRESHOLDS,
) -> pd.DataFrame:
    """Chỉ số biến thiên thế nào theo ngưỡng.

    Hàm này CỐ Ý không trả về ngưỡng tốt nhất. Chọn ngưỡng là task 14, và phải
    làm sau khi hiệu chuẩn — chọn trên xác suất chưa hiệu chuẩn thì con số
    ngưỡng không mang ý nghĩa xác suất nào.
    """
    rows = []
    for evaluation in evaluations:
        for threshold in thresholds:
            metrics = binary_metrics(
                evaluation.y_true, evaluation.y_proba, threshold=threshold)
            rows.append({
                "algo": evaluation.algo,
                "feature_set": evaluation.feature_set,
                "threshold": threshold,
                "f1_positive": metrics["f1_positive"],
                "recall_positive": metrics["recall_positive"],
                "precision_positive": metrics["precision_positive"],
                "accuracy": metrics["accuracy"],
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Dữ liệu đường cong — để vẽ và để kiểm chứng lại
# --------------------------------------------------------------------------
#: Số điểm giữ lại của mỗi đường cong. `precision_recall_curve` trả về một
#: điểm cho MỖI ngưỡng phân biệt — trên 46.127 hồ sơ là hàng chục nghìn dòng,
#: ghi hết ra CSV là vô ích. Lấy mẫu đều theo chỉ số giữ nguyên hình dạng.
CURVE_POINTS: Final[int] = 200


def _thin(*arrays: np.ndarray, n: int = CURVE_POINTS) -> tuple[np.ndarray, ...]:
    length = len(arrays[0])
    if length <= n:
        return arrays
    index = np.unique(np.linspace(0, length - 1, n).astype(int))
    return tuple(array[index] for array in arrays)


def curve_table(evaluations: list[ModelEvaluation]) -> pd.DataFrame:
    """Điểm của đường PR và đường ROC, dạng bảng dài."""
    rows = []
    for evaluation in evaluations:
        precision, recall, _ = precision_recall_curve(
            evaluation.y_true, evaluation.y_proba)
        recall_t, precision_t = _thin(recall, precision)
        for r, p in zip(recall_t, precision_t):
            rows.append({"algo": evaluation.algo,
                         "feature_set": evaluation.feature_set,
                         "curve": "pr", "x": float(r), "y": float(p)})

        fpr, tpr, _ = roc_curve(evaluation.y_true, evaluation.y_proba)
        fpr_t, tpr_t = _thin(fpr, tpr)
        for x, y in zip(fpr_t, tpr_t):
            rows.append({"algo": evaluation.algo,
                         "feature_set": evaluation.feature_set,
                         "curve": "roc", "x": float(x), "y": float(y)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Ghi kết quả
# --------------------------------------------------------------------------
def output_dir() -> Path:
    return CONFIG.paths.runs / EVAL_SUBDIR


def write_evaluation(evaluations: list[ModelEvaluation]) -> dict[str, Path]:
    """Ghi năm bảng đánh giá + metadata."""
    out_dir = output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    tables = {
        "metrics": metrics_table(evaluations),
        "confusion": confusion_long(evaluations),
        "calibration": calibration_table(evaluations),
        "capture_by_alert_rate": capture_table(evaluations),
        "threshold_sweep": threshold_sweep(evaluations),
        "curves": curve_table(evaluations),
    }

    written: dict[str, Path] = {}
    for name, table in tables.items():
        path = out_dir / f"{name}.csv"
        table.to_csv(path, index=False, encoding="utf-8")
        written[name] = path

    metadata = {
        "task": "ML02 task 11 — Đánh giá model",
        "evaluated_on": "validation",
        "test_set_touched": False,
        "n_models": len(evaluations),
        "reporting_threshold": REPORTING_THRESHOLD,
        "threshold_note": "0,5 chỉ là quy ước để bảng có một cột so được. "
                          "Ngưỡng nghiệp vụ LOW_RISK/HIGH_RISK chốt ở task 14 "
                          "sau khi hiệu chuẩn.",
        "selection_metric": "pr_auc",
        "ranking_done_here": False,
        "ranking_note": "Xếp hạng và chọn model là task 12 và 14.",
    }
    path = out_dir / "evaluation_metadata.json"
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    written["metadata"] = path

    log.info("Đã ghi %d file đánh giá → %s", len(written), out_dir)
    return written
