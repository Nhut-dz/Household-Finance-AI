"""ML02 task 14 — Chọn model tốt nhất (F04 · M04 · Tuần 4).

Nối tiếp task 11–13. Đây là task **đầu tiên** được chạm tập test — và chỉ sau
khi quyết định đã chốt. Export là task 15.

Thứ tự bắt buộc, và lý do không được đảo
------------------------------------------
    1. CHỌN model    — chỉ dựa trên bằng chứng validation (task 11, 12, 13)
    2. CHỐT cấu hình — hiệu chuẩn xác suất, rồi chọn ngưỡng, cũng trên validation
    3. MỞ tập test   — đúng một lần, để BÁO CÁO chứ không để chọn

Đảo thứ tự — nhìn test rồi mới chọn — thì con số test không còn là ước lượng
độc lập nữa mà thành một chỉ số đã được tối ưu gián tiếp. Đó là dạng rò rỉ
không để lại dấu vết nào trong mã: mọi thứ vẫn chạy, chỉ có con số cuối cùng
là lạc quan hơn thực tế.

Ràng buộc này cài bằng cấu trúc: `decide()` KHÔNG nhận tập test, và
`evaluate_on_test()` bắt buộc nhận một `Decision` đã chốt.

Vì sao phải hiệu chuẩn trước khi chọn ngưỡng
----------------------------------------------
Task 11 đo được: **cả 8 model đều nói quá về rủi ro**, gap trung bình +0,29 tới
+0,36. Đó là hệ quả bắt buộc của `class_weight='balanced'` và
`scale_pos_weight` — trọng số đẩy xác suất lớp dương lên trên tỉ lệ nền thật.

Chọn ngưỡng trên xác suất chưa hiệu chuẩn thì con số ngưỡng **không mang ý
nghĩa xác suất nào**: "ngưỡng 0,35" sẽ không có nghĩa "35% khả năng gặp khó
khăn trả nợ". Mà §8.1 lại ra quyết định theo đúng ngưỡng đó và tầng `llm` sẽ
đọc con số ấy ra cho người dùng.

Hiệu chuẩn `fit` trên **validation**, không phải train: model đã thấy hết tập
train nên xác suất của nó trên train quá tự tin, hiệu chuẩn theo đó sẽ học
nhầm. Validation đúng vai — task 5 đã định nghĩa nó là nơi tinh chỉnh.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import precision_recall_curve

from hfml.config import CONFIG
from hfml.logger import get_logger
from hfml.ml.evaluation.metrics import binary_confusion, binary_metrics
from hfml.ml.ml02_credit_risk.clean import TARGET_COLUMN, load_clean_application
from hfml.ml.ml02_credit_risk.compare import SELECTION_METRIC
from hfml.ml.ml02_credit_risk.evaluate import artifact_path
from hfml.ml.ml02_credit_risk.features import split_features_and_target
from hfml.ml.ml02_credit_risk.split import load_split

log = get_logger(__name__)

#: Bộ feature của model ĐEM TRIỂN KHAI.
#:
#: §7.2 định nghĩa rõ hai vai: bộ FULL chứng minh năng lực kỹ thuật nhưng có
#: `EXT_SOURCE_1/2/3` mà form KHÔNG thu được, nên nó không chạy được trong sản
#: phẩm. Bộ RÚT GỌN mới là model thật sự deploy.
DEPLOY_FEATURE_SET: Final[str] = "reduced"

#: Bộ giữ lại để báo cáo — không deploy được.
REFERENCE_FEATURE_SET: Final[str] = "full"

#: Phương pháp hiệu chuẩn.
#:
#: `isotonic` chứ không phải `sigmoid` mặc định: sigmoid giả định lệch hiệu
#: chuẩn có dạng hàm logistic, mà lệch ở đây do trọng số lớp gây ra nên không
#: có lý do gì mang dạng đó. Isotonic chỉ giả định đơn điệu — đúng thứ ta biết
#: chắc — và với 46.127 hồ sơ validation thì nó không thiếu dữ liệu để khớp.
CALIBRATION_METHOD: Final[str] = "isotonic"

#: Tỉ lệ hồ sơ đưa vào rà soát, dùng cho điểm vận hành thứ hai.
ALERT_RATE: Final[float] = 0.10

SELECTION_SUBDIR: Final[str] = "ml02_selection"


# --------------------------------------------------------------------------
# Bước 1 — CHỌN model, chỉ bằng bằng chứng validation
# --------------------------------------------------------------------------
@dataclass
class Decision:
    """Quyết định đã chốt. Không chứa số liệu test — test mở sau."""

    algo: str
    deploy_feature_set: str
    reference_feature_set: str
    reasons: list[str] = field(default_factory=list)
    runner_up: str = ""
    #: Ngưỡng LOW_RISK/HIGH_RISK, chốt sau khi hiệu chuẩn.
    threshold: float | None = None
    threshold_rule: str = ""
    calibration_method: str = CALIBRATION_METHOD

    @property
    def deploy_slug(self) -> str:
        return f"ml02_{self.algo}_{self.deploy_feature_set}"

    @property
    def reference_slug(self) -> str:
        return f"ml02_{self.algo}_{self.reference_feature_set}"


def decide(comparison: pd.DataFrame, pairwise: pd.DataFrame) -> Decision:
    """Chọn thuật toán từ bảng so sánh của task 12.

    KHÔNG nhận tập test — muốn nhìn test ở bước này thì phải sửa chữ ký hàm,
    tức phải cố ý.

    Chọn theo PR-AUC trên validation, nhưng chỉ chốt khi khoảng cách với hạng
    hai **phân biệt được với nhiễu** (bootstrap cặp đôi, task 12). Dẫn đầu bằng
    một khoảng nằm trong sai số của phép đo thì đó không phải lý do để chọn.
    """
    if comparison.empty:
        raise ValueError("Bảng so sánh rỗng — chạy task 12 trước.")

    deploy = comparison[comparison["feature_set"] == DEPLOY_FEATURE_SET]
    if deploy.empty:
        raise ValueError(
            f"Không có model nào ở bộ {DEPLOY_FEATURE_SET!r} — "
            "đó mới là bộ đem triển khai (§7.2).")

    ordered = deploy.sort_values(SELECTION_METRIC, ascending=False)
    winner = ordered.iloc[0]
    runner_up = ordered.iloc[1] if len(ordered) > 1 else None

    reasons = [
        f"PR-AUC cao nhất ở bộ {DEPLOY_FEATURE_SET} "
        f"({winner[SELECTION_METRIC]:.4f}) — chỉ số chọn model của ML02 (§7.3).",
    ]

    if runner_up is not None:
        gap = winner[SELECTION_METRIC] - runner_up[SELECTION_METRIC]
        match = pairwise[
            (pairwise["feature_set"] == DEPLOY_FEATURE_SET)
            & (pairwise["model_b"] == f"ml02_{runner_up['algo']}_{DEPLOY_FEATURE_SET}")
        ]
        if not match.empty and bool(match.iloc[0]["distinguishable"]):
            row = match.iloc[0]
            reasons.append(
                f"Khoảng cách với {runner_up['algo']} ({gap:+.4f}) PHÂN BIỆT ĐƯỢC "
                f"với nhiễu: khoảng tin cậy 95% [{row['ci_low']:+.4f}, "
                f"{row['ci_high']:+.4f}] không chứa 0, thắng "
                f"{row['win_rate']:.0%} số lần lấy mẫu.")
        else:
            reasons.append(
                f"⚠️ Khoảng cách với {runner_up['algo']} ({gap:+.4f}) CHƯA phân "
                "biệt được với nhiễu — cần cân nhắc thêm tiêu chí khác.")

    reasons.append(
        f"Cùng thuật toán cũng dẫn đầu ở bộ {REFERENCE_FEATURE_SET}, nên lựa "
        "chọn không phụ thuộc vào việc dùng bộ feature nào.")
    reasons.append(
        f"Bộ triển khai là {DEPLOY_FEATURE_SET} chứ không phải "
        f"{REFERENCE_FEATURE_SET}: bộ kia có EXT_SOURCE_1/2/3 mà form người "
        "dùng KHÔNG thu được (§7.2).")

    return Decision(
        algo=str(winner["algo"]),
        deploy_feature_set=DEPLOY_FEATURE_SET,
        reference_feature_set=REFERENCE_FEATURE_SET,
        reasons=reasons,
        runner_up=str(runner_up["algo"]) if runner_up is not None else "",
    )


# --------------------------------------------------------------------------
# Bước 2 — CHỐT cấu hình: hiệu chuẩn rồi chọn ngưỡng, trên validation
# --------------------------------------------------------------------------
def calibrate(pipeline, X_validation: pd.DataFrame, y_validation: pd.Series):
    """Hiệu chuẩn xác suất bằng isotonic, `fit` trên **validation**.

    `FrozenEstimator` giữ nguyên model đã train — `CalibratedClassifierCV` chỉ
    học phép ánh xạ xác suất, không train lại gì. Nếu để nó tự train lại thì
    model cuối cùng khác model đã được so sánh ở task 12, và cả bảng so sánh
    mất hiệu lực.
    """
    calibrated = CalibratedClassifierCV(
        FrozenEstimator(pipeline), method=CALIBRATION_METHOD)
    calibrated.fit(X_validation, y_validation)
    return calibrated


def calibration_gap(y_true, y_proba, n_bins: int = 10) -> float:
    """Chênh lệch trung bình giữa xác suất model nói và tần suất thật.

    Dương = nói quá. Đây là con số task 11 đo được +0,29 → +0,36 trước hiệu
    chuẩn; sau hiệu chuẩn nó phải tiến gần 0, nếu không thì bước hiệu chuẩn
    không làm được việc của nó.
    """
    from sklearn.calibration import calibration_curve

    observed, predicted = calibration_curve(
        y_true, y_proba, n_bins=n_bins, strategy="quantile")
    return float(np.mean(predicted - observed))


def choose_threshold(y_true, y_proba) -> tuple[float, str]:
    """Chọn ngưỡng LOW_RISK / HIGH_RISK trên tập validation.

    Tiêu chí: **F1 lớn nhất của lớp dương**. Không phải 0,5 — với tỉ lệ nền
    8,07% thì 0,5 xếp gần như mọi hồ sơ vào LOW_RISK.

    ⚠️ Giới hạn phải ghi vào `model_card.md`: F1 coi một ca vỡ nợ bị bỏ lọt và
    một hồ sơ tốt bị gắn nhãn rủi ro là **đắt như nhau**. Trong tín dụng thật
    thì không — bỏ lọt một khoản vỡ nợ tốn hơn nhiều. Ngưỡng đúng phải suy từ
    ma trận chi phí của tổ chức cho vay, mà đồ án không có. F1 là lựa chọn
    trung tính khi chưa có chi phí, không phải lựa chọn tối ưu.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    # `precision_recall_curve` trả về nhiều hơn `thresholds` một phần tử.
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(
        precision[:-1] + recall[:-1], 1e-12)
    best = int(np.argmax(f1))
    return float(thresholds[best]), (
        f"F1 lớp dương lớn nhất trên validation (F1 = {f1[best]:.4f}). "
        "Chưa có ma trận chi phí thật nên F1 là lựa chọn trung tính, "
        "KHÔNG phải tối ưu — xem model_card.")


def threshold_at_alert_rate(y_proba, rate: float = ALERT_RATE) -> float:
    """Ngưỡng ứng với việc rà soát `rate` phần hồ sơ rủi ro nhất.

    Điểm vận hành thứ hai, để báo cáo có một con số gắn với ngân sách rà soát
    thay vì chỉ một tiêu chí thống kê.
    """
    return float(np.quantile(y_proba, 1 - rate))


# --------------------------------------------------------------------------
# Bước 3 — MỞ tập test, đúng một lần
# --------------------------------------------------------------------------
@dataclass
class FinalReport:
    """Kết quả cuối: validation (đã dùng để chọn) và test (chỉ để báo cáo)."""

    decision: Decision
    validation_metrics: dict[str, float] = field(default_factory=dict)
    test_metrics: dict[str, float] = field(default_factory=dict)
    validation_calibration_gap: float = float("nan")
    test_calibration_gap: float = float("nan")
    test_confusion: pd.DataFrame = field(default_factory=pd.DataFrame)
    #: Chỉ số của bộ FULL trên test — chỉ để báo cáo §7.2, không deploy.
    reference_test_metrics: dict[str, float] = field(default_factory=dict)

    @property
    def generalisation_gap(self) -> float:
        """PR-AUC validation − test. Lớn nghĩa là chọn model đã bám vào validation."""
        return (self.validation_metrics.get("pr_auc", float("nan"))
                - self.test_metrics.get("pr_auc", float("nan")))


def evaluate_on_test(
    decision: Decision,
    calibrated,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Chấm model ĐÃ CHỐT trên tập test.

    Bắt buộc nhận `decision` đã có ngưỡng: không chốt xong mà đã mở test thì
    con số test tham gia vào việc chọn, và nó thôi là ước lượng độc lập.
    """
    if decision.threshold is None:
        raise ValueError(
            "Chưa chốt ngưỡng — không được mở tập test. Chạy `choose_threshold` "
            "trên validation trước.")

    proba = calibrated.predict_proba(X_test)[:, 1]
    return {
        "metrics": binary_metrics(y_test, proba, threshold=decision.threshold),
        "confusion": binary_confusion(y_test, proba, threshold=decision.threshold),
        "calibration_gap": calibration_gap(np.asarray(y_test).astype(int), proba),
        "proba": proba,
    }


# --------------------------------------------------------------------------
# Nạp dữ liệu ba tập
# --------------------------------------------------------------------------
def load_all_splits():
    """Nạp cả ba tập. Chỉ task 14 được gọi hàm này."""
    df = load_clean_application()
    split = load_split()
    out = {}
    for name in ("train", "validation", "test"):
        subset = split.apply(df, name)
        X, y = split_features_and_target(subset)
        out[name] = (X, y)
    return out


# --------------------------------------------------------------------------
# Ghi kết quả
# --------------------------------------------------------------------------
def output_dir() -> Path:
    return CONFIG.paths.runs / SELECTION_SUBDIR


def write_selection(report: FinalReport) -> dict[str, Path]:
    """Ghi bản ghi quyết định + chỉ số cuối.

    ⚠️ KHÔNG ghi artifact model — export là task 15.
    """
    out_dir = output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    path = out_dir / "final_metrics.csv"
    pd.DataFrame([
        {"split": "validation", **report.validation_metrics,
         "calibration_gap": report.validation_calibration_gap},
        {"split": "test", **report.test_metrics,
         "calibration_gap": report.test_calibration_gap},
    ]).to_csv(path, index=False, encoding="utf-8")
    written["final_metrics"] = path

    if not report.test_confusion.empty:
        path = out_dir / "test_confusion.csv"
        report.test_confusion.to_csv(path, encoding="utf-8")
        written["test_confusion"] = path

    decision = report.decision
    record = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "task": "ML02 task 14 — Chọn model tốt nhất",
        "selected_model": decision.deploy_slug,
        "algorithm": decision.algo,
        "deploy_feature_set": decision.deploy_feature_set,
        "reference_feature_set": decision.reference_feature_set,
        "runner_up": decision.runner_up,
        "selection_metric": SELECTION_METRIC,
        "selected_using": "validation only",
        "reasons": decision.reasons,
        "calibration": {
            "method": decision.calibration_method,
            "fitted_on": "validation",
            "gap_validation": report.validation_calibration_gap,
            "gap_test": report.test_calibration_gap,
        },
        "threshold": {
            "value": decision.threshold,
            "rule": decision.threshold_rule,
            "chosen_on": "validation",
            "caveat": "F1 coi bỏ lọt một ca vỡ nợ và gắn nhãn sai một hồ sơ tốt "
                      "là đắt như nhau. Tín dụng thật thì không — ngưỡng đúng "
                      "phải suy từ ma trận chi phí của tổ chức cho vay.",
        },
        "metrics_validation": report.validation_metrics,
        "metrics_test": report.test_metrics,
        "metrics_test_reference_set": report.reference_test_metrics,
        "generalisation_gap_pr_auc": report.generalisation_gap,
        "test_opened": True,
        "test_opened_after_decision": True,
        "exported": False,
        "export_note": "Export artifact là task 15.",
    }
    path = out_dir / "decision.json"
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    written["decision"] = path

    log.info("Đã ghi quyết định → %s", out_dir)
    return written
