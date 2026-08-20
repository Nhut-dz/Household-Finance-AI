"""ML02 task 15 — Export model (F04 · M04 · Tuần 4).

Task cuối của F04. Đóng gói model đã chốt ở task 14 thành một artifact dùng
được ở tầng inference, kèm đủ metadata để lần sau không ai phải đoán.

Artifact phải TỰ CHỨA, không phải chỉ có model
------------------------------------------------
Một file `.joblib` chỉ chứa estimator là artifact chưa dùng được: bên gọi vẫn
phải tự dựng feature, tự nhớ thứ tự cột, tự nhớ ngưỡng. Ba thứ đó mà nhớ sai
thì model **vẫn chạy và vẫn trả xác suất** — chỉ có điều xác suất đó vô nghĩa.

`Ml02CreditRiskModel` vì vậy gói cả bốn phần:

    1. Pipeline feature (task 3)   nối bureau → dựng tỉ lệ → tiền xử lý
    2. Model đã train (task 10)    XGBoost, `scale_pos_weight` từ tập train
    3. Lớp hiệu chuẩn (task 14)    isotonic, fit trên validation
    4. Ngưỡng nghiệp vụ (task 14)  0,1303 — KHÔNG phải 0,5

Điểm quan trọng nhất: **`predict()` dùng ngưỡng đã chốt**, không phải quy tắc
`argmax` mặc định của sklearn (tương đương ngưỡng 0,5). Với tỉ lệ nền 8,07%,
ngưỡng 0,5 xếp gần như mọi hồ sơ vào `LOW_RISK` — model trông như đang chạy
trong khi nó không phân loại gì. Gói ngưỡng vào artifact là cách duy nhất để
tầng gọi không thể quên nó.

Nhãn trả về là chuỗi nghiệp vụ
-------------------------------
`LOW_RISK` / `HIGH_RISK` chứ không phải 0/1. Tầng `api` và tầng `llm` đọc nhãn
này ra cho người dùng; để 0/1 thì mỗi nơi tự đặt tên một kiểu, và sớm muộn có
nơi đảo ngược ý nghĩa.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from hfml.config import CONFIG
from hfml.logger import get_logger
from hfml.ml.base import BaseClassifier

log = get_logger(__name__)

#: Nhãn nghiệp vụ, theo đúng thứ tự cột của `predict_proba`.
LOW_RISK: Final[str] = "LOW_RISK"
HIGH_RISK: Final[str] = "HIGH_RISK"
RISK_LABELS: Final[tuple[str, str]] = (LOW_RISK, HIGH_RISK)

#: Nhãn tiếng Việt — tầng `llm` dùng để diễn đạt, không tự dịch lại.
RISK_LABELS_VI: Final[dict[str, str]] = {
    LOW_RISK: "Rủi ro thấp",
    HIGH_RISK: "Rủi ro cao",
}

#: Phiên bản artifact. `vfinal` theo đúng quy ước ML01 (`ml01_xgboost_vfinal`).
VERSION: Final[str] = "final"


class Ml02CreditRiskModel(BaseClassifier):
    """Model ML02 đã chốt, tự chứa mọi thứ cần cho inference.

    Không có `fit()` thật: model được train ở task 10 và hiệu chuẩn ở task 14.
    Lớp này chỉ ĐÓNG GÓI, và `fit()` ném lỗi để không ai vô tình train lại một
    artifact đã chốt — train lại thì mọi con số trong `metadata.json` mô tả một
    model khác.
    """

    task = "ml02"

    def __init__(
        self,
        calibrated,
        feature_names: list[str],
        threshold: float,
        algo: str = "xgboost",
        feature_set: str = "reduced",
        version: str = VERSION,
    ):
        self.calibrated = calibrated
        self.feature_names_ = list(feature_names)
        self.threshold = float(threshold)
        self.algo = algo
        self.feature_set = feature_set
        self.version = version
        self.classes_ = list(RISK_LABELS)

    # -- Giao diện BaseClassifier -----------------------------------------
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Ml02CreditRiskModel":
        raise NotImplementedError(
            "Artifact đã chốt ở task 14 — không train lại. Muốn model mới thì "
            "chạy lại task 7–15, đừng fit lên bản export.")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Xác suất hai lớp, cột 1 là `HIGH_RISK`. ĐÃ hiệu chuẩn."""
        return self.calibrated.predict_proba(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Nhãn nghiệp vụ, cắt tại **ngưỡng đã chốt** chứ không phải 0,5.

        Đây là khác biệt duy nhất so với `predict()` của sklearn, và là lý do
        chính lớp này tồn tại: sklearn cắt ở 0,5, mà với tỉ lệ nền 8,07% thì
        0,5 xếp gần như mọi hồ sơ vào `LOW_RISK`.
        """
        risky = self.risk_probability(X) >= self.threshold
        return np.where(risky, HIGH_RISK, LOW_RISK)

    # -- Tiện ích cho tầng api / llm ---------------------------------------
    def risk_probability(self, X: pd.DataFrame) -> np.ndarray:
        """Xác suất gặp khó khăn trả nợ — cột `HIGH_RISK`."""
        return self.predict_proba(X)[:, 1]

    def explain(self, X: pd.DataFrame) -> list[dict]:
        """Kết quả cho từng hồ sơ, dạng tầng `llm` dùng được ngay.

        Trả cả `threshold` trong từng bản ghi: nói "rủi ro cao" mà không cho
        biết ngưỡng nào phân định là một khẳng định không kiểm chứng được.
        """
        proba = self.risk_probability(X)
        return [{
            "label": HIGH_RISK if p >= self.threshold else LOW_RISK,
            "label_vi": RISK_LABELS_VI[
                HIGH_RISK if p >= self.threshold else LOW_RISK],
            "probability": float(p),
            "threshold": self.threshold,
            "model_version": self.slug,
        } for p in proba]


# --------------------------------------------------------------------------
# Metadata
# --------------------------------------------------------------------------
def build_metadata(
    model: Ml02CreditRiskModel,
    decision: dict,
    dataset_manifest: dict | None = None,
) -> dict:
    """Metadata đi kèm artifact.

    Có `feature_names` theo ĐÚNG THỨ TỰ vì F06 task 3 đối chiếu danh sách này
    lúc inference — thứ tự cột sai là lỗi im lặng.

    Có `data_version` (SHA-256 của dataset) vì không có nó thì ba tháng sau
    không ai chứng minh được model này train trên đúng file nào, và §4.3 đã
    dựng sẵn manifest cho việc đó.
    """
    return {
        "task": "ML02 — Home Credit Risk Classification",
        "artifact_kind": "final_export",
        "slug": model.slug,
        "algo": model.algo,
        "feature_set": model.feature_set,
        "version": model.version,
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "random_seed": CONFIG.random_seed,

        # -- Hợp đồng inference ------------------------------------------
        "feature_names": model.feature_names_,
        "n_features": len(model.feature_names_),
        "label_mapping": {
            "0": LOW_RISK,
            "1": HIGH_RISK,
            "labels_vi": RISK_LABELS_VI,
            "positive_class": HIGH_RISK,
            "proba_column_1": HIGH_RISK,
        },
        "threshold": {
            "value": model.threshold,
            "rule": decision.get("threshold", {}).get("rule", ""),
            "chosen_on": "validation",
            "caveat": decision.get("threshold", {}).get("caveat", ""),
            "note": "predict() dùng ngưỡng này, KHÔNG dùng argmax 0,5 của "
                    "sklearn. Tỉ lệ nền 8,07% nên 0,5 xếp gần như mọi hồ sơ "
                    "vào LOW_RISK.",
        },
        "calibration": decision.get("calibration", {}),

        # -- Cấu hình model ------------------------------------------------
        "model_config": _model_config(model),

        # -- Chỉ số ---------------------------------------------------------
        "metrics_validation": decision.get("metrics_validation", {}),
        "metrics_test": decision.get("metrics_test", {}),
        "selection_metric": decision.get("selection_metric", "pr_auc"),
        "selection_reasons": decision.get("reasons", []),

        # -- Truy vết dữ liệu ------------------------------------------------
        "data_version": _data_version(dataset_manifest),

        # -- Giới hạn sử dụng -------------------------------------------------
        "limitations": [
            "Ước lượng THAM KHẢO, không phải kết quả thẩm định tín dụng và "
            "không thay thế quyết định của tổ chức cho vay.",
            "Train trên Home Credit (không phải dữ liệu Việt Nam). Mọi feature "
            "tiền tệ là TỈ LỆ để bất biến với đơn vị tiền — xem §2.1.",
            "Ngưỡng chọn bằng F1, tức coi bỏ lọt một ca vỡ nợ và gắn nhãn sai "
            "một hồ sơ tốt là đắt như nhau. Tín dụng thật thì không.",
            "`dti` của Home Credit là kỳ trả của khoản ĐANG XIN VAY, còn của "
            "form là khoản nợ ĐANG CÓ — hai khoản nợ khác nhau.",
            "Ô 'số lần trả chậm' của form ánh xạ sang số KHOẢN đang quá hạn ở "
            "bureau, không phải số LẦN trả chậm.",
        ],
    }


def _model_config(model: Ml02CreditRiskModel) -> dict:
    """Siêu tham số của estimator bên trong, dạng chuỗi để JSON hoá được."""
    try:
        inner = model.calibrated.estimator.estimator.named_steps["model"]
    except AttributeError:
        return {}
    return {k: str(v) for k, v in inner.get_params(deep=False).items()}


def _data_version(manifest: dict | None) -> dict:
    """SHA-256 rút gọn của các file dataset, lấy từ manifest §4.3."""
    if not manifest:
        return {}
    return {
        info["file"]: info.get("sha256", "")[:16]
        for info in manifest.get("files", {}).values()
    }


# --------------------------------------------------------------------------
# Ghi và nạp lại
# --------------------------------------------------------------------------
def export(
    model: Ml02CreditRiskModel,
    decision: dict,
    dataset_manifest: dict | None = None,
    directory: Path | None = None,
) -> dict[str, Path]:
    """Ghi artifact + metadata theo đúng quy ước `hfml.ml.registry`."""
    import joblib

    out_dir = directory or CONFIG.paths.runs
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / f"{model.slug}.joblib"
    joblib.dump(model, model_path)

    metadata_path = out_dir / f"{model.slug}.metadata.json"
    metadata_path.write_text(
        json.dumps(build_metadata(model, decision, dataset_manifest),
                   ensure_ascii=False, indent=2),
        encoding="utf-8")

    log.info("Đã export → %s", model_path)
    return {"model": model_path, "metadata": metadata_path}


def verify_export(
    slug: str,
    X_sample: pd.DataFrame,
    expected_proba: np.ndarray,
    directory: Path | None = None,
) -> dict:
    """Nạp lại artifact và kiểm nó cho ĐÚNG kết quả cũ.

    Không kiểm bước này thì "đã export" chỉ nghĩa là "đã ghi ra một file" —
    file hỏng, thiếu phụ thuộc, hay lệch phiên bản thư viện đều chỉ lộ ra khi
    có người thật gọi tới, mà lúc đó là lúc tệ nhất.

    Kiểm CẢ BỐN thứ, vì mỗi thứ hỏng theo một kiểu riêng:
        · nạp được không            → file/phụ thuộc
        · xác suất có trùng không   → model và pipeline
        · thứ tự feature có khớp    → hợp đồng inference (F06 task 3)
        · ngưỡng có được áp không   → `predict` dùng 0,5 hay dùng ngưỡng chốt
    """
    from hfml.ml.registry import load_metadata, load_model

    loaded = load_model(slug, directory)
    metadata = load_metadata(slug, directory)

    proba = loaded.risk_probability(X_sample)
    labels = loaded.predict(X_sample)

    return {
        "loaded": True,
        "proba_matches": bool(np.allclose(proba, expected_proba)),
        "max_proba_diff": float(np.abs(proba - expected_proba).max()),
        "feature_names_match": (
            metadata["feature_names"] == loaded.feature_names_),
        "n_features": len(loaded.feature_names_),
        "threshold_applied": bool(
            np.array_equal(labels,
                           np.where(proba >= loaded.threshold, HIGH_RISK, LOW_RISK))),
        "threshold_is_not_half": loaded.threshold != 0.5,
        "labels_returned": sorted(set(labels.tolist())),
        "slug": loaded.slug,
    }
