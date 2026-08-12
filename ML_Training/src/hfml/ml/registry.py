"""Lưu & tải artifact đã train (F03 task 15, F04 task 15).

Mỗi artifact gồm hai file trong `src/training/runs/` (`CONFIG.paths.runs`):

    ml02_xgboost_reduced_v1.joblib         preprocessing Pipeline + estimator
    ml02_xgboost_reduced_v1.metadata.json  seed, ngày train, metric, feature list

`metadata.json` không phải để cho đẹp — F06 task 3 kiểm tra thứ tự feature
lúc inference khớp với danh sách trong file này. Sai thứ tự cột là lỗi im
lặng: model vẫn chạy, vẫn trả xác suất, chỉ có điều xác suất đó vô nghĩa.

Một thư mục mặc định, không hai
-------------------------------
Cả bốn hàm dưới đây mặc định trỏ vào `CONFIG.paths.runs`, và `directory` chỉ
để ghi/đọc ở nơi khác khi cần (test dùng thư mục tạm).

Trước 12/08/2026 mặc định là `artifacts/` trong khi các lần train ML01 ghi
vào `runs/`. Hậu quả cụ thể: `load_metadata()` không đọc nổi metadata của
chính bản export cuối, `list_artifacts()` chỉ thấy 1 trong 5 artifact, và
`load_model()` vẫn nạp được từ `artifacts/` — nghĩa là một bản cũ còn sót ở
đó sẽ được nạp mà KHÔNG báo lỗi gì. Đúng loại lỗi im lặng nói ở trên, chỉ
khác là ở tầng file thay vì tầng cột.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib

from hfml.config import CONFIG
from hfml.logger import get_logger
from hfml.ml.base import BaseClassifier

log = get_logger("hfml.ml.registry")


def save_model(
    model: BaseClassifier,
    metrics: dict | None = None,
    directory: Path | None = None,
    extra: dict | None = None,
) -> Path:
    """Lưu model + metadata. Trả về đường dẫn file .joblib.

    `directory` để trống thì ghi vào `CONFIG.paths.runs`; truyền tham số này
    khi cần ghi nơi khác (test dùng thư mục tạm để không đè output thật).

    `extra` được trộn thẳng vào `metadata.json` — chỗ để ghi cấu hình lần
    chạy (siêu tham số, số fold, tỉ lệ test). Cấu hình KHÔNG nhét vào
    `metrics`: hai thứ khác loại, gộp lại thì bảng chỉ số hết đọc được.
    """
    artifacts = directory or CONFIG.paths.runs
    artifacts.mkdir(parents=True, exist_ok=True)
    model_path = artifacts / f"{model.slug}.joblib"

    joblib.dump(model, model_path)
    meta = {
        "task": model.task,
        "algo": model.algo,
        "feature_set": model.feature_set,
        "version": model.version,
        "random_seed": CONFIG.random_seed,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "feature_names": list(getattr(model, "feature_names_", [])),
        "classes": list(getattr(model, "classes_", [])),
        "metrics": metrics or {},
        **(extra or {}),
    }
    (artifacts / f"{model.slug}.metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log.info("Đã lưu artifact → %s", model_path)
    return model_path


def load_model(slug: str, directory: Path | None = None) -> BaseClassifier:
    """Tải model theo slug, ví dụ `ml01_random_forest_v1`.

    `directory` để trống thì đọc từ `CONFIG.paths.runs` — đối xứng với
    `save_model`, nên lưu ở đâu thì tải lại từ đúng đó.
    """
    path = (directory or CONFIG.paths.runs) / f"{slug}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy artifact: {path}. Cần train trước.")
    return joblib.load(path)


def load_metadata(slug: str, directory: Path | None = None) -> dict:
    """Đọc metadata.json đi kèm — dùng để kiểm tra thứ tự feature."""
    path = (directory or CONFIG.paths.runs) / f"{slug}.metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy metadata: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def list_artifacts(task: str | None = None,
                   directory: Path | None = None) -> list[str]:
    """Liệt kê slug của các artifact đã có, lọc theo task nếu cần."""
    pattern = f"{task}_*.joblib" if task else "*.joblib"
    return sorted(p.stem for p in (directory or CONFIG.paths.runs).glob(pattern))
