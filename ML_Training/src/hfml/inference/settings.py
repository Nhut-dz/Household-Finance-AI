"""AI-03 task 5 — Cấu hình runtime của module inference (F05 · M07).

Mọi thứ có thể đổi khi vận hành mà KHÔNG phải sửa mã nguồn đều nằm ở đây:
slug artifact, ngưỡng, tham số LLM, số lượt lịch sử, thư mục artifact.

Vì sao slug model phải là cấu hình chứ không phải hằng số
----------------------------------------------------------
Trước AI-03 slug được khai bằng hằng số ở HAI nơi, và hai nơi đã lệch nhau:

    pipeline/predictor.py   ML02_SLUG = "ml02_xgboost_reduced_vfinal"   ← có thật
    api/main.py             ML02_SLUG = "ml02_best_reduced_vfinal"      ← KHÔNG có

Bản ở `api/main.py` trỏ vào một artifact không tồn tại trên đĩa. Không ai phát
hiện được vì nhánh dùng tới nó vẫn đang `raise HTTPException` trước khi chạm
vào model. Đó chính là cái giá của việc để cấu hình nằm trong mã: hai bản sao
trôi khỏi nhau lặng lẽ, và bản sai chỉ lộ ra khi có người dùng thật gọi vào.

Ngưỡng ML02 mặc định là None, và đó là chủ ý
----------------------------------------------
Ngưỡng 0.1303 được chốt ở F04 task 14 trên tập validation rồi đóng vào
artifact. `ml02_threshold: null` nghĩa là **dùng ngưỡng đã chốt trong
artifact** — đúng thứ đã được đánh giá.

Cho phép ghi đè vì vận hành đôi khi cần siết tỉ lệ cảnh báo, nhưng ghi đè là
một quyết định đánh đổi precision/recall chứ không phải một nút chỉnh, nên nó
được ghi log ở mức WARNING mỗi lần nạp.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from hfml.config import CONFIG
from hfml.logger import get_logger

log = get_logger(__name__)

#: Tên hai model trong toàn bộ module. Dùng chuỗi thay vì enum vì chúng còn là
#: khoá trong structured result gửi ra ngoài.
ML01: Final[str] = "ml01"
ML02: Final[str] = "ml02"


def _env_str(name: str, fallback: str) -> str:
    return os.getenv(name, fallback)


def _env_float(name: str, fallback: float | None) -> float | None:
    raw = os.getenv(name)
    return float(raw) if raw else fallback


def _env_int(name: str, fallback: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw else fallback


@dataclass
class InferenceSettings:
    """Toàn bộ tham số runtime của module. Không có gì hardcode ngoài file này."""

    #: Slug artifact đang phục vụ. Đổi ở config/env là đổi model, không sửa mã.
    ml01_slug: str = "ml01_xgboost_vfinal"
    ml02_slug: str = "ml02_xgboost_reduced_vfinal"

    #: Thư mục chứa artifact. None → `CONFIG.paths.runs`.
    artifact_dir: Path | None = None

    #: None → dùng ngưỡng đã chốt trong artifact. Xem docstring đầu file.
    ml02_threshold: float | None = None

    #: max(predict_proba) dưới ngưỡng này thì gắn cờ low_confidence (§8.1).
    confidence_threshold: float = 0.60

    #: Số lượt hội thoại gần nhất đưa vào context.
    history_turns: int = 3

    #: Số lần gọi lại LLM khi câu trả lời không qua kiểm.
    llm_max_retries: int = 1

    #: Bật/tắt tầng LLM mà không cần gỡ API key — hữu ích khi đo tốc độ hoặc
    #: khi muốn chắc chắn câu trả lời chỉ đến từ rule và ML.
    llm_enabled: bool = True

    llm: dict = field(default_factory=dict)

    def slug_for(self, model: str) -> str:
        return self.ml01_slug if model == ML01 else self.ml02_slug

    def to_dict(self) -> dict:
        return {
            "ml01_slug": self.ml01_slug,
            "ml02_slug": self.ml02_slug,
            "artifact_dir": str(self.artifact_dir) if self.artifact_dir else None,
            "ml02_threshold": self.ml02_threshold,
            "confidence_threshold": self.confidence_threshold,
            "history_turns": self.history_turns,
            "llm_max_retries": self.llm_max_retries,
            "llm_enabled": self.llm_enabled,
            "llm_model": self.llm.get("model", ""),
        }


def load_settings(overrides: dict[str, Any] | None = None) -> InferenceSettings:
    """Dựng cấu hình: mặc định < config.yaml < biến môi trường < `overrides`.

    `overrides` đứng cuối để test và script dựng được một cấu hình riêng mà
    không phải đụng tới file hay biến môi trường của máy.
    """
    block = dict(getattr(CONFIG, "inference", {}) or {})

    settings = InferenceSettings(
        ml01_slug=_env_str("HFML_ML01_SLUG",
                           block.get("ml01_slug", InferenceSettings.ml01_slug)),
        ml02_slug=_env_str("HFML_ML02_SLUG",
                           block.get("ml02_slug", InferenceSettings.ml02_slug)),
        ml02_threshold=_env_float("HFML_ML02_THRESHOLD",
                                  block.get("ml02_threshold")),
        confidence_threshold=float(CONFIG.confidence_threshold),
        history_turns=_env_int("HFML_HISTORY_TURNS",
                               int(block.get("history_turns", 3))),
        llm_max_retries=_env_int("HFML_LLM_MAX_RETRIES",
                                 int(block.get("llm_max_retries", 1))),
        llm_enabled=os.getenv("HFML_LLM_ENABLED", "1") not in ("0", "false", "False"),
        llm=dict(CONFIG.llm),
    )

    directory = os.getenv("HFML_ARTIFACT_DIR") or block.get("artifact_dir")
    if directory:
        settings.artifact_dir = Path(directory)

    for key, value in (overrides or {}).items():
        if not hasattr(settings, key):
            raise ValueError(f"Không có tham số cấu hình `{key}`.")
        setattr(settings, key, value)

    if settings.ml02_threshold is not None:
        # Không hạ xuống INFO: đây là đánh đổi precision/recall so với ngưỡng
        # đã được đánh giá trên tập validation, người vận hành phải thấy.
        log.warning(
            "Ngưỡng ML02 bị ghi đè thành %.4f — KHÁC ngưỡng đã chốt trong "
            "artifact. Mọi nhãn LOW_RISK/HIGH_RISK sẽ lệch so với báo cáo "
            "đánh giá của F04 task 14.", settings.ml02_threshold)

    return settings


#: Cấu hình dùng chung. Gọi `reload_settings()` sau khi đổi env.
SETTINGS = load_settings()


def reload_settings(overrides: dict[str, Any] | None = None) -> InferenceSettings:
    """Nạp lại cấu hình tại chỗ, giữ nguyên đối tượng `SETTINGS`.

    Sửa tại chỗ chứ không thay tên: các module khác đã `from ... import
    SETTINGS`, nên gán lại tên ở đây sẽ để chúng ôm bản cũ mà không ai biết.
    """
    fresh = load_settings(overrides)
    SETTINGS.__dict__.update(fresh.__dict__)
    return SETTINGS
