"""AI-03 task 4 — Vòng đời model (F05 · M07).

Một nơi duy nhất chịu trách nhiệm: nạp artifact, giữ trong bộ nhớ, báo trạng
thái, và cho phép đổi model lúc đang chạy.

Nạp LƯỜI và nạp MỘT LẦN
------------------------
Lười: thiếu artifact chỉ làm hỏng đúng model đó chứ không làm chết service lúc
khởi động — `health()` vẫn phải nói ra được là thiếu gì.

Một lần: đọc XGBoost từ đĩa tốn vài trăm mili-giây, và đó là phần lớn thời gian
trả lời nếu nạp lại mỗi request.

Vì sao KHÔNG dùng `lru_cache` như bản trước
---------------------------------------------
`lru_cache` gắn chặt bộ nhớ đệm vào tên hàm, nên muốn đổi slug lúc đang chạy
thì phải gọi `get_ml01.cache_clear()` — tức nơi gọi phải biết chi tiết cài đặt
của nơi bị gọi. Đổi sang một `dict` do `ModelManager` giữ thì việc đổi model
thành một lời gọi bình thường (`swap`), và cùng cơ chế đó phục vụ luôn cho
`reload` sau khi train xong bản mới.

Thiếu model KHÔNG phải lỗi lập trình
-------------------------------------
`ModelUnavailable` là trạng thái vận hành bình thường: artifact chưa train,
chưa copy sang máy chủ, hoặc slug trong cấu hình sai. Tầng trên bắt nó và trả
về kết quả thiếu đúng phần đó kèm lý do — phần rule vẫn chạy độc lập.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Callable

from hfml.inference.settings import ML01, ML02, SETTINGS
from hfml.logger import get_logger
from hfml.ml.registry import load_model
from hfml.pipeline.adapters import neutralise_bureau_lookup

log = get_logger(__name__)


class ModelUnavailable(RuntimeError):
    """Không nạp được model. Trạng thái vận hành, không phải lỗi lập trình."""


@dataclass
class LoadedModel:
    """Model đã nạp, kèm đúng slug đã nạp nó.

    Giữ `slug` riêng thay vì đọc `model.slug`: khi cấu hình trỏ sai và ai đó
    đổi slug lúc đang chạy, thứ cần truy là "đã nạp từ tên nào", không phải
    "artifact tự khai tên gì".
    """

    name: str
    slug: str
    model: Any

    @property
    def threshold(self) -> float | None:
        value = getattr(self.model, "threshold", None)
        return float(value) if value is not None else None


def _prepare_ml02(model: Any) -> Any:
    """Tắt tra cứu bureau ngay khi nạp ML02.

    Bảng tra trong artifact là của khách hàng Home Credit; để nguyên thì nó GHI
    ĐÈ mục C mà form vừa thu được — xem `adapters.neutralise_bureau_lookup`.
    Đặt ở đây để không nơi nào nạp được ML02 mà quên bước này.
    """
    neutralise_bureau_lookup(model)
    return model


#: Việc phải làm thêm sau khi nạp, theo từng model.
_PREPARE: dict[str, Callable[[Any], Any]] = {ML02: _prepare_ml02}


class ModelManager:
    """Giữ model đã nạp và cho phép đổi chúng lúc đang chạy."""

    def __init__(self) -> None:
        self._loaded: dict[str, LoadedModel] = {}
        # Nạp model không phải thao tác nguyên tử: hai request cùng tới lúc
        # đệm còn rỗng sẽ cùng đọc đĩa. Không sai kết quả, nhưng tốn gấp đôi
        # thời gian ở đúng lúc tệ nhất — lúc service vừa khởi động.
        self._lock = threading.Lock()

    def get(self, name: str) -> LoadedModel:
        """Model đang phục vụ. Nạp nếu chưa có trong bộ nhớ."""
        slug = SETTINGS.slug_for(name)

        cached = self._loaded.get(name)
        # Cấu hình đổi slug thì bản đang giữ đã cũ — nạp lại theo slug mới.
        if cached is not None and cached.slug == slug:
            return cached

        with self._lock:
            cached = self._loaded.get(name)
            if cached is not None and cached.slug == slug:
                return cached

            try:
                model = load_model(slug, directory=SETTINGS.artifact_dir)
            except FileNotFoundError as exc:
                raise ModelUnavailable(
                    f"Chưa có artifact {name.upper()} ({slug}). {exc}") from exc
            except Exception as exc:  # noqa: BLE001 — đọc đĩa, giải mã pickle
                raise ModelUnavailable(
                    f"Không nạp được {name.upper()} ({slug}): "
                    f"{type(exc).__name__}: {exc}") from exc

            prepare = _PREPARE.get(name)
            if prepare:
                model = prepare(model)

            entry = LoadedModel(name=name, slug=slug, model=model)
            self._loaded[name] = entry
            log.info("Đã nạp %s: %s%s", name.upper(), slug,
                     f" (ngưỡng {entry.threshold:.4f})"
                     if entry.threshold is not None else "")
            return entry

    def threshold_for(self, name: str) -> float | None:
        """Ngưỡng đang dùng — cấu hình ghi đè nếu có, không thì lấy từ artifact.

        Tập trung ở một chỗ để không có bước nào lỡ đọc thẳng `model.threshold`
        và bỏ qua phần ghi đè, gây ra hai ngưỡng khác nhau trong cùng một lượt
        dự đoán: một cái dùng để phân nhãn, một cái đem đi báo cáo.
        """
        if name == ML02 and SETTINGS.ml02_threshold is not None:
            return float(SETTINGS.ml02_threshold)
        try:
            return self.get(name).threshold
        except ModelUnavailable:
            return None

    def reload(self, name: str | None = None) -> None:
        """Bỏ bản đang giữ để lượt gọi sau nạp lại từ đĩa.

        Dùng sau khi train xong bản mới và ghi đè lên cùng slug.
        """
        with self._lock:
            if name is None:
                self._loaded.clear()
                log.info("Đã bỏ toàn bộ model khỏi bộ nhớ — sẽ nạp lại khi cần.")
            else:
                self._loaded.pop(name, None)
                log.info("Đã bỏ %s khỏi bộ nhớ — sẽ nạp lại khi cần.", name)

    def swap(self, name: str, slug: str) -> LoadedModel:
        """Đổi sang artifact khác lúc đang chạy.

        Nạp bản mới TRƯỚC rồi mới thay: nếu slug mới hỏng thì ngoại lệ bay ra
        và bản cũ vẫn đang phục vụ. Thay trước rồi nạp sau thì một slug gõ sai
        làm service mất luôn model đang chạy được.
        """
        current = SETTINGS.slug_for(name)
        attribute = "ml01_slug" if name == ML01 else "ml02_slug"
        setattr(SETTINGS, attribute, slug)
        try:
            entry = self.get(name)
        except ModelUnavailable:
            setattr(SETTINGS, attribute, current)
            raise
        log.info("Đã đổi %s: %s → %s", name.upper(), current, slug)
        return entry

    def status(self) -> dict:
        """Trạng thái từng model — cho `/health`.

        Service sống mà thiếu artifact thì inference hỏng, và người vận hành
        cần biết trước khi có người dùng gọi tới rồi mới phát hiện.
        """
        report = {}
        for name in (ML01, ML02):
            slug = SETTINGS.slug_for(name)
            try:
                entry = self.get(name)
            except ModelUnavailable as exc:
                report[name] = {"loaded": False, "slug": slug, "error": str(exc)}
                continue

            report[name] = {
                "loaded": True,
                "slug": entry.slug,
                "n_features": len(getattr(entry.model, "feature_names_", [])),
            }
            threshold = self.threshold_for(name)
            if threshold is not None:
                report[name]["threshold"] = threshold
                report[name]["threshold_overridden"] = (
                    name == ML02 and SETTINGS.ml02_threshold is not None)
        return report


#: Bộ quản lý dùng chung cho cả tiến trình.
MANAGER = ModelManager()


def get_model(name: str) -> LoadedModel:
    return MANAGER.get(name)


def model_status() -> dict:
    return MANAGER.status()
