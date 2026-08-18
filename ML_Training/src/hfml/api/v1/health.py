"""AI-04 task 4 — Health check ba trạng thái (F05 · M08).

Ranh giới giữa ba trạng thái
------------------------------
    healthy     mọi thành phần sẵn sàng
    degraded    còn phục vụ được, nhưng KÉM hơn bình thường
    unhealthy   không phục vụ được yêu cầu chính nào

Ba mức chứ không hai, vì `degraded` là trạng thái hệ thống này thật sự hay
rơi vào và nó KHÔNG phải hỏng:

    · Thiếu API key LLM  → vẫn trả lời được bằng câu dựng từ dữ liệu đã tính
    · Thiếu ML02         → vẫn chẩn đoán được sức khỏe tài chính bằng ML01
    · Thiếu ML01         → vẫn đánh giá được rủi ro khoản vay bằng ML02

Gộp `degraded` vào `unhealthy` thì bộ cân bằng tải rút service khỏi vòng phục
vụ trong khi nó vẫn trả lời tốt phần lớn câu hỏi. Gộp vào `healthy` thì không
ai biết chất lượng đã tụt cho tới khi người dùng phàn nàn.

`unhealthy` chỉ khi MẤT CẢ HAI model — khi đó không còn gì để suy luận, và
tầng quy tắc một mình không phải là thứ endpoint này hứa hẹn.

Vì sao LLM không bao giờ làm `unhealthy`
------------------------------------------
Tầng LLM chỉ DIỄN ĐẠT thứ đã tính. Mất nó thì câu trả lời khô hơn nhưng vẫn
đúng và vẫn đủ số liệu — đó đúng là định nghĩa của `degraded`.
"""
from __future__ import annotations

from typing import Final

from hfml.api.v1.config import API_VERSION
from hfml.api.v1.schemas import HealthComponent, HealthResponse
from hfml.logger import get_logger

log = get_logger(__name__)

HEALTHY: Final[str] = "healthy"
DEGRADED: Final[str] = "degraded"
UNHEALTHY: Final[str] = "unhealthy"

SERVICE_NAME: Final[str] = "Household Finance AI Service"


def _component(status: str, detail: str = "", **info) -> HealthComponent:
    return HealthComponent(status=status, detail=detail, info=info)


def build_health() -> HealthResponse:
    """Trạng thái từng thành phần và kết luận chung.

    Không ném ngoại lệ trong bất cứ ca nào: một endpoint health tự nó hỏng là
    endpoint vô dụng — đúng lúc cần nó nhất thì nó im lặng.
    """
    from hfml.inference import engine as inference_engine

    components: dict[str, HealthComponent] = {
        "api": _component(HEALTHY, "FastAPI đang phục vụ", version=API_VERSION),
    }

    try:
        report = inference_engine.health()
    except Exception as exc:  # noqa: BLE001 — health không được phép tự sập
        log.exception("Không đọc được trạng thái module inference")
        components["inference"] = _component(
            UNHEALTHY, f"Không đọc được trạng thái: {type(exc).__name__}: {exc}")
        return HealthResponse(status=UNHEALTHY, service=SERVICE_NAME,
                              api_version=API_VERSION, components=components)

    models = report.get("models", {})
    for name in ("ml01", "ml02"):
        item = models.get(name, {})
        if item.get("loaded"):
            info = {"slug": item.get("slug"),
                    "n_features": item.get("n_features")}
            if "threshold" in item:
                info["threshold"] = item["threshold"]
                info["threshold_overridden"] = item.get("threshold_overridden")
            components[name] = _component(HEALTHY, "Đã nạp", **info)
        else:
            components[name] = _component(
                UNHEALTHY, item.get("error", "Chưa nạp được artifact."))

    # Preprocessing đi kèm artifact: pipeline tiền xử lý được đóng gói CÙNG
    # model trong file .joblib (F03/F04 task 15), nên nó sẵn sàng đúng khi và
    # chỉ khi có ít nhất một model nạp được. Báo riêng để người vận hành không
    # phải suy ra điều đó từ hai dòng trên.
    loaded = [n for n in ("ml01", "ml02")
              if components[n].status == HEALTHY]
    components["preprocessing"] = (
        _component(HEALTHY, "Pipeline tiền xử lý nạp kèm artifact",
                   loaded_with=loaded)
        if loaded else
        _component(UNHEALTHY, "Không có artifact nào nạp được nên không có "
                              "pipeline tiền xử lý."))

    llm = report.get("llm", {})
    if not llm.get("enabled"):
        components["llm"] = _component(
            DEGRADED, "Tầng LLM đang tắt theo cấu hình — câu trả lời sẽ dựng "
                      "từ dữ liệu đã tính.")
    elif llm.get("available"):
        components["llm"] = _component(HEALTHY, "Đã cấu hình",
                                       model=llm.get("model"))
    else:
        components["llm"] = _component(
            DEGRADED, "Chưa cấu hình GEMINI_API_KEY — câu trả lời sẽ dựng từ "
                      "dữ liệu đã tính, vẫn đúng nhưng khô hơn.")

    return HealthResponse(status=_overall(components), service=SERVICE_NAME,
                          api_version=API_VERSION, components=components)


def _overall(components: dict[str, HealthComponent]) -> str:
    """Kết luận chung — xem docstring đầu file về ranh giới."""
    models_down = sum(1 for name in ("ml01", "ml02")
                      if components[name].status == UNHEALTHY)

    if models_down == 2:
        return UNHEALTHY
    if models_down or any(c.status != HEALTHY for c in components.values()):
        return DEGRADED
    return HEALTHY
