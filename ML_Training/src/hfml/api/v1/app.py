"""AI-04 task 3, 6 — Dựng ứng dụng FastAPI (F05 · M08).

Nạp model lúc KHỞI ĐỘNG, không phải lúc có request đầu tiên
-------------------------------------------------------------
Nạp lười thì người dùng đầu tiên sau mỗi lần triển khai phải chờ thêm vài trăm
mili-giây đọc artifact — và tệ hơn, một artifact thiếu chỉ lộ ra khi có người
thật gọi vào. Nạp lúc khởi động thì `/health` nói ra ngay lập tức.

Thiếu artifact KHÔNG được làm chết tiến trình
-----------------------------------------------
Ném ngoại lệ trong lifespan thì service không lên nổi, và khi đó **không còn
gì để hỏi trạng thái** — người vận hành chỉ thấy container restart liên tục mà
không biết vì sao. Nên lỗi lúc khởi động được ghi log rồi đi tiếp; `/health`
báo `degraded` hoặc `unhealthy` kèm lý do cụ thể.

Đó là lý do lifespan này chỉ *cố gắng* nạp chứ không *bắt buộc* nạp thành công.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from hfml.api.v1.config import API_PREFIX, API_SETTINGS, API_VERSION
from hfml.api.v1.errors import register_error_handlers
from hfml.api.v1.routes import infra_router, router
from hfml.logger import get_logger

log = get_logger(__name__)

TITLE = "Household Finance AI Service"

DESCRIPTION = """\
API phục vụ AI tư vấn tài chính hộ gia đình.

Tầng này CHỈ làm việc phơi API: nhận request, gọi module `hfml.inference`, và
chiếu kết quả sang schema công khai. Toàn bộ nghiệp vụ — quy tắc RB01–RB05,
hai model, tầng diễn đạt — nằm trong module và test được mà không cần HTTP.
"""


def _warm_up() -> None:
    """Nạp sẵn model và khởi tạo client LLM.

    Mỗi thành phần được thử RIÊNG: thiếu ML02 không được ngăn ML01 nạp, vì hai
    model phục vụ hai câu hỏi khác nhau và mất một cái không làm hỏng cái kia.
    """
    from hfml.inference.lifecycle import MANAGER, ModelUnavailable
    from hfml.inference.settings import ML01, ML02

    for name in (ML01, ML02):
        try:
            entry = MANAGER.get(name)
            log.info("Khởi động: đã nạp %s (%s)", name.upper(), entry.slug)
        except ModelUnavailable as exc:
            # Không dừng — `/health` sẽ báo. Xem docstring đầu file.
            log.warning("Khởi động: chưa nạp được %s — %s", name.upper(), exc)
        except Exception as exc:  # noqa: BLE001 — biên khởi động
            log.exception("Khởi động: lỗi khi nạp %s: %s", name.upper(), exc)

    # Khởi tạo client LLM một lần để lượt gọi đầu không phải chờ dựng client.
    # Thiếu API key KHÔNG phải lỗi: tầng LLM tự chạy chế độ template.
    try:
        from hfml.llm import client

        log.info("Khởi động: LLM %s",
                 "đã cấu hình" if client.is_llm_available()
                 else "chưa cấu hình — sẽ dùng câu trả lời dựng từ dữ liệu")
    except Exception as exc:  # noqa: BLE001
        log.warning("Khởi động: không khởi tạo được client LLM: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Khởi động %s (API v%s)", TITLE, API_VERSION)
    if API_SETTINGS.warm_up_models:
        _warm_up()
    else:
        log.info("Bỏ qua nạp sẵn model (HFML_API_WARM_UP=0).")
    yield
    log.info("Dừng %s", TITLE)


def create_app() -> FastAPI:
    """Dựng ứng dụng. Hàm factory để test dựng được bản riêng của mình."""
    app = FastAPI(title=TITLE, description=DESCRIPTION, version=API_VERSION,
                  lifespan=lifespan)

    register_error_handlers(app)

    if API_SETTINGS.cors_origins:
        from fastapi.middleware.cors import CORSMiddleware

        # Khai đúng origin, không dùng `"*"`: hồ sơ tài chính không nên mở cho
        # mọi trang web gọi bằng trình duyệt của người dùng.
        app.add_middleware(
            CORSMiddleware,
            allow_origins=API_SETTINGS.cors_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )
        log.info("CORS bật cho: %s", ", ".join(API_SETTINGS.cors_origins))

    app.include_router(router, prefix=API_PREFIX, tags=["v1"])

    # `/health` cũng gắn ở gốc: bộ điều phối hạ tầng (Docker, k8s, load
    # balancer) trỏ vào một đường cố định và không nên phải biết phiên bản API.
    #
    # Gắn RIÊNG một router chỉ có health, không mount lại cả `router`. Mount
    # lại cả router thì `/inference` và `/chat` cũng xuất hiện ở gốc — tức có
    # một đường đi vòng qua phiên bản, và client lỡ dùng nó sẽ hỏng đúng vào
    # lúc v2 ra đời mà không ai lường trước.
    app.include_router(infra_router, prefix="", tags=["infra"],
                       include_in_schema=False)

    return app


app = create_app()
