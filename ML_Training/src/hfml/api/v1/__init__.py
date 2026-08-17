"""API v1 — phơi `hfml.inference` qua HTTP (Epic AI-04 · F05 · M08).

    POST /api/v1/inference   Rule + ML, không gọi LLM
    POST /api/v1/chat        trọn pipeline tới câu trả lời
    GET  /api/v1/health      healthy · degraded · unhealthy
    GET  /health             cùng nội dung, đường cố định cho hạ tầng

Bố cục
-------
    schemas.py   hợp đồng công khai — CHIẾU từ cấu trúc nội bộ, không sao chép
    config.py    tham số phục vụ — phiên bản, CORS, timeout
    errors.py    mọi đường hỏng ra cùng một vỏ `ErrorResponse`
    health.py    ba trạng thái và ranh giới giữa chúng
    routes.py    ba endpoint, mỗi cái gọi đúng một hàm của module
    app.py       factory + nạp model lúc khởi động

Tầng này KHÔNG chứa nghiệp vụ
-------------------------------
Không có phép tính, không có luật, không có quyết định nhãn. Tất cả nằm ở
`hfml.inference` và test được mà không cần dựng HTTP client.

Hiện có HAI ứng dụng FastAPI — đọc kỹ trước khi triển khai
------------------------------------------------------------
    hfml.api.main:app     cũ — /advise, /predict, /health   (Laravel đang gọi)
    hfml.api.v1.app:app   mới — /api/v1/*, /health           (Epic AI-04)

Chúng CỐ Ý tách rời và chưa gộp. Gộp được về mặt kỹ thuật, nhưng hai bên cùng
khai `/health` với hai hình dạng thân response khác nhau (`status: "ok"` so với
`status: "healthy"`), nên bản nào đăng ký trước sẽ lặng lẽ che bản kia — và
phía Laravel đang đọc hình dạng cũ.

Gộp là một quyết định triển khai, không phải chi tiết kỹ thuật: nó cần đổi cả
phía gọi. Chừng nào chưa đổi, chạy hai service hoặc chọn đúng một bản theo
client đang phục vụ.
"""
from hfml.api.v1.app import app, create_app
from hfml.api.v1.config import API_PREFIX, API_SETTINGS, API_VERSION

__all__ = ["app", "create_app", "API_PREFIX", "API_VERSION", "API_SETTINGS"]
