"""AI-04 task 5 — Xử lý lỗi thống nhất (F05 · M08).

Mọi đường hỏng đều ra đúng một hình dạng: `ErrorResponse`. Client vì vậy viết
MỘT nhánh xử lý lỗi thay vì sáu, và một loại lỗi mới thêm sau này không bắt
client sửa gì.

Mã lỗi ổn định, thông điệp thì không
--------------------------------------
`error` là thứ client dùng để phân nhánh, nên nó là hợp đồng và không đổi.
`message` là câu cho người đọc, được phép sửa cho dễ hiểu hơn bất cứ lúc nào.
Trộn hai vai đó vào một trường là buộc client so khớp chuỗi tiếng Việt — và
mọi lần sửa chính tả trở thành một thay đổi phá vỡ.

Vì sao mã HTTP phân biệt kỹ đến vậy
-------------------------------------
    422  request sai       người gọi sửa được — sai kiểu, thiếu trường
    503  chưa phục vụ được  người vận hành sửa — thiếu artifact, thiếu cấu hình
    504  quá hạn giờ        hệ thống còn sống, chỉ là chậm
    500  ngoài dự liệu      lỗi lập trình

Gộp tất cả thành 500 thì người nhận không có cách nào biết nên sửa payload,
gọi lại sau, hay báo cho người vận hành.

Không để traceback lọt ra ngoài
---------------------------------
Chi tiết nội bộ đi vào log kèm `request_id`; client chỉ nhận mã và câu giải
thích. Traceback nói ra đường dẫn file, tên hàm và phiên bản thư viện — thứ
không có lý do gì để trình duyệt người dùng biết.
"""
from __future__ import annotations

import uuid
from typing import Final

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from hfml.api.v1.schemas import Diagnostic, ErrorResponse
from hfml.inference.lifecycle import ModelUnavailable
from hfml.logger import get_logger

log = get_logger(__name__)

#: Mã 422. Starlette đã đổi tên hằng (`..._ENTITY` → `..._CONTENT`) và bản cũ
#: phát cảnh báo deprecation mỗi lần chạm tới. Đọc động để chạy được trên cả
#: hai phiên bản mà không phải ghim một mốc thư viện chỉ vì một con số.
HTTP_422: Final[int] = getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422)

#: Mã lỗi — HỢP ĐỒNG với client, đổi là phá vỡ tích hợp.
VALIDATION_ERROR: Final[str] = "validation_error"
MODEL_UNAVAILABLE: Final[str] = "model_unavailable"
LLM_ERROR: Final[str] = "llm_error"
TIMEOUT: Final[str] = "timeout"
CONFIGURATION_ERROR: Final[str] = "configuration_error"
INTERNAL_ERROR: Final[str] = "internal_error"
NOT_FOUND: Final[str] = "not_found"


class ApiError(Exception):
    """Lỗi đã biết, đã có mã và mã HTTP tương ứng."""

    def __init__(self, error: str, message: str, status_code: int,
                 details: list[Diagnostic] | None = None) -> None:
        super().__init__(message)
        self.error = error
        self.message = message
        self.status_code = status_code
        self.details = details or []


class ConfigurationError(ApiError):
    """Thiếu hoặc sai cấu hình — người vận hành sửa, không phải người gọi."""

    def __init__(self, message: str) -> None:
        super().__init__(CONFIGURATION_ERROR, message,
                         status.HTTP_503_SERVICE_UNAVAILABLE)


def _respond(status_code: int, error: str, message: str,
             details: list[Diagnostic] | None = None,
             request_id: str | None = None) -> JSONResponse:
    body = ErrorResponse(error=error, message=message,
                         details=details or [], request_id=request_id)
    return JSONResponse(status_code=status_code,
                        content=body.model_dump(mode="json"))


def register_error_handlers(app: FastAPI) -> None:
    """Gắn toàn bộ handler. Gọi một lần lúc dựng app."""

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError):
        """422 — request sai. Nêu ĐÍCH DANH từng trường.

        Pydantic đã biết chính xác trường nào sai và sai thế nào; bỏ phí thông
        tin đó rồi trả "dữ liệu không hợp lệ" là bắt người tích hợp đoán.
        """
        details = [
            Diagnostic(
                code=item.get("type", "invalid"),
                message=str(item.get("msg", "")),
                severity="error",
                # Bỏ mục đầu ("body"/"query") — client quan tâm tên trường.
                field=".".join(str(p) for p in item.get("loc", [])[1:]),
            )
            for item in exc.errors()
        ]
        return _respond(HTTP_422, VALIDATION_ERROR,
                        "Dữ liệu gửi lên không hợp lệ.", details)

    @app.exception_handler(ModelUnavailable)
    async def _model(request: Request, exc: ModelUnavailable):
        """503 — thiếu artifact. Trạng thái vận hành, không phải lỗi request."""
        log.warning("Model không khả dụng: %s", exc)
        return _respond(status.HTTP_503_SERVICE_UNAVAILABLE, MODEL_UNAVAILABLE,
                        str(exc))

    @app.exception_handler(TimeoutError)
    async def _timeout(request: Request, exc: TimeoutError):
        """504 — còn sống, chỉ là chậm. Khác hẳn 500."""
        log.warning("Request quá hạn giờ: %s %s", request.method, request.url.path)
        return _respond(
            status.HTTP_504_GATEWAY_TIMEOUT, TIMEOUT,
            "Yêu cầu xử lý quá lâu nên đã dừng. Vui lòng thử lại.")

    @app.exception_handler(ApiError)
    async def _api(request: Request, exc: ApiError):
        return _respond(exc.status_code, exc.error, exc.message, exc.details)

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException):
        """Đưa cả lỗi HTTP sẵn có về cùng một vỏ.

        Không có handler này thì 404 trả `{"detail": ...}` còn mọi lỗi khác
        trả `ErrorResponse` — client lại phải xử lý hai hình dạng.
        """
        code = NOT_FOUND if exc.status_code == 404 else INTERNAL_ERROR
        return _respond(exc.status_code, code, str(exc.detail))

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        """500 — ngoài dự liệu. Chi tiết vào log, client chỉ nhận mã.

        `request_id` là sợi dây nối câu trả lời người dùng nhìn thấy với dòng
        log chứa traceback. Không có nó thì "hệ thống lỗi" là tất cả những gì
        người vận hành có để đi tìm.
        """
        request_id = uuid.uuid4().hex[:12]
        log.exception("Lỗi không lường trước [%s] %s %s",
                      request_id, request.method, request.url.path)
        return _respond(
            status.HTTP_500_INTERNAL_SERVER_ERROR, INTERNAL_ERROR,
            "Đã xảy ra lỗi không lường trước. Vui lòng thử lại hoặc liên hệ "
            f"quản trị kèm mã {request_id}.", request_id=request_id)
