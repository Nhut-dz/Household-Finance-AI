"""AI-04 task 2 — Endpoint của API v1 (F05 · M08).

Ba endpoint, và cả ba đều MỎNG một cách có chủ ý: nhận request đã validate,
gọi đúng một hàm của `hfml.inference`, chiếu kết quả sang schema công khai.
Không endpoint nào chứa phép tính, luật nghiệp vụ hay quyết định nhãn.

Vì sao phải chạy pipeline ở threadpool
----------------------------------------
Pipeline là mã đồng bộ và tốn CPU (XGBoost) lẫn tốn chờ (gọi LLM ra mạng
ngoài). Gọi thẳng trong hàm `async` thì nó chiếm event loop, và trong lúc đó
**mọi request khác đứng im** — kể cả `/health`. Nên nó được đẩy sang
threadpool, còn event loop rảnh để nhận request mới.

Timeout không giết được luồng đang chạy
-----------------------------------------
`asyncio.wait_for` bỏ chờ, nhưng luồng trong threadpool vẫn chạy nốt tới khi
xong — Python không có cách an toàn nào để giết một luồng đang thực thi.

Đây là giới hạn thật, không phải chi tiết bỏ qua được: một loạt request treo
sẽ làm cạn threadpool dù client đã nhận 504 từ lâu. Chấp nhận vì cách duy
nhất để cắt cứng là chạy pipeline ở tiến trình riêng, và cái giá của nó
(tuần tự hoá dữ liệu, nạp model hai lần) lớn hơn nhiều so với vấn đề đang có.
Giới hạn thời gian của chính lượt gọi LLM mới là chỗ chặn gốc.
"""
from __future__ import annotations

import asyncio
from typing import Any, Callable

from fastapi import APIRouter, Response, status

from hfml.api.v1.config import API_SETTINGS
from hfml.api.v1.health import UNHEALTHY, build_health
from hfml.api.v1.schemas import (
    AIResult,
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    HealthResponse,
    InferenceRequest,
    LLMResponse,
)
from hfml.logger import get_logger

log = get_logger(__name__)

router = APIRouter()

#: Chỉ chứa `/health`. Gắn ở gốc cho hạ tầng — xem `app.create_app`.
infra_router = APIRouter()

#: Vỏ lỗi chung cho mọi endpoint — để OpenAPI nói đúng client sẽ nhận gì.
ERROR_RESPONSES: dict = {
    422: {"model": ErrorResponse, "description": "Request không hợp lệ"},
    500: {"model": ErrorResponse, "description": "Lỗi không lường trước"},
    503: {"model": ErrorResponse, "description": "Model hoặc cấu hình chưa sẵn sàng"},
    504: {"model": ErrorResponse, "description": "Xử lý quá hạn giờ"},
}


async def _run(work: Callable[[], Any], timeout: float, label: str) -> Any:
    """Chạy pipeline đồng bộ ở threadpool, có giới hạn thời gian.

    Ném `TimeoutError` khi quá hạn — `errors.py` đổi nó thành 504.
    """
    try:
        return await asyncio.wait_for(asyncio.to_thread(work), timeout=timeout)
    except asyncio.TimeoutError as exc:
        log.warning("%s vượt quá %.0fs", label, timeout)
        raise TimeoutError(
            f"{label} vượt quá giới hạn {timeout:.0f} giây.") from exc


@router.post("/inference", response_model=AIResult, responses=ERROR_RESPONSES,
             summary="Chạy Rule + ML, không gọi LLM")
async def run_inference(request: InferenceRequest) -> AIResult:
    """Input → Validation → Preprocessing → Rule → ML01 → ML02 → Aggregation.

    Không gọi LLM nên nhanh và không tốn quota. Dùng cho màn hình chỉ cần chỉ
    số tài chính.
    """
    from hfml.inference import analyze

    payload = request.household.to_payload()
    result = await _run(lambda: analyze(payload),
                        API_SETTINGS.inference_timeout, "Inference")
    return AIResult.from_internal(result)


@router.post("/chat", response_model=ChatResponse, responses=ERROR_RESPONSES,
             summary="Chạy trọn pipeline và trả câu trả lời")
async def run_chat(request: ChatRequest) -> ChatResponse:
    """Input → … → LLM → Output Validation → Response.

    Trả về cả câu trả lời lẫn phần phân tích đứng sau nó, để client đối chiếu
    được — không có phần đó thì không có cách nào kiểm chứng ngoài việc tin.
    """
    from hfml.inference import chat

    payload = request.household.to_payload()
    history = ([turn.model_dump() for turn in request.history]
               if request.history else None)

    result = await _run(
        lambda: chat(payload, request.question,
                     intent_code=request.intent_code,
                     history=history,
                     previous_intent=request.previous_intent),
        API_SETTINGS.chat_timeout, "Chat")

    return ChatResponse(
        ok=result.ok,
        answer=LLMResponse.from_internal(result),
        analysis=AIResult.from_internal(result),
    )


@router.get("/health", response_model=HealthResponse,
            summary="Trạng thái service và từng thành phần")
@infra_router.get("/health", response_model=HealthResponse)
async def health(response: Response) -> HealthResponse:
    """`healthy` · `degraded` · `unhealthy`.

    Trả 503 khi `unhealthy` để bộ cân bằng tải rút service khỏi vòng phục vụ
    mà không cần đọc thân response. `degraded` vẫn 200 — service còn phục vụ
    tốt phần lớn yêu cầu, rút nó ra là tự làm hỏng thêm.
    """
    report = build_health()
    if report.status == UNHEALTHY:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return report
