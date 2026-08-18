"""Epic AI-03 — Module inference độc lập (F05 · M07).

Đóng gói toàn bộ pipeline AI thành một module dùng được từ script, notebook,
test hay FastAPI — mà không phụ thuộc vào bất cứ thứ nào trong số đó.

    from hfml.inference import analyze, chat, health

    result = analyze(payload)                      # tới Aggregation
    result = chat(payload, "Tôi vay được bao nhiêu?")   # trọn sơ đồ

Bố cục
-------
    payloads.py    quy đổi tên trường của client về schema chuẩn
    settings.py    cấu hình runtime — slug model, ngưỡng, tham số LLM
    result.py      vỏ kết quả chuẩn — StageResult, InferenceResult, Diagnostic
    lifecycle.py   nạp/giữ/đổi model
    stages.py      mười bước, mỗi bước uỷ quyền cho module đã có
    engine.py      điều phối + điểm vào

Module này KHÔNG cài lại nghiệp vụ
------------------------------------
Rule ở `hfml.rules`, model ở `hfml.ml`, chuẩn hoá ở `hfml.pipeline`, diễn đạt ở
`hfml.llm`. Ở đây chỉ có điều phối, cấu hình và xử lý lỗi. Chép một công thức
vào đây là tạo nguồn sự thật thứ hai cho cùng một con số — và bản chép sẽ không
được sửa khi bản gốc sửa.
"""
from hfml.inference.engine import SCHEMA_VERSION, analyze, chat, health
from hfml.inference.lifecycle import MANAGER, ModelUnavailable
from hfml.inference.result import (
    ERROR,
    INFO,
    WARNING,
    Diagnostic,
    InferenceResult,
    StageResult,
)
from hfml.inference.payloads import normalize_payload
from hfml.inference.settings import SETTINGS, InferenceSettings, reload_settings

__all__ = [
    "analyze", "chat", "health", "SCHEMA_VERSION",
    "MANAGER", "ModelUnavailable",
    "InferenceResult", "StageResult", "Diagnostic", "ERROR", "WARNING", "INFO",
    "SETTINGS", "InferenceSettings", "reload_settings", "normalize_payload",
]
