"""AI-04 task 6 — Cấu hình tầng API (F05 · M08).

Chỉ những thứ thuộc về việc PHỤC VỤ: đường dẫn phiên bản, CORS, timeout, cỡ
request. Cấu hình của AI (slug model, ngưỡng, tham số LLM) nằm ở
`hfml.inference.settings` và KHÔNG lặp lại ở đây — hai nơi cùng khai một tham
số là hai nơi có thể lệch nhau.

Vì sao CORS mặc định là danh sách rỗng
----------------------------------------
Rỗng nghĩa là không bật CORS. Mặc định `"*"` thì mọi trang web bất kỳ gọi được
API này bằng trình duyệt của người dùng, và hồ sơ tài chính là thứ không nên
mở như vậy chỉ vì tiện lúc dev. Ai cần thì khai đúng origin qua
`HFML_API_CORS_ORIGINS`.

Vì sao timeout tồn tại
------------------------
Một lượt gọi LLM đi ra mạng ngoài và có thể treo. Không có giới hạn thì
request treo theo, giữ chỗ trong pool, và người dùng nhìn màn hình quay vô
hạn. Hết giờ thì trả 504 — nói rõ là chậm, khác hẳn với 500.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Final

#: Tiền tố phiên bản. Đổi hợp đồng thì thêm `/api/v2`, KHÔNG sửa v1 tại chỗ:
#: client cũ vẫn đang gọi và không có cách nào biết hình dạng vừa đổi.
API_PREFIX: Final[str] = "/api/v1"
API_VERSION: Final[str] = "1.0"


def _env_float(name: str, fallback: float) -> float:
    raw = os.getenv(name)
    try:
        return float(raw) if raw else fallback
    except ValueError:
        return fallback


def _env_list(name: str) -> list[str]:
    raw = os.getenv(name, "").strip()
    return [item.strip() for item in raw.split(",") if item.strip()]


@dataclass
class ApiSettings:
    """Tham số phục vụ. Mọi giá trị đều phủ được bằng biến môi trường."""

    #: Giới hạn cho `POST /api/v1/chat` — có gọi LLM nên chậm hơn hẳn.
    chat_timeout: float = 60.0
    #: Giới hạn cho `POST /api/v1/inference` — chỉ rule + ML.
    inference_timeout: float = 20.0

    #: Origin được phép gọi từ trình duyệt. Rỗng = không bật CORS.
    cors_origins: list[str] = field(default_factory=list)

    #: Nạp model ngay lúc khởi động thay vì chờ request đầu tiên.
    warm_up_models: bool = True

    def to_dict(self) -> dict:
        return {
            "api_version": API_VERSION,
            "prefix": API_PREFIX,
            "chat_timeout": self.chat_timeout,
            "inference_timeout": self.inference_timeout,
            "cors_enabled": bool(self.cors_origins),
            "warm_up_models": self.warm_up_models,
        }


def load_api_settings() -> ApiSettings:
    return ApiSettings(
        chat_timeout=_env_float("HFML_API_CHAT_TIMEOUT", 60.0),
        inference_timeout=_env_float("HFML_API_INFERENCE_TIMEOUT", 20.0),
        cors_origins=_env_list("HFML_API_CORS_ORIGINS"),
        warm_up_models=os.getenv("HFML_API_WARM_UP", "1")
        not in ("0", "false", "False"),
    )


API_SETTINGS = load_api_settings()
