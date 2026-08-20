"""Logging dùng chung cho toàn pipeline (F01 task 3).

Cấu hình MỘT LẦN ở logger gốc `hfml`, các module con chỉ lấy logger con và
để log propagate lên. Không gắn handler vào từng logger con — làm vậy là
log bị in hai lần (con in một lần, cha in thêm một lần nữa).

    # trong module bất kỳ
    from hfml.logger import get_logger
    log = get_logger(__name__)     # -> "hfml.ml.registry"

Ghi ra hai nơi:

    console   theo dõi lúc chạy
    logs/hfml.log   lưu vết, xoay vòng 5 MB × 3 file

CẢ HAI đều bắt buộc UTF-8: thông điệp trong project này là tiếng Việt, mà
mặc định của Windows là codepage hệ thống (cp1252). File log thì truyền
`encoding="utf-8"`; console thì phải `reconfigure` `sys.stdout` — xem
`_force_utf8()`.

Có file log là điều kiện để F06 task 6 kiểm tra tái lập: chạy lại với seed
42 rồi đối chiếu metric giữa hai lần chạy, không phải đọc bằng mắt trên
terminal rồi đóng đi là mất.
"""
from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler

from hfml.config import CONFIG

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"

#: Logger gốc của package. Mọi logger con propagate lên đây.
ROOT_NAME = "hfml"

_configured = False


def _force_utf8(stream):
    """Ép `stream` sang UTF-8 tại chỗ. Trả về chính nó.

    Trên Windows `sys.stdout` mặc định là codepage hệ thống (cp1252), không
    mã hóa nổi chữ có dấu: **mọi** dòng log tiếng Việt ném `UnicodeEncodeError`
    và `logging` nuốt nó thành "--- Logging error ---". Chạy demo trước hội
    đồng mà console đầy traceback là hỏng.

    Dùng `reconfigure()` chứ không bọc `TextIOWrapper` mới quanh
    `stream.buffer`: hai wrapper trên cùng một buffer sẽ chèn nhau, và cái
    thứ hai lúc bị GC sẽ đóng buffer của cái thứ nhất.

    Sửa tại chỗ nên `print()` trong script/notebook cũng hết hỏng dấu, không
    riêng handler của logging.

    `errors="replace"` là lớp chắn cuối: nếu đến UTF-8 cũng không ghi được
    (console kỳ lạ, stream bị chuyển hướng) thì mất dấu còn hơn mất dòng log.

    Stream không có `reconfigure` (`StringIO`, capture của pytest) thì để
    nguyên — chúng vốn đã xử lý được unicode.
    """
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is None:
        return stream
    if (getattr(stream, "encoding", "") or "").lower().replace("-", "") == "utf8":
        return stream
    try:
        reconfigure(encoding="utf-8", errors="replace")
    except (ValueError, OSError):
        # Stream đã đóng hoặc không cho đổi encoding — không đáng để việc
        # cấu hình log làm sập cả tiến trình.
        pass
    return stream


def setup_logging(level: str | int | None = None, log_file: str | None = None) -> logging.Logger:
    """Cấu hình logger gốc `hfml`. Gọi lại nhiều lần cũng chỉ chạy một lần.

    Tham số để trống thì lấy từ `config/config.yaml` (mục `logging`).
    """
    global _configured
    root = logging.getLogger(ROOT_NAME)
    if _configured:
        return root

    conf = CONFIG.logging
    level = level if level is not None else conf.get("level", "INFO")
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    root.setLevel(level)
    # Không để log lọt lên root logger của Python (tránh in trùng khi có
    # thư viện khác đã basicConfig).
    root.propagate = False
    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    if conf.get("console", True):
        console = logging.StreamHandler(_force_utf8(sys.stdout))
        console.setFormatter(formatter)
        root.addHandler(console)

    file_name = log_file if log_file is not None else conf.get("file", "")
    if file_name:
        path = CONFIG.paths.root / file_name
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            path,
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",      # bắt buộc — thông điệp là tiếng Việt
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    _configured = True
    return root


def get_logger(name: str = ROOT_NAME) -> logging.Logger:
    """Lấy logger con. Truyền `__name__` để tên logger khớp cây module."""
    setup_logging()
    if name != ROOT_NAME and not name.startswith(ROOT_NAME + "."):
        name = f"{ROOT_NAME}.{name}"
    return logging.getLogger(name)


def log_run_context(log: logging.Logger) -> None:
    """In bối cảnh một lần chạy — để đọc lại log biết chạy với seed nào.

    Gọi ở đầu mỗi script train / experiment (F07 task 1).
    """
    # Ghi tỉ lệ chia thay cho `n_splits`: ML01 bỏ K-Fold từ 14/08/2026, và
    # cái quyết định con số của một lần chạy bây giờ là ba tỉ lệ này.
    val_size = CONFIG.training["val_size"]
    test_size = CONFIG.training["test_size"]
    log.info("seed=%d | confidence_threshold=%.2f | split %.0f/%.0f/%.0f",
             CONFIG.random_seed,
             CONFIG.confidence_threshold,
             (1.0 - val_size - test_size) * 100, val_size * 100, test_size * 100)
