"""F01 task 3 — kiểm tra config và logging.

Ba thứ được kiểm ở đây đều là lỗi im lặng: log in hai lần thì vẫn chạy,
tiếng Việt hỏng dấu trong file log thì vẫn chạy, ngưỡng hardcode thì vẫn
chạy. Có test mới phát hiện được.
"""
from __future__ import annotations

import io
import logging
import sys
import uuid
from logging.handlers import RotatingFileHandler

import pytest

import hfml.logger
from hfml.config import CONFIG, load_config
from hfml.logger import (
    ROOT_NAME,
    _force_utf8,
    get_logger,
    log_run_context,
    setup_logging,
)

#: Thông điệp mẫu — có đủ dấu thanh, dấu mũ và ký tự đ.
VIETNAMESE = "Sinh 20000 hộ — tỉ lệ tiết kiệm, dư nợ đã tính xong"


# ---------------------------------------------------------------- config

def test_config_loads_yaml_values():
    """Mọi tham số phải đọc được từ config.yaml, không hardcode trong code."""
    cfg = load_config()
    assert cfg.random_seed == 42
    assert 0 < cfg.confidence_threshold <= 1
    # Không còn `n_splits`: ML01 bỏ K-Fold Cross-Validation (14/08/2026).
    assert "n_splits" not in cfg.training
    assert cfg.training["val_size"] == pytest.approx(0.15)
    assert cfg.training["test_size"] == pytest.approx(0.15)
    assert cfg.logging["level"] in {"DEBUG", "INFO", "WARNING", "ERROR"}
    assert cfg.llm["model"]


def test_env_overrides_yaml(monkeypatch):
    """Thứ tự ưu tiên: env > config.yaml > mặc định."""
    monkeypatch.setenv("HFML_RANDOM_SEED", "7")
    monkeypatch.setenv("HFML_LOG_LEVEL", "DEBUG")
    cfg = load_config()
    assert cfg.random_seed == 7
    assert cfg.logging["level"] == "DEBUG"


def test_writable_paths_exist():
    """Thư mục ghi ra được tạo sẵn; `dataset` thì không (phải tải thủ công)."""
    for path in (CONFIG.paths.runs, CONFIG.paths.logs,
                 CONFIG.paths.data_processed):
        assert path.is_dir(), f"chưa tạo {path}"


def test_api_key_not_hardcoded():
    """Secret chỉ đến từ env, không được nằm trong config.yaml."""
    yaml_text = (CONFIG.paths.config / "config.yaml").read_text(encoding="utf-8")
    assert "api_key" not in yaml_text.lower()


# --------------------------------------------------------------- logging

def test_only_root_logger_holds_handlers():
    """Chống log in hai lần: chỉ logger gốc `hfml` được gắn handler."""
    setup_logging()
    child = get_logger("hfml.ml.registry")

    assert not child.handlers, "logger con có handler riêng → log sẽ in trùng"
    assert child.propagate, "logger con phải propagate lên gốc"
    assert logging.getLogger(ROOT_NAME).handlers


def test_get_logger_prefixes_package_name():
    """get_logger(__name__) cho tên khớp cây module."""
    assert get_logger("hfml.rules.engine").name == "hfml.rules.engine"
    assert get_logger("rules.engine").name == "hfml.rules.engine"
    assert get_logger().name == ROOT_NAME


def test_log_file_keeps_vietnamese_diacritics():
    """File log phải là UTF-8 — mặc định Windows (cp1252) sẽ hỏng dấu."""
    setup_logging()
    log = get_logger("hfml.test")
    marker = uuid.uuid4().hex[:8]
    message = f"Kiểm tra tiếng Việt có dấu — tỉ lệ tiết kiệm, dư nợ [{marker}]"
    log.info(message)

    handlers = [h for h in logging.getLogger(ROOT_NAME).handlers
                if isinstance(h, RotatingFileHandler)]
    assert handlers, "chưa cấu hình ghi file log"
    handler = handlers[0]
    assert handler.encoding == "utf-8"

    handler.flush()
    written = open(handler.baseFilename, encoding="utf-8").read()
    assert message in written


def test_force_utf8_converts_windows_codepage_stream():
    """cp1252 — mặc định của console Windows — không mã hóa nổi chữ có dấu."""
    broken = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    with pytest.raises(UnicodeEncodeError):
        broken.write(VIETNAMESE)
        broken.flush()

    raw = io.BytesIO()
    fixed = hfml.logger._force_utf8(io.TextIOWrapper(raw, encoding="cp1252"))
    assert fixed.encoding.lower().replace("-", "") == "utf8"
    fixed.write(VIETNAMESE)
    fixed.flush()
    assert raw.getvalue().decode("utf-8") == VIETNAMESE


def test_force_utf8_leaves_stream_without_reconfigure_alone():
    """`StringIO` / capture của pytest vốn xử lý được unicode — đừng đụng vào."""
    stream = io.StringIO()
    assert hfml.logger._force_utf8(stream) is stream
    stream.write(VIETNAMESE)
    assert stream.getvalue() == VIETNAMESE


def test_console_handler_survives_vietnamese_on_codepage_console(monkeypatch):
    """Console cp1252 không được nuốt dòng log tiếng Việt.

    Lỗi thật: chạy `generate_households()` trên Windows thì MỌI dòng log ném
    `UnicodeEncodeError`, `logging` biến nó thành "--- Logging error ---".
    """
    raw = io.BytesIO()
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(raw, encoding="cp1252"))
    monkeypatch.setattr(hfml.logger, "_configured", False)

    root = logging.getLogger(ROOT_NAME)
    saved = root.handlers[:]
    root.handlers.clear()
    try:
        setup_logging()
        get_logger("hfml.test").info(VIETNAMESE)
        for handler in root.handlers:
            handler.flush()
        assert VIETNAMESE in raw.getvalue().decode("utf-8")
    finally:
        for handler in root.handlers:
            if handler not in saved:
                handler.close()
        root.handlers[:] = saved


def test_log_run_context_records_seed():
    """Đọc lại log phải biết lần chạy đó dùng seed nào (F06 task 6)."""
    setup_logging()
    log = get_logger("hfml.test")
    log_run_context(log)

    handler = next(h for h in logging.getLogger(ROOT_NAME).handlers
                   if isinstance(h, RotatingFileHandler))
    handler.flush()
    written = open(handler.baseFilename, encoding="utf-8").read()
    assert f"seed={CONFIG.random_seed}" in written
