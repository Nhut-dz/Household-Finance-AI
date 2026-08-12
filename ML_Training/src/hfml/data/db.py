"""Kết nối PostgreSQL Household_Finance_V2_Dev (pgAdmin 4).

ĐIỂM VÀO DUY NHẤT để đọc dữ liệu nghiệp vụ từ DB — song song với `loader.py`
(đọc CSV Home Credit). Không module nào khác tự tạo engine hay gọi
`pd.read_sql` thẳng, để chuỗi kết nối chỉ nằm một chỗ (`hfml.config.Database`).

Cấu hình host/port/name/user ở `config/config.yaml`; password (nếu có) đặt ở
env `HFML_DB_PASSWORD`. Server dev local dùng trust auth nên password rỗng vẫn
kết nối được.

Engine tạo lười (lazy) và tái dùng: chỉ mở connection pool khi thật sự cần,
tránh treo lúc import nếu DB chưa bật.
"""
from __future__ import annotations

from typing import Final

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from hfml.config import CONFIG
from hfml.logger import get_logger

log = get_logger(__name__)

#: Các bảng nghiệp vụ trong Household_Finance_V2_Dev (theo thứ tự phụ thuộc FK).
TABLES: Final[tuple[str, ...]] = (
    "tblusers",
    "tblhouseholds",
    "tblassets",
    "tblfinancial_goals",
    "tblconsultations",
    "tblcalculation_results",
    "tblai_responses",
    "tblrecommendation_packages",
)

_engine: Engine | None = None


def get_engine() -> Engine:
    """Trả về engine dùng chung, tạo lần đầu khi được gọi.

    `pool_pre_ping=True` để tự phát hiện connection chết (server restart) thay
    vì ném lỗi giữa chừng một query.
    """
    global _engine
    if _engine is None:
        url = CONFIG.database.url()
        log.info("Tạo engine PostgreSQL → %s@%s:%d/%s",
                 CONFIG.database.user, CONFIG.database.host,
                 CONFIG.database.port, CONFIG.database.name)
        _engine = create_engine(url, pool_pre_ping=True, future=True)
    return _engine


def ping() -> bool:
    """Kiểm tra kết nối. True nếu `SELECT 1` chạy được, ngược lại raise.

    Dùng cho health check trước khi chạy pipeline đọc từ DB.
    """
    with get_engine().connect() as conn:
        conn.execute(text("SELECT 1"))
    log.info("Kết nối DB OK: %s", CONFIG.database.name)
    return True


def read_table(
    name: str,
    columns: list[str] | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Đọc một bảng nghiệp vụ vào DataFrame.

    `columns` giới hạn cột (nhẹ RAM hơn); `limit` giới hạn số dòng khi chạy thử.
    """
    if name not in TABLES:
        raise KeyError(f"Không biết bảng '{name}'. Chọn một trong: {list(TABLES)}")
    cols = ", ".join(f'"{c}"' for c in columns) if columns else "*"
    sql = f'SELECT {cols} FROM "{name}"'
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    log.info("Đọc bảng %s%s", name, f" (limit {limit})" if limit else "")
    df = pd.read_sql(text(sql), get_engine())
    log.info("→ %s: %d dòng × %d cột", name, len(df), df.shape[1])
    return df


def read_sql(query: str, params: dict | None = None) -> pd.DataFrame:
    """Chạy một câu SQL tùy ý (SELECT/JOIN) và trả DataFrame.

    Dùng `params` với bind `:ten` để tránh SQL injection, không nối chuỗi tay.
    """
    return pd.read_sql(text(query), get_engine(), params=params)


def table_counts() -> dict[str, int]:
    """`{tên bảng: số dòng}` cho cả 8 bảng — dùng khi verify import/health check."""
    with get_engine().connect() as conn:
        return {
            t: conn.execute(text(f'SELECT COUNT(*) FROM "{t}"')).scalar_one()
            for t in TABLES
        }
