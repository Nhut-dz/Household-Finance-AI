"""Ghi lại mọi lần train vào `experiments/results.csv` (F07 task 1).

Vì sao phải có file này chứ không đọc log trên terminal
-------------------------------------------------------
Bảng so sánh 4 thuật toán trong báo cáo phải dựng lại được. Nếu số chỉ tồn
tại trong scrollback của terminal thì đến lúc viết báo cáo là chép tay, mà
chép tay thì sai — và không ai phát hiện được vì không còn gì để đối chiếu.

Mỗi lần chạy ghi thêm dòng, KHÔNG ghi đè: lịch sử các lần chỉnh tham số
chính là phần "quá trình thực nghiệm" của báo cáo.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from hfml.config import CONFIG
from hfml.logger import get_logger

log = get_logger(__name__)

#: Cột đứng đầu, theo thứ tự cố định. Cột chỉ số nối vào sau và có thể khác
#: nhau giữa ML01 (macro-F1) và ML02 (PR-AUC) — `append_results` tự hợp nhất.
LEADING_COLUMNS = ("run_at", "task", "algo", "feature_set", "random_seed",
                   "n_rows", "n_splits", "note")


def results_path() -> Path:
    """`src/training/runs/results.csv` — cùng chỗ với artifact.

    Trước 12/08/2026 hàm này trỏ vào `experiments/`, tách khỏi nơi chứa
    artifact. Hai nơi thì bảng chỉ số và model mà nó mô tả có thể lệch nhau
    mà không ai thấy.
    """
    return CONFIG.paths.runs / "results.csv"


def append_results(rows: pd.DataFrame, path: Path | None = None) -> Path:
    """Nối các dòng kết quả vào `results.csv`, giữ nguyên lịch sử cũ.

    Lần chạy sau có thêm cột chỉ số mới thì hợp nhất tập cột thay vì làm hỏng
    file — dòng cũ để trống ở cột mới. Ném đi lịch sử chỉ vì đổi bộ chỉ số là
    đánh mất đúng thứ mà file này tồn tại để giữ.
    """
    path = path or results_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    stamped = rows.copy()
    stamped.insert(0, "run_at", datetime.now(timezone.utc).isoformat(timespec="seconds"))

    if path.exists():
        previous = pd.read_csv(path)
        stamped = pd.concat([previous, stamped], ignore_index=True)

    ordered = [c for c in LEADING_COLUMNS if c in stamped.columns]
    ordered += [c for c in stamped.columns if c not in ordered]
    stamped[ordered].to_csv(path, index=False, encoding="utf-8")
    log.info("Ghi %d dòng kết quả → %s", len(rows), path)
    return path
