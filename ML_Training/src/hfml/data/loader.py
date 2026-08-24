"""Nạp dữ liệu Home Credit (F01 task 5).

Đây là ĐIỂM VÀO DUY NHẤT để đọc dataset. Không module nào khác được gọi
`pd.read_csv` thẳng vào `dataset/` — làm vậy thì đường dẫn và tên file rải
khắp nơi, đổi một chỗ là hỏng chỗ khác.

Loader chỉ ĐỌC, không làm sạch. Sentinel `DAYS_EMPLOYED = 365243`, missing
value, giá trị bất hợp lệ — tất cả để nguyên như trong file. Việc xử lý
thuộc `hfml.data.preprocessing.cleaner` (task 8, 9). Tách như vậy để lúc
kiểm tra chất lượng dữ liệu (task 6) còn nhìn thấy dữ liệu thật.

Dataset KHÔNG commit vào git (166–723 MB/file). Tải từ Kaggle:
https://www.kaggle.com/c/home-credit-default-risk/data

Hiệu năng: `application_train.csv` đọc đủ 122 cột mất ~4s và tốn ~505 MB RAM;
truyền `columns=` để chỉ lấy cột cần thì còn ~1,4s và ~68 MB. Trong vòng lặp
thử nghiệm, dùng `nrows=` để chạy nhanh trên mẫu nhỏ.
"""
from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd

from hfml.config import CONFIG
from hfml.logger import get_logger

log = get_logger(__name__)

#: Tên logic → tên file trong `dataset/home-credit-default-risk/`.
HOME_CREDIT_FILES: Final[dict[str, str]] = {
    "application_train": "application_train.csv",
    "previous_application": "previous_application.csv",
    "bureau": "bureau.csv",
    "installments_payments": "installments_payments.csv",
    "columns_description": "HomeCredit_columns_description.csv",
}

#: Bảng gốc của F04: mỗi hồ sơ một dòng, mang cột nhãn `TARGET`.
#:
#: KHÔNG phải file duy nhất được dùng — `bureau` cũng được đọc và gộp lại thành
#: 9 trong 16 feature của model đang deploy (`aggregate_bureau` ở
#: `ml02_credit_risk/features.py`). Chú thích cũ ở đây ghi "file duy nhất F04
#: thực sự dùng", viết từ lúc chưa gộp bureau và không được sửa theo; ai đọc
#: rồi tưởng thiếu `bureau.csv` vẫn train được sẽ mất thời gian.
#:
#: Hai file THỰC SỰ không dùng là `previous_application` và
#: `installments_payments` — chưa nằm trong phạm vi (PLAN.md §4.3). Chúng đã bị
#: xoá khỏi đĩa ngày 24/08/2026; muốn dùng phải tải lại từ Kaggle.
PRIMARY_FILE: Final[str] = "application_train"

#: File BẮT BUỘC phải có trên đĩa. Thiếu một trong ba là không chạy được F04:
#: hai file đầu nuôi feature, file thứ ba nuôi `describe()` và `docs/dataset.md`.
REQUIRED_FILES: Final[tuple[str, ...]] = (
    "application_train", "bureau", "columns_description")

#: File NGOÀI PHẠM VI — vắng mặt là bình thường, không phải lỗi.
#:
#: Đã bị xoá khỏi đĩa ngày 24/08/2026 để lấy lại 1,13 GB. Chúng vẫn nằm trong
#: `HOME_CREDIT_FILES` để `resolve()` biết đường dẫn nếu sau này mở rộng phạm
#: vi, và để thông báo lỗi chỉ đúng chỗ tải lại thay vì báo "không biết file".
OPTIONAL_FILES: Final[tuple[str, ...]] = (
    "previous_application", "installments_payments")


#: Cột nhãn của ML02.
TARGET_COLUMN: Final[str] = "TARGET"
#: Khóa hồ sơ. Không được đưa vào feature set.
ID_COLUMN: Final[str] = "SK_ID_CURR"

_DOWNLOAD_HINT = (
    "Tải từ https://www.kaggle.com/c/home-credit-default-risk/data "
    f"rồi giải nén vào {CONFIG.paths.dataset}"
)


class DatasetNotFoundError(FileNotFoundError):
    """Thiếu file dataset. Nêu rõ thiếu file nào và tải ở đâu."""


def resolve(name: str) -> Path:
    """Đường dẫn tuyệt đối tới một file dataset theo tên logic."""
    if name not in HOME_CREDIT_FILES:
        raise KeyError(
            f"Không biết file '{name}'. Chọn một trong: {sorted(HOME_CREDIT_FILES)}")
    return CONFIG.paths.dataset / HOME_CREDIT_FILES[name]


def available_files() -> dict[str, bool]:
    """`{tên logic: có trên đĩa hay không}` — dùng cho health check và task 7."""
    return {name: resolve(name).exists() for name in HOME_CREDIT_FILES}


def missing_required() -> list[str]:
    """File BẮT BUỘC còn thiếu. Rỗng = đủ điều kiện chạy F04.

    Tách khỏi `available_files()` vì đó là hai câu hỏi khác nhau: cái kia hỏi
    "trên đĩa có những gì", cái này hỏi "đã đủ để chạy chưa". Gộp hai câu vào
    một phép kiểm chính là lý do trước đây vắng `previous_application.csv`
    cũng bị báo động ngang với vắng `application_train.csv`.
    """
    return [name for name in REQUIRED_FILES if not resolve(name).exists()]


def require(name: str) -> Path:
    """Như `resolve` nhưng báo lỗi rõ ràng nếu file chưa có."""
    path = resolve(name)
    if not path.exists():
        raise DatasetNotFoundError(f"Thiếu file '{path.name}' tại {path}. {_DOWNLOAD_HINT}")
    return path


def load_raw(
    name: str,
    columns: list[str] | None = None,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Đọc một file dataset nguyên trạng, không làm sạch.

    `columns` giới hạn cột đọc (nhanh và nhẹ RAM hơn hẳn);
    `nrows` giới hạn số dòng, tiện khi chạy thử.
    """
    path = require(name)
    log.info("Đọc %s%s", path.name,
             f" ({len(columns)} cột)" if columns else "")
    df = pd.read_csv(path, usecols=columns, nrows=nrows)
    log.info("→ %s: %d dòng × %d cột", path.name, len(df), df.shape[1])
    return df


def load_application_train(
    columns: list[str] | None = None,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Nạp `application_train.csv` — 307.511 hồ sơ, nhãn thật `TARGET`.

    Nếu truyền `columns` thì `SK_ID_CURR` và `TARGET` luôn được thêm vào,
    vì thiếu hai cột đó là không train được.
    """
    if columns is not None:
        required = [c for c in (ID_COLUMN, TARGET_COLUMN) if c not in columns]
        columns = required + list(columns)

    df = load_raw(PRIMARY_FILE, columns=columns, nrows=nrows)

    missing = {ID_COLUMN, TARGET_COLUMN} - set(df.columns)
    if missing:
        raise ValueError(f"{HOME_CREDIT_FILES[PRIMARY_FILE]} thiếu cột {sorted(missing)}")
    return df


def load_columns_description() -> pd.DataFrame:
    """Nạp từ điển mô tả cột — dùng khi viết `docs/dataset.md` (F07 task 2).

    File của Kaggle không thống nhất encoding giữa các bản tải, nên thử lần
    lượt utf-8 → cp1252 → latin-1 thay vì cứng một loại rồi chết giữa chừng.
    """
    path = require("columns_description")
    for encoding in ("utf-8", "cp1252", "latin-1"):
        try:
            df = pd.read_csv(path, encoding=encoding, index_col=0)
        except UnicodeDecodeError:
            log.debug("%s không đọc được bằng %s, thử encoding khác", path.name, encoding)
            continue
        log.info("Đọc %s bằng encoding %s: %d dòng", path.name, encoding, len(df))
        return df
    raise UnicodeDecodeError(
        "utf-8", b"", 0, 1,
        f"Không đọc được {path.name} bằng utf-8, cp1252 hay latin-1")


def describe_column(column: str) -> str | None:
    """Mô tả chính thức của một cột Home Credit, `None` nếu không có.

    Tiện khi giải trình feature importance trước hội đồng — biết chắc
    `EXT_SOURCE_2` nghĩa là gì thay vì đoán.
    """
    desc = load_columns_description()
    hit = desc[desc["Row"].str.strip() == column]
    if hit.empty:
        return None
    return str(hit.iloc[0]["Description"]).strip()


def target_distribution(nrows: int | None = None) -> dict[str, float | int]:
    """Phân bố nhãn — con số phải trích dẫn trong báo cáo (PLAN.md §7.3)."""
    df = load_raw(PRIMARY_FILE, columns=[ID_COLUMN, TARGET_COLUMN], nrows=nrows)
    total = len(df)
    positive = int(df[TARGET_COLUMN].sum())
    return {
        "n_rows": total,
        "n_positive": positive,
        "n_negative": total - positive,
        "positive_rate": positive / total if total else 0.0,
        # Tỉ lệ dùng cho scale_pos_weight của XGBoost (PLAN.md §7.3).
        "scale_pos_weight": (total - positive) / positive if positive else float("nan"),
        # Accuracy của model đoán tất cả là 0 — mốc để thấy accuracy vô nghĩa.
        "majority_class_accuracy": (total - positive) / total if total else 0.0,
    }
