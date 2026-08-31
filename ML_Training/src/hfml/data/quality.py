"""Kiểm tra chất lượng và quản lý phiên bản dataset (F01 task 6, 7).

Hai việc gộp một file vì chúng trả lời hai nửa của cùng một câu hỏi mà hội
đồng chắc chắn sẽ hỏi: *"dữ liệu của bạn là dữ liệu nào, và nó sạch tới đâu?"*

    task 6  `check_application_train()`  → báo cáo chất lượng
    task 7  `build_manifest()`           → SHA-256 + số dòng/cột, chốt phiên bản

Vì sao cần manifest: dataset không commit vào git (1,4 GB). Không có SHA-256
thì ba tháng sau không ai chứng minh được model đã train trên đúng file này —
và F06 task 6 (chạy lại seed 42 ra metric trùng đến 4 chữ số) mất căn cứ.

Module này chỉ ĐO và BÁO CÁO, không sửa dữ liệu. Việc sửa thuộc
`hfml.data.preprocessing.cleaner` (task 8, 9).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Final

import pandas as pd

from hfml.config import CONFIG
from hfml.data import loader
from hfml.logger import get_logger

log = get_logger(__name__)

# --------------------------------------------------------------------------
# Bất thường đã biết của Home Credit — phát hiện bằng khảo sát thực tế
# --------------------------------------------------------------------------

#: Giá trị "canh gác" thay cho NaN. `DAYS_EMPLOYED = 365243` ≈ 1000 năm.
#: 55.374 dòng (18,01%), trùng khít với `ORGANIZATION_TYPE = 'XNA'` và
#: `OCCUPATION_TYPE` rỗng — đó là nhóm nghỉ hưu (55.352) và thất nghiệp (22).
#: Nhóm này vỡ nợ 5,40% so với 8,66% của nhóm có việc làm, nên khi chuyển
#: sang NaN BẮT BUỘC giữ lại cờ nhị phân, xem `cleaner` (task 8).
KNOWN_SENTINELS: Final[dict[str, float]] = {
    "DAYS_EMPLOYED": 365243,
}

#: Chuỗi Home Credit dùng thay cho "không rõ". Phải coi như missing, không
#: được để lọt vào encoder thành một hạng mục thật.
PLACEHOLDER_VALUES: Final[frozenset[str]] = frozenset({"XNA", "XAP", "Unknown"})

#: Cột thiếu quá ngưỡng này thì impute cũng chỉ là bịa số.
HIGH_MISSING_THRESHOLD: Final[float] = 0.50
#: Categorical nhiều hạng mục hơn ngưỡng này thì one-hot sẽ nổ chiều.
HIGH_CARDINALITY_THRESHOLD: Final[int] = 30

MANIFEST_FILENAME: Final[str] = "dataset_manifest.json"
DATASET_DOC_FILENAME: Final[str] = "dataset.md"
KAGGLE_URL: Final[str] = "https://www.kaggle.com/c/home-credit-default-risk/data"


class Severity(str, Enum):
    ERROR = "error"        # không xử lý thì không train được
    WARNING = "warning"    # train được nhưng kết quả sẽ lệch
    INFO = "info"          # cần biết, không cần hành động ngay


@dataclass(frozen=True)
class QualityIssue:
    severity: Severity
    code: str
    message: str
    columns: tuple[str, ...] = ()

    def __str__(self) -> str:
        return f"[{self.severity.value.upper()}] {self.code}: {self.message}"


@dataclass
class QualityReport:
    """Kết quả kiểm tra chất lượng một bảng."""
    table: str
    n_rows: int
    n_cols: int
    issues: list[QualityIssue] = field(default_factory=list)
    columns: pd.DataFrame | None = None

    def by_severity(self, severity: Severity) -> list[QualityIssue]:
        return [i for i in self.issues if i.severity is severity]

    @property
    def has_errors(self) -> bool:
        return bool(self.by_severity(Severity.ERROR))

    def to_markdown(self) -> str:
        lines = [f"### `{self.table}` — {self.n_rows:,} dòng × {self.n_cols} cột", ""]
        if not self.issues:
            lines.append("Không phát hiện vấn đề.")
            return "\n".join(lines)

        lines += ["| Mức | Mã | Mô tả |", "|---|---|---|"]
        for sev in (Severity.ERROR, Severity.WARNING, Severity.INFO):
            for issue in self.by_severity(sev):
                lines.append(f"| {sev.value} | `{issue.code}` | {issue.message} |")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Task 6 — kiểm tra chất lượng
# --------------------------------------------------------------------------
def column_report(df: pd.DataFrame) -> pd.DataFrame:
    """Một dòng cho mỗi cột: kiểu, tỉ lệ thiếu, số giá trị phân biệt."""
    n = len(df)
    rows = [
        {
            "column": col,
            "dtype": str(df[col].dtype),
            "missing": int(df[col].isna().sum()),
            "missing_rate": float(df[col].isna().mean()) if n else 0.0,
            "n_unique": int(df[col].nunique(dropna=True)),
        }
        for col in df.columns
    ]
    return pd.DataFrame(rows).sort_values("missing_rate", ascending=False, ignore_index=True)


def find_issues(df: pd.DataFrame, target: str | None = None) -> list[QualityIssue]:
    """Rà toàn bộ các bất thường đã biết. Không sửa gì."""
    n = len(df)
    issues: list[QualityIssue] = []
    if n == 0:
        return [QualityIssue(Severity.ERROR, "empty_table", "Bảng không có dòng nào")]

    # -- Trùng lặp -------------------------------------------------------
    if loader.ID_COLUMN in df.columns:
        dup = int(df[loader.ID_COLUMN].duplicated().sum())
        if dup:
            issues.append(QualityIssue(
                Severity.ERROR, "duplicate_id",
                f"{dup:,} dòng trùng `{loader.ID_COLUMN}`", (loader.ID_COLUMN,)))

    # -- Sentinel --------------------------------------------------------
    for col, value in KNOWN_SENTINELS.items():
        if col not in df.columns:
            continue
        hits = int((df[col] == value).sum())
        if hits:
            issues.append(QualityIssue(
                Severity.ERROR, "sentinel_value",
                f"`{col}` có {hits:,} dòng ({hits / n:.2%}) mang giá trị canh gác "
                f"{value:,.0f} — phải chuyển NaN VÀ giữ cờ nhị phân, "
                f"vì nhóm này vỡ nợ ít hơn hẳn", (col,)))

    # -- Chuỗi thay cho missing ------------------------------------------
    placeholder_cols: list[str] = []
    for col in df.columns:
        if df[col].dtype == object or isinstance(df[col].dtype, pd.StringDtype):
            hits = int(df[col].isin(PLACEHOLDER_VALUES).sum())
            if hits:
                placeholder_cols.append(f"{col} ({hits:,})")
    if placeholder_cols:
        issues.append(QualityIssue(
            Severity.WARNING, "placeholder_as_category",
            "Giá trị " + "/".join(sorted(PLACEHOLDER_VALUES))
            + " đang nằm như một hạng mục thật ở: " + ", ".join(placeholder_cols),
            tuple(c.split(" ")[0] for c in placeholder_cols)))

    # -- Thiếu nhiều ------------------------------------------------------
    missing_rate = df.isna().mean()
    heavy = missing_rate[missing_rate > HIGH_MISSING_THRESHOLD]
    if len(heavy):
        issues.append(QualityIssue(
            Severity.WARNING, "high_missing",
            f"{len(heavy)} cột thiếu trên {HIGH_MISSING_THRESHOLD:.0%} dữ liệu "
            f"(cao nhất `{heavy.idxmax()}` {heavy.max():.1%}) — impute cũng chỉ là bịa số",
            tuple(heavy.index)))

    # -- Cột hằng số ------------------------------------------------------
    constant = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    if constant:
        issues.append(QualityIssue(
            Severity.WARNING, "constant_column",
            f"{len(constant)} cột chỉ có một giá trị, không mang thông tin",
            tuple(constant)))

    # -- Cardinality cao --------------------------------------------------
    high_card = [
        c for c in df.columns
        if (df[c].dtype == object or isinstance(df[c].dtype, pd.StringDtype))
        and df[c].nunique(dropna=True) > HIGH_CARDINALITY_THRESHOLD
    ]
    if high_card:
        issues.append(QualityIssue(
            Severity.INFO, "high_cardinality",
            f"Categorical nhiều hạng mục (>{HIGH_CARDINALITY_THRESHOLD}): "
            + ", ".join(f"`{c}` ({df[c].nunique()})" for c in high_card)
            + " — one-hot sẽ nổ chiều, cân nhắc gộp nhóm",
            tuple(high_card)))

    # -- Tiền âm ----------------------------------------------------------
    money_cols = [c for c in df.columns if c.startswith("AMT_")]
    negative = [c for c in money_cols if (df[c] < 0).any()]
    if negative:
        issues.append(QualityIssue(
            Severity.ERROR, "negative_amount",
            "Cột tiền có giá trị âm: " + ", ".join(f"`{c}`" for c in negative),
            tuple(negative)))

    # -- Mất cân bằng nhãn -------------------------------------------------
    if target and target in df.columns:
        rate = float(df[target].mean())
        if rate < 0.20 or rate > 0.80:
            majority = max(rate, 1 - rate)
            issues.append(QualityIssue(
                Severity.INFO, "class_imbalance",
                f"`{target}` mất cân bằng: {rate:.2%} dương. Model đoán toàn lớp đa số "
                f"đã đạt {majority:.2%} accuracy — chọn model bằng PR-AUC, "
                f"không dùng accuracy", (target,)))

    return issues


def check_application_train(df: pd.DataFrame | None = None) -> QualityReport:
    """Chạy toàn bộ kiểm tra trên `application_train.csv` (task 6).

    Truyền sẵn `df` để khỏi đọc lại file (đọc đủ 122 cột mất ~4s).
    """
    if df is None:
        df = loader.load_application_train()
    return QualityReport(
        table=loader.HOME_CREDIT_FILES[loader.PRIMARY_FILE],
        n_rows=len(df),
        n_cols=df.shape[1],
        issues=find_issues(df, target=loader.TARGET_COLUMN),
        columns=column_report(df),
    )


# --------------------------------------------------------------------------
# Task 7 — quản lý phiên bản dataset
# --------------------------------------------------------------------------
def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 đọc theo khối 1 MB — file lớn nhất 723 MB, không nạp hết vào RAM."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def file_fingerprint(name: str, compute_hash: bool = True) -> dict:
    """Dấu vân tay một file dataset: kích thước, ngày sửa, SHA-256."""
    path = loader.require(name)
    stat = path.stat()
    info = {
        "file": path.name,
        "size_bytes": stat.st_size,
        "modified": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
    }
    if compute_hash:
        log.info("Băm SHA-256 %s (%.0f MB)…", path.name, stat.st_size / 1024**2)
        info["sha256"] = sha256_file(path)
    return info


def _relative_dataset_dir() -> str:
    """Đường dẫn dataset tương đối từ gốc project.

    Manifest có commit vào git nên không được nhúng đường dẫn tuyệt đối của
    một máy cụ thể — máy khác đọc vào sẽ thấy ổ đĩa không tồn tại.
    """
    try:
        return CONFIG.paths.dataset.relative_to(CONFIG.paths.root).as_posix()
    except ValueError:
        return CONFIG.paths.dataset.as_posix()


def build_manifest(compute_hash: bool = True) -> dict:
    """Manifest của toàn bộ dataset — thành phẩm chính của task 7."""
    files = {
        name: file_fingerprint(name, compute_hash=compute_hash)
        for name, exists in loader.available_files().items() if exists
    }
    missing = [name for name, exists in loader.available_files().items() if not exists]
    return {
        "source": KAGGLE_URL,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": _relative_dataset_dir(),
        "files": files,
        "missing_files": missing,
        "application_train": loader.target_distribution(),
    }


def _format_size(size_bytes: int) -> str:
    """Kích thước dễ đọc — 37 KB không nên hiện thành `0.0 MB`."""
    if size_bytes >= 1024**3:
        return f"{size_bytes / 1024**3:,.2f} GB"
    if size_bytes >= 1024**2:
        return f"{size_bytes / 1024**2:,.1f} MB"
    return f"{size_bytes / 1024:,.0f} KB"


def manifest_path() -> Path:
    return CONFIG.paths.docs / MANIFEST_FILENAME


def write_manifest(manifest: dict | None = None) -> Path:
    """Ghi `docs/dataset_manifest.json`. File này CÓ commit vào git."""
    manifest = manifest if manifest is not None else build_manifest()
    path = manifest_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Đã ghi manifest → %s", path)
    return path


def load_manifest() -> dict:
    path = manifest_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Chưa có {path}. Chạy `python scripts/build_dataset_docs.py` để tạo.")
    return json.loads(path.read_text(encoding="utf-8"))


def verify_manifest(compute_hash: bool = True) -> list[str]:
    """So dataset hiện tại với manifest đã chốt. Rỗng = khớp hoàn toàn.

    Dùng trước mỗi lần train lại: nếu file đổi mà metric cũng đổi, biết ngay
    nguyên nhân là dữ liệu chứ không phải code.
    """
    saved = load_manifest()
    drift: list[str] = []

    for name, expected in saved["files"].items():
        path = loader.resolve(name)
        if not path.exists():
            drift.append(f"{expected['file']}: đã có trong manifest nhưng nay không còn")
            continue
        actual = file_fingerprint(name, compute_hash=compute_hash)
        if actual["size_bytes"] != expected["size_bytes"]:
            drift.append(
                f"{expected['file']}: kích thước {actual['size_bytes']:,} "
                f"≠ manifest {expected['size_bytes']:,}")
        elif compute_hash and "sha256" in expected and actual.get("sha256") != expected["sha256"]:
            drift.append(f"{expected['file']}: SHA-256 khác manifest — nội dung đã đổi")

    for name, exists in loader.available_files().items():
        if exists and name not in saved["files"]:
            drift.append(f"{loader.HOME_CREDIT_FILES[name]}: file mới, chưa có trong manifest")

    return drift


def render_dataset_doc(manifest: dict, report: QualityReport | None = None) -> str:
    """Sinh nội dung `docs/dataset.md` (task 7, dùng lại cho F07 task 2)."""
    dist = manifest["application_train"]
    lines = [
        "# Dataset — Home Credit Default Risk",
        "",
        "> File này sinh tự động bởi `scripts/build_dataset_docs.py`. Đừng sửa tay.",
        "",
        f"- **Nguồn:** {manifest['source']}",
        "- **License:** theo điều khoản cuộc thi Kaggle — dùng cho mục đích học tập, "
        "không phân phối lại dữ liệu",
        f"- **Thư mục:** `{manifest['dataset_dir']}` (**không commit vào git**)",
        f"- **Chốt phiên bản lúc:** {manifest['generated_at']}",
        "",
        "## Phiên bản file (SHA-256)",
        "",
        "| File | Kích thước | SHA-256 |",
        "|---|---:|---|",
    ]
    for info in manifest["files"].values():
        digest = info.get("sha256", "—")
        short = f"`{digest[:16]}…`" if digest != "—" else "—"
        lines.append(f"| `{info['file']}` | {_format_size(info['size_bytes'])} | {short} |")

    # Tách hai loại vắng mặt. Gọi chung là "Thiếu" thì file cố ý bỏ đi trông
    # như sự cố, và người đọc tài liệu sẽ đi tải lại 1,13 GB không cần dùng.
    absent = list(manifest["missing_files"])
    out_of_scope = [n for n in absent if n in loader.OPTIONAL_FILES]
    truly_missing = [n for n in absent if n not in loader.OPTIONAL_FILES]

    if out_of_scope:
        lines += [
            "",
            f"Không lưu trên đĩa (ngoài phạm vi F04, **không cần tải**): "
            f"{', '.join(out_of_scope)}",
        ]
    if truly_missing:
        lines += ["", f"⚠️ THIẾU file bắt buộc: {', '.join(truly_missing)}"]

    lines += [
        "",
        "## Nhãn — `application_train.csv`",
        "",
        "| | |",
        "|---|---:|",
        f"| Số hồ sơ | {dist['n_rows']:,} |",
        f"| `TARGET = 1` (khó khăn trả nợ) | {dist['n_positive']:,} |",
        f"| `TARGET = 0` | {dist['n_negative']:,} |",
        f"| Tỉ lệ dương | {dist['positive_rate']:.4%} |",
        f"| `scale_pos_weight` (XGBoost) | {dist['scale_pos_weight']:.2f} |",
        f"| Accuracy của model đoán toàn `0` | {dist['majority_class_accuracy']:.4%} |",
        "",
        "Con số cuối là lý do **không dùng accuracy để chọn model** ở ML02: "
        "một model không học gì đã đạt hơn 91%.",
        "",
    ]

    if report is not None:
        lines += ["## Chất lượng dữ liệu", "", report.to_markdown(), ""]

    lines += [
        "## Dữ liệu synthetic của ML01",
        "",
        "ML01 không dùng dataset này. Cách sinh dân số hộ gia đình và hàm sinh nhãn "
        "`g(·)` mô tả trong `hfml.ml.ml01_recommendation` (F03).",
        "",
    ]
    return "\n".join(lines)


def write_dataset_doc(manifest: dict, report: QualityReport | None = None) -> Path:
    path = CONFIG.paths.docs / DATASET_DOC_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_dataset_doc(manifest, report), encoding="utf-8")
    log.info("Đã ghi tài liệu dataset → %s", path)
    return path
