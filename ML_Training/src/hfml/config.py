"""Nạp cấu hình tập trung.

Ưu tiên: biến môi trường (.env / OS) > config/config.yaml > mặc định.
Giữ mọi đường dẫn & tham số ở một nơi để module khác không hardcode.

Ngưỡng của tầng rule KHÔNG nằm ở đây — chúng ở `config/rules.yaml` và được
`hfml.rules.thresholds` nạp riêng, vì mỗi ngưỡng còn kèm nguồn trích dẫn
(PLAN.md §5).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Gốc project = .../Household-Finance-ML-Python (lên 3 cấp từ file này)
ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Paths:
    root: Path = ROOT
    config: Path = ROOT / "config"
    # Home Credit tải từ Kaggle — không commit vào git.
    dataset: Path = ROOT / "dataset" / "home-credit-default-risk"
    data_raw: Path = ROOT / "data" / "raw"
    data_interim: Path = ROOT / "data" / "interim"
    data_processed: Path = ROOT / "data" / "processed"
    #: Output của một lần train: metric, kết quả, cấu hình, model artifact.
    #: KHÔNG chứa log — log đi vào `logs/`.
    #:
    #: Đây là NƠI DUY NHẤT chứa output của tầng ML. Trước 12/08/2026 còn hai
    #: thư mục `artifacts/` và `experiments/` làm việc đó; chúng đã bị bỏ vì
    #: hai nơi cùng giữ một loại file là chỗ nạp nhầm bản cũ mà không báo lỗi.
    runs: Path = ROOT / "src" / "training" / "runs"
    docs: Path = ROOT / "docs"
    logs: Path = ROOT / "logs"

    def ensure(self) -> None:
        """Tạo các thư mục ghi ra. Không tạo `dataset` — nó phải tải sẵn."""
        for p in (self.data_raw, self.data_interim, self.data_processed,
                  self.runs, self.logs):
            p.mkdir(parents=True, exist_ok=True)


@dataclass
class Database:
    """Kết nối PostgreSQL (pgAdmin 4) chứa dữ liệu nghiệp vụ Household Finance.

    Host/port/name/user để trong config/config.yaml; riêng `password` chỉ đọc
    từ env `HFML_DB_PASSWORD` để không commit secret. Server dev local dùng
    trust auth nên password rỗng vẫn kết nối được.
    """
    host: str = "127.0.0.1"
    port: int = 5432
    name: str = "Household_Finance_V2_Dev"
    user: str = "postgres"
    password: str = ""

    def url(self) -> str:
        """Chuỗi kết nối SQLAlchemy (psycopg2). Escape password cho an toàn."""
        from urllib.parse import quote_plus
        auth = self.user if not self.password else f"{self.user}:{quote_plus(self.password)}"
        return f"postgresql+psycopg2://{auth}@{self.host}:{self.port}/{self.name}"


@dataclass
class Config:
    paths: Paths = field(default_factory=Paths)
    # Kết nối PostgreSQL nguồn dữ liệu nghiệp vụ (pgAdmin 4).
    database: Database = field(default_factory=Database)
    # Seed dùng chung cho MỌI bước có ngẫu nhiên: sinh synthetic, split, train.
    # F06 task 6 kiểm tra chạy lại với seed này ra metric trùng đến 4 chữ số.
    random_seed: int = 42

    # Ngưỡng hạ cấp khi ML thiếu tin cậy (PLAN.md §8.1 task 7).
    # max(predict_proba) dưới ngưỡng → dùng kết luận rule + low_confidence.
    confidence_threshold: float = 0.60

    # Tham số train dùng chung cho ML01 và ML02 (PLAN.md §11).
    # Task 5: train 70% · validation 15% · test 15%. Phần train là số còn lại,
    # không khai riêng để ba con số không thể lệch tổng.
    #
    # Không có `n_splits`: ML01 bỏ K-Fold Cross-Validation từ 14/08/2026, chọn
    # model theo macro-F1 trên tập validation.
    training: dict = field(default_factory=lambda: {
        "val_size": 0.15,
        "test_size": 0.15,
    })

    # Logging — xem hfml.logger. `file` là đường dẫn tương đối từ gốc project.
    logging: dict = field(default_factory=lambda: {
        "level": "INFO",
        "console": True,
        "file": "logs/hfml.log",
    })

    # Cấu hình LLM (Gemini). API key đọc từ env để không commit secret.
    # Bỏ trống key → tầng llm chạy chế độ template, pipeline vẫn chạy đủ.
    llm: dict = field(default_factory=lambda: {
        "model": "gemini-3.6-flash",
        "max_tokens": 1500,
        "temperature": 0.3,
        # Giới hạn thời gian một lượt gọi, và ngân sách cho cả lượt sinh
        # (gồm lần sinh lại). Xem config.yaml về lý do phải có hai con số.
        "timeout_seconds": 18,
        "budget_seconds": 32,
    })
    llm_api_key: str = ""

    # Cấu hình runtime của module inference (Epic AI-03). Nội dung được
    # `hfml.inference.settings` đọc và diễn giải — ở đây chỉ giữ nguyên khối
    # dict để `config.py` không phải biết gì về tầng inference.
    inference: dict = field(default_factory=dict)


def load_config(path: str | Path | None = None) -> Config:
    """Đọc config.yaml (nếu có) rồi phủ biến môi trường lên trên."""
    # Nạp .env vào os.environ trước khi đọc getenv bên dưới. `override=False`
    # để biến đã đặt sẵn ở OS/CI thắng file .env của máy dev.
    load_dotenv(ROOT / ".env", override=False)
    cfg = Config()
    yaml_path = Path(path) if path else ROOT / "config" / "config.yaml"
    if yaml_path.exists():
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        cfg.random_seed = data.get("random_seed", cfg.random_seed)
        cfg.confidence_threshold = data.get("confidence_threshold", cfg.confidence_threshold)
        cfg.training.update(data.get("training", {}))
        cfg.logging.update(data.get("logging", {}))
        cfg.llm.update(data.get("llm", {}))
        cfg.inference.update(data.get("inference", {}) or {})
        db = data.get("database", {}) or {}
        cfg.database.host = db.get("host", cfg.database.host)
        cfg.database.port = int(db.get("port", cfg.database.port))
        cfg.database.name = db.get("name", cfg.database.name)
        cfg.database.user = db.get("user", cfg.database.user)

    # Env override (ưu tiên cao nhất)
    if os.getenv("HFML_RANDOM_SEED"):
        cfg.random_seed = int(os.environ["HFML_RANDOM_SEED"])
    if os.getenv("HFML_LOG_LEVEL"):
        cfg.logging["level"] = os.environ["HFML_LOG_LEVEL"]
    cfg.llm_api_key = os.getenv("GEMINI_API_KEY", cfg.llm_api_key)
    # Kết nối DB — cho phép env phủ lên yaml. Password chỉ từ env.
    cfg.database.host = os.getenv("HFML_DB_HOST", cfg.database.host)
    if os.getenv("HFML_DB_PORT"):
        cfg.database.port = int(os.environ["HFML_DB_PORT"])
    cfg.database.name = os.getenv("HFML_DB_NAME", cfg.database.name)
    cfg.database.user = os.getenv("HFML_DB_USER", cfg.database.user)
    cfg.database.password = os.getenv("HFML_DB_PASSWORD", cfg.database.password)

    cfg.paths.ensure()
    return cfg


# Instance dùng chung, import ở nơi khác: from hfml.config import CONFIG
CONFIG = load_config()
