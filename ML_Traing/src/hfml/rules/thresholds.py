"""Nạp và quản lý cấu hình ngưỡng (thresholds) cho F02 Rule-Based Engine.

Đọc từ `config/rules.yaml` — tuyệt đối KHÔNG hardcode hệ số trong mã nguồn Python.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import yaml

from hfml.config import ROOT
from hfml.logger import get_logger

log = get_logger("hfml.rules.thresholds")

RULES_CONFIG_PATH = ROOT / "config" / "rules.yaml"


@dataclass
class RB01Thresholds:
    zero_tolerance: float = 0.0


@dataclass
class RB02Thresholds:
    dti_excellent_max: float = 0.20
    dti_good_max: float = 0.36
    dti_warning_max: float = 0.50
    emergency_critical_less: float = 1.0
    emergency_warning_less: float = 3.0
    emergency_good_min: float = 3.0
    emergency_excellent_min: float = 6.0
    savings_rate_critical_less: float = 0.0
    savings_rate_warning_less: float = 0.10
    savings_rate_good_min: float = 0.10
    savings_rate_excellent_min: float = 0.20


@dataclass
class RB03Thresholds:
    max_surplus_allocation_ratio: float = 0.80
    default_timeline_months: int = 12


@dataclass
class RB04Thresholds:
    needs_ratio: float = 0.50
    wants_ratio: float = 0.30
    savings_ratio: float = 0.20


@dataclass
class RB05Thresholds:
    max_dti: float = 0.40
    max_ltv: float = 0.70
    assumed_annual_interest_rate: float = 0.10


@dataclass
class RuleThresholds:
    rb01: RB01Thresholds = field(default_factory=RB01Thresholds)
    rb02: RB02Thresholds = field(default_factory=RB02Thresholds)
    rb03: RB03Thresholds = field(default_factory=RB03Thresholds)
    rb04: RB04Thresholds = field(default_factory=RB04Thresholds)
    rb05: RB05Thresholds = field(default_factory=RB05Thresholds)
    raw_config: dict = field(default_factory=dict)


def load_rule_thresholds(config_path: Path | str | None = None) -> RuleThresholds:
    """Nạp cấu hình ngưỡng từ file YAML."""
    path = Path(config_path) if config_path else RULES_CONFIG_PATH
    if not path.exists():
        log.warning("File %s không tồn tại, dùng ngưỡng mặc định.", path)
        return RuleThresholds()

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        thresh_data = data.get("thresholds", {})

        rb01_d = thresh_data.get("rb01_cashflow", {})
        rb01 = RB01Thresholds(
            zero_tolerance=float(rb01_d.get("zero_tolerance", 0.0))
        )

        rb02_d = thresh_data.get("rb02_health", {})
        dti_d = rb02_d.get("dti", {})
        emerg_d = rb02_d.get("emergency_months", {})
        sav_d = rb02_d.get("savings_rate", {})
        rb02 = RB02Thresholds(
            dti_excellent_max=float(dti_d.get("excellent_max", 0.20)),
            dti_good_max=float(dti_d.get("good_max", 0.36)),
            dti_warning_max=float(dti_d.get("warning_max", 0.50)),
            emergency_critical_less=float(emerg_d.get("critical_less", 1.0)),
            emergency_warning_less=float(emerg_d.get("warning_less", 3.0)),
            emergency_good_min=float(emerg_d.get("good_min", 3.0)),
            emergency_excellent_min=float(emerg_d.get("excellent_min", 6.0)),
            savings_rate_critical_less=float(sav_d.get("critical_less", 0.0)),
            savings_rate_warning_less=float(sav_d.get("warning_less", 0.10)),
            savings_rate_good_min=float(sav_d.get("good_min", 0.10)),
            savings_rate_excellent_min=float(sav_d.get("excellent_min", 0.20)),
        )

        rb03_d = thresh_data.get("rb03_savings_goal", {})
        rb03 = RB03Thresholds(
            max_surplus_allocation_ratio=float(rb03_d.get("max_surplus_allocation_ratio", 0.80)),
            default_timeline_months=int(rb03_d.get("default_timeline_months", 12)),
        )

        rb04_d = thresh_data.get("rb04_503020", {})
        rb04 = RB04Thresholds(
            needs_ratio=float(rb04_d.get("needs_ratio", 0.50)),
            wants_ratio=float(rb04_d.get("wants_ratio", 0.30)),
            savings_ratio=float(rb04_d.get("savings_ratio", 0.20)),
        )

        rb05_d = thresh_data.get("rb05_loan_capacity", {})
        rb05 = RB05Thresholds(
            max_dti=float(rb05_d.get("max_dti", 0.40)),
            max_ltv=float(rb05_d.get("max_ltv", 0.70)),
            assumed_annual_interest_rate=float(rb05_d.get("assumed_annual_interest_rate", 0.10)),
        )

        return RuleThresholds(
            rb01=rb01,
            rb02=rb02,
            rb03=rb03,
            rb04=rb04,
            rb05=rb05,
            raw_config=data,
        )
    except Exception as e:
        log.error("Lỗi khi đọc %s: %s, dùng ngưỡng mặc định.", path, e)
        return RuleThresholds()


# Global instance dùng mặc định
DEFAULT_THRESHOLDS = load_rule_thresholds()
