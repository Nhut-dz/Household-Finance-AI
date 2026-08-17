"""AI-01 task 3, 6, 8 — Chạy rule, gom kết quả, xử lý lỗi (F05 · M05).

Đây là tầng ĐIỀU PHỐI. Nó không tính gì cả — mọi con số đến từ `rules/engine`
hoặc từ hai model. Ranh giới đó là điều làm cho §8.2 guardrail 1 kiểm chứng
được: prompt gửi cho LLM chỉ chứa JSON đã tính sẵn, và nếu tầng này tự tính
thì không ai còn phân biệt được con số nào do rule quyết, con số nào do
orchestrator nghĩ ra.

Nguyên tắc chịu lỗi: hỏng một phần, không hỏng cả kết quả
----------------------------------------------------------
Bốn nguồn kết quả — 5 rule, ML01, ML02 — độc lập với nhau. Một cái hỏng không
được kéo theo ba cái kia:

    Thiếu input bắt buộc  → dừng ngay, trả về lỗi có tên trường. Đây là ca DUY
                            NHẤT không có kết quả nào, vì không có gì để tính.
    Rule engine lỗi       → mất phần rule, ML vẫn chạy và vẫn báo cáo được
    ML01 / ML02 lỗi       → mất đúng model đó, ba phần kia giữ nguyên
    Model chưa có         → `available: false` kèm lý do, KHÔNG phải lỗi
    Xác suất không hợp lệ → coi như model lỗi; thà thiếu còn hơn báo cáo một
                            con số đã biết là sai
    Confidence thấp       → VẪN trả kết quả, nhưng gắn cờ để `llm` nói ra

Ca cuối là ca dễ làm sai nhất: im lặng bỏ qua nghĩa là người dùng đọc một
phỏng đoán mong manh như một kết luận chắc chắn (§8.1 task 7).

Output ổn định cho Epic AI-02
------------------------------
`AiResult.to_dict()` luôn trả về ĐỦ các khoá, kể cả khi một phần hỏng — phần
hỏng có `available: false` và `error`, không biến mất. Tầng tiêu thụ nhờ vậy
không phải viết `if "ml02" in result`, và một khoá thiếu không thể bị đọc
nhầm thành "rủi ro thấp".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from hfml.config import CONFIG
from hfml.logger import get_logger
from hfml.pipeline.normalizer import NormalizedInput, normalize_input
from hfml.pipeline.predictor import PredictionResult, predict_ml01, predict_ml02
from hfml.rules.engine import RuleEngine

log = get_logger(__name__)

#: Phiên bản schema của structured result. Epic AI-02 đọc khoá này để biết
#: mình đang nhận cấu trúc nào — đổi shape mà không đổi số là cách chắc chắn
#: nhất để tầng tiêu thụ hỏng âm thầm.
SCHEMA_VERSION = "1.0"

#: Năm rule, đúng thứ tự báo cáo.
RULE_CODES = ("RB01", "RB02", "RB03", "RB04", "RB05")


@dataclass
class AiResult:
    """Structured result — đầu ra duy nhất của Epic AI-01."""

    ok: bool = True
    input_summary: dict = field(default_factory=dict)
    rules: dict = field(default_factory=dict)
    overall_status: str | None = None
    ml01: dict = field(default_factory=dict)
    ml02: dict = field(default_factory=dict)
    warnings: list[dict] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)
    generated_at: str = ""
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict:
        """Luôn đủ khoá, kể cả khi một phần hỏng.

        Phần hỏng mang `available: false` chứ không biến mất — khoá thiếu rất
        dễ bị tầng trên đọc nhầm thành "không có rủi ro".
        """
        return {
            "ok": self.ok,
            "schema_version": self.schema_version,
            "generated_at": self.generated_at,
            "input_summary": self.input_summary,
            "rules": self.rules,
            "overall_status": self.overall_status,
            "ml01": self.ml01,
            "ml02": self.ml02,
            "warnings": self.warnings,
            "errors": self.errors,
        }

    @property
    def has_low_confidence(self) -> bool:
        """Có phần nào kém tin cậy không — `llm` phải nói ra nếu có."""
        return any(part.get("confidence", {}).get("low_confidence")
                   for part in (self.ml01, self.ml02) if part.get("available"))


def _error(code: str, message: str, field_name: str = "") -> dict:
    return {"code": code, "message": message, "field": field_name}


# --------------------------------------------------------------------------
# Task 3 — chạy 5 rule
# --------------------------------------------------------------------------
def run_rules(normalized: NormalizedInput) -> tuple[dict, str | None, dict | None]:
    """Chạy cả 5 rule qua `RuleEngine` có sẵn.

    KHÔNG cài lại rule nào ở đây. `RuleEngine` là nơi duy nhất định nghĩa
    RB01–RB05, và nó đã có test riêng ở F02 task 7 — viết lại một bản thứ hai
    là tạo ra hai nguồn sự thật cho cùng một phép tính.

    Trả `(rules, overall_status, error)`. Rule hỏng thì hai giá trị đầu rỗng
    nhưng ML vẫn chạy được — bốn nguồn kết quả độc lập với nhau.
    """
    profile = normalized.profile
    loan = normalized.loan

    try:
        engine = RuleEngine()
        result = engine.evaluate(
            profile,
            # RB05 cần ba số này. Ưu tiên form khoản vay; không có thì lấy từ
            # khối vay của hồ sơ. Không có cả hai thì RB05 tự xử lý thiếu.
            requested_loan=float(loan.loan_amount) if loan else (
                float(profile.loan_amount) if profile.loan_amount else None),
            asset_price=float(loan.asset_price) if loan else (
                float(profile.asset_price) if profile.asset_price else None),
            loan_term_months=(loan.loan_term_months if loan
                              else profile.loan_term_months),
            need=profile.financial_needs[0] if profile.financial_needs else None,
        )
    except Exception as exc:  # noqa: BLE001 — biên ngoài
        log.exception("RuleEngine lỗi")
        return {}, None, _error(
            "rule_engine_error", f"Tầng quy tắc lỗi: {exc}", "rules")

    rules = {code: result["rules"].get(code, {}) for code in RULE_CODES}
    return rules, result.get("overall_status"), None


# --------------------------------------------------------------------------
# Task 6 + 8 — gom kết quả, xử lý lỗi
# --------------------------------------------------------------------------
def analyze(payload: dict[str, Any]) -> AiResult:
    """Chạy toàn bộ pipeline inference cho một hồ sơ.

    Đây là điểm vào DUY NHẤT của Epic AI-01. Không ném ngoại lệ — mọi hỏng hóc
    đều thành một mục trong `errors`, vì ở biên của một service thì ngoại lệ
    nghĩa là 500 và người dùng không biết mình sai ở đâu.
    """
    result = AiResult(
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))

    # -- Task 1: chuẩn hoá input ------------------------------------------
    normalized = normalize_input(payload)
    result.input_summary = normalized.summary()
    result.warnings = [i.to_dict() for i in normalized.warnings]

    if not normalized.is_valid:
        # Ca DUY NHẤT không có kết quả nào — không có gì hợp lệ để tính.
        result.ok = False
        result.errors = [
            _error("invalid_input", issue.message, issue.field)
            for issue in normalized.errors
        ]
        result.ml01 = PredictionResult.unavailable(
            "ml01", "Đầu vào không hợp lệ.").to_dict()
        result.ml02 = PredictionResult.unavailable(
            "ml02", "Đầu vào không hợp lệ.").to_dict()
        log.warning("Dừng inference: %d lỗi đầu vào", len(result.errors))
        return result

    # -- Task 3: 5 rule ----------------------------------------------------
    rules, overall, rule_error = run_rules(normalized)
    result.rules = rules
    result.overall_status = overall
    if rule_error:
        result.errors.append(rule_error)

    # -- Task 4, 5: hai model ---------------------------------------------
    age = result.input_summary.get("age")
    result.ml01 = predict_ml01(normalized.profile, age).to_dict()
    result.ml02 = predict_ml02(normalized.profile, normalized.loan).to_dict()

    # -- Task 7: confidence thấp thì NÓI RA, không im lặng -----------------
    for part in (result.ml01, result.ml02):
        if part.get("available") and part.get("confidence", {}).get("low_confidence"):
            result.warnings.append({
                "field": part["model"],
                "code": "low_confidence",
                "severity": "warning",
                "message": _low_confidence_message(part),
            })

    # Model không có kết quả → warning, không phải error: phần rule và model
    # còn lại vẫn báo cáo được.
    #
    # Giữ nguyên `reason_code` của predictor thay vì gộp tất cả thành
    # "model_unavailable". Bốn mã đó không cùng loại: `missing_input` nghĩa là
    # bảo người dùng khai tiếp, còn `model_unavailable` là vấn đề vận hành.
    # Nói "hệ thống không khả dụng" cho người chưa điền form là nói sai.
    for part in (result.ml01, result.ml02):
        if not part.get("available") and part.get("error"):
            result.warnings.append({
                "field": part["model"],
                "code": part.get("reason_code") or "model_unavailable",
                "severity": "warning",
                "message": part["error"],
            })

    # `ok` phản ánh "có tính được gì không", không phải "có hoàn hảo không".
    # Mất ML02 vì người dùng chưa khai khoản vay là bình thường; đánh dấu cả
    # request là thất bại vì chuyện đó sẽ làm tầng trên xử lý sai.
    result.ok = not result.errors
    return result


def _low_confidence_message(part: dict) -> str:
    """Câu cảnh báo, khác nhau giữa hai model vì hai loại ngưỡng khác nhau."""
    if part["model"] == "ml01":
        return (
            f"Hồ sơ nằm gần ranh giới giữa các nhóm: xác suất cao nhất "
            f"{part['confidence']['confidence']:.1%}, dưới ngưỡng tin cậy "
            f"{part['confidence']['threshold']:.0%}. Nên đọc kèm kết quả của "
            "tầng quy tắc thay vì chỉ nhìn một nhãn.")
    return (
        "Xác suất rủi ro nằm sát ngưỡng phân loại "
        f"({part['probability']:.1%} so với ngưỡng "
        f"{part['confidence']['threshold']:.1%}) — thay đổi nhỏ ở dữ liệu đầu "
        "vào có thể làm đổi nhãn.")
