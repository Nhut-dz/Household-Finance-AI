"""AI-02 task 8 — Validate output của LLM (F05 · M06).

Đây là chốt chặn cuối. Mọi guardrail ở prompt đều là YÊU CẦU; file này là chỗ
KIỂM xem yêu cầu có được tuân thủ không. Prompt bảo "đừng bịa số" chỉ có giá
trị khi có ai đó đếm lại.

Năm phép kiểm, xếp theo mức nguy hiểm nếu bỏ sót
--------------------------------------------------
1. **Số bịa** — số xuất hiện trong câu trả lời mà không có trong
   `numeric_facts`. Nguy hiểm nhất vì nó trông y hệt số thật: người dùng không
   có cách nào phân biệt "35.000.000" lấy từ hồ sơ với "35.000.000" LLM tự
   nghĩ ra.
2. **Mâu thuẫn với model** — câu trả lời nói khác nhãn mà model đã trả về.
3. **Biến dự đoán thành chắc chắn** — "chắc chắn", "đảm bảo", "cam kết".
4. **Vượt phạm vi khuyến nghị** — gọi tên mã cổ phiếu, quỹ, ngân hàng cụ thể.
5. **Sai schema** — thiếu khoá bắt buộc, sai kiểu.

Vì sao đối chiếu số phải LINH HOẠT VỀ ĐỊNH DẠNG nhưng CHẶT VỀ GIÁ TRỊ
-----------------------------------------------------------------------
LLM viết "35 triệu", "35.000.000", "35,000,000" cho cùng một con số. Bắt khớp
chuỗi thì báo động giả liên tục và cuối cùng sẽ có người tắt phép kiểm đi.

Nên: chuẩn hoá về số rồi so theo dung sai tương đối. Đồng thời chấp nhận các
biến thể **suy ra được** mà không cần tính toán — phần trăm của một tỉ lệ
(0,163 → 16,3%), và số đã làm tròn hợp lý. Ngoài những dạng đó thì báo.

Số nhỏ được miễn, và đây là nhượng bộ có ý thức
------------------------------------------------
Số nguyên từ 0 đến 12 bị bỏ qua: chúng gần như luôn là số đếm trong câu văn
("3 việc nên làm", "2 con", "6 tháng") chứ không phải số liệu tài chính. Bắt
chúng thì mọi câu trả lời đều bị đánh dấu và phép kiểm mất tác dụng.

Cái giá: LLM có thể bịa "trong 6 tháng tới" mà không bị bắt. Chấp nhận được vì
đó là mốc thời gian trong lời khuyên, không phải số liệu về hộ gia đình.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Final

from hfml.llm.prompts import REQUIRED_KEYS
from hfml.logger import get_logger

log = get_logger(__name__)

#: Dung sai tương đối khi so một con số trong câu trả lời với `numeric_facts`.
#: 1% đủ rộng cho việc làm tròn khi trình bày, đủ chặt để bắt số bịa.
NUMBER_TOLERANCE: Final[float] = 0.01

#: Số nguyên ≤ ngưỡng này được bỏ qua — xem docstring đầu file.
SMALL_INTEGER_LIMIT: Final[int] = 12

#: Cụm biến dự đoán thành sự chắc chắn (§8.2 guardrail 4).
#:
#: Phải là CỤM gắn với một kết quả, không phải từ đơn. Bắt trần "đảm bảo" hay
#: "chắc chắn" thì báo động giả liên tục vì tiếng Việt dùng chúng như động từ
#: thường: "đảm bảo quỹ dự phòng 6 tháng" là lời khuyên đúng.
#:
#: Đã sập đúng bẫy đó khi chạy thật — phép kiểm bắt chính câu MIỄN TRỪ mà LLM
#: viết ra: *"các dự đoán mang tính xác suất và không đảm bảo chắc chắn diễn
#: biến trong tương lai"*. Câu đó là hành vi mong muốn, vậy mà bị đánh dấu vi
#: phạm và cả câu trả lời bị vứt.
CERTAINTY_PHRASES: Final[tuple[str, ...]] = (
    "chắc chắn sẽ", "chắc chắn có lãi", "chắc chắn thành công",
    "đảm bảo lợi nhuận", "đảm bảo sinh lời", "đảm bảo có lãi",
    "bảo đảm lợi nhuận", "cam kết lợi nhuận", "cam kết sinh lời",
    "nhất định sẽ", "không thể thua", "luôn luôn có lãi", "chắc chắn trả được",
)

#: Từ phủ định đứng ngay trước làm cụm đổi nghĩa hoàn toàn.
#: "KHÔNG chắc chắn sẽ…" là nói đúng mức, không phải khẳng định.
_NEGATIONS: Final[tuple[str, ...]] = ("không", "chưa", "chẳng", "khó")

#: Số ký tự nhìn ngược để tìm phủ định.
_NEGATION_WINDOW: Final[int] = 20

#: Dấu hiệu khuyến nghị vượt phạm vi: mã chứng khoán, tên sản phẩm cụ thể.
#: §8.2 guardrail 3 chỉ cho phép khuyến nghị theo LỚP tài sản.
_TICKER = re.compile(r"\b(?:mã\s+)?[A-Z]{3}\b")
OUT_OF_SCOPE_HINTS: Final[tuple[str, ...]] = (
    "cổ phiếu", "mã chứng khoán", "bitcoin", "crypto", "tiền ảo", "forex",
)

#: Số trong văn bản: 1.234.567 · 1,234,567 · 35 · 0,163 · 16,3%
#:
#: Bắt buộc KẾT THÚC bằng chữ số. Không có ràng buộc đó thì dấu câu bị nuốt
#: vào: "0.9982," khớp cả dấu phẩy, và bước chuẩn hoá đọc dấu phẩy cuối như
#: dấu ngăn nghìn → 0,9982 thành 9982. Con số hợp lệ bị báo là bịa, câu trả
#: lời đúng bị vứt. Đã sập đúng lỗi này khi chạy thật.
_NUMBER = re.compile(r"\d(?:[\d.,]*\d)?\s*%?")

#: Đơn vị nhân sau con số — "35 triệu" phải hiểu là 35.000.000.
_MULTIPLIERS: Final[tuple[tuple[str, float], ...]] = (
    ("nghìn tỷ", 1e12), ("nghìn tỉ", 1e12),
    ("tỷ", 1e9), ("tỉ", 1e9),
    ("triệu", 1e6),
    ("nghìn", 1e3), ("ngàn", 1e3),
)


@dataclass
class ValidationIssue:
    check: str
    message: str
    severity: str = "error"      # "error" → phải fallback; "warning" → ghi nhận

    def to_dict(self) -> dict:
        return {"check": self.check, "message": self.message,
                "severity": self.severity}


@dataclass
class ValidationReport:
    issues: list[ValidationIssue] = field(default_factory=list)
    ungrounded_numbers: list[str] = field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict:
        return {
            "valid": self.is_valid,
            "issues": [i.to_dict() for i in self.issues],
            "ungrounded_numbers": list(self.ungrounded_numbers),
        }


# --------------------------------------------------------------------------
# Trích và chuẩn hoá số
# --------------------------------------------------------------------------
def _parse(token: str) -> list[float]:
    """Mọi cách đọc hợp lý của một chuỗi số, dạng `float`.

    Trả về DANH SÁCH chứ không phải một giá trị, vì `0,163` thật sự mơ hồ:

        · quy ước Việt Nam — phẩy là dấu thập phân → 0,163
        · quy ước Anh Mỹ   — phẩy ngăn nghìn      → 163

    Không có ngữ cảnh nào trong chính chuỗi đó để phân xử. Đoán một cách rồi
    đoán sai thì phép kiểm **báo bịa cho một con số hợp lệ** và vứt cả câu trả
    lời đúng — đúng lỗi đã sập khi chạy thật với `0.9982`.

    Với một bộ canh, hướng sai lệch phải là "chấp nhận khi còn nghi ngờ": khả
    năng một số bịa tình cờ khớp dưới MỘT trong hai cách đọc là rất nhỏ, còn
    khả năng từ chối nhầm thì vừa xảy ra.
    """
    text = token.strip().rstrip("%").strip()
    if not text:
        return []

    candidates: list[str] = []
    if "," in text and "." in text:
        # Dấu xuất hiện SAU CÙNG là dấu thập phân — ca này không mơ hồ.
        if text.rfind(",") > text.rfind("."):
            candidates.append(text.replace(".", "").replace(",", "."))
        else:
            candidates.append(text.replace(",", ""))
    elif "," in text:
        candidates += [text.replace(",", "."), text.replace(",", "")]
    elif "." in text:
        candidates += [text, text.replace(".", "")]
    else:
        candidates.append(text)

    values = []
    for candidate in candidates:
        try:
            values.append(float(candidate))
        except ValueError:
            continue
    return values


def extract_numbers(text: str) -> list[tuple[str, list[float]]]:
    """Mọi con số trong văn bản, kèm MỌI cách đọc hợp lý của nó.

    Nhân theo đơn vị đứng ngay sau: "35 triệu" → 35.000.000. Không nhân thì
    "35 triệu" bị so với 35 và báo bịa, dù nó hoàn toàn đúng.
    """
    found: list[tuple[str, list[float]]] = []
    for match in _NUMBER.finditer(text):
        token = match.group()
        values = _parse(token)
        if not values:
            continue

        tail = text[match.end():match.end() + 12].lower().lstrip()
        for word, factor in _MULTIPLIERS:
            if tail.startswith(word):
                values = [v * factor for v in values]
                token = f"{token} {word}"
                break

        found.append((token.strip(), values))
    return found


def _matches_any(value: float, allowed: list[float]) -> bool:
    """Giá trị có khớp một mục nào trong danh sách trắng không.

    Chấp nhận ba dạng, tất cả đều **suy ra được mà không cần tính**:
        · chính con số đó
        · dạng phần trăm của một tỉ lệ (0,163 ↔ 16,3)
        · số đã làm tròn khi trình bày
    """
    for fact in allowed:
        for candidate in (fact, fact * 100.0, fact / 100.0):
            if candidate == 0:
                if abs(value) < 1e-9:
                    return True
                continue
            if abs(value - candidate) / abs(candidate) <= NUMBER_TOLERANCE:
                return True
    return False


def check_numbers(text: str, numeric_facts: dict[str, float]) -> list[str]:
    """Các con số trong văn bản KHÔNG có căn cứ trong context.

    Đây là hiện thực của §8.2 guardrail 2. Trả về danh sách chuỗi bịa; rỗng
    nghĩa là mọi con số đều truy được về dữ liệu đã tính sẵn.
    """
    allowed = list(numeric_facts.values())
    ungrounded = []

    for token, values in extract_numbers(text):
        # Số đếm nhỏ trong câu văn — xem docstring đầu file.
        if any(v == int(v) and 0 <= v <= SMALL_INTEGER_LIMIT for v in values):
            continue
        # Khớp theo BẤT KỲ cách đọc nào — xem docstring của `_parse`.
        if not any(_matches_any(v, allowed) for v in values):
            ungrounded.append(token)

    return ungrounded


# --------------------------------------------------------------------------
# Các phép kiểm còn lại
# --------------------------------------------------------------------------
def _all_text(payload: dict) -> str:
    """Gộp mọi phần văn bản của câu trả lời để quét một lượt."""
    parts = [str(payload.get("explanation", ""))]
    for item in payload.get("recommendations") or []:
        if isinstance(item, dict):
            parts += [str(item.get("action", "")), str(item.get("reason", ""))]
    parts += [str(x) for x in (payload.get("caveats") or [])]
    return "\n".join(parts)


def check_schema(payload: Any) -> list[ValidationIssue]:
    """Đúng kiểu và đủ khoá bắt buộc."""
    if not isinstance(payload, dict):
        return [ValidationIssue("schema", f"Không phải object JSON: {type(payload).__name__}")]

    issues = []
    for key in REQUIRED_KEYS:
        if key not in payload:
            issues.append(ValidationIssue("schema", f"Thiếu khoá bắt buộc `{key}`."))

    if "explanation" in payload and not str(payload["explanation"]).strip():
        issues.append(ValidationIssue("schema", "`explanation` rỗng."))

    recommendations = payload.get("recommendations")
    if recommendations is not None and not isinstance(recommendations, list):
        issues.append(ValidationIssue(
            "schema", "`recommendations` phải là danh sách."))
    elif isinstance(recommendations, list):
        for index, item in enumerate(recommendations):
            if not isinstance(item, dict):
                issues.append(ValidationIssue(
                    "schema", f"recommendations[{index}] không phải object."))
            elif not str(item.get("action", "")).strip():
                issues.append(ValidationIssue(
                    "schema", f"recommendations[{index}] thiếu `action`."))
    return issues


def check_contradiction(text: str, context) -> list[ValidationIssue]:
    """Câu trả lời có nói ngược nhãn model không.

    Model trả `LOW_RISK` mà câu trả lời nói "rủi ro cao" là mâu thuẫn thẳng —
    và đó là dạng sai tệ nhất, vì người dùng tin vào câu chữ chứ không đọc JSON.
    """
    issues = []
    lowered = text.lower()

    ml02 = context.ml02 or {}
    if ml02.get("available"):
        # Chỉ là mâu thuẫn khi câu trả lời nhắc nhãn NGƯỢC LẠI mà KHÔNG hề
        # nhắc nhãn model đã trả — cùng luật với nhánh ML01 bên dưới.
        #
        # Thiếu vế thứ hai thì phép kiểm bắt nhầm câu trả lời tốt nhất: model
        # trả LOW_RISK, LLM viết "phân loại ở nhãn 'Rủi ro thấp' với xác suất
        # 0.9796 (xác suất rủi ro cao chỉ 0.0204)" — nêu đúng nhãn kèm xác
        # suất bù, chính xác hơn hẳn mức tối thiểu, vậy mà bị đánh trượt hai
        # lượt liền và hạ cấp về template. Đã xảy ra thật khi chạy demo.
        nhan = ml02.get("label")
        nhan_vi = str(ml02.get("label_vi", "")).lower()
        nhac_dung_nhan = bool(nhan_vi) and nhan_vi in lowered

        nguoc = {"LOW_RISK": "rủi ro cao", "HIGH_RISK": "rủi ro thấp"}.get(nhan)
        if nguoc and nguoc in lowered and not nhac_dung_nhan:
            issues.append(ValidationIssue(
                "contradiction",
                f"Model trả {nhan} nhưng câu trả lời nói '{nguoc}' mà không "
                f"nhắc tới nhãn thật '{ml02.get('label_vi')}'."))

    ml01 = context.ml01 or {}
    if ml01.get("available") and ml01.get("label_vi"):
        # Nhãn ML01 phải được nhắc tới, không được thay bằng nhóm khác.
        khac = {"Cần xử lý khẩn cấp dòng tiền", "Cần tập trung xử lý nợ",
                "Cần xây dựng quỹ dự phòng", "Có thể hướng tới tăng trưởng"}
        khac.discard(ml01["label_vi"])
        for nhan_khac in khac:
            if nhan_khac.lower() in lowered and ml01["label_vi"].lower() not in lowered:
                issues.append(ValidationIssue(
                    "contradiction",
                    f"Model trả '{ml01['label_vi']}' nhưng câu trả lời chỉ "
                    f"nhắc tới '{nhan_khac}'."))
                break
    return issues


def _is_negated(text: str, position: int) -> bool:
    """Cụm ở vị trí này có bị phủ định ngay trước không."""
    window = text[max(0, position - _NEGATION_WINDOW):position]
    return any(word in window for word in _NEGATIONS)


def check_certainty(text: str) -> list[ValidationIssue]:
    """Có biến dự đoán thành sự chắc chắn không (§8.2 guardrail 4).

    Bỏ qua cụm bị phủ định: "KHÔNG đảm bảo chắc chắn" là nói đúng mức, và đó
    chính là câu miễn trừ mà prompt yêu cầu LLM viết ra.
    """
    lowered = text.lower()
    hits = []
    for phrase in CERTAINTY_PHRASES:
        start = 0
        while (index := lowered.find(phrase, start)) >= 0:
            if not _is_negated(lowered, index):
                hits.append(phrase)
                break
            start = index + len(phrase)

    return [ValidationIssue(
        "certainty",
        f"Khẳng định kết quả là chắc chắn: {', '.join(hits)}.")] if hits else []


def check_scope(text: str) -> list[ValidationIssue]:
    """Khuyến nghị có vượt phạm vi cho phép không (§8.2 guardrail 3).

    Chỉ được khuyến nghị theo LỚP tài sản. Gọi tên một mã cụ thể là tư vấn đầu
    tư có định danh — thứ hệ thống này không được phép làm.
    """
    lowered = text.lower()
    hits = [hint for hint in OUT_OF_SCOPE_HINTS if hint in lowered]
    if hits:
        return [ValidationIssue(
            "scope", f"Nhắc tới sản phẩm ngoài phạm vi: {', '.join(hits)}.")]
    return []


# --------------------------------------------------------------------------
# Chạy đủ năm phép kiểm
# --------------------------------------------------------------------------
def validate(payload: Any, context) -> ValidationReport:
    """Kiểm một câu trả lời của LLM trước khi đưa ra cho người dùng.

    `is_valid` False thì tầng gọi PHẢI fallback hoặc sinh lại — không được
    "sửa nhẹ rồi dùng tạm". Một câu trả lời chứa số bịa mà được sửa qua loa
    vẫn là câu trả lời không truy được về dữ liệu.
    """
    report = ValidationReport()
    report.issues.extend(check_schema(payload))

    if not isinstance(payload, dict) or report.errors:
        return report

    text = _all_text(payload)

    ungrounded = check_numbers(text, context.numeric_facts)
    if ungrounded:
        report.ungrounded_numbers = ungrounded
        report.issues.append(ValidationIssue(
            "ungrounded_number",
            f"{len(ungrounded)} con số không có trong dữ liệu đã cấp: "
            f"{', '.join(ungrounded[:5])}"))

    report.issues.extend(check_contradiction(text, context))
    report.issues.extend(check_certainty(text))
    report.issues.extend(check_scope(text))

    if not report.is_valid:
        log.warning("Câu trả lời LLM không đạt: %s",
                    "; ".join(i.check for i in report.errors))
    return report
