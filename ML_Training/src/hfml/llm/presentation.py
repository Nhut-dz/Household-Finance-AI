"""Tầng trình bày — biên cuối cùng giữa hệ thống và mắt người dùng (F05 · M06).

Vì sao tầng này phải tồn tại riêng
------------------------------------
Rule và ML nói bằng một thứ tiếng khác với người dùng. `RB01`, `CRITICAL`,
`DEFICIT`, `REJECTED`, `ml01_xgboost_vfinal` là từ vựng NỘI BỘ: chúng có nghĩa
với người viết mã, và không có nghĩa với người đang hỏi tháng này nhà mình có
dư tiền không.

Chừng nào những chuỗi đó còn nằm trong `AiResult` — và chúng phải nằm đó, vì
tầng kiểm chứng cần chúng — thì luôn có một đường nào đó rò ra màn hình. Đã rò
qua đúng ba đường cùng lúc:

    api/main.py `/advise`      ghép thẳng trạng thái rule vào chuỗi f-string
    llm/client.py `_template`  ghép mã rule kèm trạng thái vào từng gạch đầu dòng
    llm                        chép lại `CRITICAL` nó đọc được trong prompt

Ba chỗ, ba nguyên nhân khác nhau, nhưng cùng một hậu quả. Nên chỗ chặn không
thể là "sửa cả ba rồi cẩn thận hơn" — chỗ chặn phải là MỘT hàm mà mọi câu chữ
đi ra ngoài đều phải qua, và nó chặn được cả đường thứ tư chưa ai nghĩ tới.

Ba lớp, và chỉ lớp thứ ba là bảo đảm
--------------------------------------
    1. prompt cấm LLM nhắc mã          giảm số lần phải sinh lại
    2. context mang sẵn chữ tiếng Việt cho LLM từ đúng để dùng
    3. `to_plain_text` quét lần cuối   BẢO ĐẢM — không phụ thuộc ai tuân thủ

Hai lớp đầu là yêu cầu, lớp thứ ba là phép kiểm. Bỏ lớp thứ ba đi thì mọi thứ
lại quay về chỗ dựa vào thiện chí của một model ở đầu kia đường mạng.

Vì sao ra chữ thuần chứ không phải Markdown
---------------------------------------------
Màn Chatbot render bằng `whitespace-pre-line` và không có bộ dựng Markdown, nên
chuỗi in đậm hiện nguyên hai dấu sao. Tệ hơn: nút "Đọc bằng giọng nói" đưa
thẳng chuỗi đó cho `SpeechSynthesis`, và người dùng nghe máy đọc cả dấu sao.

Nên tầng này trả chữ thuần. Chỗ cần nhấn mạnh thì dùng xuống dòng và gạch đầu
dòng — hai thứ `whitespace-pre-line` hiển thị đúng và máy đọc bỏ qua êm.
"""
from __future__ import annotations

import re
from typing import Final

#: Tên nghiệp vụ của 5 rule. Dùng khi buộc phải nhắc tới một rule cụ thể —
#: "khả năng vay" nói được điều mà "RB05" không nói được với ai ngoài nhóm.
RULE_NAMES_VI: Final[dict[str, str]] = {
    "RB01": "dòng tiền hằng tháng",
    "RB02": "sức khỏe tài chính",
    "RB03": "mục tiêu tiết kiệm",
    "RB04": "phân bổ ngân sách",
    "RB05": "khả năng vay",
}

#: Trạng thái nội bộ → chữ tiếng Việt, tra theo (mã rule, trạng thái).
#:
#: Khoá phải là CẶP chứ không phải riêng trạng thái: `WARNING` của RB02 nghĩa
#: là sức khỏe tài chính đáng lo, còn `WARNING` của RB05 nghĩa là khoản vay
#: vượt khả năng trả. `BALANCED` của RB01 là dòng tiền vừa đủ, của RB04 là
#: ngân sách chia hợp lý. Gộp chung một bảng thì một nửa số câu sẽ nói sai ý.
STATUS_VI: Final[dict[tuple[str, str], str]] = {
    # Đánh giá tổng quan gộp từ cả 5 rule
    ("OVERALL", "CRITICAL"): "cần xử lý ngay",
    ("OVERALL", "WARNING"): "cần lưu ý",
    ("OVERALL", "STABLE"): "ổn định",
    ("OVERALL", "EXCELLENT"): "rất tốt",

    ("RB01", "POSITIVE"): "đang dư",
    ("RB01", "BALANCED"): "vừa đủ, không dư không thiếu",
    ("RB01", "DEFICIT"): "đang thiếu hụt",

    ("RB02", "CRITICAL"): "cần xử lý ngay",
    ("RB02", "WARNING"): "cần lưu ý",
    ("RB02", "GOOD"): "tốt",
    ("RB02", "EXCELLENT"): "rất tốt",

    ("RB03", "COMPLETED"): "đã đạt mục tiêu",
    ("RB03", "FEASIBLE"): "trong tầm với",
    ("RB03", "STRETCHED"): "khá chật vật",
    ("RB03", "INFEASIBLE"): "chưa khả thi với mức dư hiện tại",

    ("RB04", "BALANCED"): "chia hợp lý",
    ("RB04", "OVERBUDGET"): "chi vượt mức khuyến nghị",
    ("RB04", "UNDER_SAVING"): "phần để dành còn thấp hơn khuyến nghị",

    ("RB05", "APPROVED"): "khoản vay nằm trong khả năng trả",
    ("RB05", "ELIGIBLE"): "đủ điều kiện vay",
    ("RB05", "WARNING"): "khoản vay vượt khả năng trả",
    ("RB05", "REJECTED"): "chưa nên vay thêm lúc này",

    # Năng lực thế chấp (`RB05.value.collateral_quality`). Cần mục riêng chứ
    # không dùng bản dịch chung: bản chung cho `HIGH` → "cao" và `NONE` →
    # "chưa có", ghép vào câu thành "Mức chưa có" — đúng nghĩa nhưng đọc như
    # một câu chưa viết xong. Ở đây nói thẳng ra tài sản nào tạo nên mức đó.
    ("COLLATERAL", "HIGH"): "cao — có bất động sản làm tài sản bảo đảm",
    ("COLLATERAL", "MEDIUM"): "trung bình — có phương tiện làm tài sản bảo đảm",
    ("COLLATERAL", "LOW"): "thấp — tài sản hiện có khó dùng làm bảo đảm",
    ("COLLATERAL", "NONE"): "chưa có tài sản nào dùng làm bảo đảm",
}

#: Bản dịch không cần ngữ cảnh, dùng cho bước quét cuối — khi một mã đã lọt
#: vào câu chữ thì không còn biết nó vốn thuộc rule nào nữa.
_GENERIC_VI: Final[dict[str, str]] = {
    "CRITICAL": "cần xử lý ngay",
    "WARNING": "cần lưu ý",
    "STABLE": "ổn định",
    "EXCELLENT": "rất tốt",
    "GOOD": "tốt",
    "POSITIVE": "đang dư",
    "BALANCED": "cân đối",
    "DEFICIT": "đang thiếu hụt",
    "COMPLETED": "đã đạt",
    "FEASIBLE": "khả thi",
    "STRETCHED": "chật vật",
    "INFEASIBLE": "chưa khả thi",
    "OVERBUDGET": "vượt mức khuyến nghị",
    "UNDER_SAVING": "để dành còn thấp",
    "APPROVED": "trong khả năng trả",
    "ELIGIBLE": "đủ điều kiện",
    "REJECTED": "chưa nên vay thêm",
    # Nhãn ML01
    "EMERGENCY": "cần xử lý khẩn cấp dòng tiền",
    "DEBT_FOCUS": "cần tập trung xử lý nợ",
    "BUILD_BUFFER": "cần xây dựng quỹ dự phòng",
    "GROWTH": "có thể hướng tới tăng trưởng",
    # Nhãn ML02
    "LOW_RISK": "rủi ro thấp",
    "HIGH_RISK": "rủi ro cao",
    # Năng lực thế chấp (RB05) và mức DTI (RB02)
    "HIGH": "cao",
    "MEDIUM": "trung bình",
    "LOW": "thấp",
    "NONE": "chưa có",
}

#: Khoá kỹ thuật không bao giờ được xuất hiện trong câu chữ. Chúng là tên
#: trường của `AiResult`, lọt ra khi một chuỗi f-string in cả khoá lẫn giá trị.
_TECHNICAL_KEYS: Final[tuple[str, ...]] = (
    "label_vi", "label", "status_vi", "status", "available", "reason_code",
    "model_version", "schema_version", "generated_at", "numeric_facts",
    "overall_status", "intent_code", "low_confidence", "prompt_version",
    "summary_vi", "message_key", "threshold", "probabilities",
)

_RULE_CODE = re.compile(r"\bRB0[1-5]\b")
_ALLCAPS_TOKEN = re.compile(r"\b[A-Z][A-Z_]{2,}\b")
#: Slug artifact — `ml01_xgboost_vfinal`, `ml02_xgboost_reduced_vfinal`.
_MODEL_SLUG = re.compile(r"\bml0[12]_[a-z0-9_]+\b", re.IGNORECASE)
#: Đường dẫn khoá trong context — `rules.RB02.value.dti`, `ml01.probability`.
_DOTTED_PATH = re.compile(
    r"\b(?:rules|ml01|ml02|profile|question|input_summary)"
    r"(?:\.[A-Za-z_][A-Za-z0-9_]*|\[\d+\])+")
#: Chỗ trống của f-string chưa được điền.
_PLACEHOLDER = re.compile(r"\{[A-Za-z_][A-Za-z0-9_.\[\]]*\}")
#: Giá trị rỗng của Python/JSON lọt vào câu chữ.
_EMPTY_VALUE = re.compile(r"\b(?:None|null|nan|NaN|N/A|undefined)\b")


def label_status(code: str, status: str | None) -> str:
    """Trạng thái nội bộ của một rule → chữ tiếng Việt.

    `code` là `RB01`–`RB05` hoặc `OVERALL`. Không tra được thì trả chuỗi rỗng
    chứ KHÔNG trả lại `status` thô: trả về mã gốc nghĩa là mở lại đúng đường
    rò mà cả module này sinh ra để bịt.
    """
    if not status:
        return ""
    key = (code.upper(), str(status).upper())
    return STATUS_VI.get(key) or _GENERIC_VI.get(str(status).upper(), "")


def rule_name(code: str) -> str:
    """Tên nghiệp vụ của một rule, dùng thay cho mã khi phải nhắc tới nó."""
    return RULE_NAMES_VI.get(code.upper(), "")


def percent(value: float | None, digits: int = 1) -> str:
    """Tỉ lệ 0–1 → phần trăm kiểu Việt Nam: 0.9449 thành "94,5%".

    Xác suất thô là thứ người dùng đọc sai nhiều nhất: `0.9449` trông như một
    con số tiền hoặc một mã, không ai đọc ra "gần như chắc chắn". Tầng kiểm số
    ở `validator` đã chấp nhận dạng phần trăm (`_matches_any` xét cả
    `fact * 100`) nên đổi cách trình bày không làm câu trả lời bị đánh trượt.
    """
    if value is None:
        return ""
    return f"{value * 100:.{digits}f}".replace(".", ",") + "%"


def money(value: float | None) -> str:
    """Số tiền thành "3.000.000đ", dấu chấm ngăn nghìn theo quy ước Việt Nam."""
    if value is None:
        return ""
    return f"{value:,.0f}".replace(",", ".") + "đ"


# --------------------------------------------------------------------------
# Quét cuối
# --------------------------------------------------------------------------
def _strip_markdown(text: str) -> str:
    """Bỏ cú pháp Markdown, giữ nguyên nội dung.

    Không đổi sang thẻ HTML mà bỏ hẳn dấu: đích đến là một thẻ đoạn văn render
    bằng `whitespace-pre-line`, nơi mọi thứ đều là chữ thuần.
    """
    text = re.sub(r"```[a-zA-Z]*\n?", "", text)
    text = text.replace("```", "")
    text = re.sub(r"`([^`]*)`", r"\1", text)
    # Liên kết dạng [nhãn](đích) — giữ nhãn, bỏ đích vì URL không đọc ra được.
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    # Đường kẻ ngang trên một dòng riêng.
    text = re.sub(r"^[ \t]*([-*_])\1{2,}[ \t]*$", "", text, flags=re.MULTILINE)
    # Tiêu đề mở đầu bằng dấu thăng.
    text = re.sub(r"^[ \t]*#{1,6}[ \t]*", "", text, flags=re.MULTILINE)
    # Trích dẫn mở đầu bằng dấu lớn hơn.
    text = re.sub(r"^[ \t]*>[ \t]?", "", text, flags=re.MULTILINE)
    # Gạch đầu dòng thành dấu chấm tròn.
    text = re.sub(r"^[ \t]*[-*+][ \t]+", "• ", text, flags=re.MULTILINE)
    # Đậm/nghiêng. Chạy SAU phần gạch đầu dòng, để một dòng mở đầu bằng dấu
    # sao không bị hiểu nhầm là dấu nghiêng đang mở.
    text = re.sub(r"\*\*\*(.+?)\*\*\*", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"\*(.+?)\*", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"__(.+?)__", r"\1", text, flags=re.DOTALL)
    # Nghiêng bằng gạch dưới: chỉ khi hai đầu là ranh giới từ, để không cắt
    # nhầm một tên trường như has_asset_cash nếu nó lọt được tới đây.
    text = re.sub(r"(?<![\w])_([^_\n]+)_(?![\w])", r"\1", text)
    return text


def _strip_internal_vocabulary(text: str) -> str:
    """Bỏ mã rule, trạng thái nội bộ, slug model và khoá JSON."""
    # Đường dẫn khoá và slug model phải đi TRƯỚC mã rule.
    #
    # Thứ tự ngược lại thì `rules.RB02.value.dti` bị bước dịch mã rule biến
    # thành `rules.sức khỏe tài chính.value.dti`, và đường dẫn không còn khớp
    # `_DOTTED_PATH` nữa — nó chỉ ăn được `rules.s`, để lại một mảnh chữ cụt
    # giữa câu. Bỏ nguyên cụm khi nó còn nguyên hình dạng thì sạch hơn hẳn.
    text = _MODEL_SLUG.sub("", text)
    text = _DOTTED_PATH.sub("", text)
    text = _PLACEHOLDER.sub("", text)

    # Mã rule trong ngoặc đơn — bỏ hẳn, vì cụm đứng trước đã tự đủ nghĩa.
    text = re.sub(r"[ \t]*\(\s*RB0[1-5]\s*\)", "", text)
    # Mã rule mở đầu một mục — bỏ, giữ phần mô tả phía sau.
    text = re.sub(r"\bRB0[1-5]\s*[:\-–]\s*", "", text)
    # Còn sót ở giữa câu thì thay bằng tên nghiệp vụ.
    text = _RULE_CODE.sub(lambda m: RULE_NAMES_VI.get(m.group(), ""), text)

    # Trạng thái trong ngoặc đơn → bản dịch; không dịch được thì bỏ cả ngoặc.
    def _paren(match: re.Match) -> str:
        vi = _GENERIC_VI.get(match.group(1))
        return f" ({vi})" if vi else ""

    text = re.sub(r"[ \t]*\(\s*([A-Z][A-Z_]{2,})\s*\)", _paren, text)

    # Khoá kỹ thuật đứng trước dấu hai chấm — bỏ khoá, giữ giá trị phía sau
    # để bước dịch mã trần bên dưới còn xử lý được.
    keys = "|".join(re.escape(k) for k in _TECHNICAL_KEYS)
    text = re.sub(rf"\b(?:{keys})\s*[:=]\s*", "", text)

    # Mã trần còn lại trong câu — CHỈ dịch những mã đã biết, không đụng tới
    # chữ viết hoa lạ.
    #
    # Xoá mọi cụm viết hoa là quét quá tay: `DTI`, `LTV`, `PMT` là thuật ngữ
    # tài chính thật, người dùng đọc hiểu được và câu văn cần chúng. Xoá đi
    # thì "tỷ lệ DTI 34,3%" thành "tỷ lệ 34,3%" — mất luôn thứ đang được đo.
    # Từ vựng nội bộ là một tập ĐÓNG và đã liệt kê đủ ở trên, nên chặn theo
    # danh sách vừa đủ chặt vừa không đụng tới phần còn lại.
    text = _ALLCAPS_TOKEN.sub(
        lambda m: _GENERIC_VI.get(m.group(), m.group()), text)

    text = _EMPTY_VALUE.sub("chưa có", text)
    return text


def _tidy(text: str) -> str:
    """Dọn dấu vết của việc cắt bỏ: ngoặc rỗng, khoảng trắng thừa, dòng cụt."""
    text = re.sub(r"\(\s*\)", "", text)
    text = re.sub(r"\[\s*\]", "", text)
    # Khoảng trắng lặp trong một dòng, không đụng tới ký tự xuống dòng.
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"[ \t]+([,.;:!?%])", r"\1", text)
    # Dấu câu dính nhau sau khi bỏ chữ ở giữa.
    text = re.sub(r"([,.;:])[ \t]*(?=[,.;:])", "", text)
    # Dòng chỉ còn dấu chấm tròn hoặc dấu câu thì bỏ hẳn.
    text = re.sub(r"^[ \t]*[•\-–:,.]+[ \t]*$", "", text, flags=re.MULTILINE)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def to_plain_text(text: str | None) -> str:
    """Biên cuối — mọi câu chữ gửi ra FE đều phải đi qua đây.

    Bỏ Markdown → bỏ từ vựng nội bộ → bỏ Markdown lần nữa → dọn dấu vết.

    Thứ tự có ý nghĩa, và Markdown phải quét HAI lần:

        lần 1  để lộ mã đang bị bọc trong dấu — `**CRITICAL**` chỉ khớp bảng
               trạng thái sau khi hai dấu sao đã biến mất
        lần 2  để dọn dấu mà chính bước bỏ mã làm hở ra — dòng
               `_Mô hình: ml01_xgboost_vfinal._` có gạch dưới nằm giữa nên
               lần 1 không nhận ra đó là chữ nghiêng; bỏ slug xong thì nó mới
               thành một cặp gạch dưới hoàn chỉnh

    Hàm này KHÔNG đụng tới con số. Nó chỉ bỏ đi thứ vốn không nên có mặt, nên
    không có đường nào để nó làm sai lệch một số liệu đã tính đúng.
    """
    if not text:
        return ""
    return _tidy(_strip_markdown(_strip_internal_vocabulary(
        _strip_markdown(str(text)))))


def has_internal_vocabulary(text: str | None) -> bool:
    """Còn sót mã rule hay trạng thái nội bộ không — dùng cho test và log."""
    if not text:
        return False
    return bool(_RULE_CODE.search(text)
                or _MODEL_SLUG.search(text)
                or _DOTTED_PATH.search(text)
                or any(token in _GENERIC_VI
                       for token in _ALLCAPS_TOKEN.findall(text)))
