"""Diễn đạt kết quả ML ra tiếng Việt (F05 · M06, phần task 10–11).

Ràng buộc cứng của tầng này, và là lý do nó tồn tại tách khỏi tầng ml:

    Nhãn ĐÃ ĐƯỢC QUYẾT trước khi vào đây.

Mọi hàm dưới đây nhận `label` như một tham số **bắt buộc**, không nhận
`predict_proba` thô rồi tự chọn ngưỡng, không có nhánh nào đổi nhãn. Muốn
biết nhãn là gì thì phải hỏi model — tầng này chỉ trả lời câu "nhãn đó nghĩa
là gì với hộ này". Ràng buộc đó thể hiện bằng chữ ký hàm chứ không phải bằng
lời hứa trong tài liệu, nên vi phạm nó là phải sửa chữ ký — và test sẽ thấy.

Chế độ chạy
-----------
Bản này là **template điền chỗ trống**, chưa gọi Gemini. Đó là phương án đã
ghi sẵn trong PLAN.md §13 (mục cắt giảm) và trong `hfml.config`: *"Bỏ trống
key → tầng llm chạy chế độ template, pipeline vẫn chạy đủ"*.

Chọn template trước, LLM sau, vì thứ tự ngược lại là bẫy: có LLM rồi thì
không ai còn phân biệt được câu nào do model quyết và câu nào do LLM tự nghĩ
ra. Dựng template trước thì phần "được phép nói gì" đã cố định thành mã, và
khi cắm LLM vào (task 10) nó chỉ được viết lại cho mượt trong đúng khuôn đó.

Con số trong mọi câu đều đến từ tham số truyền vào, không có phép tính nào
sinh thêm số mới — điều kiện để guardrail "đối chiếu số bằng regex" ở task 14
có nghĩa.
"""
from __future__ import annotations

from typing import Final

#: Câu bắt buộc gắn cuối mọi lời tư vấn (PLAN.md §8.2 guardrail 3).
DISCLAIMER: Final[str] = (
    "_Thông tin trên là khuyến nghị tham khảo được tính từ dữ liệu bạn cung cấp, "
    "không phải tư vấn tài chính chuyên nghiệp._"
)

#: Câu bắt buộc riêng cho ML02 (guardrail 4): kết quả là ƯỚC LƯỢNG, không
#: phải quyết định cho vay. Nhầm hai thứ này là chỗ dễ gây hiểu sai nhất của
#: cả hệ thống.
LOAN_RISK_DISCLAIMER: Final[str] = (
    "_Đây là ước lượng tham khảo dựa trên dữ liệu bạn tự khai, **không phải kết quả "
    "thẩm định** và không thay thế quyết định của tổ chức tín dụng._"
)

#: Việc cần làm ứng với từng nhóm ML01. Đây là phần "tư vấn" — nó gắn với
#: NHÃN, không phải với hộ cụ thể, nên viết sẵn được và kiểm chứng được.
_ML01_GUIDANCE: Final[dict[str, tuple[str, tuple[str, ...]]]] = {
    "EMERGENCY": (
        "Dòng tiền đang ở mức cần xử lý ngay: chi tiêu vượt thu nhập, hoặc quỹ "
        "dự phòng chưa đủ sống một tháng.",
        (
            "Rà lại các khoản chi lớn trong tháng và cắt những khoản hoãn được.",
            "Ưu tiên gom đủ 1 tháng chi tiêu làm đệm trước khi tính bất cứ mục tiêu nào khác.",
            "Chưa nên vay thêm ở giai đoạn này — thêm một kỳ trả nợ sẽ làm dòng tiền âm sâu hơn.",
        ),
    ),
    "DEBT_FOCUS": (
        "Phần thu nhập dành trả nợ đang chiếm tỉ trọng cao, nên xử lý nợ trước "
        "khi tính tới tích lũy hay đầu tư.",
        (
            "Trả trước các khoản lãi suất cao nhất, giữ nguyên kỳ trả của các khoản còn lại.",
            "Cân nhắc gộp nợ nếu đang có nhiều khoản nhỏ lãi cao.",
            "Giữ quỹ dự phòng tối thiểu song song, đừng dồn toàn bộ số dư vào trả nợ.",
        ),
    ),
    "BUILD_BUFFER": (
        "Dòng tiền đã dương và nợ trong tầm kiểm soát, việc còn thiếu là một "
        "quỹ dự phòng đủ dày.",
        (
            "Đặt mục tiêu quỹ dự phòng 3–6 tháng chi tiêu, gửi ở nơi rút được ngay.",
            "Trích tự động một phần thu nhập ngay khi nhận, trước khi chi tiêu.",
            "Đủ quỹ dự phòng rồi mới tính tới các kênh sinh lời dài hạn.",
        ),
    ),
    "GROWTH": (
        "Không còn việc cấp thiết nào chắn đường: dòng tiền dương, nợ nhẹ, quỹ "
        "dự phòng đã đủ.",
        (
            "Duy trì quỹ dự phòng hiện có, đừng rút vào để dồn cho mục tiêu khác.",
            "Phân bổ phần dư theo lớp tài sản (tiền gửi / trái phiếu / chứng chỉ quỹ) "
            "thay vì dồn vào một chỗ.",
            "Rà lại tỉ lệ phân bổ mỗi 6–12 tháng hoặc khi thu nhập đổi đáng kể.",
        ),
    ),
}


def _bullets(items: tuple[str, ...] | list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def _probability_lines(probabilities: list[dict]) -> str:
    """Bảng xác suất bốn nhóm, giữ nguyên thứ tự model trả về.

    Hiển thị đủ bốn dòng chứ không chỉ nhóm thắng: một hồ sơ 0,41 / 0,39 rất
    khác một hồ sơ 0,95 / 0,02, mà chỉ nhìn nhãn thì hai ca đó giống hệt nhau.
    """
    return "\n".join(
        f"- {p.get('label_vi') or p.get('label')}: **{float(p['probability']):.1%}**"
        for p in probabilities
    )


def explain_ml01(
    *,
    label: str,
    label_vi: str,
    confidence: float,
    probabilities: list[dict],
    low_confidence: bool,
    rule_summary: str = "",
    model_version: str = "",
) -> str:
    """Diễn đạt kết quả ML01 — nhóm định hướng tài chính của hộ.

    `label` là tham số BẮT BUỘC và là thứ duy nhất quyết định nội dung phần
    tư vấn. Hàm này không nhận `predict_proba` thô để tự chọn nhãn, và không
    có nhánh nào đổi `label` — kể cả khi `low_confidence` bật.

    `low_confidence` chỉ làm hàm NÓI RA rằng kết quả chưa chắc chắn, không
    làm nó đổi kết luận. Im lặng ở đây là để người dùng đọc một phỏng đoán
    mong manh như thể một kết luận chắc chắn (PLAN.md §8.1 task 7).
    """
    headline, actions = _ML01_GUIDANCE.get(
        label,
        ("Chưa có diễn giải cho nhóm này.", ("Vui lòng liên hệ để được hỗ trợ thêm.",)),
    )

    parts = [
        f"🧭 **Chẩn đoán sức khỏe tài chính: {label_vi}**",
        "",
        headline,
        "",
        f"Mức tin cậy của mô hình: **{confidence:.1%}**.",
    ]

    if low_confidence:
        parts += [
            "",
            "⚠️ Hồ sơ của bạn nằm gần ranh giới giữa hai nhóm nên kết quả này "
            "**chưa chắc chắn**. Hãy đọc nó cùng phần đánh giá theo quy tắc bên "
            "dưới thay vì chỉ nhìn một nhãn.",
        ]

    if probabilities:
        parts += ["", "**Xác suất từng nhóm:**", _probability_lines(probabilities)]

    parts += ["", "**Việc nên làm tiếp theo:**", _bullets(actions)]

    if rule_summary:
        parts += ["", rule_summary]

    if model_version:
        parts += ["", f"_Mô hình: `{model_version}`._"]

    parts += ["", DISCLAIMER]
    return "\n".join(parts)


def explain_ml02(
    *,
    label: str,
    label_vi: str,
    probability: float,
    threshold: float,
    loan_summary: str = "",
    top_factors: list[dict] | None = None,
    model_version: str = "",
) -> str:
    """Diễn đạt kết quả ML02 — mức rủi ro của khoản vay đang xét.

    Như `explain_ml01`, `label` là đầu vào chứ không phải kết quả tính ở đây.
    `threshold` được in ra chứ không giấu đi: nói "rủi ro cao" mà không cho
    biết ngưỡng nào phân định là một khẳng định không kiểm chứng được, và
    ngưỡng đó **không phải 0,5** — tỉ lệ vỡ nợ nền chỉ 8,07% nên 0,5 sẽ xếp
    gần như mọi hồ sơ vào nhóm rủi ro thấp.
    """
    parts = [
        f"⚖️ **Chẩn đoán rủi ro vay vốn: {label_vi}**",
        "",
        f"Xác suất gặp khó khăn trả nợ ước tính: **{probability:.1%}** "
        f"(ngưỡng phân loại đang dùng: {threshold:.1%}).",
    ]

    if loan_summary:
        parts += ["", loan_summary]

    if top_factors:
        parts += [
            "",
            "**Những yếu tố ảnh hưởng nhiều nhất tới kết quả này:**",
            _bullets([
                f"{f['feature_vi']}: {f['direction']}"
                for f in top_factors
            ]),
        ]

    parts += ["", LOAN_RISK_DISCLAIMER]

    if model_version:
        parts += ["", f"_Mô hình: `{model_version}`._"]

    return "\n".join(parts)
