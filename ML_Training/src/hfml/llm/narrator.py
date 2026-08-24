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
#:
#: Chữ thuần, không Markdown: màn Chatbot render nguyên trạng nên dấu gạch
#: dưới bao quanh sẽ hiện ra thành ký tự, và nút đọc bằng giọng nói đọc luôn
#: cả dấu. Xem `hfml.llm.presentation`.
DISCLAIMER: Final[str] = (
    "Thông tin trên là khuyến nghị tham khảo được tính từ dữ liệu bạn cung cấp, "
    "không phải tư vấn tài chính chuyên nghiệp."
)

#: Câu bắt buộc riêng cho ML02 (guardrail 4): kết quả là ƯỚC LƯỢNG, không
#: phải quyết định cho vay. Nhầm hai thứ này là chỗ dễ gây hiểu sai nhất của
#: cả hệ thống.
LOAN_RISK_DISCLAIMER: Final[str] = (
    "Đây là ước lượng tham khảo dựa trên dữ liệu bạn tự khai, không phải kết quả "
    "thẩm định và không thay thế quyết định của tổ chức tín dụng."
)

#: Việc cần làm ứng với từng nhóm ML01. Đây là phần "tư vấn" — nó gắn với
#: NHÃN, không phải với hộ cụ thể, nên viết sẵn được và kiểm chứng được.
_ML01_GUIDANCE: Final[dict[str, tuple[str, tuple[str, ...]]]] = {
    "EMERGENCY": (
        "Dòng tiền gia đình đang ở trạng thái cần can thiệp xử lý ngay: Chi tiêu sinh hoạt vượt quá thu nhập hàng tháng, hoặc số tiền tiết kiệm tích lũy chưa đủ trang trải chi phí tối thiểu cho một tháng.",
        (
            "Tập trung rà soát và thắt chặt các khoản chi tiêu chưa thực sự cấp thiết trong ngắn hạn.",
            "Ưu tiên hàng đầu là tích lũy đủ đệm dự phòng 1 tháng chi tiêu sinh hoạt trước khi thực hiện bất kỳ kế hoạch tài chính nào khác.",
            "Tạm dừng việc đăng ký thêm các khoản vay mới để tránh làm gia tăng gánh nặng nghĩa vụ trả nợ hàng tháng.",
        ),
    ),
    "DEBT_FOCUS": (
        "Nghĩa vụ trả nợ hiện tại đang chiếm tỷ trọng lớn trong tổng thu nhập gia đình. Hệ thống khuyến nghị ưu tiên tập trung xử lý và cân bằng lại khoản nợ trước khi tính đến các mục tiêu tích lũy hay đầu tư dài hạn.",
        (
            "Ưu tiên thanh toán trước các khoản nợ có lãi suất cao nhất để giảm thiểu chi phí lãi phát sinh.",
            "Cân nhắc giải pháp gộp nợ hoặc tái cơ cấu kỳ hạn vay nếu đang gánh nhiều khoản vay nhỏ lẻ.",
            "Duy trì song song quỹ dự phòng tối thiểu để chủ động ứng phó trước các tình huống bất ngờ mà không phải phát sinh thêm nợ mới.",
        ),
    ),
    "BUILD_BUFFER": (
        "Gia đình bạn đã kiểm soát tốt dòng tiền hàng tháng và dư thặng dư ổn định. Bước quan trọng tiếp theo là củng cố thêm Quỹ dự phòng khẩn cấp để đạt độ dày an toàn tối ưu.",
        (
            "Đặt mục tiêu tích lũy Quỹ dự phòng khẩn cấp đạt mốc 3 đến 6 tháng chi tiêu sinh hoạt (gửi tại các kênh thanh khoản cao, dễ rút).",
            "Thiết lập cơ chế tự động trích lập một phần thặng dư hàng tháng vào quỹ tích lũy ngay khi nhận thu nhập.",
            "Khi Quỹ dự phòng đạt độ dày an toàn, gia đình có thể tự tin chuyển hướng sang các kênh đầu tư sinh lời dài hạn.",
        ),
    ),
    "GROWTH": (
        "Sức khỏe tài chính của gia đình bạn đang ở trạng thái rất lành mạnh và lý tưởng: Dòng tiền thặng dư dồi dào, tỷ lệ nợ trong tầm kiểm soát an toàn và đệm dự phòng đã đạt chuẩn.",
        (
            "Tiếp tục duy trì và bảo vệ Quỹ dự phòng hiện có, tránh rút tiền quỹ cho các mục tiêu chi tiêu ngẫu hứng.",
            "Chủ động phân bổ phần thặng dư hàng tháng vào danh mục tài sản đa dạng (Tiền gửi tích lũy, Trái phiếu, Chứng chỉ quỹ sinh lời an toàn).",
            "Định kỳ rà soát và tái cân đối tỷ lệ phân bổ tài sản mỗi 6 đến 12 tháng để tối ưu hóa hiệu quả tài chính dài hạn.",
        ),
    ),
}


def _bullets(items: tuple[str, ...] | list[str]) -> str:
    return "\n".join(f"• {item}" for item in items)


def _probability_lines(probabilities: list[dict]) -> str:
    """Bảng xác suất bốn nhóm, giữ nguyên thứ tự model trả về.

    Hiển thị đủ bốn dòng chứ không chỉ nhóm thắng: một hồ sơ 0,41 / 0,39 rất
    khác một hồ sơ 0,95 / 0,02, mà chỉ nhìn nhãn thì hai ca đó giống hệt nhau.

    Gọi nhóm bằng `label_vi`, và KHÔNG lùi về `label` khi thiếu: mã nhóm là
    `EMERGENCY`, `DEBT_FOCUS` — đúng thứ không được để lọt ra màn hình. Thiếu
    bản tiếng Việt thì bỏ hẳn dòng đó, vì một dòng thiếu vẫn tốt hơn một dòng
    người dùng không đọc được.
    """
    from hfml.llm.presentation import percent

    return "\n".join(
        f"• {p['label_vi']}: {percent(float(p['probability']))}"
        for p in probabilities if p.get("label_vi")
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
    from hfml.llm.presentation import percent

    headline, actions = _ML01_GUIDANCE.get(
        label,
        ("Chưa có diễn giải cho nhóm này.", ("Vui lòng liên hệ để được hỗ trợ thêm.",)),
    )

    parts = [
        f"🧭 Chẩn đoán sức khỏe tài chính: {label_vi}",
        "",
        headline,
        "",
        f"Mức tin cậy của mô hình: {percent(confidence)}.",
    ]

    if low_confidence:
        parts += [
            "",
            "⚠️ Hồ sơ của bạn nằm gần ranh giới giữa hai nhóm nên kết quả này "
            "chưa chắc chắn. Hãy đọc nó cùng phần đánh giá theo quy tắc bên "
            "dưới thay vì chỉ nhìn một nhãn.",
        ]

    if probabilities:
        parts += ["", "Xác suất từng nhóm:", _probability_lines(probabilities)]

    parts += ["", "Việc nên làm tiếp theo:", _bullets(actions)]

    if rule_summary:
        parts += ["", rule_summary]

    # `model_version` KHÔNG được in ra.
    #
    # Nó vẫn là tham số của hàm, và vẫn đi theo `AiResult` để đối chiếu khi cần
    # truy một kết quả về đúng artifact đã sinh ra nó. Nhưng `ml01_xgboost_
    # vfinal` không nói gì với người đang hỏi chuyện tiền nong của nhà mình —
    # nó chỉ làm câu trả lời trông như một trang log.

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

    `threshold` vẫn là tham số bắt buộc nhưng KHÔNG còn được in ra
    ---------------------------------------------------------------
    Bản trước in kèm "(ngưỡng phân loại đang dùng: 13,0%)" với lý do chính
    đáng: nói "rủi ro cao" mà giấu ngưỡng phân định là một khẳng định không
    kiểm chứng được, nhất là khi ngưỡng đó không phải 0,5 — tỉ lệ vỡ nợ nền
    chỉ 8,07%.

    Lý do đó đúng với người đọc báo cáo đánh giá, không đúng với người dùng
    cuối: ngưỡng phân loại là tham số nội bộ của model, và trong một câu tư
    vấn nó chỉ làm người đọc bối rối giữa hai con số phần trăm cạnh nhau.
    Tính kiểm chứng được không mất đi — `threshold` vẫn nằm nguyên trong
    `AiResult` mà `/inference` trả về, nơi người cần đối chiếu sẽ đọc.
    """
    from hfml.llm.presentation import percent

    parts = [
        f"⚖️ Chẩn đoán rủi ro vay vốn: {label_vi}",
        "",
        f"Xác suất gặp khó khăn trả nợ ước tính: {percent(probability)}.",
    ]

    if loan_summary:
        parts += ["", loan_summary]

    if top_factors:
        parts += [
            "",
            "Những yếu tố ảnh hưởng nhiều nhất tới kết quả này:",
            _bullets([
                f"{f['feature_vi']}: {f['direction']}"
                for f in top_factors
            ]),
        ]

    parts += ["", LOAN_RISK_DISCLAIMER]

    # `model_version` không in ra — xem `explain_ml01`.

    return "\n".join(parts)
