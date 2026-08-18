"""AI-03 task 3, 6 — Structured result chuẩn cho mọi tầng (F05 · M07).

Mỗi bước trong pipeline trả về CÙNG một kiểu vỏ: `StageResult`. Nhờ vậy tầng
điều phối không cần biết bước đó làm gì mới xử lý được kết quả của nó, và một
bước mới thêm vào sau này không kéo theo sửa đổi ở nơi gọi.

Vì sao lỗi là DỮ LIỆU chứ không phải ngoại lệ
-----------------------------------------------
Trong một pipeline có 12 bước mà bước 5 hỏng, thứ người dùng cần KHÔNG phải là
một trang 500 trắng — mà là bốn bước đầu đã tính được cộng một dòng nói rõ bước
5 hỏng vì sao. Ngoại lệ bay lên tới biên service thì mọi thứ đã tính được đều
mất theo.

Nên `StageResult` mang `ok`, `data`, `errors`, `warnings`; tầng điều phối bắt
mọi ngoại lệ tại biên mỗi bước và đổi nó thành một `StageResult` hỏng. Chỉ có
đúng một loại hỏng làm dừng cả pipeline — đầu vào không hợp lệ, vì khi đó không
có gì hợp lệ để tính.

Ba mức, không phải hai
-----------------------
    error     bước này không cho ra kết quả dùng được
    warning   có kết quả nhưng người đọc cần biết thêm điều gì đó
    info      ghi nhận để truy vết, không cần nói với người dùng

Gộp `warning` vào `error` là cách chắc chắn nhất để mọi hồ sơ thiếu khoản vay
bị báo là hỏng, trong khi thiếu khoản vay là chuyện hoàn toàn bình thường.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final

#: Mức nghiêm trọng của một mục chẩn đoán.
ERROR: Final[str] = "error"
WARNING: Final[str] = "warning"
INFO: Final[str] = "info"


@dataclass
class Diagnostic:
    """Một mục chẩn đoán — lỗi, cảnh báo, hoặc ghi chú.

    `stage` và `field` cùng có mặt vì chúng trả lời hai câu hỏi khác nhau:
    stage nói hỏng ở ĐÂU trong pipeline (để người vận hành đi sửa), field nói
    hỏng ở TRƯỜNG nào của hồ sơ (để người dùng biết điền lại chỗ nào).
    """

    code: str
    message: str
    severity: str = ERROR
    stage: str = ""
    field: str = ""

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message,
                "severity": self.severity, "stage": self.stage,
                "field": self.field}


@dataclass
class StageResult:
    """Kết quả của MỘT bước. Mọi bước trả về đúng kiểu này."""

    stage: str
    ok: bool = True
    data: Any = None
    diagnostics: list[Diagnostic] = field(default_factory=list)
    #: Mili-giây bước này tốn. Cần khi đi tìm bước làm chậm cả request.
    elapsed_ms: float = 0.0

    @property
    def errors(self) -> list[Diagnostic]:
        return [d for d in self.diagnostics if d.severity == ERROR]

    @property
    def warnings(self) -> list[Diagnostic]:
        return [d for d in self.diagnostics if d.severity == WARNING]

    def add(self, code: str, message: str, severity: str = ERROR,
            field_name: str = "") -> "StageResult":
        """Thêm một mục chẩn đoán. `error` tự động đánh dấu bước là hỏng."""
        self.diagnostics.append(Diagnostic(
            code=code, message=message, severity=severity,
            stage=self.stage, field=field_name))
        if severity == ERROR:
            self.ok = False
        return self

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "ok": self.ok,
            "diagnostics": [d.to_dict() for d in self.diagnostics],
            "elapsed_ms": round(self.elapsed_ms, 2),
        }

    @classmethod
    def failed(cls, stage: str, code: str, message: str,
               field_name: str = "") -> "StageResult":
        return cls(stage=stage, ok=False).add(code, message, ERROR, field_name)


@dataclass
class InferenceResult:
    """Đầu ra của cả pipeline — đầu vào tới câu trả lời.

    Gộp phần phân tích (AI-01) và phần diễn đạt (AI-02) vào MỘT vỏ, vì phía
    tiêu thụ luôn cần cả hai: câu trả lời để hiển thị, và structured result để
    đối chiếu khi cần biết câu trả lời đó dựa trên cái gì.
    """

    ok: bool = True
    #: Structured result của tầng phân tích — `AiResult.to_dict()`.
    analysis: dict = field(default_factory=dict)
    #: Câu trả lời đã qua kiểm — `Answer.to_dict()`. Rỗng nếu chỉ chạy phân tích.
    answer: dict = field(default_factory=dict)
    #: Câu trả lời dạng văn bản, sẵn sàng hiển thị.
    text: str = ""
    intent: str = ""
    topic: str = ""
    diagnostics: list[Diagnostic] = field(default_factory=list)
    #: Từng bước đã chạy, theo đúng thứ tự — dùng để gỡ lỗi và đo tốc độ.
    trace: list[dict] = field(default_factory=list)
    settings: dict = field(default_factory=dict)
    schema_version: str = ""
    generated_at: str = ""

    @property
    def errors(self) -> list[Diagnostic]:
        return [d for d in self.diagnostics if d.severity == ERROR]

    @property
    def warnings(self) -> list[Diagnostic]:
        return [d for d in self.diagnostics if d.severity == WARNING]

    def to_dict(self) -> dict:
        """Luôn đủ khoá, kể cả khi một phần hỏng.

        Cùng lý do với `AiResult.to_dict()`: một khoá biến mất rất dễ bị tầng
        trên đọc nhầm thành "không có vấn đề gì".
        """
        return {
            "ok": self.ok,
            "schema_version": self.schema_version,
            "generated_at": self.generated_at,
            "intent": self.intent,
            "topic": self.topic,
            "text": self.text,
            "answer": self.answer,
            "analysis": self.analysis,
            "errors": [d.to_dict() for d in self.errors],
            "warnings": [d.to_dict() for d in self.warnings],
            "trace": self.trace,
            "settings": self.settings,
        }
