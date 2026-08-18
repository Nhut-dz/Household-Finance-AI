r"""Entry-point cho Epic AI-02 — LLM Layer (F05 · M06).

    .venv\Scripts\python.exe scripts/chat_advisor.py
    .venv\Scripts\python.exe scripts/chat_advisor.py --intent LOAN_RISK_DIAGNOSIS
    .venv\Scripts\python.exe scripts/chat_advisor.py --ask "Tôi vay được bao nhiêu?"
    .venv\Scripts\python.exe scripts/chat_advisor.py --demo
    .venv\Scripts\python.exe scripts/chat_advisor.py --json

Chạy AI-01 một lần để có structured result, rồi đưa qua tầng LLM. Không có
`--ask` thì dùng bộ câu hỏi mẫu bao đủ các nhánh: chip ML, câu tự nhập, câu nối
tiếp, và câu ngoài phạm vi.

Vì sao script này in cả PHẦN KIỂM chứ không chỉ câu trả lời
--------------------------------------------------------------
Câu trả lời của LLM đọc lên lúc nào cũng trôi chảy — kể cả khi nó vừa bịa một
con số. Nhìn văn bản không phân biệt được. Nên mỗi lượt đều in kèm nguồn
(`llm` · `llm_retry` · `template`) và kết quả kiểm: đó mới là thứ nói cho người
vận hành biết câu trả lời vừa rồi có đáng tin không.

Không có API key thì script vẫn chạy — mọi lượt hạ cấp về template. Đó là hành
vi đúng, không phải lỗi.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from analyze_profile import SAMPLE                          # noqa: E402
from hfml.api.intents import IntentCode                     # noqa: E402
from hfml.llm import client                                 # noqa: E402
from hfml.llm.chat import answer                            # noqa: E402
from hfml.logger import get_logger                          # noqa: E402
from hfml.pipeline.orchestrator import analyze              # noqa: E402

log = get_logger(__name__)

#: Kịch bản mẫu — mỗi mục là (câu hỏi, intent_code từ chip hoặc None).
#:
#: Cố ý xếp đủ bốn nhánh để một lượt chạy là thấy được toàn bộ hành vi:
#: hai chip ML, một câu tự nhập, một câu nối tiếp (dựa vào lượt ngay trước),
#: và một câu ngoài phạm vi phải bị chặn TRƯỚC khi gọi LLM.
DEMO: tuple[tuple[str, str | None], ...] = (
    ("Sức khỏe tài chính của gia đình tôi thế nào?",
     IntentCode.FINANCIAL_HEALTH_DIAGNOSIS.value),
    ("Khoản vay tôi định vay có rủi ro không?",
     IntentCode.LOAN_RISK_DIAGNOSIS.value),
    ("Tôi nên phân bổ thu nhập theo quy tắc 50/30/20 thế nào?", None),
    ("Thế còn 2 tỷ?", None),
    ("Tôi nên mua bitcoin không?", None),
)

_MARK = {client.SOURCE_LLM: "🟢", client.SOURCE_LLM_RETRY: "🟡",
         client.SOURCE_TEMPLATE: "⚪", client.SOURCE_OUT_OF_SCOPE: "🚫"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=None,
                        help="file JSON chứa hồ sơ")
    parser.add_argument("--ask", default=None, help="một câu hỏi cụ thể")
    parser.add_argument("--intent", default=None,
                        help=f"mã intent từ chip ({', '.join(i.value for i in IntentCode)})")
    parser.add_argument("--demo", action="store_true",
                        help="chạy trọn bộ câu hỏi mẫu")
    parser.add_argument("--json", action="store_true",
                        help="in JSON thô, không diễn giải")
    args = parser.parse_args()

    payload = (json.loads(args.input.read_text(encoding="utf-8"))
               if args.input else SAMPLE)

    result = analyze(payload).to_dict()
    if not result["ok"]:
        print("❌ AI-01 không chạy được — không có gì để diễn đạt.")
        for item in result["errors"]:
            print(f"   [{item['code']}] {item['field']}: {item['message']}")
        return 1

    if args.ask:
        turns = ((args.ask, args.intent),)
    elif args.demo or args.intent is None:
        turns = DEMO
    else:
        turns = ((f"Chẩn đoán giúp tôi.", args.intent),)

    if not args.json:
        print("\n" + "=" * 72)
        print("AI-02 · LLM LAYER")
        print(f"  LLM: {'sẵn sàng' if client.is_llm_available() else 'CHƯA CẤU HÌNH — mọi lượt sẽ dùng template'}")
        print("=" * 72)

    # Lịch sử và intent lượt trước được mang theo — đó là thứ làm câu nối tiếp
    # ("thế còn 2 tỷ?") hiểu được, thay vì rơi vào GENERAL với context rỗng.
    history: list[dict] = []
    previous_intent: str | None = None
    output: list[dict] = []

    for question, intent_code in turns:
        turn = answer(question, result, intent_code=intent_code,
                      history=history, previous_intent=previous_intent)
        data = turn.to_dict()
        output.append(data)

        if not args.json:
            _print_turn(data, intent_code)

        history += [{"role": "user", "content": question},
                    {"role": "assistant", "content": data["text"]}]
        previous_intent = turn.intent

    if args.json:
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return 0

    print("\n" + "=" * 72)
    counts: dict[str, int] = {}
    for data in output:
        source = data["answer"]["source"]
        counts[source] = counts.get(source, 0) + 1
    print("  " + " · ".join(f"{_MARK.get(k, '·')} {k}: {v}"
                            for k, v in sorted(counts.items())))
    print("=" * 72)
    return 0


def _print_turn(data: dict, intent_code: str | None) -> None:
    reply = data["answer"]
    source = reply["source"]

    print(f"\n{'─' * 72}")
    print(f"❓ {data['question']}")
    print(f"   intent={data['intent']} · chủ đề={data['topic']}"
          + ("  (từ chip)" if intent_code else "  (tự nhận từ câu hỏi)"))

    print(f"\n{_MARK.get(source, '·')} {reply['explanation']}")

    for item in reply["recommendations"]:
        mark = {"high": "🔴", "medium": "🟠", "low": "🟢"}.get(
            str(item.get("priority", "")).lower(), "·")
        print(f"   {mark} {item.get('action', '')}")
        if item.get("reason"):
            print(f"      └ {item['reason']}")

    for caveat in reply["caveats"]:
        print(f"   ⚠️  {caveat}")
    for missing in reply["needs_more_data"]:
        print(f"   📋 cần bổ sung: {missing}")
    for suggestion in reply.get("suggested_questions", []):
        print(f"   💡 {suggestion}")

    # Phần kiểm — xem docstring đầu file về lý do luôn in ra.
    check = reply["validation"]
    # Ba trạng thái, không phải hai: `None` nghĩa là chưa từng có câu trả lời
    # nào để kiểm (mạng, quota, model hỏng) — khác hẳn với bị đánh trượt.
    status = {True: "đạt", False: "KHÔNG ĐẠT",
              None: "không chạy được"}[check.get("valid")]
    detail = f" · {check['note']}" if check.get("note") else ""
    if check.get("issues"):
        detail = " · " + ", ".join(i["check"] for i in check["issues"])
    if check.get("ungrounded_numbers"):
        detail += " · số bịa: " + ", ".join(check["ungrounded_numbers"])
    print(f"\n   [nguồn {source} · prompt {reply['prompt_version']}"
          f" · kiểm {status}{detail}]")


if __name__ == "__main__":
    sys.exit(main())
