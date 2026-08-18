r"""Entry-point chạy TOÀN BỘ ML02 trong một lệnh (F04 · M04 · task 3–15).

    .venv\Scripts\python.exe scripts/train_ml02.py
    .venv\Scripts\python.exe scripts/train_ml02.py --rows 20000    # chạy thử nhanh
    .venv\Scripts\python.exe scripts/train_ml02.py --from train    # bỏ bước dựng dữ liệu
    .venv\Scripts\python.exe scripts/train_ml02.py --skip importance
    .venv\Scripts\python.exe scripts/train_ml02.py --only evaluate,compare
    .venv\Scripts\python.exe scripts/train_ml02.py --dry-run

Đối xứng với `scripts/train_ml01.py`: script này chỉ ĐIỀU PHỐI, không tự cài
đặt lại bước nào. Mỗi stage gọi đúng script lẻ đã có, nên chạy qua đây hay gõ
tay từng lệnh đều cho cùng kết quả — và khi một stage đổi hành vi thì chỗ này
không phải sửa theo.

Vì sao gọi subprocess chứ không import hàm
--------------------------------------------
Mỗi script lẻ có `main()` riêng kèm phần in bảng và các mặc định đối số của
nó. Gọi lại hàm thư viện bên dưới nghĩa là chép lại phần điều phối ấy vào đây,
và hai bản sao sẽ trôi khỏi nhau — đúng lỗi mà `hfml.inference.settings` đã
gặp với `ML02_SLUG`. Chạy thẳng script con thì không có bản sao nào.

Bù lại là ~2 giây khởi động mỗi stage, không đáng kể so với vài phút train.

Thứ tự KHÔNG đảo được
----------------------
Task 14 chọn model từ bảng của task 12, task 15 export model task 14 đã chọn.
Chạy `select` khi chưa có `compare` thì nó dừng và báo thiếu, chứ không tự
chạy bù — đó là thiết kế của F04, script này giữ nguyên.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from hfml.config import CONFIG
from hfml.logger import get_logger

log = get_logger(__name__)

SCRIPTS = Path(__file__).parent


@dataclass(frozen=True)
class Stage:
    name: str
    task: str
    script: str
    #: Nhận `--rows` để chạy thử nhanh hay không.
    takes_rows: bool = False
    #: Nằm ngoài mặc định — chỉ chạy khi được gọi tên.
    optional: bool = False


#: Toàn bộ F04, đúng thứ tự phụ thuộc. `explore` và `clean` để ngoài mặc định:
#: chúng đọc dataset 1,4 GB và chỉ cần chạy lại khi dữ liệu thô đổi, còn vòng
#: lặp thường ngày là sửa feature rồi train lại.
STAGES: tuple[Stage, ...] = (
    Stage("explore",       "1",  "explore_ml02.py",            optional=True),
    Stage("clean",         "2",  "clean_ml02.py",              optional=True),
    Stage("features",      "3",  "build_features_ml02.py"),
    Stage("imbalance",     "4",  "imbalance_ml02.py"),
    Stage("split",         "5",  "split_ml02.py"),
    Stage("baseline",      "6",  "baseline_ml02.py",           takes_rows=True),
    Stage("decision_tree", "7",  "train_ml02_decision_tree.py", takes_rows=True),
    Stage("bagging",       "8",  "train_ml02_bagging.py",      takes_rows=True),
    Stage("random_forest", "9",  "train_ml02_random_forest.py", takes_rows=True),
    Stage("xgboost",       "10", "train_ml02_xgboost.py",      takes_rows=True),
    Stage("evaluate",      "11", "evaluate_ml02.py"),
    Stage("compare",       "12", "compare_ml02.py"),
    Stage("importance",    "13", "importance_ml02.py"),
    Stage("select",        "14", "select_ml02.py"),
    Stage("export",        "15", "export_ml02.py"),
)

BY_NAME = {s.name: s for s in STAGES}
#: Nhóm tắt cho `--from` / `--only`, đặt theo cách người ta hay nói.
ALIASES: dict[str, tuple[str, ...]] = {
    "data":   ("features", "imbalance", "split", "baseline"),
    "train":  ("decision_tree", "bagging", "random_forest", "xgboost"),
    "report": ("evaluate", "compare", "importance"),
    "ship":   ("select", "export"),
}


def resolve(names: str) -> list[str]:
    """Bung danh sách tên stage, chấp nhận cả alias."""
    out: list[str] = []
    for raw in names.split(","):
        key = raw.strip()
        if not key:
            continue
        if key in ALIASES:
            out.extend(ALIASES[key])
        elif key in BY_NAME:
            out.append(key)
        else:
            raise SystemExit(
                f"Stage không có: {key!r}\n"
                f"  stage : {', '.join(BY_NAME)}\n"
                f"  nhóm  : {', '.join(ALIASES)}")
    return out


def plan(args) -> list[Stage]:
    if args.only:
        chosen = set(resolve(args.only))
        return [s for s in STAGES if s.name in chosen]

    running = [s for s in STAGES if not s.optional]
    if args.start:
        first = resolve(args.start)[0]
        order = [s.name for s in STAGES]
        # `--from clean` kéo cả stage optional vào lại.
        running = [s for s in STAGES if order.index(s.name) >= order.index(first)]
    if args.stop:
        last = resolve(args.stop)[-1]
        order = [s.name for s in STAGES]
        running = [s for s in running
                   if order.index(s.name) <= order.index(last)]
    if args.skip:
        dropped = set(resolve(args.skip))
        running = [s for s in running if s.name not in dropped]
    return running


def run(stage: Stage, rows: int | None) -> tuple[bool, float]:
    cmd = [sys.executable, str(SCRIPTS / stage.script)]
    if rows and stage.takes_rows:
        cmd += ["--rows", str(rows)]
    started = time.time()
    result = subprocess.run(cmd)
    return result.returncode == 0, time.time() - started


def check_deployed_slug() -> list[str]:
    """Đối chiếu model vừa chọn với slug mà tầng inference đang trỏ tới.

    Đây là cái bẫy đã suýt sập hai lần: task 14 sinh slug theo THUẬT TOÁN
    THẮNG (`ml02_{algo}_{feature_set}`), nên chỉ cần lần train lại có model
    khác dẫn đầu là tên artifact đổi — trong khi `InferenceSettings.ml02_slug`
    vẫn trỏ vào tên cũ. Hệ thống khi đó nạp model CŨ và không báo gì cả.
    """
    notes: list[str] = []
    path = CONFIG.paths.runs / "ml02_selection" / "decision.json"
    if not path.exists():
        return ["Chưa có decision.json — chưa chạy task 14."]

    decision = json.loads(path.read_text(encoding="utf-8"))
    expected = f"{decision['selected_model']}_vfinal"

    from hfml.inference.settings import SETTINGS
    configured = SETTINGS.ml02_slug

    if expected == configured:
        notes.append(f"✅ slug khớp: {configured}")
    else:
        notes.append(
            f"❌ LỆCH SLUG — inference sẽ nạp model CŨ:\n"
            f"     task 14 chọn : {expected}\n"
            f"     config trỏ   : {configured}\n"
            f"     Sửa `ml02_slug` trong config/config.yaml (hoặc đặt "
            f"HFML_ML02_SLUG) rồi chạy lại phần kiểm chứng.")

    artifact = CONFIG.paths.runs / f"{expected}.joblib"
    if not artifact.exists():
        notes.append(f"❌ Chưa có artifact {artifact.name} — task 15 chưa chạy?")
    elif not decision.get("exported"):
        notes.append("⚠️  decision.json ghi `exported = false`.")
    return notes


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from", dest="start", default=None,
                        help="chạy từ stage này trở đi")
    parser.add_argument("--to", dest="stop", default=None,
                        help="dừng sau stage này")
    parser.add_argument("--only", default=None,
                        help="chỉ chạy các stage này (ngăn cách bằng dấu phẩy)")
    parser.add_argument("--skip", default=None, help="bỏ qua các stage này")
    parser.add_argument("--rows", type=int, default=None,
                        help="giới hạn số dòng — CHỈ để chạy thử")
    parser.add_argument("--dry-run", action="store_true",
                        help="in kế hoạch rồi thoát, không chạy gì")
    args = parser.parse_args()

    stages = plan(args)
    if not stages:
        raise SystemExit("Không còn stage nào để chạy.")

    width = 78
    print("\n" + "=" * width)
    print(f"  ML02 · F04 — chạy {len(stages)} stage"
          + (f" · giới hạn {args.rows:,} dòng" if args.rows else ""))
    print("=" * width)
    for s in stages:
        extra = "  (--rows)" if args.rows and s.takes_rows else ""
        print(f"  task {s.task:>2} · {s.name:<14} {s.script}{extra}")
    print("=" * width)

    if args.dry_run:
        print("\n--dry-run: không chạy gì.")
        return 0

    done: list[tuple[Stage, float]] = []
    for index, stage in enumerate(stages, 1):
        print(f"\n{'─' * width}")
        print(f"▶ [{index}/{len(stages)}] task {stage.task} · {stage.name}")
        print("─" * width)
        ok, seconds = run(stage, args.rows)
        done.append((stage, seconds))
        if not ok:
            print(f"\n{'=' * width}")
            print(f"  ❌ DỪNG ở task {stage.task} · {stage.name} "
                  f"({stage.script}) sau {seconds:.1f}s")
            print(f"  Các stage sau KHÔNG chạy vì chúng phụ thuộc stage này.")
            print(f"  Sửa xong chạy tiếp bằng:  --from {stage.name}")
            print("=" * width)
            return 1

    total = sum(s for _, s in done)
    print(f"\n{'=' * width}")
    print(f"  ✅ Xong {len(done)} stage · tổng {total / 60:.1f} phút")
    print("=" * width)
    for stage, seconds in sorted(done, key=lambda x: -x[1]):
        share = seconds / total if total else 0
        print(f"  task {stage.task:>2} · {stage.name:<14}"
              f"{seconds:>8.1f}s  {'█' * max(1, round(share * 28))}")

    ran_export = any(s.name == "export" for s, _ in done)
    if ran_export:
        print(f"\n{'─' * width}")
        print("  Kiểm tra artifact đang deploy")
        print("─" * width)
        for note in check_deployed_slug():
            print(f"  {note}")

    print(f"\n{'─' * width}")
    print("  Bước kiểm chứng (không nằm trong F04):")
    print("    .venv\\Scripts\\python.exe scripts/property_test_ml02.py")
    print("    .venv\\Scripts\\python.exe scripts/shap_overdue_ml02.py")
    print("    .venv\\Scripts\\python.exe scripts/testcases_ml01_ml02.py")
    print("─" * width)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
