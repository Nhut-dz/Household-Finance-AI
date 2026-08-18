"""ML02 task 5 — Chia train / validation / test (F04 · M04 · Tuần 4).

Nối tiếp task 4. Task này KHÔNG train model — nó chốt **một** phép chia và ghi
ra đĩa để cả task 6 → 15 dùng chung đúng bộ đó.

    train       70%   fit model
    validation  15%   **căn cứ CHỌN MODEL** — so PR-AUC giữa 4 thuật toán,
                      và là nơi tinh chỉnh siêu tham số nếu cần
    test        15%   khoá lại, CHỈ mở đúng một lần ở task 14

**KHÔNG dùng K-Fold Cross-Validation** (chốt 14/08/2026, áp cho cả ML01 và
ML02). Mỗi thuật toán được fit đúng một lần trên train và chấm đúng một lần
trên validation.

Cái mất khi bỏ CV, ghi ra để đọc số cho đúng: mỗi chỉ số là MỘT điểm đo trên
46.127 hồ sơ validation, không phải trung bình 5 lần kèm độ lệch. Không còn
`pr_auc_std`, nên chênh vài phần nghìn giữa hai model **không** quy chiếu được
về độ nhiễu. Task 12 phải phát biểu đúng như vậy, đừng nói "hơn hẳn" cho một
khoảng chênh không đo được độ tin cậy.

Vì sao lưu DANH SÁCH `SK_ID_CURR` chứ không chỉ lưu seed
--------------------------------------------------------
Lưu mỗi `random_state=42` rồi mỗi task tự chia lại nghe có vẻ đủ, vì phép chia
là tất định. Nhưng nó chỉ tất định **khi đầu vào không đổi**: thêm một dòng,
đổi thứ tự dòng, hay chạy lại task 2 trên dữ liệu mới là cả ba tập đổi hết —
và không có gì báo. Lúc đó model của task 7 và model của task 10 được train
trên hai tập khác nhau, còn bảng so sánh ở task 12 thì vẫn trông bình thường.

Lưu danh sách id thì lệch là thấy ngay: `load_split()` kiểm số lượng và giao
nhau, `verify_split()` kiểm phủ kín.

Ba tập rời nhau — điểm phải canh nhất
--------------------------------------
Một hồ sơ nằm ở cả train lẫn test là rò rỉ theo nghĩa đen: model đã thấy đáp
án. Task 2 đã bỏ dòng trùng `SK_ID_CURR` (0 dòng) nên id là khoá thật sự, và
`SplitResult.assert_disjoint()` kiểm lại lần nữa ngay sau khi chia.

Dòng bất hợp lệ: bỏ khỏi RIÊNG train
-------------------------------------
Task 2 cố ý chỉ gắn cờ `INVALID_ROW` chứ không bỏ dòng, vì bỏ trước khi chia
thì tập test cũng sạch theo và chỉ số báo cáo sẽ cao hơn năng lực thật — lúc
chạy thật hồ sơ bất hợp lệ vẫn cứ đến. Task này thực hiện đúng lời hẹn đó: chia
ba tập trên toàn bộ dòng, rồi chỉ loại cờ khỏi train.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from hfml.config import CONFIG
from hfml.logger import get_logger
from hfml.ml.ml02_credit_risk.clean import (
    ID_COLUMN,
    INVALID_ROW_FLAG,
    TARGET_COLUMN,
)

log = get_logger(__name__)

#: Tên ba tập, thứ tự cố định — mọi bảng báo cáo dùng đúng thứ tự này.
SPLIT_NAMES: Final[tuple[str, ...]] = ("train", "validation", "test")

#: Nơi lưu. `runs/` là nơi DUY NHẤT chứa output của tầng ML (xem `config.Paths`).
SPLIT_SUBDIR: Final[str] = "ml02_split"
ASSIGNMENT_FILE: Final[str] = "split_assignment.csv"
METADATA_FILE: Final[str] = "split_metadata.json"

#: Chênh lệch tỉ lệ dương tối đa giữa một tập và toàn bộ dữ liệu, tính theo
#: bội số sai số chuẩn. `stratify` bảo đảm chênh lệch chỉ do làm tròn, nên
#: ngưỡng này rất rộng — nó tồn tại để bắt lỗi quên `stratify`, không phải để
#: đo chất lượng phép chia.
STRATIFY_TOLERANCE_SIGMA: Final[float] = 3.0


@dataclass(frozen=True)
class SplitResult:
    """Kết quả chia, lưu theo `SK_ID_CURR` chứ không theo vị trí dòng.

    Vị trí dòng chỉ có nghĩa với đúng một lần đọc file; `SK_ID_CURR` có nghĩa
    với mọi lần đọc, kể cả khi thứ tự dòng đổi.
    """

    train_ids: np.ndarray
    validation_ids: np.ndarray
    test_ids: np.ndarray
    seed: int
    val_size: float
    test_size: float
    #: Số dòng bị loại khỏi RIÊNG tập train vì mang cờ `INVALID_ROW`.
    n_invalid_excluded_from_train: int = 0

    def ids(self, split: str) -> np.ndarray:
        if split not in SPLIT_NAMES:
            raise ValueError(f"Tập không hợp lệ: {split!r}. Chọn {SPLIT_NAMES}.")
        return getattr(self, f"{split}_ids")

    @property
    def n_total(self) -> int:
        return sum(len(self.ids(s)) for s in SPLIT_NAMES)

    def assert_disjoint(self) -> None:
        """Ba tập không được giao nhau. Ném lỗi ngay nếu có."""
        for a, b in (("train", "validation"), ("train", "test"),
                     ("validation", "test")):
            overlap = np.intersect1d(self.ids(a), self.ids(b))
            if overlap.size:
                raise ValueError(
                    f"{overlap.size} hồ sơ nằm ở cả {a} lẫn {b} — rò rỉ dữ liệu. "
                    f"Ví dụ: {overlap[:5].tolist()}")

    def frame(self) -> pd.DataFrame:
        """Bảng hai cột `SK_ID_CURR, split` — file lưu ra đĩa."""
        return pd.DataFrame({
            ID_COLUMN: np.concatenate([self.ids(s) for s in SPLIT_NAMES]),
            "split": np.concatenate(
                [np.repeat(s, len(self.ids(s))) for s in SPLIT_NAMES]),
        })

    def apply(self, df: pd.DataFrame, split: str) -> pd.DataFrame:
        """Lấy phần dữ liệu thuộc một tập, theo `SK_ID_CURR`."""
        return df[df[ID_COLUMN].isin(self.ids(split))].reset_index(drop=True)


def split_train_val_test(
    df: pd.DataFrame,
    val_size: float | None = None,
    test_size: float | None = None,
    seed: int | None = None,
    exclude_invalid_from_train: bool = True,
) -> SplitResult:
    """Chia 70/15/15, phân tầng theo nhãn.

    Cắt hai lần chứ không một lần: lần đầu tách `test`, lần sau tách
    `validation` ra khỏi phần còn lại. Tỉ lệ lần hai phải quy đổi theo phần còn
    lại (`0,15 / 0,85 = 0,1765`) — lấy thẳng 0,15 của phần còn lại thì
    validation chỉ được 12,75% tổng, và tập train phình lên 72,25%.

    `stratify` ở cả hai lần là **bắt buộc**, không phải cho đẹp: lớp dương chỉ
    chiếm 8,07%: cắt ngẫu nhiên thì tỉ lệ dương giữa ba tập lệch nhau và PR-AUC
    — vốn phụ thuộc trực tiếp vào tỉ lệ nền — hết so được giữa các tập.
    """
    val_size = CONFIG.training["val_size"] if val_size is None else val_size
    test_size = CONFIG.training["test_size"] if test_size is None else test_size
    seed = CONFIG.random_seed if seed is None else seed

    if val_size + test_size >= 1.0:
        raise ValueError(
            f"val_size ({val_size}) + test_size ({test_size}) phải nhỏ hơn 1,0 — "
            "không còn dòng nào cho tập train.")

    ids = df[ID_COLUMN].to_numpy()
    y = df[TARGET_COLUMN].to_numpy()

    rest_ids, test_ids, y_rest, _ = train_test_split(
        ids, y, test_size=test_size, stratify=y, random_state=seed)

    val_of_rest = val_size / (1.0 - test_size)
    train_ids, validation_ids, _, _ = train_test_split(
        rest_ids, y_rest, test_size=val_of_rest, stratify=y_rest,
        random_state=seed)

    # Cờ `INVALID_ROW` chỉ loại khỏi TRAIN. Tập validation và test giữ nguyên
    # để phản ánh đúng phân bố lúc chạy thật.
    n_excluded = 0
    if exclude_invalid_from_train and INVALID_ROW_FLAG in df.columns:
        invalid = set(df.loc[df[INVALID_ROW_FLAG] == 1, ID_COLUMN])
        if invalid:
            before = len(train_ids)
            train_ids = np.array([i for i in train_ids if i not in invalid])
            n_excluded = before - len(train_ids)

    result = SplitResult(
        train_ids=np.sort(train_ids),
        validation_ids=np.sort(validation_ids),
        test_ids=np.sort(test_ids),
        seed=seed,
        val_size=val_size,
        test_size=test_size,
        n_invalid_excluded_from_train=n_excluded,
    )
    result.assert_disjoint()

    log.info("Chia dữ liệu: train %d (%.1f%%) · validation %d (%.1f%%) · test %d (%.1f%%)",
             len(result.train_ids), len(result.train_ids) / len(df) * 100,
             len(result.validation_ids), len(result.validation_ids) / len(df) * 100,
             len(result.test_ids), len(result.test_ids) / len(df) * 100)
    return result


# --------------------------------------------------------------------------
# Kiểm chứng
# --------------------------------------------------------------------------
def distribution_table(df: pd.DataFrame, split: SplitResult) -> pd.DataFrame:
    """Phân bố nhãn ở từng tập, kèm chênh lệch so với toàn bộ dữ liệu.

    `deviation_sigma` là chênh lệch tính theo bội số sai số chuẩn của tỉ lệ —
    con số đó mới so được giữa các tập có kích thước khác nhau. Chênh 0,1 điểm
    phần trăm trên 46.000 dòng nghiêm trọng hơn hẳn chênh 0,1 điểm trên 3.000.
    """
    base_rate = float(df[TARGET_COLUMN].mean())
    rows = []
    for name in SPLIT_NAMES:
        subset = split.apply(df, name)
        n = len(subset)
        rate = float(subset[TARGET_COLUMN].mean()) if n else float("nan")
        std_error = float(np.sqrt(base_rate * (1 - base_rate) / n)) if n else float("nan")
        rows.append({
            "split": name,
            "n_rows": n,
            "share": n / len(df) if len(df) else 0.0,
            "n_positive": int(subset[TARGET_COLUMN].sum()) if n else 0,
            "positive_rate": rate,
            "deviation_sigma": abs(rate - base_rate) / std_error if std_error else 0.0,
        })
    return pd.DataFrame(rows)


def verify_split(df: pd.DataFrame, split: SplitResult) -> pd.DataFrame:
    """Năm phép kiểm, mỗi phép trả về một dòng có ĐO ĐƯỢC.

    Cùng tinh thần với kiểm toán rò rỉ ở task 2: trả lời *"làm sao biết phép
    chia này đúng?"* bằng số chứ không bằng "tôi đã cẩn thận".
    """
    checks: list[dict] = []

    def add(check: str, passed: bool, measured: str, note: str) -> None:
        checks.append({"check": check, "passed": passed,
                       "measured": measured, "note": note})

    # 1. Ba tập rời nhau.
    try:
        split.assert_disjoint()
        disjoint, detail = True, "0 hồ sơ nằm ở hai tập"
    except ValueError as exc:
        disjoint, detail = False, str(exc)
    add("splits_are_disjoint", disjoint, detail,
        "Một hồ sơ ở cả train lẫn test là rò rỉ theo nghĩa đen.")

    # 2. Ba tập phủ kín dữ liệu (trừ dòng bất hợp lệ đã loại khỏi train).
    expected = len(df) - split.n_invalid_excluded_from_train
    add("splits_cover_every_row", split.n_total == expected,
        f"{split.n_total:,} / {expected:,} dòng "
        f"(đã loại {split.n_invalid_excluded_from_train} dòng bất hợp lệ khỏi train)",
        "Dòng rơi ra ngoài cả ba tập là dữ liệu bị bỏ quên không ai biết.")

    # 3. Tỉ lệ ba tập đúng như đã chốt.
    table = distribution_table(df, split)
    shares = dict(zip(table["split"], table["share"]))
    target = {"train": 1 - split.val_size - split.test_size,
              "validation": split.val_size, "test": split.test_size}
    worst = max(abs(shares[s] - target[s]) for s in SPLIT_NAMES)
    add("split_sizes_match_the_plan", worst < 0.01,
        " · ".join(f"{s} {shares[s]:.2%} (chốt {target[s]:.0%})" for s in SPLIT_NAMES),
        "Cắt hai lần mà quên quy đổi tỉ lệ lần hai thì validation chỉ được 12,75%.")

    # 4. Phân tầng có tác dụng — tỉ lệ dương ba tập không lệch quá nhiễu.
    worst_sigma = float(table["deviation_sigma"].max())
    add("target_rate_is_preserved", worst_sigma <= STRATIFY_TOLERANCE_SIGMA,
        f"lệch nhiều nhất {worst_sigma:.2f}σ so với tỉ lệ chung "
        f"{df[TARGET_COLUMN].mean():.4%} (ngưỡng {STRATIFY_TOLERANCE_SIGMA}σ)",
        "Quên `stratify` thì tỉ lệ dương ba tập lệch nhau và PR-AUC hết so được.")

    # 5. Không dòng bất hợp lệ nào lọt vào train.
    if INVALID_ROW_FLAG in df.columns:
        train = split.apply(df, "train")
        n_invalid = int(train[INVALID_ROW_FLAG].sum())
        add("train_has_no_invalid_rows", n_invalid == 0,
            f"{n_invalid} dòng mang cờ {INVALID_ROW_FLAG} trong train",
            "Task 2 hẹn loại chúng khỏi RIÊNG train; validation/test giữ nguyên.")

    return pd.DataFrame(checks)


# --------------------------------------------------------------------------
# Lưu và nạp lại
# --------------------------------------------------------------------------
def output_dir() -> Path:
    return CONFIG.paths.runs / SPLIT_SUBDIR


def save_split(split: SplitResult, df: pd.DataFrame | None = None) -> dict[str, Path]:
    """Ghi danh sách id + metadata. Task 6 → 15 nạp lại bằng `load_split()`."""
    out_dir = output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    assignment_path = out_dir / ASSIGNMENT_FILE
    split.frame().to_csv(assignment_path, index=False, encoding="utf-8")

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "task": "ML02 task 5 — Chia train/validation/test",
        "method": "stratified two-stage holdout",
        "cross_validation": False,
        "note": "KHÔNG dùng K-Fold Cross-Validation. Mỗi thuật toán fit một "
                "lần trên train, chấm một lần trên validation. Test chỉ mở ở "
                "task 14.",
        "seed": split.seed,
        "sizes": {
            "train": len(split.train_ids),
            "validation": len(split.validation_ids),
            "test": len(split.test_ids),
        },
        "ratios": {
            "train": 1 - split.val_size - split.test_size,
            "validation": split.val_size,
            "test": split.test_size,
        },
        "n_invalid_excluded_from_train": split.n_invalid_excluded_from_train,
        "id_column": ID_COLUMN,
        "target_column": TARGET_COLUMN,
    }
    if df is not None:
        metadata["distribution"] = distribution_table(df, split).to_dict("records")

    metadata_path = out_dir / METADATA_FILE
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    log.info("Đã lưu phép chia → %s", out_dir)
    return {"assignment": assignment_path, "metadata": metadata_path}


def load_split() -> SplitResult:
    """Nạp lại phép chia đã lưu — điểm vào của task 6 → 15.

    Mọi task sau PHẢI đi qua đây thay vì tự chia lại. Tự chia lại thì chỉ cần
    một tham số khác nhau ở một chỗ là hai model được train trên hai tập khác
    nhau, và bảng so sánh ở task 12 so nhầm mà không có gì để lộ ra.
    """
    out_dir = output_dir()
    assignment_path = out_dir / ASSIGNMENT_FILE
    metadata_path = out_dir / METADATA_FILE

    if not assignment_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"Chưa có phép chia ở {out_dir}. "
            "Chạy `python scripts/split_ml02.py` trước.")

    frame = pd.read_csv(assignment_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    by_split = {name: group[ID_COLUMN].to_numpy()
                for name, group in frame.groupby("split")}

    missing = [s for s in SPLIT_NAMES if s not in by_split]
    if missing:
        raise ValueError(f"File phép chia thiếu tập: {missing}")

    result = SplitResult(
        train_ids=by_split["train"],
        validation_ids=by_split["validation"],
        test_ids=by_split["test"],
        seed=metadata["seed"],
        val_size=metadata["ratios"]["validation"],
        test_size=metadata["ratios"]["test"],
        n_invalid_excluded_from_train=metadata["n_invalid_excluded_from_train"],
    )
    # Kiểm lại ngay lúc nạp: file có thể bị sửa tay giữa hai task.
    result.assert_disjoint()
    return result
