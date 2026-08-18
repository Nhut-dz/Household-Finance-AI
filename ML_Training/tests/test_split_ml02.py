"""Test phép chia train/validation/test của ML02 (`...ml02_credit_risk.split`).

Bốn bất biến được canh, mỗi cái ứng với một cách hỏng KHÔNG báo lỗi:

    · ba tập giao nhau      → rò rỉ theo nghĩa đen, chỉ số đẹp giả
    · quên `stratify`       → tỉ lệ dương lệch giữa ba tập, PR-AUC hết so được
    · quên quy đổi tỉ lệ    → validation chỉ được 12,75% thay vì 15%
    · mỗi task tự chia lại  → hai model train trên hai tập khác nhau, bảng so
                              sánh ở task 12 so nhầm
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hfml.config import CONFIG
from hfml.ml.ml02_credit_risk.clean import (
    ID_COLUMN,
    INVALID_ROW_FLAG,
    TARGET_COLUMN,
)
from hfml.ml.ml02_credit_risk.split import (
    SPLIT_NAMES,
    SplitResult,
    distribution_table,
    load_split,
    save_split,
    split_train_val_test,
    verify_split,
)


def frame(n: int = 20_000, positive_rate: float = 0.08,
          invalid: int = 0) -> pd.DataFrame:
    """Dữ liệu đã làm sạch, thu nhỏ, có tỉ lệ dương định trước."""
    n_positive = int(n * positive_rate)
    target = np.array([1] * n_positive + [0] * (n - n_positive))
    rng = np.random.default_rng(0)
    rng.shuffle(target)

    flags = np.zeros(n, dtype=int)
    if invalid:
        flags[:invalid] = 1

    return pd.DataFrame({
        ID_COLUMN: np.arange(100_001, 100_001 + n),
        TARGET_COLUMN: target,
        INVALID_ROW_FLAG: flags,
        "feature": rng.normal(size=n),
    })


# ------------------------------------------------------------- tỉ lệ ba tập
def test_split_is_seventy_fifteen_fifteen():
    df = frame()
    split = split_train_val_test(df)

    shares = {s: len(split.ids(s)) / len(df) for s in SPLIT_NAMES}

    assert shares["train"] == pytest.approx(0.70, abs=0.005)
    assert shares["validation"] == pytest.approx(0.15, abs=0.005)
    assert shares["test"] == pytest.approx(0.15, abs=0.005)


def test_validation_ratio_is_converted_against_the_remainder():
    """Cắt hai lần mà quên quy đổi thì validation chỉ được 12,75%.

    Lần cắt thứ hai chạy trên 85% còn lại, nên muốn validation bằng 15% TỔNG
    thì phải lấy 0,15/0,85 = 17,65% của phần còn lại. Lấy thẳng 0,15 là sai —
    và sai một cách rất khó thấy, vì 12,75% vẫn "trông như" 15%.
    """
    df = frame(20_000)
    split = split_train_val_test(df)

    n_val = len(split.validation_ids)

    assert n_val == pytest.approx(3_000, abs=30)
    assert n_val > 2_900, f"validation chỉ có {n_val} — nhiều khả năng quên quy đổi"


def test_sizes_that_leave_no_training_data_are_rejected():
    with pytest.raises(ValueError, match="nhỏ hơn 1"):
        split_train_val_test(frame(1_000), val_size=0.5, test_size=0.5)


# ------------------------------------------------------------- ba tập rời nhau
def test_the_three_splits_never_overlap():
    """Một hồ sơ ở cả train lẫn test là rò rỉ theo nghĩa đen."""
    split = split_train_val_test(frame())

    for a, b in (("train", "validation"), ("train", "test"),
                 ("validation", "test")):
        assert np.intersect1d(split.ids(a), split.ids(b)).size == 0


def test_overlapping_split_is_rejected_on_construction():
    """`assert_disjoint` phải bắt được, kể cả khi file bị sửa tay."""
    overlapping = SplitResult(
        train_ids=np.array([1, 2, 3]),
        validation_ids=np.array([3, 4]),      # id 3 nằm ở hai tập
        test_ids=np.array([5]),
        seed=42, val_size=0.15, test_size=0.15,
    )

    with pytest.raises(ValueError, match="rò rỉ"):
        overlapping.assert_disjoint()


def test_every_row_lands_in_exactly_one_split():
    """Dòng rơi ra ngoài cả ba tập là dữ liệu bị bỏ quên không ai biết."""
    df = frame()
    split = split_train_val_test(df)

    tat_ca = np.concatenate([split.ids(s) for s in SPLIT_NAMES])

    assert len(tat_ca) == len(df)
    assert set(tat_ca) == set(df[ID_COLUMN])


# ------------------------------------------------------------------ phân tầng
def test_target_rate_is_preserved_across_splits():
    """Lớp dương chỉ 8,07%; cắt ngẫu nhiên thì ba tập lệch nhau.

    PR-AUC phụ thuộc trực tiếp vào tỉ lệ nền, nên ba tập lệch tỉ lệ dương là
    ba con số không so được với nhau.
    """
    df = frame(20_000, positive_rate=0.08)
    split = split_train_val_test(df)

    table = distribution_table(df, split)

    assert (table["deviation_sigma"] <= 3.0).all(), table.to_string(index=False)
    for rate in table["positive_rate"]:
        assert rate == pytest.approx(0.08, abs=0.005)


def test_a_rare_class_survives_in_every_split():
    """Ngay cả khi lớp dương rất hiếm, không tập nào được rỗng lớp đó."""
    df = frame(20_000, positive_rate=0.01)
    split = split_train_val_test(df)

    for name in SPLIT_NAMES:
        assert split.apply(df, name)[TARGET_COLUMN].sum() > 0


# ---------------------------------------------------- dòng bất hợp lệ
def test_invalid_rows_are_removed_from_train_only():
    """Task 2 hẹn: bỏ khỏi RIÊNG train, validation/test giữ nguyên.

    Bỏ khỏi cả ba tập thì tập test cũng sạch theo, và chỉ số báo cáo sẽ cao
    hơn năng lực thật — lúc chạy thật hồ sơ bất hợp lệ vẫn cứ đến.
    """
    df = frame(20_000, invalid=200)
    split = split_train_val_test(df)

    train = split.apply(df, "train")
    con_lai = pd.concat([split.apply(df, "validation"), split.apply(df, "test")])

    assert train[INVALID_ROW_FLAG].sum() == 0
    assert con_lai[INVALID_ROW_FLAG].sum() > 0, "validation/test không được dọn"
    assert split.n_invalid_excluded_from_train > 0


def test_keeping_invalid_rows_is_possible_for_inspection():
    df = frame(5_000, invalid=100)

    split = split_train_val_test(df, exclude_invalid_from_train=False)

    assert split.n_invalid_excluded_from_train == 0
    assert split.n_total == len(df)


# --------------------------------------------------------------- tái lập
def test_the_same_seed_gives_the_same_split():
    df = frame()

    a = split_train_val_test(df, seed=42)
    b = split_train_val_test(df, seed=42)

    np.testing.assert_array_equal(a.train_ids, b.train_ids)
    np.testing.assert_array_equal(a.test_ids, b.test_ids)


def test_a_different_seed_gives_a_different_split():
    """Nếu đổi seed mà kết quả không đổi thì `random_state` không có tác dụng."""
    df = frame()

    a = split_train_val_test(df, seed=42)
    b = split_train_val_test(df, seed=7)

    assert not np.array_equal(a.train_ids, b.train_ids)


def test_split_uses_the_project_seed_by_default():
    df = frame()

    assert split_train_val_test(df).seed == CONFIG.random_seed


# ------------------------------------------------------------ lưu và nạp lại
def test_saved_split_reloads_identically(tmp_path, monkeypatch):
    """Task 6 → 15 nạp lại bằng `load_split()`; nó phải trả về đúng bộ đã ghi."""
    monkeypatch.setattr(CONFIG.paths, "runs", tmp_path)
    df = frame()
    goc = split_train_val_test(df)

    save_split(goc, df)
    nap_lai = load_split()

    for name in SPLIT_NAMES:
        np.testing.assert_array_equal(
            np.sort(nap_lai.ids(name)), np.sort(goc.ids(name)))
    assert nap_lai.seed == goc.seed
    assert nap_lai.n_invalid_excluded_from_train == goc.n_invalid_excluded_from_train


def test_loading_without_a_saved_split_fails_loudly(tmp_path, monkeypatch):
    """Thiếu file phải báo lỗi kèm cách sửa, không trả về phép chia rỗng."""
    monkeypatch.setattr(CONFIG.paths, "runs", tmp_path)

    with pytest.raises(FileNotFoundError, match="split_ml02"):
        load_split()


def test_split_is_stored_by_id_not_by_row_position(tmp_path, monkeypatch):
    """Lưu theo `SK_ID_CURR` để đổi thứ tự dòng không làm hỏng phép chia.

    Vị trí dòng chỉ có nghĩa với đúng một lần đọc file; id có nghĩa với mọi
    lần đọc.
    """
    monkeypatch.setattr(CONFIG.paths, "runs", tmp_path)
    df = frame(5_000)
    save_split(split_train_val_test(df), df)

    xao_tron = df.sample(frac=1.0, random_state=1).reset_index(drop=True)
    nap_lai = load_split()

    train = nap_lai.apply(xao_tron, "train")

    assert len(train) == len(nap_lai.train_ids)
    assert set(train[ID_COLUMN]) == set(nap_lai.train_ids)


# ------------------------------------------------------------------ kiểm chứng
def test_verification_passes_on_a_correct_split():
    df = frame(20_000, invalid=100)
    split = split_train_val_test(df)

    checks = verify_split(df, split)

    assert checks["passed"].all(), checks.to_string(index=False)
    assert len(checks) == 5


def test_verification_catches_an_unstratified_split():
    """Phép kiểm phải bắt được lỗi quên `stratify`, không chỉ chạy cho có."""
    df = frame(20_000, positive_rate=0.08).sort_values(
        TARGET_COLUMN, ignore_index=True)

    # Cắt theo thứ tự đã sắp: toàn bộ lớp dương dồn về cuối, tức vào test.
    n = len(df)
    lech = SplitResult(
        train_ids=df[ID_COLUMN].to_numpy()[: int(n * 0.70)],
        validation_ids=df[ID_COLUMN].to_numpy()[int(n * 0.70): int(n * 0.85)],
        test_ids=df[ID_COLUMN].to_numpy()[int(n * 0.85):],
        seed=42, val_size=0.15, test_size=0.15,
    )

    checks = verify_split(df, lech).set_index("check")

    assert not checks.loc["target_rate_is_preserved", "passed"]


def test_verification_catches_rows_that_fell_outside_every_split():
    df = frame(1_000)
    ids = df[ID_COLUMN].to_numpy()
    thieu = SplitResult(
        train_ids=ids[:600], validation_ids=ids[600:750], test_ids=ids[750:900],
        seed=42, val_size=0.15, test_size=0.15,
    )

    checks = verify_split(df, thieu).set_index("check")

    assert not checks.loc["splits_cover_every_row", "passed"]


def test_apply_returns_only_the_rows_of_that_split():
    df = frame(5_000)
    split = split_train_val_test(df)

    train = split.apply(df, "train")

    assert set(train[ID_COLUMN]) == set(split.train_ids)
    assert len(train) == len(split.train_ids)


def test_unknown_split_name_is_rejected():
    split = split_train_val_test(frame(1_000))

    with pytest.raises(ValueError, match="Tập không hợp lệ"):
        split.ids("val")
