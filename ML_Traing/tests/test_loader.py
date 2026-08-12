"""F01 task 5 — kiểm tra nạp dữ liệu Home Credit.

Test đọc dataset thật (166 MB), nên chỉ lấy vài cột và giới hạn số dòng ở
những chỗ không cần toàn bộ. Nếu chưa tải dataset, cả module tự skip thay vì
đỏ toàn bảng — người mới clone repo về chưa có `dataset/`.
"""
from __future__ import annotations

import pytest

from hfml.data import loader
from hfml.data.loader import (
    HOME_CREDIT_FILES,
    ID_COLUMN,
    PRIMARY_FILE,
    TARGET_COLUMN,
    DatasetNotFoundError,
)

pytestmark = pytest.mark.skipif(
    not loader.resolve(PRIMARY_FILE).exists(),
    reason="chưa tải dataset Home Credit vào dataset/home-credit-default-risk/",
)


# ------------------------------------------------------------- đường dẫn

def test_all_five_files_present():
    """PLAN.md §4.3 khẳng định đủ 5 file — kiểm lại chứ không tin tài liệu."""
    missing = [name for name, ok in loader.available_files().items() if not ok]
    assert not missing, f"thiếu file: {missing}"


def test_unknown_file_name_rejected():
    with pytest.raises(KeyError, match="Không biết file"):
        loader.resolve("application_test")


def test_missing_file_error_mentions_download_source(monkeypatch, tmp_path):
    """Lỗi thiếu file phải nói rõ tải ở đâu, không chỉ 'file not found'."""
    monkeypatch.setattr(loader.CONFIG.paths, "dataset", tmp_path)
    with pytest.raises(DatasetNotFoundError, match="kaggle.com"):
        loader.require(PRIMARY_FILE)


# ----------------------------------------------------------------- đọc

def test_load_subset_of_columns():
    df = loader.load_application_train(columns=["AMT_INCOME_TOTAL"], nrows=100)
    assert len(df) == 100
    # ID và TARGET được thêm tự động dù không yêu cầu.
    assert set(df.columns) == {ID_COLUMN, TARGET_COLUMN, "AMT_INCOME_TOTAL"}


def test_columns_not_duplicated_when_explicitly_requested():
    df = loader.load_application_train(
        columns=[ID_COLUMN, TARGET_COLUMN, "DAYS_BIRTH"], nrows=10)
    assert list(df.columns).count(ID_COLUMN) == 1
    assert list(df.columns).count(TARGET_COLUMN) == 1


def test_row_ids_are_unique():
    df = loader.load_application_train(columns=[], nrows=5000)
    assert df[ID_COLUMN].is_unique


def test_columns_needed_by_plan_exist():
    """Các cột PLAN §2.1 và §7.2 nhắc đích danh phải có thật."""
    needed = [
        "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
        "DAYS_BIRTH", "DAYS_EMPLOYED", "CNT_CHILDREN", "CNT_FAM_MEMBERS",
        "OCCUPATION_TYPE", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
    ]
    df = loader.load_application_train(columns=needed, nrows=5)
    assert set(needed) <= set(df.columns)


# ------------------------------------------------------------- phân bố

def test_target_distribution_matches_plan():
    """Con số trong PLAN §7.3 phải khớp dữ liệu thật, sai lệch là hỏng báo cáo."""
    dist = loader.target_distribution()
    assert dist["n_rows"] == 307_511
    assert dist["n_positive"] == 24_825
    assert dist["positive_rate"] == pytest.approx(0.0807, abs=1e-4)
    assert dist["scale_pos_weight"] == pytest.approx(11.4, abs=0.1)
    assert dist["majority_class_accuracy"] == pytest.approx(0.9193, abs=1e-4)


def test_loader_does_not_clean_sentinel():
    """Loader chỉ đọc. Sentinel 365243 phải còn nguyên cho task 6 nhìn thấy."""
    df = loader.load_application_train(columns=["DAYS_EMPLOYED"])
    assert (df["DAYS_EMPLOYED"] == 365243).any()


# ------------------------------------------------------------ mô tả cột

def test_columns_description_readable():
    desc = loader.load_columns_description()
    assert {"Table", "Row", "Description"} <= set(desc.columns)
    assert len(desc) > 100


def test_describe_known_column():
    assert "difficulties" in (loader.describe_column(TARGET_COLUMN) or "").lower()


def test_describe_unknown_column_returns_none():
    assert loader.describe_column("KHONG_TON_TAI") is None


def test_file_registry_covers_primary():
    assert PRIMARY_FILE in HOME_CREDIT_FILES
