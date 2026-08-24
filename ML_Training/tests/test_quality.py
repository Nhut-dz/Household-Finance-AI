"""F01 task 6 + 7 — kiểm tra chất lượng dữ liệu và chốt phiên bản dataset.

Phần lớn test chạy trên DataFrame dựng tay: nhanh, và quan trọng hơn là
kiểm được cả trường hợp dữ liệu bẩn mà dataset thật không có (ID trùng, tiền
âm, cột hằng số). Vài test cuối mới đụng dataset thật, và tự skip nếu chưa tải.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from hfml.data import loader, quality
from hfml.data.quality import QualityReport, Severity

HAS_DATASET = loader.resolve(loader.PRIMARY_FILE).exists()
needs_dataset = pytest.mark.skipif(HAS_DATASET is False, reason="chưa tải dataset Home Credit")


def frame(**overrides) -> pd.DataFrame:
    """Bảng nhỏ sạch sẽ, giống cấu trúc application_train."""
    data = {
        "SK_ID_CURR": [1, 2, 3, 4],
        "TARGET": [0, 0, 1, 0],
        "DAYS_EMPLOYED": [-1000, -2000, -500, -3000],
        "AMT_INCOME_TOTAL": [100.0, 200.0, 300.0, 400.0],
        "OCCUPATION_TYPE": ["Laborers", "Managers", "Laborers", "Drivers"],
    }
    data.update(overrides)
    return pd.DataFrame(data)


# ------------------------------------------------------- task 6: phát hiện

def codes(df: pd.DataFrame, target: str | None = "TARGET") -> set[str]:
    return {i.code for i in quality.find_issues(df, target=target)}


def test_clean_frame_reports_no_error():
    issues = quality.find_issues(frame(), target="TARGET")
    assert not [i for i in issues if i.severity is Severity.ERROR]


def test_empty_table_is_error():
    issues = quality.find_issues(pd.DataFrame(), target=None)
    assert issues[0].code == "empty_table"
    assert issues[0].severity is Severity.ERROR


def test_duplicate_id_detected():
    assert "duplicate_id" in codes(frame(SK_ID_CURR=[1, 1, 2, 3]))


def test_sentinel_detected():
    df = frame(DAYS_EMPLOYED=[365243, -2000, 365243, -3000])
    issues = [i for i in quality.find_issues(df) if i.code == "sentinel_value"]
    assert issues and issues[0].severity is Severity.ERROR
    assert "365,243" in issues[0].message
    assert issues[0].columns == ("DAYS_EMPLOYED",)


def test_placeholder_value_detected():
    df = frame(OCCUPATION_TYPE=["XNA", "Managers", "Unknown", "Drivers"])
    assert "placeholder_as_category" in codes(df)


def test_constant_column_detected():
    df = frame()
    df["FLAG_X"] = 1
    assert "constant_column" in codes(df)


def test_negative_amount_is_error():
    df = frame(AMT_INCOME_TOTAL=[-1.0, 200.0, 300.0, 400.0])
    issues = [i for i in quality.find_issues(df) if i.code == "negative_amount"]
    assert issues and issues[0].severity is Severity.ERROR


def test_high_missing_detected():
    df = frame()
    df["MOSTLY_EMPTY"] = [1.0, None, None, None]      # 75% thiếu
    assert "high_missing" in codes(df)


def test_high_cardinality_detected():
    n = quality.HIGH_CARDINALITY_THRESHOLD + 5
    df = pd.DataFrame({
        "SK_ID_CURR": range(n),
        "TARGET": [0] * n,
        "ORG": [f"org_{i}" for i in range(n)],
    })
    assert "high_cardinality" in codes(df)


def test_class_imbalance_detected():
    n = 100
    df = pd.DataFrame({"SK_ID_CURR": range(n), "TARGET": [1] * 5 + [0] * 95})
    issues = [i for i in quality.find_issues(df, target="TARGET") if i.code == "class_imbalance"]
    assert issues and "PR-AUC" in issues[0].message


def test_column_report_sorted_by_missing():
    df = frame()
    df["A"] = [1.0, None, None, None]
    df["B"] = [1.0, 2.0, 3.0, 4.0]
    rep = quality.column_report(df)
    assert rep.iloc[0]["column"] == "A"
    assert rep.iloc[0]["missing"] == 3
    assert set(rep["column"]) == set(df.columns)


def test_report_markdown_renders():
    report = QualityReport(table="t.csv", n_rows=4, n_cols=5,
                           issues=quality.find_issues(frame(SK_ID_CURR=[1, 1, 2, 3])))
    md = report.to_markdown()
    assert "duplicate_id" in md and report.has_errors


def test_report_markdown_when_clean():
    report = QualityReport(table="t.csv", n_rows=4, n_cols=5, issues=[])
    assert "Không phát hiện vấn đề" in report.to_markdown()
    assert not report.has_errors


# ------------------------------------------------------ task 7: phiên bản

def test_sha256_matches_hashlib(tmp_path):
    import hashlib

    path = tmp_path / "x.bin"
    payload = b"h" * (3 * 1024 * 1024 + 7)      # nhiều hơn 1 khối, lẻ khối
    path.write_bytes(payload)
    assert quality.sha256_file(path) == hashlib.sha256(payload).hexdigest()


def test_sha256_of_empty_file(tmp_path):
    import hashlib

    path = tmp_path / "empty.bin"
    path.write_bytes(b"")
    assert quality.sha256_file(path) == hashlib.sha256(b"").hexdigest()


@pytest.mark.parametrize("size,expected", [
    (37_383, "37 KB"),
    (166_133_370, "158.4 MB"),
    (2 * 1024**3, "2.00 GB"),
])
def test_size_formatting(size, expected):
    assert quality._format_size(size) == expected


def test_manifest_path_is_relative_not_absolute():
    """Manifest có commit vào git — không được nhúng ổ đĩa của một máy."""
    rel = quality._relative_dataset_dir()
    assert not rel.startswith(("/", "\\"))
    assert ":" not in rel


# --------------------------------------------------- chạy trên dataset thật

@needs_dataset
def test_fingerprint_without_hash_is_cheap():
    info = quality.file_fingerprint(loader.PRIMARY_FILE, compute_hash=False)
    assert info["size_bytes"] > 0
    assert "sha256" not in info


@needs_dataset
def test_manifest_structure():
    m = quality.build_manifest(compute_hash=False)
    # `missing_files` nay chứa đúng hai file ngoài phạm vi đã xoá khỏi đĩa
    # (24/08/2026), không phải rỗng như trước.
    assert sorted(m["missing_files"]) == sorted(loader.OPTIONAL_FILES)
    assert set(m["files"]) == set(loader.REQUIRED_FILES)
    assert m["application_train"]["n_rows"] == 307_511
    assert m["source"].startswith("https://")


@needs_dataset
def test_written_manifest_is_valid_json_and_matches_disk():
    saved = json.loads(quality.manifest_path().read_text(encoding="utf-8"))
    for name, info in saved["files"].items():
        assert loader.resolve(name).stat().st_size == info["size_bytes"], name
        assert len(info["sha256"]) == 64


@needs_dataset
def test_verify_manifest_clean():
    """Dataset trên đĩa phải khớp manifest đã chốt. Bỏ băm cho nhanh."""
    assert quality.verify_manifest(compute_hash=False) == []


@needs_dataset
def test_verify_detects_size_drift(monkeypatch):
    real = quality.load_manifest()
    tampered = json.loads(json.dumps(real))
    key = next(iter(tampered["files"]))
    tampered["files"][key]["size_bytes"] += 1
    monkeypatch.setattr(quality, "load_manifest", lambda: tampered)

    drift = quality.verify_manifest(compute_hash=False)
    assert any("kích thước" in d for d in drift)


@needs_dataset
def test_dataset_doc_contains_key_numbers():
    doc = (quality.CONFIG.paths.docs / quality.DATASET_DOC_FILENAME).read_text(encoding="utf-8")
    for expected in ("307,511", "24,825", "11.39", "91.9", "kaggle.com"):
        assert expected in doc, expected
