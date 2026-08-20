"""Test bước làm sạch của ML02 (`hfml.ml.ml02_credit_risk.clean`).

Không test nào chạm dataset thật. Mỗi test dựng một khung dữ liệu nhỏ có
KHUYẾT ĐIỂM ĐÃ BIẾT rồi kiểm bước làm sạch có bắt được đúng khuyết điểm đó
không.

Trọng tâm là những cách làm sạch có thể hỏng **âm thầm** — tức vẫn cho ra một
bảng trông sạch sẽ, model vẫn train được, chỉ số vẫn đẹp, và cái sai chỉ lộ
ra khi hệ thống chạy thật. Ba nhóm như vậy:

    · rò rỉ nhãn      → chỉ số đẹp giả, không có dấu hiệu nào
    · bỏ nhầm dòng    → đếm thiếu, sai lệch đúng feature mà form đang hỏi
    · làm sạch có học → metric lạc quan vì thống kê tính trên cả tập test
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hfml.ml.ml02_credit_risk.clean import (
    BUREAU_FUTURE_LOOKING_OK,
    BUREAU_PAST_ONLY_COLUMNS,
    CleaningReport,
    ID_COLUMN,
    INVALID_ROW_FLAG,
    NON_FEATURE_COLUMNS,
    PIPELINE_STEPS_REMAINING,
    TARGET_COLUMN,
    bureau_future_information,
    clean_application,
    clean_bureau,
    dtype_issues,
    dtype_report,
    feature_columns,
    leakage_audit,
    normalize_dtypes,
)


def application_frame(n: int = 200) -> pd.DataFrame:
    """Khung `application_train` thu nhỏ, đủ cột cho mọi quy tắc chạy."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "SK_ID_CURR": np.arange(100_001, 100_001 + n),
        "TARGET": rng.binomial(1, 0.08, size=n),
        "CODE_GENDER": rng.choice(["M", "F"], size=n),
        "NAME_FAMILY_STATUS": rng.choice(["Married", "Single / not married"], size=n),
        "AMT_INCOME_TOTAL": rng.uniform(50_000, 300_000, size=n),
        "AMT_CREDIT": rng.uniform(100_000, 900_000, size=n),
        "AMT_ANNUITY": rng.uniform(5_000, 40_000, size=n),
        "CNT_CHILDREN": rng.integers(0, 3, size=n),
        "CNT_FAM_MEMBERS": rng.integers(3, 6, size=n).astype(float),
        "DAYS_BIRTH": -rng.integers(8_000, 22_000, size=n),
        "DAYS_EMPLOYED": -rng.integers(100, 7_000, size=n),
        "EXT_SOURCE_1": rng.uniform(0, 1, size=n),
        "EXT_SOURCE_3": rng.uniform(0, 1, size=n),
        "OCCUPATION_TYPE": rng.choice(["Laborers", "Core staff"], size=n),
        "FLAG_MOBIL": np.ones(n, dtype=int),
    })


def bureau_frame() -> pd.DataFrame:
    """Năm bản ghi tín dụng, mỗi bản ghi minh hoạ một tình huống:

        khách 1  hai khoản bình thường
        khách 2  đang quá hạn, `DAYS_CREDIT_ENDDATE` dương (còn hiệu lực)
        khách 3  hai khoản TRÙNG NỘI DUNG, khác mỗi `SK_ID_BUREAU`
        khách 4  có thông tin cập nhật SAU ngày nộp đơn → phải bỏ
    """
    return pd.DataFrame({
        "SK_ID_CURR": [1, 1, 2, 3, 3, 4],
        "SK_ID_BUREAU": [10, 11, 12, 13, 14, 15],
        "CREDIT_ACTIVE": ["Closed", "Active", "Active", "Closed", "Closed", "Closed"],
        "DAYS_CREDIT": [-730, -365, -1_095, -500, -500, -600],
        "DAYS_ENDDATE_FACT": [-153, np.nan, np.nan, -100, -100, -200],
        # Chỉ dòng cuối có thông tin cập nhật SAU ngày nộp đơn.
        "DAYS_CREDIT_UPDATE": [-131, -20, -7, -15, -15, 42],
        # 35% giá trị dương ở dữ liệu thật — HỢP LỆ, không được bỏ.
        "DAYS_CREDIT_ENDDATE": [-153.0, 900.0, 1_200.0, -100.0, -100.0, -200.0],
        "AMT_CREDIT_SUM": [91_323.0, 50_000.0, 20_000.0, 7_500.0, 7_500.0, 3_000.0],
        "CREDIT_DAY_OVERDUE": [0, 0, 45, 0, 0, 0],
    })


# ------------------------------------------------------------ rò rỉ dữ liệu
def test_target_and_id_are_never_features():
    """Ba cột kỹ thuật không bao giờ được lọt vào X.

    Một nơi duy nhất trả lời câu "cột nào được vào X". Mỗi chỗ tự viết
    `drop(columns=['TARGET'])` là kiểu chỗ nhớ chỗ quên.
    """
    df = application_frame().assign(**{INVALID_ROW_FLAG: 0})

    features = feature_columns(df)

    assert TARGET_COLUMN not in features
    assert ID_COLUMN not in features
    assert INVALID_ROW_FLAG not in features
    assert set(NON_FEATURE_COLUMNS).isdisjoint(features)


def test_audit_catches_the_target_hiding_under_another_name():
    """Bản sao của nhãn dưới tên khác phải bị bắt.

    Đây là dạng rò rỉ nguy hiểm nhất vì nó KHÔNG lộ ra ở đâu cả: model đạt
    accuracy gần 100%, mọi thuật toán ngang nhau, và bảng so sánh trở nên vô
    nghĩa mà không có thông báo lỗi nào.
    """
    df = application_frame()
    df["WILL_DEFAULT"] = df[TARGET_COLUMN]        # nhãn đội lốt feature

    audit = leakage_audit(df)
    row = audit[audit["check"] == "no_column_duplicates_the_target"].iloc[0]

    assert row["passed"] is False or row["passed"] == False  # noqa: E712
    assert "WILL_DEFAULT" in row["measured"]


def test_audit_catches_a_feature_correlating_with_the_label():
    """Cột tương quan gần tuyệt đối với nhãn gần như luôn là nhãn trá hình."""
    rng = np.random.default_rng(0)
    df = application_frame()
    # Nhãn cộng chút nhiễu — không trùng khít nên `equals()` không bắt được,
    # nhưng tương quan thì rất cao.
    df["RISK_SCORE"] = df[TARGET_COLUMN] + rng.normal(0, 0.05, size=len(df))

    audit = leakage_audit(df)
    row = audit[audit["check"] == "no_feature_correlates_with_target"].iloc[0]

    assert not row["passed"]
    assert "RISK_SCORE" in row["measured"]


def test_audit_passes_on_clean_data():
    """Dữ liệu sạch phải qua CẢ SÁU phép kiểm — không có phép nào luôn trượt."""
    df, _ = clean_application(application_frame())

    audit = leakage_audit(df)

    assert audit["passed"].all(), audit.to_string(index=False)
    assert len(audit) == 6


def test_audit_measures_id_time_signal_instead_of_assuming_it():
    """Câu "ID có mang thông tin thời gian không" phải trả lời bằng số đo.

    Dựng ID tăng dần đi kèm rủi ro tăng dần — đúng dạng "mã cấp theo ngày nộp
    đơn" — và phép kiểm phải bắt được.
    """
    n = 1_000
    df = application_frame(n)
    # Nửa sau (ID lớn hơn) vỡ nợ gấp nhiều lần nửa đầu.
    df[TARGET_COLUMN] = np.r_[np.zeros(n // 2, dtype=int), np.ones(n // 2, dtype=int)]

    audit = leakage_audit(df)
    row = audit[audit["check"] == "id_carries_no_time_signal"].iloc[0]

    assert not row["passed"]


def test_duplicate_customers_are_removed_before_any_split():
    """Cùng một khách ở cả train lẫn test là rò rỉ theo nghĩa đen."""
    df = application_frame(50)
    doubled = pd.concat([df, df.iloc[:5]], ignore_index=True)

    out, steps = clean_application(doubled)

    assert len(out) == 50
    assert out[ID_COLUMN].duplicated().sum() == 0
    step = next(s for s in steps if s.name == "drop_duplicate_customers")
    assert step.rows_removed == 5


# ------------------------------------------------- dòng bất hợp lệ: cờ, không bỏ
def test_invalid_rows_are_flagged_not_dropped():
    """Bỏ dòng bất hợp lệ TRƯỚC khi chia tập làm tập test đẹp hơn thực tế.

    Lúc chạy thật hồ sơ bất hợp lệ vẫn cứ đến. Nếu tập test đã được dọn sạch
    chúng thì chỉ số báo cáo cao hơn năng lực thật, và không có gì trong bảng
    kết quả để lộ ra điều đó.
    """
    df = application_frame(50)
    df.loc[0, "AMT_INCOME_TOTAL"] = 0            # vi phạm: thu nhập ≤ 0
    df.loc[1, "CNT_CHILDREN"] = 9                # vi phạm: số con ≥ nhân khẩu
    df.loc[1, "CNT_FAM_MEMBERS"] = 3.0

    out, _ = clean_application(df)

    assert len(out) == 50, "không được bỏ dòng nào"
    assert INVALID_ROW_FLAG in out.columns
    assert out[INVALID_ROW_FLAG].sum() == 2
    assert out.loc[0, INVALID_ROW_FLAG] == 1
    assert out.loc[1, INVALID_ROW_FLAG] == 1


def test_invalid_flag_is_not_a_feature():
    """Cờ kỹ thuật là siêu dữ liệu về dòng, không phải thuộc tính khách hàng."""
    out, _ = clean_application(application_frame(30))

    assert INVALID_ROW_FLAG not in feature_columns(out)


# ------------------------------------------------------------- missing values
def test_employment_sentinel_becomes_missing_and_keeps_its_flag():
    """Sentinel → NaN NHƯNG phải giữ cờ.

    Ở dataset này việc thiếu dữ liệu tự nó dự báo được vỡ nợ: nhóm thiếu
    `DAYS_EMPLOYED` vỡ nợ 5,40% so với 8,66% của nhóm có. Chuyển NaN rồi vứt
    cờ đi là xoá mất tín hiệu đó — mà bảng dữ liệu vẫn trông hoàn toàn bình
    thường.
    """
    df = application_frame(20)
    df.loc[:4, "DAYS_EMPLOYED"] = 365_243

    out, _ = clean_application(df)

    assert out.loc[:4, "DAYS_EMPLOYED"].isna().all()
    assert "DAYS_EMPLOYED_MISSING" in out.columns
    assert out.loc[:4, "DAYS_EMPLOYED_MISSING"].eq(1).all()
    assert out.loc[5:, "DAYS_EMPLOYED_MISSING"].eq(0).all()


def test_placeholder_strings_become_missing():
    """'XNA'/'Unknown' không được nằm lại như một hạng mục thật."""
    df = application_frame(20)
    df.loc[:1, "CODE_GENDER"] = "XNA"
    df.loc[2:3, "NAME_FAMILY_STATUS"] = "Unknown"

    out, _ = clean_application(df)

    assert out.loc[:1, "CODE_GENDER"].isna().all()
    assert out.loc[2:3, "NAME_FAMILY_STATUS"].isna().all()


def test_cleaning_never_imputes_anything():
    """Bước làm sạch KHÔNG được điền giá trị thiếu.

    Điền thiếu cần trung vị của cả tập — một thống kê CÓ HỌC. Làm ở đây rồi
    lưu ra đĩa nghĩa là trung vị đã được tính trên cả những dòng sau này là
    tập test, và mọi chỉ số sau đó lạc quan hơn thực tế mà không có dấu hiệu gì.
    """
    df = application_frame(50)
    df.loc[:9, "EXT_SOURCE_1"] = np.nan

    out, _ = clean_application(df)

    assert out.loc[:9, "EXT_SOURCE_1"].isna().all(), "giá trị thiếu đã bị điền"


def test_cleaning_does_not_drop_high_missing_columns():
    """Bỏ cột thiếu quá ngưỡng cũng CÓ HỌC — thuộc Pipeline, không thuộc đây."""
    df = application_frame(100)
    df.loc[:89, "EXT_SOURCE_1"] = np.nan       # thiếu 90%

    out, _ = clean_application(df)

    assert "EXT_SOURCE_1" in out.columns


def test_cleaning_does_not_clip_outliers():
    """Kẹp biên học phân vị từ tập train — cũng thuộc Pipeline."""
    df = application_frame(100)
    df.loc[0, "AMT_INCOME_TOTAL"] = 117_000_000.0

    out, _ = clean_application(df)

    assert out.loc[0, "AMT_INCOME_TOTAL"] == 117_000_000.0


def test_remaining_pipeline_steps_are_declared():
    """Danh sách việc CHƯA làm phải tồn tại và nêu rõ từng bước học cái gì."""
    names = [name for name, _ in PIPELINE_STEPS_REMAINING]

    assert "SimpleImputer" in names
    assert "OutlierClipper" in names
    assert "SupervisedFeatureSelector" in names
    assert all(reason for _, reason in PIPELINE_STEPS_REMAINING)


# --------------------------------------------------------------- kiểu dữ liệu
def test_string_columns_become_category():
    df = normalize_dtypes(application_frame(20))

    assert isinstance(df["CODE_GENDER"].dtype, pd.CategoricalDtype)
    assert pd.api.types.is_numeric_dtype(df["AMT_INCOME_TOTAL"])


def test_dtype_report_separates_binary_from_numeric():
    """Cột kiểu số nhưng chỉ nhận {0,1} phải được gọi đúng tên là nhị phân."""
    df = application_frame(50)
    df["FLAG_DOCUMENT_3"] = np.random.default_rng(1).integers(0, 2, size=50)

    report = dtype_report(df).set_index("column")

    assert report.loc["FLAG_DOCUMENT_3", "semantic"] == "binary"
    assert report.loc["AMT_INCOME_TOTAL", "semantic"] == "numeric"
    assert report.loc["CODE_GENDER", "semantic"] == "categorical"


def test_dtype_issues_finds_numbers_stored_as_text():
    """Số bị đọc thành chuỗi là lỗi kiểu thật sự — phải phát hiện được."""
    df = application_frame(20)
    df["AMT_GOODS_PRICE"] = ["123456"] * 20        # số nhưng lưu dạng chuỗi

    issues = dtype_issues(df)

    assert "AMT_GOODS_PRICE" in set(issues["column"])


def test_dtype_issues_is_empty_on_well_typed_data():
    assert dtype_issues(application_frame(20)).empty


# ------------------------------------------------------------------- bureau
def test_bureau_drops_rows_with_information_from_after_the_application():
    """`DAYS_CREDIT_UPDATE > 0` = thông tin đến sau ngày nộp đơn."""
    out, steps = clean_bureau(bureau_frame())

    assert len(out) == 5
    assert (out["DAYS_CREDIT_UPDATE"] <= 0).all()
    step = next(s for s in steps if s.name == "drop_future_information")
    assert step.rows_removed == 1


def test_bureau_keeps_positive_credit_enddate():
    """`DAYS_CREDIT_ENDDATE` dương là HỢP LỆ, không được coi là dữ liệu tương lai.

    Đó là ngày kết thúc dự kiến của khoản vay còn hiệu lực — đã biết ngay lúc
    ký hợp đồng. 35,11% giá trị của cột này là số dương ở dữ liệu thật; "sửa"
    nó là mất một feature hợp lệ.
    """
    out, _ = clean_bureau(bureau_frame())

    assert (out["DAYS_CREDIT_ENDDATE"] > 0).any()
    assert "DAYS_CREDIT_ENDDATE" not in BUREAU_PAST_ONLY_COLUMNS
    assert "DAYS_CREDIT_ENDDATE" in BUREAU_FUTURE_LOOKING_OK


def test_bureau_keeps_content_duplicate_records():
    """Hai khoản vay giống hệt nhau là chuyện có thật, không phải trùng lặp.

    `SK_ID_BUREAU` khác nhau nên đó là hai bản ghi riêng biệt ở trung tâm tín
    dụng. Bỏ một cái sẽ làm `previous_loan_count` — đúng ô "Số khoản vay trước
    đây" của form — đếm thiếu.
    """
    out, _ = clean_bureau(bureau_frame())
    khach_3 = out[out["SK_ID_CURR"] == 3]

    assert len(khach_3) == 2, "hai khoản vay trùng nội dung đã bị bỏ mất một"


def test_bureau_report_does_not_break_the_leakage_gate():
    """Báo cáo bureau không có bảng kiểm toán — không được vì thế mà nổ.

    Lỗi đã sập khi chạy thật: bảng "thông tin tương lai" của bureau bị nhét
    vào cùng ô với bảng kiểm toán của application, mà hai bảng khác cột hoàn
    toàn. `passed_leakage_audit` đọc cột `passed` không tồn tại và ném
    `KeyError` — sau khi mọi phép đo đã chạy xong 2,5 phút, ngay lúc ghi file.
    """
    report = CleaningReport(
        table="bureau.csv",
        future_information=bureau_future_information(bureau_frame()),
    )

    assert report.passed_leakage_audit is True
    assert "passed" not in report.future_information.columns


def test_bureau_future_information_counts_affected_customers():
    report = bureau_future_information(bureau_frame()).set_index("column")

    assert report.loc["DAYS_CREDIT_UPDATE", "n_future_rows"] == 1
    assert report.loc["DAYS_CREDIT_UPDATE", "n_customers"] == 1
    assert report.loc["DAYS_CREDIT", "n_future_rows"] == 0


# ------------------------------------------------------------------ nhật ký
def test_every_step_is_logged_even_when_it_changes_nothing():
    """Bước không tìm thấy gì vẫn phải có mặt trong nhật ký.

    "Đã kiểm và không thấy gì" khác hẳn "không biết có kiểm hay không" — mà
    nhìn một bảng thiếu dòng thì hai thứ đó giống hệt nhau.
    """
    _, steps = clean_application(application_frame(30))
    names = [s.name for s in steps]

    assert names == [
        "drop_duplicate_customers",
        "normalize_missing",
        "normalize_dtypes",
        "flag_invalid_rows",
    ]
    assert all(s.description for s in steps)


def test_step_records_row_and_column_deltas():
    _, steps = clean_application(application_frame(30))
    flags = next(s for s in steps if s.name == "normalize_missing")

    assert flags.cols_added > 0, "bước sinh cờ phải làm tăng số cột"
    assert flags.rows_removed == 0, "bước sinh cờ không được bỏ dòng nào"
