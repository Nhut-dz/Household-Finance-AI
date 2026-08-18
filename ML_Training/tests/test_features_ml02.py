"""Test bước sinh feature của ML02 (`hfml.ml.ml02_credit_risk.features`).

Bốn nhóm bất biến được canh ở đây, mỗi nhóm ứng với một cách hệ thống hỏng mà
**không** báo lỗi:

    · rò rỉ           thống kê của tập test lọt vào phép biến đổi → chỉ số đẹp giả
    · lệch đơn vị     một cột tiền tuyệt đối lọt vào bộ rút gọn → hồ sơ VN ra
                      ngoài phân phối huấn luyện, model trả số vô nghĩa
    · NaN / inf       phép chia sinh `inf` → scaler nổ, `SimpleImputer` không
                      bắt được vì nó chỉ xử lý NaN
    · lệch train ↔ inference   tên hoặc thứ tự cột khác nhau giữa hai đường
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hfml.data.features.builder import SHARED_FEATURES
from hfml.ml.ml02_credit_risk.clean import ID_COLUMN, TARGET_COLUMN
from hfml.ml.ml02_credit_risk.features import (
    FULL_ONLY_FEATURES,
    REDUCED_EXCLUDED,
    REDUCED_FEATURES,
    BureauJoiner,
    HomeCreditFeatureBuilder,
    absolute_money_columns,
    aggregate_bureau,
    build_feature_pipeline,
    engineered_names_for,
    merge_bureau,
    split_features_and_target,
)


def application(n: int = 200) -> pd.DataFrame:
    """Khung `application_train` đã làm sạch, thu nhỏ."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        ID_COLUMN: np.arange(1, n + 1),
        TARGET_COLUMN: rng.binomial(1, 0.08, size=n),
        "AMT_INCOME_TOTAL": rng.uniform(100_000, 300_000, size=n),
        "AMT_CREDIT": rng.uniform(200_000, 900_000, size=n),
        "AMT_ANNUITY": rng.uniform(10_000, 40_000, size=n),
        "AMT_GOODS_PRICE": rng.uniform(180_000, 850_000, size=n),
        "CNT_CHILDREN": rng.integers(0, 3, size=n),
        "CNT_FAM_MEMBERS": rng.integers(3, 6, size=n).astype(float),
        "DAYS_BIRTH": -rng.integers(8_000, 22_000, size=n),
        "DAYS_EMPLOYED": -rng.integers(100, 7_000, size=n),
        "CODE_GENDER": rng.choice(["M", "F"], size=n),
        "NAME_EDUCATION_TYPE": rng.choice(
            ["Higher education", "Secondary / secondary special"], size=n),
    })


def bureau(n_customers: int = 200, n_without: int = 50) -> pd.DataFrame:
    """Bản ghi bureau, mỗi khách hai khoản vay.

    `n_without` khách cuối cùng KHÔNG có bản ghi nào — đó là 14,31% dân số
    thật, và là nhóm phải điền 0 chứ không phải NaN.
    """
    rng = np.random.default_rng(7)
    n_with = max(1, n_customers - n_without)
    ids = np.repeat(np.arange(1, n_with + 1), 2)
    return pd.DataFrame({
        ID_COLUMN: ids,
        "SK_ID_BUREAU": np.arange(len(ids)),
        "CREDIT_ACTIVE": rng.choice(["Active", "Closed"], size=len(ids)),
        "CREDIT_DAY_OVERDUE": rng.choice([0, 0, 0, 30], size=len(ids)),
        "AMT_CREDIT_SUM": rng.uniform(10_000, 200_000, size=len(ids)),
        "AMT_CREDIT_SUM_DEBT": rng.uniform(0, 100_000, size=len(ids)),
        "AMT_CREDIT_SUM_OVERDUE": rng.choice([0.0, 0.0, 0.0, 5_000.0], size=len(ids)),
        "DAYS_CREDIT": -rng.integers(100, 3_000, size=len(ids)),
    })


@pytest.fixture
def joined() -> pd.DataFrame:
    """Hồ sơ đã nối bureau — đầu vào của `HomeCreditFeatureBuilder`."""
    X, _ = split_features_and_target(application())
    return BureauJoiner(aggregates=aggregate_bureau(bureau())).transform(X)


# ------------------------------------------------------- bất biến đơn vị tiền
def test_reduced_set_contains_no_absolute_money_column():
    """Đây là ràng buộc gốc của cả ML02 (§2.1).

    `AMT_INCOME_TOTAL` trung vị Home Credit là 147.150, người dùng VN nhập
    50.000.000 — lệch ~340 lần. Một cột tiền tuyệt đối lọt vào bộ deploy là
    model trả về số vô nghĩa **mà không báo lỗi**.
    """
    assert absolute_money_columns(list(REDUCED_FEATURES)) == []


def test_every_shared_feature_is_in_the_reduced_set_unless_excluded_on_purpose():
    """Feature dùng chung của §2.1b phải có mặt đủ, TRỪ phần loại có lý do.

    Ràng buộc vẫn chặt: một feature chỉ được vắng mặt khi nó nằm trong
    `REDUCED_EXCLUDED` — tức có người đã viết ra lý do. Rơi rụng âm thầm thì
    test này bắt được.
    """
    missing = set(SHARED_FEATURES) - set(REDUCED_FEATURES)
    assert missing == set(REDUCED_EXCLUDED), (
        f"Feature dùng chung biến mất khỏi bộ rút gọn mà không khai lý do: "
        f"{missing - set(REDUCED_EXCLUDED)}")


def test_excluded_features_carry_a_reason():
    """Loại một feature khỏi bộ deploy phải kèm lý do đọc được, không để rỗng."""
    for name, reason in REDUCED_EXCLUDED.items():
        assert reason.strip(), f"{name} bị loại mà không ghi lý do"


def test_income_per_capita_ratio_is_out_of_the_deployed_set():
    """Khoá lại phần sửa B2 — xem `REDUCED_EXCLUDED` để biết vì sao.

    Mẫu số của feature này là trung vị Home Credit, còn tử số lúc inference là
    thu nhập hộ Việt Nam. Mọi hồ sơ VN đều vượt biên kẹp trên nên feature
    thành hằng số 9,00 lúc chạy thật. Đưa nó trở lại bộ deploy mà chưa có mức
    tham chiếu của quần thể Việt Nam là tái tạo đúng lỗi đó.
    """
    assert "income_per_capita_ratio" not in REDUCED_FEATURES
    assert "income_per_capita_ratio" in REDUCED_EXCLUDED


def test_reduced_and_full_only_sets_do_not_overlap():
    assert set(REDUCED_FEATURES).isdisjoint(FULL_ONLY_FEATURES)


def test_credit_goods_markup_is_full_only():
    """Nó KHÔNG phải LTV nên không được đưa vào bộ deploy.

    Home Credit cộng phí và bảo hiểm vào `AMT_CREDIT` nên tỉ lệ này luôn ≥ 1,0
    — nó đo MỨC ĐỘI GIÁ. Còn `loan_amount / asset_price` của form = 0,70 nghĩa
    là "vay 70%, tự có 30%". Gộp hai thứ là đẩy hồ sơ VN ra ngoài phân phối
    huấn luyện.
    """
    assert "credit_goods_markup" in FULL_ONLY_FEATURES
    assert "credit_goods_markup" not in REDUCED_FEATURES


# --------------------------------------------------------------- NaN và inf
def test_division_by_zero_gives_nan_not_infinity():
    """`inf` chảy xuống dưới làm scaler nổ, và `SimpleImputer` KHÔNG bắt được
    nó vì nó chỉ xử lý NaN. Trả NaN để bước impute trong Pipeline lo tiếp."""
    df = application(10)
    df.loc[0, "AMT_INCOME_TOTAL"] = 0.0
    df.loc[1, "AMT_ANNUITY"] = 0.0

    out = HomeCreditFeatureBuilder(feature_set="reduced").fit_transform(
        BureauJoiner(aggregates=aggregate_bureau(bureau(10, n_without=2)))
        .transform(df))

    numeric = out.select_dtypes("number")
    assert not np.isinf(numeric.to_numpy(dtype=float)).any(), "còn giá trị inf"
    assert pd.isna(out.loc[0, "dti"]), "chia cho thu nhập 0 phải ra NaN"
    assert pd.isna(out.loc[1, "credit_term_implied"])


def test_customers_without_bureau_get_zero_counts_but_missing_history():
    """Chưa từng vay: số khoản = 0 (biết chắc), số năm lịch sử = NaN (không tồn tại).

    Điền 0 cho `bureau_history_years` là khẳng định họ vừa mở quan hệ tín dụng
    hôm nay — một điều sai, và model sẽ học theo.
    """
    X, _ = split_features_and_target(application(200))
    out = BureauJoiner(aggregates=aggregate_bureau(bureau(200))).transform(X)

    khong_co = out["BUREAU_NO_RECORD"] == 1
    assert khong_co.sum() == 50
    assert (out.loc[khong_co, "BUREAU_LOAN_COUNT"] == 0).all()
    assert (out.loc[khong_co, "BUREAU_TOTAL_OVERDUE"] == 0).all()
    assert out.loc[khong_co, "BUREAU_HISTORY_YEARS"].isna().all()


def test_overdue_share_is_nan_when_there_are_no_loans():
    """0/0 không định nghĩa được — NaN mới là câu trả lời đúng, không phải 0."""
    X, _ = split_features_and_target(application(200))
    joined_frame = BureauJoiner(aggregates=aggregate_bureau(bureau(200))).transform(X)

    out = HomeCreditFeatureBuilder(feature_set="reduced").fit_transform(joined_frame)
    khong_co = joined_frame["BUREAU_NO_RECORD"] == 1

    assert out.loc[khong_co, "bureau_overdue_loan_share"].isna().all()


# ----------------------------------------------------------------- rò rỉ
def test_target_is_never_an_input_to_feature_building():
    """`split_features_and_target` phải tách sạch nhãn khỏi X."""
    X, y = split_features_and_target(application())

    assert TARGET_COLUMN not in X.columns
    assert len(y) == len(X)


def test_reference_income_is_learned_only_from_the_data_it_was_fit_on():
    """Trung vị thu nhập đầu người phải đến từ tập `fit`, không phải tập transform.

    Đây là bước DUY NHẤT của task 3 học từ quần thể. Fit trên toàn bộ dữ liệu
    rồi mới chia tập nghĩa là mẫu số đã "nhìn thấy" tập test — chỉ số sau đó
    lạc quan hơn thực tế mà không có dấu hiệu gì.
    """
    a = application(100)
    b = application(100)
    b["AMT_INCOME_TOTAL"] *= 10          # quần thể hoàn toàn khác

    # Bộ FULL vì phép kiểm này không liên quan bureau — chỉ soi đúng một
    # thống kê học được là trung vị thu nhập đầu người.
    fit_a = HomeCreditFeatureBuilder(feature_set="full").fit(a)
    fit_b = HomeCreditFeatureBuilder(feature_set="full").fit(b)

    assert fit_a.reference_income_per_capita_ != fit_b.reference_income_per_capita_
    # Cùng một dữ liệu vào, hai mẫu số khác nhau → hai kết quả khác nhau. Nếu
    # bằng nhau thì `fit` chẳng học gì và ràng buộc "fit trên train" là hình thức.
    assert not fit_a.transform(a)["income_per_capita_ratio"].equals(
        fit_b.transform(a)["income_per_capita_ratio"])


def test_transform_does_not_refit_the_reference():
    """Gọi `transform` nhiều lần không được đổi thống kê đã học."""
    frame = application(100)
    builder = HomeCreditFeatureBuilder(feature_set="full").fit(frame)
    before = builder.reference_income_per_capita_

    builder.transform(frame.assign(
        AMT_INCOME_TOTAL=lambda d: d["AMT_INCOME_TOTAL"] * 100))

    assert builder.reference_income_per_capita_ == before


# ------------------------------------------------- nhất quán train ↔ inference
def test_declared_names_match_the_columns_actually_produced():
    """`get_feature_names_out()` phải khớp KHÍT cột mà `transform()` sinh ra.

    Lỗi đã sập lúc chạy thật với bộ FULL: hàm khai tên tự liệt kê lại danh
    sách theo trí nhớ, sót 6 tỉ lệ dùng chung → sklearn ném "Length mismatch:
    Expected axis has 156 elements, new values have 162". Nay cả hai chỗ cùng
    đọc `engineered_names_for()`.
    """
    frame = BureauJoiner(aggregates=aggregate_bureau(bureau())).transform(
        split_features_and_target(application())[0])

    for feature_set in ("reduced", "full"):
        builder = HomeCreditFeatureBuilder(feature_set=feature_set).fit(frame)
        produced = list(builder.transform(frame).columns)
        declared = list(builder.get_feature_names_out())

        assert declared == produced, feature_set


def test_engineered_names_adapt_to_the_columns_available():
    """Không có bureau thì không khai feature bureau — khai thừa là lệch tên cột."""
    voi_bureau = engineered_names_for(["AMT_GOODS_PRICE", "BUREAU_LOAN_COUNT"])
    khong_bureau = engineered_names_for(["AMT_GOODS_PRICE"])

    assert "bureau_loan_count" in voi_bureau
    assert "bureau_loan_count" not in khong_bureau
    assert "credit_goods_markup" in khong_bureau


def test_reduced_pipeline_produces_a_stable_column_order():
    """Thứ tự cột sai là lỗi im lặng: model vẫn trả xác suất, chỉ là vô nghĩa."""
    df = application(300)
    X, y = split_features_and_target(df)
    pipeline = build_feature_pipeline(
        feature_set="reduced", bureau_aggregates=aggregate_bureau(bureau(300)))

    train = pipeline.fit_transform(X.iloc[:200], y.iloc[:200])
    test = pipeline.transform(X.iloc[200:])

    assert list(train.columns) == list(test.columns)


def test_the_same_row_gives_the_same_features_in_batch_and_one_at_a_time():
    """Inference chạy TỪNG DÒNG, train chạy cả lô — hai đường phải trùng khít.

    Nếu một bước nào đó lén dùng thống kê của chính lô đang transform (thay vì
    thống kê đã học lúc fit) thì kết quả một dòng sẽ khác kết quả trong lô, và
    model deploy sẽ chạy trên feature khác với lúc huấn luyện.
    """
    df = application(300)
    X, y = split_features_and_target(df)
    pipeline = build_feature_pipeline(
        feature_set="reduced", bureau_aggregates=aggregate_bureau(bureau(300)))
    pipeline.fit(X.iloc[:200], y.iloc[:200])

    theo_lo = pipeline.transform(X.iloc[200:])
    tung_dong = pd.concat(
        [pipeline.transform(X.iloc[[i]]) for i in range(200, 210)])

    pd.testing.assert_frame_equal(theo_lo.head(10), tung_dong, check_dtype=False)


def test_pipeline_survives_dump_and_load(tmp_path):
    """Pipeline phải nạp lại từ đĩa và cho kết quả trùng khít.

    Đây là điều kiện để "dùng lại y hệt khi inference" có nghĩa: artifact được
    `joblib.dump` cùng model và nạp lại ở service.
    """
    import joblib

    df = application(200)
    X, y = split_features_and_target(df)
    pipeline = build_feature_pipeline(
        feature_set="reduced", bureau_aggregates=aggregate_bureau(bureau(200)))
    truoc = pipeline.fit_transform(X, y)

    path = tmp_path / "pipeline.joblib"
    joblib.dump(pipeline, path)
    sau = joblib.load(path).transform(X)

    pd.testing.assert_frame_equal(truoc, sau)


def test_unseen_category_does_not_crash_inference():
    """Người dùng gửi hạng mục chưa từng thấy lúc train — KHÔNG được sập.

    Đây là yêu cầu của F06 task 1. Encoder ordinal ánh xạ giá trị lạ sang
    `UNKNOWN_CODE = -1`, khác `MISSING_CODE = -2` để cây phân biệt được.
    """
    df = application(200)
    X, y = split_features_and_target(df)
    pipeline = build_feature_pipeline(
        feature_set="full", bureau_aggregates=aggregate_bureau(bureau(200)))
    pipeline.fit(X, y)

    la = X.iloc[[0]].copy()
    la["NAME_EDUCATION_TYPE"] = "Bằng cấp chưa từng có trong tập train"

    out = pipeline.transform(la)

    assert len(out) == 1
    assert not out.isna().any().any()


# --------------------------------------------------------------- hợp đồng
def test_invalid_feature_set_is_rejected_early():
    """Sai tên bộ feature phải báo lỗi ngay, không âm thầm chạy bộ mặc định."""
    with pytest.raises(ValueError, match="feature_set"):
        build_feature_pipeline(feature_set="rut_gon")

    with pytest.raises(ValueError, match="feature_set"):
        HomeCreditFeatureBuilder(feature_set="sai").fit(application(10))


def test_reduced_set_requires_bureau_to_be_joined_first():
    """Thiếu bureau thì báo lỗi rõ ràng, không trả về bộ feature thiếu cột."""
    with pytest.raises(ValueError, match="bureau"):
        HomeCreditFeatureBuilder(feature_set="reduced").fit_transform(
            split_features_and_target(application(20))[0])


def test_merge_bureau_and_the_transformer_agree():
    """Hàm tiện dụng và transformer phải là MỘT phép biến đổi, không phải hai.

    Hai định nghĩa cho cùng một phép gộp là chỗ sớm muộn cho ra hai con số
    khác nhau về cùng một khách hàng.
    """
    X, _ = split_features_and_target(application(100))
    aggregates = aggregate_bureau(bureau(100))

    pd.testing.assert_frame_equal(
        merge_bureau(X, aggregates),
        BureauJoiner(aggregates=aggregates).transform(X))


def test_bureau_joiner_without_aggregates_is_a_no_op():
    """Không truyền bảng gộp thì trả lại nguyên hồ sơ — dùng cho ML01/tầng rule."""
    X, _ = split_features_and_target(application(50))

    pd.testing.assert_frame_equal(BureauJoiner().transform(X), X)
