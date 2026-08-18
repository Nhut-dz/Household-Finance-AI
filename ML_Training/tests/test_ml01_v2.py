"""Test ML01 sau redesign — nhãn đa chiều, feature tỉ số (F03 · 17/08/2026).

File này canh những tính chất mà bản cũ đã VI PHẠM, và mỗi test ghi lại con số
đo được lúc phát hiện để lần sau không ai vô tình quay lại:

    P(EMERGENCY | net_cashflow < 0) = 1.000    một phép so sánh định đoạt nhóm
    P(dti ≥ 0.40 | DEBT_FOCUS)      = 1.000    nhóm chỉ là một ngưỡng
    cây sâu 5 trên 3 tỉ số          = 1.0000   nhãn không có phần dư để học
    savings_amount + has_savings    = 41,5%    một biến áp đảo bảng trọng số

Test ở đây chạy trên tập nhỏ để nhanh; các ngưỡng vì vậy đặt rộng hơn ngưỡng
báo cáo. Bản đo đầy đủ nằm ở `scripts/validate_ml01_dataset.py`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from hfml.ml.ml01_recommendation import features as feature_mod
from hfml.ml.ml01_recommendation import scoring
from hfml.ml.ml01_recommendation.dataset import (
    REPORTING_NOISE,
    apply_reporting_noise,
    build_dataset,
)
from hfml.rules import indicators as rule_indicators

GROUPS = scoring.GROUPS


@pytest.fixture(scope="module")
def dataset():
    return build_dataset(n_rows=6_000, seed=42)


def _household(**overrides) -> pd.DataFrame:
    base = {
        "average_monthly_income": 30_000_000.0,
        "average_monthly_expense": 13_000_000.0,
        "monthly_debt_payment": 3_000_000.0,
        "total_current_debt": 200_000_000.0,
        "savings_amount": 60_000_000.0,
        "household_size": 4,
        "children_count": 2,
        "has_dependents": False,
        "age": 38,
    }
    base.update(overrides)
    return pd.DataFrame([base])


# ==========================================================================
# Định nghĩa tài chính dùng chung
# ==========================================================================
class TestSingleSourceOfTruth:
    def test_rb01_rb02_va_feature_cung_mot_con_so(self):
        """Ba tầng phải cho CÙNG `savings_rate` — trước đây chúng cho ba số.

        Đo được lúc phát hiện: 72,0% số hộ nhận giá trị khác nhau giữa
        `labeler` và RB02; 39,4% số hộ dòng tiền âm bị RB02 kẹp về 0,0.
        """
        from hfml.rules.rb01_cashflow import evaluate_cashflow
        from hfml.rules.rb02_health import evaluate_financial_health

        df = _household(average_monthly_expense=20_000_000.0,
                        monthly_debt_payment=15_000_000.0)   # dòng tiền ÂM
        payload = df.iloc[0].to_dict()
        payload["assets"] = []

        rb01 = evaluate_cashflow(payload)["value"]["savings_rate"]
        rb02 = evaluate_financial_health(payload)["value"]["savings_rate"]
        frame = rule_indicators.compute_frame(df)["savings_rate"].iloc[0]

        assert rb01 < 0, "hộ này thâm hụt — con số phải mang dấu âm"
        assert rb01 == pytest.approx(rb02, abs=1e-4)
        assert rb01 == pytest.approx(frame, abs=1e-4)

    def test_compute_va_compute_frame_khop_nhau(self):
        """Bản một hộ và bản vector hoá phải trùng — hai đường, một công thức."""
        df = _household()
        one = rule_indicators.compute(df.iloc[0].to_dict())
        many = rule_indicators.compute_frame(df).iloc[0]

        for name in ("net_cashflow", "savings_rate", "dti"):
            assert getattr(one, name) == pytest.approx(many[name], abs=1e-9), name


# ==========================================================================
# Thiết kế nhãn
# ==========================================================================
class TestScoringDesign:
    def test_khong_tin_hieu_nao_ap_dao_trong_mot_diem(self):
        """Trọng số lớn nhất trong mỗi nhóm phải dưới 0,35 tổng trọng số.

        Vượt mức đó thì tín hiệu ấy một mình định đoạt nhóm, và ta quay lại
        đúng bài toán bậc thang cũ.
        """
        weights = scoring.DEFAULT_WEIGHTS
        for name, block in (("emergency", weights.emergency),
                            ("debt", weights.debt),
                            ("buffer", weights.buffer),
                            ("growth", weights.growth)):
            total = sum(abs(w) for w in block.values())
            largest = max(abs(w) for w in block.values())
            assert largest / total < 0.35, f"{name}: {largest / total:.3f}"

    def test_moi_diem_dung_it_nhat_bon_tin_hieu(self):
        weights = scoring.DEFAULT_WEIGHTS
        for block in (weights.emergency, weights.debt,
                      weights.buffer, weights.growth):
            assert len(block) >= 4

    def test_cung_dti_khac_nhan(self):
        """Hai hộ cùng DTI = 45% phải rơi vào hai nhóm khác nhau.

        Đây là yêu cầu nghiệp vụ trực tiếp: DTI một mình không được quyết định
        nhóm. Khác biệt nằm ở đệm dự phòng và dòng tiền.
        """
        day = _household(monthly_debt_payment=13_500_000.0,
                         savings_amount=120_000_000.0)          # đệm dày
        mong = _household(monthly_debt_payment=13_500_000.0,
                          savings_amount=4_000_000.0,
                          average_monthly_expense=20_000_000.0)  # đệm mỏng

        for frame in (day, mong):
            dti = rule_indicators.compute_frame(frame)["dti"].iloc[0]
            assert dti == pytest.approx(0.45, abs=0.01)

        assert scoring.label_frame(day).iloc[0] != scoring.label_frame(mong).iloc[0]

    def test_nhan_la_argmax_cua_diem(self):
        df = _household()
        scores = scoring.compute_scores(df)
        assert (scoring.label_from_scores(scores).iloc[0]
                == GROUPS[int(scores.iloc[0].values.argmax())])

    def test_diem_thay_doi_TRON_khi_dau_vao_thay_doi(self):
        """Không có bậc thang: nhích chi tiêu một chút thì điểm nhích một chút.

        Bậc thang tạo ra ranh giới mà một nhát cắt học được trọn vẹn — đó
        chính là cơ chế làm bản cũ trở thành luật if/else.
        """
        expenses = np.linspace(10_000_000, 22_000_000, 25)
        emergency = [
            scoring.compute_scores(
                _household(average_monthly_expense=e))["EMERGENCY"].iloc[0]
            for e in expenses]

        steps = np.abs(np.diff(emergency))
        assert steps.max() < 0.06, "có bước nhảy — điểm không còn trơn"
        assert emergency[-1] > emergency[0], "chi tăng thì cấp thiết phải tăng"


# ==========================================================================
# Tập dữ liệu
# ==========================================================================
class TestDataset:
    def test_moi_lop_du_ti_le(self, dataset):
        share = dataset.labels.value_counts(normalize=True)
        assert set(share.index) == set(GROUPS)
        assert share.min() >= 0.08

    def test_nhan_tinh_tren_gia_tri_that(self, dataset):
        """Nhãn phải khớp điểm số của giá trị THẬT, không phải giá trị khai.

        Đây là chỗ tạo ra sai số Bayes. Nếu nhãn tính trên giá trị đã khai thì
        quan hệ đầu vào ↔ nhãn lại thành hàm số và bài toán mất phần dư.
        """
        expected = scoring.label_from_scores(
            scoring.compute_scores(dataset.truth))
        assert (dataset.labels == expected).all()

    def test_khong_dao_nhan_ngau_nhien(self, dataset):
        """Nhiễu nằm ở BIẾN QUAN SÁT, không nằm ở nhãn.

        Đảo nhãn tạo ra những hộ mà nhãn mâu thuẫn với hoàn cảnh của họ; model
        học từ đó chỉ học được cách bắt chước một phép tung đồng xu.
        """
        again = scoring.label_from_scores(scoring.compute_scores(dataset.truth))
        assert (dataset.labels == again).all()

    def test_gia_tri_khai_lech_khoi_gia_tri_that(self, dataset):
        """Có sai số khai báo thật, và nó khác nhau giữa các cột."""
        for column in REPORTING_NOISE:
            truth = dataset.truth[column].astype(float).fillna(0.0)
            observed = dataset.observed[column].astype(float).fillna(0.0)
            positive = truth > 0
            if positive.sum() < 50:
                continue
            differs = (truth[positive] - observed[positive]).abs() > 1.0
            assert differs.mean() > 0.5, column

    def test_so_khong_van_la_so_khong(self, dataset):
        """Hộ không nợ thì khai lệch cũng vẫn là không nợ."""
        no_debt = dataset.truth["monthly_debt_payment"].fillna(0.0) == 0
        if no_debt.sum():
            assert (dataset.observed.loc[no_debt,
                                         "monthly_debt_payment"] == 0).all()

    def test_lap_lai_duoc(self):
        a = build_dataset(n_rows=800, seed=7)
        b = build_dataset(n_rows=800, seed=7)
        assert (a.labels == b.labels).all()
        pd.testing.assert_frame_equal(a.observed, b.observed)


# ==========================================================================
# Feature
# ==========================================================================
class TestFeatures:
    def test_dung_thu_tu_va_du_cot(self, dataset):
        X = feature_mod.build_features(dataset.observed)
        assert list(X.columns) == list(feature_mod.FEATURES)
        assert not X.isna().any().any()

    def test_khong_con_cot_tien_tuyet_doi(self):
        """Số tuyệt đối bị thay bằng tỉ số — nguyên nhân của bảng trọng số lệch.

        Đo được ở bản cũ: `savings_amount` + `has_savings` chiếm 41,5% trọng số,
        áp đảo thu nhập + chi tiêu (28,6%).
        """
        banned = {"savings_amount", "total_current_debt", "monthly_debt_payment",
                  "average_monthly_income", "average_monthly_expense",
                  "has_savings", "has_debt"}
        assert not (set(feature_mod.FEATURES) & banned)

    def test_moi_cot_bi_loai_deu_co_ly_do(self):
        assert feature_mod.DROPPED
        for name, reason in feature_mod.DROPPED.items():
            assert len(reason) > 30, name

    def test_bat_bien_theo_don_vi_tien(self):
        """Nhân đôi mọi khoản tiền thì các tỉ số phải giữ nguyên.

        Tính chất này là lý do dùng tỉ số: một hộ ở thành phố và một hộ ở nông
        thôn có cùng cấu trúc tài chính phải nhìn giống nhau với model.
        """
        one = _household()
        two = _household(average_monthly_income=60_000_000.0,
                         average_monthly_expense=26_000_000.0,
                         monthly_debt_payment=6_000_000.0,
                         total_current_debt=400_000_000.0,
                         savings_amount=120_000_000.0)

        a = feature_mod.build_features(one).iloc[0]
        b = feature_mod.build_features(two).iloc[0]
        for name in ("expense_ratio", "savings_rate", "dti", "payment_share",
                     "emergency_months", "savings_to_income", "debt_years"):
            assert a[name] == pytest.approx(b[name], abs=1e-6), name


# ==========================================================================
# Nhãn không còn là một luật đơn giản
# ==========================================================================
class TestNotARule:
    def test_khong_nguong_nao_dinh_doat_mot_nhom(self, dataset):
        """`P(nhóm | ngưỡng)` phải rời xa 1,0 — bản cũ đạt đúng 1.000."""
        ind = rule_indicators.compute_frame(dataset.truth)
        y = dataset.labels

        conditions = {
            "net_cashflow < 0": ind["net_cashflow"] < 0,
            "dti >= 0.40": ind["dti"] >= 0.40,
            "emergency_months < 1": ind["emergency_months"] < 1.0,
        }
        for name, cond in conditions.items():
            if cond.sum() < 30:
                continue
            worst = max(((y == cls) & cond).sum() / cond.sum() for cls in GROUPS)
            assert worst < 0.90, f"{name}: P cao nhất = {worst:.3f}"

    def test_cay_nong_khong_tai_tao_duoc_nhan(self, dataset):
        """Cây sâu 5 phải sai đáng kể — bản cũ đạt accuracy 1.0000."""
        X = feature_mod.build_features(dataset.observed)
        Xtr, Xte, ytr, yte = train_test_split(
            X, dataset.labels, test_size=0.3, random_state=42,
            stratify=dataset.labels)
        tree = DecisionTreeClassifier(max_depth=5, random_state=42).fit(Xtr, ytr)
        assert tree.score(Xte, yte) < 0.95

    def test_cac_nhom_chong_lan_that(self, dataset):
        """Nhóm phải giao nhau trên trục tài chính, không nằm tách biệt hẳn."""
        X = feature_mod.build_features(dataset.observed)
        y = dataset.labels

        bounds = {}
        for cls in GROUPS:
            values = X.loc[y == cls, "dti"]
            bounds[cls] = tuple(np.percentile(values, [10, 90]))

        pairs = [(a, b) for i, a in enumerate(GROUPS) for b in GROUPS[i + 1:]]
        overlapping = sum(
            1 for a, b in pairs
            if min(bounds[a][1], bounds[b][1]) > max(bounds[a][0], bounds[b][0]))
        assert overlapping >= len(pairs) // 2

    def test_ho_sat_bien_ton_tai(self, dataset):
        """Phải có hồ sơ mà hai nhóm gần ngang điểm — đó là vùng model nên do dự."""
        assert (dataset.margin < 0.02).mean() > 0.02


# ==========================================================================
# Model
# ==========================================================================
class TestModelContract:
    @pytest.fixture(scope="class")
    def model(self, request):
        from hfml.ml.ml01_recommendation.train_v2 import Ml01Model, build_candidates

        ds = build_dataset(n_rows=3_000, seed=11)
        X = feature_mod.build_features(ds.observed)
        estimator = build_candidates(len(X))["random_forest"]
        return Ml01Model("random_forest", estimator).fit(X, ds.labels), X

    def test_predict_la_argmax_cua_proba(self, model):
        """Không có ngưỡng cứng nào trong `predict` — chỉ là argmax."""
        fitted, X = model
        sample = X.head(200)
        expected = [fitted.classes_[i]
                    for i in fitted.predict_proba(sample).argmax(axis=1)]
        assert list(fitted.predict(sample)) == expected

    def test_proba_du_bon_lop_va_cong_bang_mot(self, model):
        fitted, X = model
        proba = fitted.predict_proba(X.head(100))
        assert proba.shape[1] == len(GROUPS)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_predict_one_tra_du_thong_tin(self, model):
        fitted, X = model
        out = fitted.predict_one(X.head(1))
        assert out["predicted_group"] in GROUPS
        assert set(out["probabilities"]) == set(GROUPS)
        assert 0.0 <= out["prediction_confidence"] <= 1.0
        assert out["margin"] >= 0.0

    def test_thu_tu_lop_khop_giua_ma_hoa_va_ten(self, model):
        """Lệch thứ tự lớp là lỗi im lặng tệ nhất: xác suất vẫn cộng thành 1,
        chỉ có điều gắn sai tên nhóm."""
        fitted, _ = model
        assert fitted.classes_ == list(GROUPS)
