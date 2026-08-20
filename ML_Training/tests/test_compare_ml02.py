"""Test tầng so sánh model của ML02 (task 12).

Bài test quan trọng nhất file này là nhóm bootstrap CẶP ĐÔI. Nó canh phần trả
lời cho câu hỏi trung tâm của task 12 — *chênh lệch giữa hai model là thật hay
chỉ là nhiễu* — mà task 5 đã lấy đi công cụ thông thường khi bỏ K-Fold.

Ba cách hỏng được canh, và cả ba đều KHÔNG tự lộ ra:

    · so hai model trên hai tập khác nhau  → phép so vô nghĩa
    · bootstrap không cặp đôi              → khoảng tin cậy rộng vô ích, kết
                                             luận "không phân biệt được" cho cả
                                             những chênh lệch có thật
    · xếp hạng chung hai bộ feature        → bộ deploy được không bao giờ hiện ra
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hfml.ml.evaluation.metrics import binary_confusion, binary_metrics
from hfml.ml.ml02_credit_risk.compare import (
    COMPARISON_COLUMNS,
    SELECTION_METRIC,
    adjacent_pairs_table,
    bootstrap_pr_auc,
    build_tables,
    comparison_table,
    confidence_table,
    feature_set_delta,
    leaders,
    paired_bootstrap,
    pairwise_table,
    write_comparison,
)
from hfml.ml.ml02_credit_risk.evaluate import ModelEvaluation

#: Ít lần lấy mẫu cho test chạy nhanh. Kết luận của test không phụ thuộc con
#: số này — chúng kiểm dấu và thứ tự, không kiểm hai chữ số cuối.
RESAMPLES = 120


def make(algo: str, feature_set: str, strength: float,
         truth: np.ndarray, seed: int) -> ModelEvaluation:
    """Model giả với sức mạnh định trước, chấm trên CÙNG một bộ nhãn.

    Cùng `truth` là bắt buộc: hai model chấm trên hai tập khác nhau thì phép
    so cặp đôi không định nghĩa được.
    """
    rng = np.random.default_rng(seed)
    proba = np.clip(rng.beta(2, 8, size=len(truth)) + truth * strength, 0.001, 0.999)
    return ModelEvaluation(
        algo=algo, feature_set=feature_set, y_true=truth, y_proba=proba,
        metrics=binary_metrics(truth, proba),
        confusion=binary_confusion(truth, proba))


@pytest.fixture
def truth() -> np.ndarray:
    return np.random.default_rng(0).binomial(1, 0.08, size=4_000)


@pytest.fixture
def evaluations(truth) -> list[ModelEvaluation]:
    """Bốn thuật toán × hai bộ feature, sức mạnh tăng dần đã biết trước."""
    return [
        make("decision_tree", "reduced", 0.05, truth, 1),
        make("bagging", "reduced", 0.10, truth, 2),
        make("random_forest", "reduced", 0.15, truth, 3),
        make("xgboost", "reduced", 0.25, truth, 4),
        make("decision_tree", "full", 0.15, truth, 5),
        make("bagging", "full", 0.25, truth, 6),
        make("random_forest", "full", 0.30, truth, 7),
        make("xgboost", "full", 0.45, truth, 8),
    ]


# ---------------------------------------------------------- bảng xếp hạng
def test_ranking_is_computed_within_each_feature_set(evaluations):
    """Bộ FULL và RÚT GỌN là hai bài toán triển khai khác nhau (§7.2).

    Xếp chung một bảng thì bốn model của bộ FULL chiếm hết đầu bảng và bộ
    deploy được — thứ thật sự chạy trong sản phẩm — không bao giờ hiện ra.
    """
    table = comparison_table(evaluations)

    for feature_set, group in table.groupby("feature_set"):
        assert sorted(group["rank"]) == [1, 2, 3, 4], feature_set
        assert group.loc[group["rank"] == 1, SELECTION_METRIC].iloc[0] == \
            group[SELECTION_METRIC].max()


def test_ranking_uses_pr_auc_not_accuracy(evaluations):
    """Chỉ số chọn model là PR-AUC (§7.3). Accuracy không được cầm lái."""
    table = comparison_table(evaluations)

    for _, group in table.groupby("feature_set"):
        ordered = group.sort_values("rank")
        assert ordered[SELECTION_METRIC].is_monotonic_decreasing


def test_comparison_table_is_sorted_unlike_the_evaluation_table(evaluations):
    """Đây là ranh giới giữa task 11 (đo) và task 12 (xếp hạng).

    Task 11 giữ nguyên thứ tự nạp; task 12 sắp xếp. Trộn hai việc thì phần
    đánh giá bị rút gọn thành "cái nào cao nhất".
    """
    table = comparison_table(evaluations)
    full = table[table["feature_set"] == "full"]

    assert full[SELECTION_METRIC].is_monotonic_decreasing
    assert "rank" in table.columns


def test_accuracy_is_last_among_comparison_columns():
    assert COMPARISON_COLUMNS[0] == SELECTION_METRIC
    assert COMPARISON_COLUMNS[-1] == "accuracy"


def test_leaders_returns_one_model_per_feature_set(evaluations):
    top = leaders(comparison_table(evaluations))

    assert len(top) == 2
    assert set(top["feature_set"]) == {"full", "reduced"}
    assert set(top["algo"]) == {"xgboost"}


# ------------------------------------------------------ khoảng tin cậy
def test_bootstrap_interval_brackets_the_point_estimate(evaluations):
    """Chỉ số đo được phải nằm trong khoảng của chính nó."""
    interval = bootstrap_pr_auc(evaluations[-1], n_resamples=RESAMPLES)

    assert interval.low < interval.point < interval.high
    assert interval.width > 0


def test_interval_narrows_as_the_sample_grows(truth):
    """Mẫu lớn hơn → khoảng hẹp hơn. Không thì bootstrap chẳng đo gì.

    Đây là tính chất định nghĩa của sai số lấy mẫu; thiếu nó thì con số khoảng
    tin cậy chỉ là trang trí.
    """
    nho = np.random.default_rng(1).binomial(1, 0.08, size=1_000)
    lon = np.random.default_rng(1).binomial(1, 0.08, size=20_000)

    hep = bootstrap_pr_auc(make("a", "full", 0.3, lon, 9), n_resamples=RESAMPLES)
    rong = bootstrap_pr_auc(make("a", "full", 0.3, nho, 9), n_resamples=RESAMPLES)

    assert hep.width < rong.width


def test_confidence_table_covers_every_model(evaluations):
    table = confidence_table(evaluations, n_resamples=RESAMPLES)

    assert len(table) == len(evaluations)
    assert (table["ci_low"] < table["pr_auc"]).all()
    assert (table["pr_auc"] < table["ci_high"]).all()


# --------------------------------------------------- bootstrap cặp đôi
def test_paired_bootstrap_detects_a_real_gap(truth):
    """Chênh lệch lớn và có thật phải cho khoảng KHÔNG chứa 0."""
    manh = make("xgboost", "full", 0.45, truth, 10)
    yeu = make("decision_tree", "full", 0.05, truth, 11)

    result = paired_bootstrap(manh, yeu, n_resamples=RESAMPLES)

    assert result["diff"] > 0
    assert result["ci_low"] > 0
    assert result["distinguishable"] is True
    assert result["win_rate"] > 0.95


def test_paired_bootstrap_admits_when_two_models_are_indistinguishable(truth):
    """Hai model gần như y hệt phải cho khoảng CHỨA 0.

    Đây là kết luận phải ghi ra, không phải làm ngơ để xếp hạng cho gọn. Nếu
    phép so luôn tuyên bố phân biệt được thì nó vô dụng.
    """
    a = make("bagging", "full", 0.20, truth, 12)
    b = make("random_forest", "full", 0.20, truth, 12)   # cùng seed → trùng khít

    result = paired_bootstrap(a, b, n_resamples=RESAMPLES)

    assert result["ci_low"] <= 0 <= result["ci_high"]
    assert result["distinguishable"] is False


def test_paired_bootstrap_is_narrower_than_comparing_two_intervals(truth):
    """Cặp đôi phải CHẶT hơn so hai khoảng rời — đó là lý do nó tồn tại.

    Hai model chấm trên cùng 46.127 hồ sơ, nên phần dao động do tập validation
    tác động lên cả hai và tự triệt tiêu khi lấy hiệu. So hai khoảng rời nhau
    là phép so quá bảo thủ: nó kết luận "không phân biệt được" cho cả những
    chênh lệch có thật.
    """
    a = make("xgboost", "full", 0.30, truth, 13)
    b = make("bagging", "full", 0.26, truth, 14)

    cap_doi = paired_bootstrap(a, b, n_resamples=RESAMPLES)
    khoang_a = bootstrap_pr_auc(a, n_resamples=RESAMPLES)
    khoang_b = bootstrap_pr_auc(b, n_resamples=RESAMPLES)

    rong_roi_rac = (khoang_a.high - khoang_a.low) + (khoang_b.high - khoang_b.low)
    assert (cap_doi["ci_high"] - cap_doi["ci_low"]) < rong_roi_rac


def test_paired_bootstrap_refuses_models_scored_on_different_sets():
    """So hai con số đo trên hai tập khác nhau là phép so vô nghĩa.

    Và nó KHÔNG tự lộ ra ở đâu cả — hàm vẫn trả về một con số trông bình thường.
    """
    a = make("a", "full", 0.3, np.random.default_rng(1).binomial(1, 0.08, 1_000), 1)
    b = make("b", "full", 0.3, np.random.default_rng(2).binomial(1, 0.08, 1_000), 2)

    with pytest.raises(ValueError, match="cùng tập validation"):
        paired_bootstrap(a, b, n_resamples=RESAMPLES)


def test_paired_bootstrap_uses_the_same_resamples_for_both_models(truth):
    """Cùng seed → cùng bộ chỉ số lấy mẫu → kết quả tái lập được."""
    a = make("a", "full", 0.30, truth, 15)
    b = make("b", "full", 0.20, truth, 16)

    x = paired_bootstrap(a, b, n_resamples=RESAMPLES, seed=42)
    y = paired_bootstrap(a, b, n_resamples=RESAMPLES, seed=42)

    assert x["ci_low"] == pytest.approx(y["ci_low"])
    assert x["ci_high"] == pytest.approx(y["ci_high"])


def test_difference_flips_sign_when_arguments_swap(truth):
    a = make("a", "full", 0.30, truth, 17)
    b = make("b", "full", 0.15, truth, 18)

    xuoi = paired_bootstrap(a, b, n_resamples=RESAMPLES)
    nguoc = paired_bootstrap(b, a, n_resamples=RESAMPLES)

    assert xuoi["diff"] == pytest.approx(-nguoc["diff"])
    assert xuoi["distinguishable"] == nguoc["distinguishable"]


# ------------------------------------------------------- các cặp cụ thể
def test_pairwise_compares_the_leader_against_everyone_else(evaluations):
    table = pairwise_table(evaluations, "full", n_resamples=RESAMPLES)

    assert len(table) == 3
    assert set(table["model_a"]) == {"ml02_xgboost_full"}


def test_adjacent_pairs_cover_the_narrowest_gaps(evaluations):
    """Bảng "dẫn đầu vs phần còn lại" không nói gì về hạng 2 so với hạng 3.

    Mà đó lại là chỗ khoảng cách hẹp nhất và dễ kết luận sai nhất — đúng câu
    task 9 để ngỏ về Random Forest vs Bagging.
    """
    table = adjacent_pairs_table(evaluations, "full", n_resamples=RESAMPLES)

    assert len(table) == 3
    assert list(table["rank_a"]) == [1, 2, 3]
    assert list(table["rank_b"]) == [2, 3, 4]


# --------------------------------------------- Full vs Rút gọn (§7.2)
def test_feature_set_delta_measures_the_cost_of_being_deployable(evaluations):
    """`gap` = cái giá của việc form không thu được `EXT_SOURCE_1/2/3`."""
    table = feature_set_delta(evaluations, n_resamples=RESAMPLES)

    assert len(table) == 4
    assert (table["gap"] > 0).all(), "bộ full phải mạnh hơn bộ rút gọn"
    assert (table["pr_auc_full"] > table["pr_auc_reduced"]).all()


def test_feature_set_delta_reports_uncertainty_not_just_a_number(evaluations):
    """Khoảng chênh cũng phải kèm khoảng tin cậy, như mọi so sánh khác."""
    table = feature_set_delta(evaluations, n_resamples=RESAMPLES)

    assert {"ci_low", "ci_high", "distinguishable"} <= set(table.columns)
    assert (table["ci_low"] < table["gap"]).all()
    assert (table["gap"] < table["ci_high"]).all()


def test_delta_skips_algorithms_missing_one_feature_set(truth):
    """Thiếu một bộ thì bỏ qua thuật toán đó, không so với chính nó."""
    chi_co_full = [make("xgboost", "full", 0.3, truth, 19)]

    assert feature_set_delta(chi_co_full, n_resamples=RESAMPLES).empty


# ------------------------------------------------------------- tính một lần
def test_build_tables_returns_every_table_needed(evaluations):
    tables = build_tables(evaluations, n_resamples=RESAMPLES)

    assert {"comparison", "confidence_interval", "pairwise_vs_leader",
            "pairwise_adjacent", "feature_set_delta"} == set(tables)
    assert all(isinstance(t, pd.DataFrame) for t in tables.values())


def test_write_accepts_precomputed_tables_without_recomputing(
        tmp_path, monkeypatch, evaluations):
    """Truyền bảng đã tính sẵn thì `write_comparison` KHÔNG bootstrap lại.

    Bản đầu tính hai lần — một lần để in, một lần để ghi. Bootstrap 1.000 lần
    trên 8 model là phần đắt nhất của task, nên gấp đôi nó là gấp đôi thời
    gian chạy mà không được gì.
    """
    from hfml.config import CONFIG
    from hfml.ml.ml02_credit_risk import compare as compare_module

    monkeypatch.setattr(CONFIG.paths, "runs", tmp_path)
    tables = build_tables(evaluations, n_resamples=RESAMPLES)

    goi = {"n": 0}

    def khong_duoc_goi(*args, **kwargs):
        goi["n"] += 1
        raise AssertionError("write_comparison đã bootstrap lại")

    monkeypatch.setattr(compare_module, "confidence_table", khong_duoc_goi)
    written = write_comparison(evaluations, RESAMPLES, tables=tables)

    assert goi["n"] == 0
    assert written["comparison"].exists()


def test_written_metadata_says_selection_is_not_final(tmp_path, monkeypatch, evaluations):
    """Task 12 xếp hạng; chốt model là task 14, export là task 15."""
    import json

    from hfml.config import CONFIG

    monkeypatch.setattr(CONFIG.paths, "runs", tmp_path)
    written = write_comparison(
        evaluations, RESAMPLES,
        tables=build_tables(evaluations, n_resamples=RESAMPLES))
    metadata = json.loads(written["metadata"].read_text(encoding="utf-8"))

    assert metadata["final_selection_done_here"] is False
    assert metadata["test_set_touched"] is False
    assert metadata["cross_validation"] is False
    assert metadata["selection_metric"] == SELECTION_METRIC
