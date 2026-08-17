"""ML02 task 12 — So sánh model (F04 · M04 · Tuần 4).

Nối tiếp task 11. Task này XẾP HẠNG và chỉ ra model hiệu quả nhất trên
validation; **chốt model và kiểm trên test là task 14**, export là task 15.

Vấn đề trung tâm: chênh lệch có thật hay chỉ là nhiễu
------------------------------------------------------
Task 5 bỏ K-Fold Cross-Validation, nên mỗi chỉ số là MỘT điểm đo trên 46.127
hồ sơ — không có `pr_auc_std` giữa các fold để quy chiếu. Task 9 đã gặp đúng
vấn đề đó: Random Forest thua Bagging **0,0018** ở bộ full, và không có cách
nào nói khoảng chênh ấy là thật hay là nhiễu.

Bỏ trống câu hỏi này thì bảng so sánh dẫn tới kết luận sai: xếp hạng theo một
con số mà không biết con số đó dao động bao nhiêu chẳng khác gì xếp hạng theo
nhiễu.

**Bootstrap trên tập validation trả lời được, mà không cần K-Fold.** Lấy lại
mẫu 46.127 hồ sơ có hoàn lại, tính PR-AUC trên mỗi lần lấy, rồi đọc phân vị:
đó là ước lượng độ dao động của chỉ số nếu tập validation là một mẫu khác từ
cùng quần thể.

Vì sao phải là bootstrap CẶP ĐÔI khi so hai model
---------------------------------------------------
So hai khoảng tin cậy rời nhau là phép so **quá bảo thủ** và trả lời sai câu
hỏi. Hai model được chấm trên CÙNG 46.127 hồ sơ, nên phần lớn dao động là do
tập validation chứ không do model — nó tác động lên cả hai theo cùng một
hướng và tự triệt tiêu khi lấy hiệu.

`paired_bootstrap()` vì vậy dùng **cùng một bộ chỉ số lấy mẫu** cho cả hai
model rồi mới lấy hiệu. Khoảng tin cậy của HIỆU mới là thứ trả lời "model A
có thật sự hơn model B không".

Không train, không chạm test
-----------------------------
Mọi con số ở đây tính lại từ xác suất mà task 11 đã đo trên validation. Tập
test khoá tới task 14.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from hfml.config import CONFIG
from hfml.logger import get_logger
from hfml.ml.ml02_credit_risk.evaluate import ModelEvaluation

log = get_logger(__name__)

#: Chỉ số CHỌN MODEL của ML02 (§7.3). Accuracy không được cầm lái.
SELECTION_METRIC: Final[str] = "pr_auc"

#: Số lần lấy lại mẫu. 1.000 đủ cho khoảng tin cậy 95% ở mức chính xác cần
#: thiết; tăng lên 10.000 chỉ làm hai chữ số cuối ổn định hơn, mà kết luận
#: không phụ thuộc hai chữ số đó.
N_BOOTSTRAP: Final[int] = 1_000

#: Mức tin cậy của khoảng.
CONFIDENCE: Final[float] = 0.95

COMPARE_SUBDIR: Final[str] = "ml02_comparison"


# --------------------------------------------------------------------------
# Bảng so sánh
# --------------------------------------------------------------------------
#: Cột của bảng so sánh, theo thứ tự đọc. Chỉ số chọn model đứng đầu;
#: `overfit_gap` và `calibration_gap` đứng cạnh nó vì task 14 phải cân nhắc
#: cả ba — model dẫn đầu PR-AUC mà học thuộc gấp bốn lần không phải lựa chọn
#: hiển nhiên.
COMPARISON_COLUMNS: Final[tuple[str, ...]] = (
    "pr_auc", "pr_auc_lift", "roc_auc", "f1_positive",
    "recall_positive", "precision_positive", "brier", "accuracy",
)


def comparison_table(
    evaluations: list[ModelEvaluation],
    metric: str = SELECTION_METRIC,
) -> pd.DataFrame:
    """Bảng so sánh, sắp xếp giảm dần theo chỉ số chọn model.

    Khác `evaluate.metrics_table()` ở đúng một điểm: bảng này CÓ sắp xếp. Đó
    là ranh giới giữa task 11 (đo) và task 12 (xếp hạng).

    `rank` tính riêng trong từng bộ feature: bộ FULL và bộ RÚT GỌN là hai bài
    toán triển khai khác nhau (§7.2), xếp chung một bảng thì bốn model của bộ
    FULL chiếm hết đầu bảng và bộ deploy được không bao giờ hiện ra.
    """
    rows = [{
        "algo": e.algo,
        "feature_set": e.feature_set,
        **{k: e.metrics[k] for k in COMPARISON_COLUMNS if k in e.metrics},
    } for e in evaluations]

    table = pd.DataFrame(rows).sort_values(
        ["feature_set", metric], ascending=[True, False], ignore_index=True)
    table["rank"] = table.groupby("feature_set")[metric].rank(
        ascending=False, method="min").astype(int)
    return table


def leaders(table: pd.DataFrame, metric: str = SELECTION_METRIC) -> pd.DataFrame:
    """Model dẫn đầu của TỪNG bộ feature.

    "Dẫn đầu" ≠ "được chọn". Task 14 mới chốt, và phải cân nhắc thêm hiệu
    chuẩn, mức học thuộc và khả năng triển khai — những thứ một cột PR-AUC
    không nói ra.
    """
    return (table[table["rank"] == 1]
            .sort_values(metric, ascending=False, ignore_index=True))


# --------------------------------------------------------------------------
# Bootstrap — độ dao động của chỉ số
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Interval:
    """Khoảng tin cậy của một chỉ số."""

    point: float
    low: float
    high: float

    @property
    def width(self) -> float:
        return self.high - self.low


def _bootstrap_indices(n: int, n_resamples: int, seed: int) -> np.ndarray:
    """Ma trận chỉ số lấy lại mẫu, dùng CHUNG cho mọi model.

    Dùng chung là điều kiện để phép so cặp đôi có nghĩa: hai model phải nhìn
    thấy đúng cùng những hồ sơ ở mỗi lần lấy mẫu, nếu không thì hiệu của chúng
    lẫn cả phần khác nhau giữa hai mẫu.
    """
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(n_resamples, n))


def bootstrap_pr_auc(
    evaluation: ModelEvaluation,
    n_resamples: int = N_BOOTSTRAP,
    confidence: float = CONFIDENCE,
    seed: int | None = None,
) -> Interval:
    """Khoảng tin cậy của PR-AUC bằng bootstrap trên tập validation.

    Đây là thứ thay cho `pr_auc_std` mà việc bỏ K-Fold đã lấy đi. Nó KHÔNG
    tương đương: bootstrap đo dao động do **mẫu validation**, còn CV đo thêm
    dao động do **mẫu train**. Nhưng nó trả lời được đúng câu hỏi của task 12
    — hai model chênh nhau ngần này thì có phân biệt được không.
    """
    seed = CONFIG.random_seed if seed is None else seed
    indices = _bootstrap_indices(len(evaluation.y_true), n_resamples, seed)

    scores = np.array([
        average_precision_score(evaluation.y_true[row], evaluation.y_proba[row])
        for row in indices
    ])
    alpha = (1 - confidence) / 2
    return Interval(
        point=float(average_precision_score(evaluation.y_true, evaluation.y_proba)),
        low=float(np.quantile(scores, alpha)),
        high=float(np.quantile(scores, 1 - alpha)),
    )


def confidence_table(
    evaluations: list[ModelEvaluation],
    n_resamples: int = N_BOOTSTRAP,
    seed: int | None = None,
) -> pd.DataFrame:
    """Khoảng tin cậy PR-AUC của từng model."""
    rows = []
    for evaluation in evaluations:
        interval = bootstrap_pr_auc(evaluation, n_resamples, seed=seed)
        rows.append({
            "algo": evaluation.algo,
            "feature_set": evaluation.feature_set,
            "pr_auc": interval.point,
            "ci_low": interval.low,
            "ci_high": interval.high,
            "ci_width": interval.width,
        })
    return pd.DataFrame(rows)


def paired_bootstrap(
    a: ModelEvaluation,
    b: ModelEvaluation,
    n_resamples: int = N_BOOTSTRAP,
    confidence: float = CONFIDENCE,
    seed: int | None = None,
) -> dict:
    """So hai model bằng bootstrap CẶP ĐÔI trên cùng bộ mẫu.

    Trả về khoảng tin cậy của **hiệu** `PR-AUC(a) − PR-AUC(b)`, cộng tỉ lệ lần
    lấy mẫu mà `a` thắng. Khoảng chứa 0 nghĩa là chênh lệch **không phân biệt
    được với nhiễu** — và đó là kết luận phải ghi ra, không phải làm ngơ để
    xếp hạng cho gọn.

    Ném lỗi nếu hai model không chấm trên cùng tập nhãn: so hai con số đo trên
    hai tập khác nhau là phép so vô nghĩa, và nó không tự lộ ra ở đâu cả.
    """
    if not np.array_equal(a.y_true, b.y_true):
        raise ValueError(
            f"{a.slug} và {b.slug} không chấm trên cùng tập validation — "
            "không so cặp đôi được.")

    seed = CONFIG.random_seed if seed is None else seed
    indices = _bootstrap_indices(len(a.y_true), n_resamples, seed)

    differences = np.array([
        average_precision_score(a.y_true[row], a.y_proba[row])
        - average_precision_score(b.y_true[row], b.y_proba[row])
        for row in indices
    ])
    alpha = (1 - confidence) / 2
    low = float(np.quantile(differences, alpha))
    high = float(np.quantile(differences, 1 - alpha))

    return {
        "model_a": a.slug,
        "model_b": b.slug,
        "diff": float(average_precision_score(a.y_true, a.y_proba)
                      - average_precision_score(b.y_true, b.y_proba)),
        "ci_low": low,
        "ci_high": high,
        "win_rate": float((differences > 0).mean()),
        # Khoảng chứa 0 → chưa phân biệt được với nhiễu.
        "distinguishable": bool(low > 0 or high < 0),
    }


def pairwise_table(
    evaluations: list[ModelEvaluation],
    feature_set: str,
    n_resamples: int = N_BOOTSTRAP,
    seed: int | None = None,
) -> pd.DataFrame:
    """So model dẫn đầu với từng model còn lại, trong một bộ feature.

    Không so tất cả các cặp: câu hỏi của task 12 là *"model dẫn đầu có thật sự
    hơn phần còn lại không"*, và n(n−1)/2 phép so làm bảng khó đọc mà không trả
    lời thêm gì.
    """
    subset = [e for e in evaluations if e.feature_set == feature_set]
    if len(subset) < 2:
        return pd.DataFrame()

    subset.sort(key=lambda e: e.metrics[SELECTION_METRIC], reverse=True)
    top, rest = subset[0], subset[1:]

    return pd.DataFrame([
        {"feature_set": feature_set,
         **paired_bootstrap(top, other, n_resamples, seed=seed)}
        for other in rest
    ])


def adjacent_pairs_table(
    evaluations: list[ModelEvaluation],
    feature_set: str,
    n_resamples: int = N_BOOTSTRAP,
    seed: int | None = None,
) -> pd.DataFrame:
    """So từng cặp model ĐỨNG LIỀN NHAU trong bảng xếp hạng.

    Bảng `pairwise_table` chỉ so model dẫn đầu với phần còn lại, nên nó không
    nói gì về khoảng cách giữa hạng 2 và hạng 3 — mà đó lại là chỗ khoảng cách
    hẹp nhất và dễ kết luận sai nhất.

    Đây là câu task 9 để ngỏ: Random Forest thua Bagging **0,0018** ở bộ full,
    và khi đó chưa có cách nào nói khoảng chênh ấy là thật hay là nhiễu. Không
    trả lời thì báo cáo sẽ xếp hạng hai model theo một con số nhỏ hơn cả sai số
    của phép đo.
    """
    subset = [e for e in evaluations if e.feature_set == feature_set]
    subset.sort(key=lambda e: e.metrics[SELECTION_METRIC], reverse=True)

    return pd.DataFrame([
        {"feature_set": feature_set,
         "rank_a": index + 1, "rank_b": index + 2,
         **paired_bootstrap(subset[index], subset[index + 1],
                            n_resamples, seed=seed)}
        for index in range(len(subset) - 1)
    ])


# --------------------------------------------------------------------------
# Full vs Rút gọn — phân tích tính khả thi triển khai (§7.2)
# --------------------------------------------------------------------------
def feature_set_delta(
    evaluations: list[ModelEvaluation],
    n_resamples: int = N_BOOTSTRAP,
    seed: int | None = None,
) -> pd.DataFrame:
    """Cái giá của việc form không thu được `EXT_SOURCE_1/2/3`.

    Với mỗi thuật toán, so bộ FULL với bộ RÚT GỌN bằng bootstrap cặp đôi. Đây
    là nội dung của mục *"phân tích tính khả thi triển khai"* trong báo cáo:
    bộ deploy được mất bao nhiêu năng lực dự báo, và khoảng mất đó có chắc chắn
    không.
    """
    by_algo: dict[str, dict[str, ModelEvaluation]] = {}
    for evaluation in evaluations:
        by_algo.setdefault(evaluation.algo, {})[evaluation.feature_set] = evaluation

    rows = []
    for algo, sets in by_algo.items():
        if "full" not in sets or "reduced" not in sets:
            continue
        full, reduced = sets["full"], sets["reduced"]
        paired = paired_bootstrap(full, reduced, n_resamples, seed=seed)
        rows.append({
            "algo": algo,
            "pr_auc_full": full.metrics[SELECTION_METRIC],
            "pr_auc_reduced": reduced.metrics[SELECTION_METRIC],
            "gap": paired["diff"],
            "gap_relative": paired["diff"] / reduced.metrics[SELECTION_METRIC],
            "ci_low": paired["ci_low"],
            "ci_high": paired["ci_high"],
            "distinguishable": paired["distinguishable"],
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Ghi kết quả
# --------------------------------------------------------------------------
def output_dir() -> Path:
    return CONFIG.paths.runs / COMPARE_SUBDIR


def build_tables(
    evaluations: list[ModelEvaluation],
    n_resamples: int = N_BOOTSTRAP,
    seed: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Tính MỘT LẦN toàn bộ năm bảng so sánh.

    Tách khỏi `write_comparison` để chỗ gọi in ra và ghi ra từ cùng một kết
    quả. Bản đầu tính hai lần — một lần để in, một lần để ghi — và bootstrap
    1.000 lần trên 8 model là phần đắt nhất của cả task, nên gấp đôi nó là
    gấp đôi thời gian chạy mà không được gì.
    """
    feature_sets = sorted({e.feature_set for e in evaluations})
    return {
        "comparison": comparison_table(evaluations),
        "confidence_interval": confidence_table(evaluations, n_resamples, seed=seed),
        "pairwise_vs_leader": pd.concat(
            [pairwise_table(evaluations, fs, n_resamples, seed=seed)
             for fs in feature_sets], ignore_index=True),
        "pairwise_adjacent": pd.concat(
            [adjacent_pairs_table(evaluations, fs, n_resamples, seed=seed)
             for fs in feature_sets], ignore_index=True),
        "feature_set_delta": feature_set_delta(evaluations, n_resamples, seed=seed),
    }


def write_comparison(
    evaluations: list[ModelEvaluation],
    n_resamples: int = N_BOOTSTRAP,
    seed: int | None = None,
    tables: dict[str, pd.DataFrame] | None = None,
) -> dict[str, Path]:
    """Ghi năm bảng so sánh + metadata.

    Truyền `tables` đã tính sẵn để khỏi chạy lại bootstrap. Không truyền thì
    hàm tự tính — tiện khi gọi từ notebook, nhưng script chính thức luôn
    truyền vào.
    """
    out_dir = output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    tables = tables if tables is not None else build_tables(
        evaluations, n_resamples, seed=seed)
    table = tables["comparison"]

    written: dict[str, Path] = {}
    for name, frame in tables.items():
        path = out_dir / f"{name}.csv"
        frame.to_csv(path, index=False, encoding="utf-8")
        written[name] = path

    top = leaders(table)
    metadata = {
        "task": "ML02 task 12 — So sánh model",
        "evaluated_on": "validation",
        "test_set_touched": False,
        "selection_metric": SELECTION_METRIC,
        "cross_validation": False,
        "uncertainty_method": f"bootstrap {n_resamples} lần trên tập validation",
        "uncertainty_note": "Thay cho pr_auc_std mà việc bỏ K-Fold lấy đi. KHÔNG "
                            "tương đương: bootstrap đo dao động do mẫu validation, "
                            "CV đo thêm dao động do mẫu train.",
        "leaders": top.to_dict("records"),
        "final_selection_done_here": False,
        "final_selection_note": "Chốt model + kiểm trên test là task 14; export "
                                "là task 15. Model dẫn đầu PR-AUC chưa chắc là "
                                "model được chọn — còn hiệu chuẩn, mức học thuộc "
                                "và khả năng triển khai.",
    }
    path = out_dir / "comparison_metadata.json"
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    written["metadata"] = path

    log.info("Đã ghi %d file so sánh → %s", len(written), out_dir)
    return written
