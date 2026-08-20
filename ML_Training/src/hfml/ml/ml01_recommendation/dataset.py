"""ML01 — Dựng tập huấn luyện (F03 · redesign 17/08/2026).

Đường đi, và thứ tự này là toàn bộ thiết kế:

    sinh hoàn cảnh THẬT  →  chấm điểm & gán nhãn trên giá trị thật
                         →  thêm SAI SỐ KHAI BÁO
                         →  dựng feature từ giá trị đã khai

Vì sao nhãn phải tính trên giá trị thật, không phải giá trị khai
------------------------------------------------------------------
Đây là chỗ tạo ra sai số Bayes — phần biến thiên mà không mô hình nào giải
thích được, và cũng là thứ bản cũ hoàn toàn không có (một cây sâu 5 đạt
accuracy 1.0000).

Cơ chế mô phỏng đúng một hiện tượng có thật: hộ gia đình khai tài chính của
mình có sai lệch. Chi tiêu hầu như luôn bị khai thiếu vì các khoản lặt vặt
không ai nhớ hết; thu nhập bất thường hay bị bỏ sót; số dư tiết kiệm thường
được làm tròn. Hoàn cảnh THẬT quyết định hộ đó cần lời khuyên gì, nhưng model
chỉ nhìn thấy bản khai.

Hệ quả: hai hộ khai giống hệt nhau có thể thuộc hai nhóm khác nhau, vì hoàn
cảnh thật của họ khác. Đó là bài toán học máy đúng nghĩa — có phần dư không
khử được, và mọi mô hình đều bị chặn trên bởi cùng một giới hạn.

Không đảo nhãn để cân bằng lớp
--------------------------------
Nhiễu đặt ở BIẾN QUAN SÁT, không đặt ở nhãn. Đảo nhãn tạo ra những hộ mà nhãn
mâu thuẫn với chính hoàn cảnh của họ; model học từ đó chỉ học được cách bắt
chước một phép tung đồng xu. Mất cân bằng lớp là tính chất của dân số và được
xử lý ở tầng huấn luyện (`class_weight`, chia tầng), không phải bằng cách sửa
ground truth.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

from hfml.data.synthetic import PopulationParams, generate_households
from hfml.logger import get_logger
from hfml.ml.ml01_recommendation import scoring

log = get_logger(__name__)

#: Cột tiền chịu sai số khai báo, kèm độ lệch chuẩn tương đối và độ chệch.
#:
#: `bias` âm nghĩa là người khai có xu hướng khai THIẾU. Con số lấy theo hướng
#: đã biết trong khảo sát hộ gia đình: chi tiêu bị khai thiếu nhiều nhất vì
#: các khoản nhỏ không ai nhớ hết.
REPORTING_NOISE: Final[dict[str, tuple[float, float]]] = {
    #  cột                        (sigma tương đối, độ chệch)
    "average_monthly_income":     (0.08, 0.00),
    "average_monthly_expense":    (0.16, -0.07),
    "monthly_debt_payment":       (0.06, 0.00),
    "savings_amount":             (0.12, 0.00),
    "total_current_debt":         (0.10, 0.00),
}

#: Giới hạn hệ số nhiễu, chặn giá trị vô lý ở đuôi phân phối.
_NOISE_CLIP: Final[tuple[float, float]] = (0.45, 1.9)


@dataclass
class Ml01Dataset:
    """Tập dữ liệu đã dựng, kèm mọi thứ cần để kiểm chứng lại.

    `truth` và `scores` KHÔNG được đưa vào `X` — chúng là ground truth và là
    đường sinh ra nhãn. Giữ lại để đo mức chồng lấn và kiểm rò rỉ.
    """

    observed: pd.DataFrame      # hồ sơ như người dùng khai — nguồn của feature
    truth: pd.DataFrame         # hoàn cảnh thật — chỉ dùng để gán nhãn
    labels: pd.Series
    scores: pd.DataFrame
    margin: pd.Series           # khoảng cách nhóm nhất ↔ nhóm nhì

    def summary(self) -> dict:
        counts = self.labels.value_counts()
        return {
            "n_rows": len(self.labels),
            "class_counts": counts.to_dict(),
            "class_share": (counts / len(self.labels)).round(4).to_dict(),
            "min_class_share": float((counts / len(self.labels)).min()),
            "median_margin": float(self.margin.median()),
            "share_margin_below_0.02": float((self.margin < 0.02).mean()),
        }


def apply_reporting_noise(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Biến hoàn cảnh thật thành bản khai của người dùng.

    Nhân với một hệ số quanh 1 chứ không cộng một lượng tuyệt đối: sai số khai
    báo tỉ lệ với độ lớn của khoản tiền — không ai khai lệch 2 triệu trên một
    khoản 3 triệu, nhưng lệch 2 triệu trên 40 triệu là bình thường.
    """
    rng = np.random.default_rng(seed)
    out = df.copy()

    for column, (sigma, bias) in REPORTING_NOISE.items():
        if column not in out.columns:
            continue
        factor = 1.0 + bias + rng.normal(0.0, sigma, len(out))
        factor = np.clip(factor, *_NOISE_CLIP)
        values = out[column].astype(float).fillna(0.0) * factor
        # Giữ số 0 đúng là 0: hộ không có nợ thì khai lệch cũng vẫn là không nợ.
        values = values.where(out[column].astype(float).fillna(0.0) > 0, 0.0)
        out[column] = values.round(-3)      # người khai làm tròn tới nghìn

    return out


def build_dataset(
    n_rows: int = 20_000,
    seed: int = 42,
    params: PopulationParams | None = None,
) -> Ml01Dataset:
    """Sinh dân số, gán nhãn trên giá trị thật, rồi thêm sai số khai báo."""
    population = params or PopulationParams(n=n_rows)
    population.n = n_rows

    truth = generate_households(population, seed=seed)

    # Nhãn tính TRÊN GIÁ TRỊ THẬT — xem docstring đầu file.
    scores = scoring.compute_scores(truth)
    labels = scoring.label_from_scores(scores)
    margin = scoring.score_margin(scores)

    observed = apply_reporting_noise(truth, seed=seed + 1)

    share = labels.value_counts(normalize=True)
    log.info("Tập ML01: %d hồ sơ · lớp nhỏ nhất %.1f%% · %.1f%% hồ sơ nằm sát biên",
             len(labels), share.min() * 100, (margin < 0.02).mean() * 100)

    return Ml01Dataset(observed=observed, truth=truth, labels=labels,
                       scores=scores, margin=margin)
