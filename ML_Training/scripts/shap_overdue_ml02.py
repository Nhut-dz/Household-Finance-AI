r"""PHẦN 6 — SHAP audit riêng cho nhóm feature quá hạn của ML02 (F06).

    .venv\Scripts\python.exe scripts/shap_overdue_ml02.py

Vì sao cần script này bên cạnh `importance_ml02.py` (task 13)
--------------------------------------------------------------
Task 13 báo `mean|SHAP|` trên TOÀN tập validation. Với một feature chỉ bật ở
**1,2%** số hồ sơ, thống kê đó bị 98,8% số hàng có giá trị 0 kéo về gần 0 —
kể cả khi feature ấy quyết định gần như toàn bộ dự đoán cho nhóm nó bật.

    mean|SHAP| toàn bộ  =  mean|SHAP| trong nhóm bật × tỉ lệ nhóm bật
                        ≈  0,206 × 1,2%  ≈  0,0058

Con số 0,0058 KHÔNG chứng minh model bỏ qua feature. Nó chỉ phản ánh feature
hiếm. Kết luận "model không dùng tín hiệu quá hạn" rút ra từ bảng task 13 là
một **lỗi đọc thống kê**, và script này tồn tại để không ai lặp lại nó.

Báo cáo cả ba con số cho mỗi feature:

    · mean|SHAP| toàn bộ      — so được với bảng task 13
    · mean|SHAP| trong nhóm   — mức đóng góp thật khi feature có giá trị
    · tỉ lệ hồ sơ bị đẩy TĂNG — chiều tác động, kiểm bằng dấu của SHAP
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from hfml.config import CONFIG
from hfml.ml.ml02_credit_risk.train import load_training_data
from hfml.ml.registry import load_model

#: Nhóm feature cần soi. `bureau_total_overdue` CỐ Ý không có mặt: nó là cột
#: TIỀN TUYỆT ĐỐI (`BUREAU_TOTAL_OVERDUE`), bị §2.1 cấm khỏi mọi feature set
#: và chỉ dùng làm tử số cho `bureau_overdue_income_ratio`. Không phải bỏ sót.
OVERDUE_FEATURES = (
    "bureau_overdue_loan_count",
    "bureau_has_overdue",
    "bureau_overdue_loan_share",
    "bureau_overdue_income_ratio",
)

SLUG = "ml02_xgboost_reduced_vfinal"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slug", default=SLUG)
    parser.add_argument("--sample", type=int, default=None,
                        help="giới hạn số dòng validation cho SHAP")
    args = parser.parse_args()
    logging.getLogger("hfml").setLevel(logging.WARNING)

    import shap

    # Nạp artifact ĐÃ EXPORT — audit đúng thứ được deploy, không phải một bản
    # train lại trong bộ nhớ. KHÔNG đi qua `predictor.get_ml02()`: hàm đó tắt
    # tra cứu bureau cho đường inference, mà ở đây ta cần bảng bureau thật.
    model = load_model(args.slug)
    inner = model.calibrated.estimator.estimator
    features = inner.named_steps["features"]
    booster = inner.named_steps["model"]

    data = load_training_data()
    X = data.X_validation if args.sample is None \
        else data.X_validation.head(args.sample)
    Xt = features.transform(X)
    names = list(Xt.columns)

    sv = pd.DataFrame(shap.TreeExplainer(booster).shap_values(Xt),
                      columns=names, index=Xt.index)
    active = Xt["bureau_has_overdue"] > 0
    share = float(active.mean())

    print("\n" + "=" * 100)
    print(f"  PHẦN 6 · SHAP AUDIT NHÓM QUÁ HẠN — {args.slug}")
    print(f"  {len(Xt):,} hồ sơ validation · {int(active.sum()):,} hồ sơ có quá hạn "
          f"({share:.3%})")
    print("=" * 100)

    rows = []
    print(f"\n  {'feature':<32}{'|SHAP| toàn bộ':>18}{'|SHAP| nhóm bật':>18}"
          f"{'gấp':>8}{'đẩy TĂNG':>11}{'hạng':>7}")
    order = sv.abs().mean().sort_values(ascending=False)
    for f in OVERDUE_FEATURES:
        overall = float(sv[f].abs().mean())
        grp = float(sv.loc[active, f].abs().mean())
        up = float((sv.loc[active, f] > 0).mean())
        rank = int(list(order.index).index(f)) + 1
        ratio = grp / overall if overall > 0 else np.nan
        print(f"  {f:<32}{overall:>18.6f}{grp:>18.6f}{ratio:>7.1f}×"
              f"{up:>11.1%}{rank:>7}")
        rows.append({"feature": f, "shap_overall": overall,
                     "shap_active_group": grp, "ratio": ratio,
                     "share_pushed_up": up, "rank_overall": rank})

    tot_all = float(sv[list(OVERDUE_FEATURES)].abs().sum(axis=1).mean())
    tot_grp = float(sv.loc[active, list(OVERDUE_FEATURES)].abs().sum(axis=1).mean())
    top = order.index[0]
    top_grp = float(sv.loc[active, top].abs().mean())

    print(f"  {'─' * 94}")
    print(f"  {'TỔNG 4 feature quá hạn':<32}{tot_all:>18.6f}{tot_grp:>18.6f}"
          f"{tot_grp / tot_all:>7.1f}×")
    print(f"  {f'đối chiếu: `{top}` trong cùng nhóm':<32}{'':>18}{top_grp:>18.6f}")
    print(f"\n  → Trong nhóm hồ sơ CÓ quá hạn, bốn feature này đóng góp "
          f"{tot_grp / top_grp:.0%} so với")
    print(f"    feature mạnh nhất của model. Đó mới là mức sử dụng thật.")

    out = CONFIG.paths.runs / "ml02_importance"
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "shap_overdue.csv", index=False, encoding="utf-8-sig")
    print(f"\n  Đã ghi → {out / 'shap_overdue.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
