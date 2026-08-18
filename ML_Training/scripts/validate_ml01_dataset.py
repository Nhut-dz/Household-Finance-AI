r"""Kiểm chứng tập dữ liệu ML01 TRƯỚC khi huấn luyện (F03 · redesign).

    .venv\Scripts\python.exe scripts/validate_ml01_dataset.py

Năm phép kiểm, chạy song song trên tập CŨ và tập MỚI để thấy rõ cái gì đã đổi:

    1. Cây nông      nhãn có phải một luật đơn giản không
    2. Cắt feature   ba cột có thay được cả bộ không
    3. Permutation   có feature nào một mình áp đảo không
    4. Chồng lấn     các nhóm có giao nhau thật không
    5. Ngưỡng đơn    một phép so sánh có định đoạt được nhóm không

Script này CỐ Ý chạy trước bước train. Một tập dữ liệu mà nhãn suy được bằng
một phép so sánh thì mọi con số huấn luyện sau đó đều vô nghĩa — và biết điều
đó trước rẻ hơn nhiều so với biết sau khi đã chọn xong model.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).parent))

from hfml.data.synthetic import PopulationParams, generate_households  # noqa: E402
from hfml.ml.ml01_recommendation import features as new_features       # noqa: E402
from hfml.ml.ml01_recommendation import labeler as old_labeler         # noqa: E402
from hfml.ml.ml01_recommendation.dataset import build_dataset          # noqa: E402
from hfml.rules import indicators as rule_indicators                   # noqa: E402

N_ROWS = 20_000
SEED = 42
GROUPS = ("EMERGENCY", "DEBT_FOCUS", "BUILD_BUFFER", "GROWTH")


def _split(X, y):
    return train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)


def _rule(title: str) -> None:
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


def build_old():
    """Tập CŨ: nhãn bậc thang, feature là số tuyệt đối."""
    df = generate_households(PopulationParams(n=N_ROWS), seed=SEED)
    y = old_labeler.label_frame(df)
    X = old_labeler.add_derived_features(df)[list(old_labeler.RAW_FEATURES)]
    return X, y, df


def build_new():
    """Tập MỚI: nhãn từ điểm đa chiều, feature là tỉ số, có sai số khai báo."""
    ds = build_dataset(n_rows=N_ROWS, seed=SEED)
    X = new_features.build_features(ds.observed)
    return X, ds.labels, ds


def test_shallow_tree(X, y, tag: str) -> dict:
    Xtr, Xte, ytr, yte = _split(X, y)
    out = {}
    for depth in (3, 5, 8, None):
        tree = DecisionTreeClassifier(max_depth=depth, random_state=SEED,
                                      class_weight="balanced").fit(Xtr, ytr)
        out[depth] = tree.score(Xte, yte)
        label = "không giới hạn" if depth is None else f"max_depth={depth}"
        print(f"   {tag} · {label:<18} accuracy = {out[depth]:.4f}")
    return out


def test_ablation(X, y, tag: str) -> dict:
    Xtr, Xte, ytr, yte = _split(X, y)
    model = RandomForestClassifier(n_estimators=120, random_state=SEED,
                                   class_weight="balanced", n_jobs=-1)
    full = model.fit(Xtr, ytr).score(Xte, yte)

    ranking = pd.Series(model.feature_importances_,
                        index=X.columns).sort_values(ascending=False)
    top3 = list(ranking.index[:3])
    partial = RandomForestClassifier(
        n_estimators=120, random_state=SEED, class_weight="balanced",
        n_jobs=-1).fit(Xtr[top3], ytr).score(Xte[top3], yte)

    print(f"   {tag} · toàn bộ {X.shape[1]:>2} feature   accuracy = {full:.4f}")
    print(f"   {tag} · 3 feature mạnh nhất   accuracy = {partial:.4f}"
          f"   ({', '.join(top3)})")
    print(f"   {tag} · chênh lệch            {full - partial:+.4f}")
    return {"full": full, "top3": partial, "gap": full - partial}


def test_permutation(X, y, tag: str) -> pd.Series:
    Xtr, Xte, ytr, yte = _split(X, y)
    model = RandomForestClassifier(n_estimators=120, random_state=SEED,
                                   class_weight="balanced", n_jobs=-1).fit(Xtr, ytr)
    result = permutation_importance(model, Xte, yte, n_repeats=5,
                                    random_state=SEED, scoring="f1_macro",
                                    n_jobs=-1)
    imp = pd.Series(result.importances_mean, index=X.columns)
    imp = imp.clip(lower=0.0)
    share = (imp / imp.sum()) if imp.sum() > 0 else imp

    print(f"   {tag} · 5 feature đóng góp nhiều nhất")
    for name, value in share.sort_values(ascending=False).head(5).items():
        print(f"        {name:<24} {value:6.1%}")
    print(f"   {tag} · feature cao nhất chiếm {share.max():.1%} tổng đóng góp")
    return share


def test_overlap(X, y, tag: str) -> dict:
    """Các nhóm có giao nhau không, đo trên vài trục tài chính chính."""
    axes = [c for c in ("dti", "savings_rate", "emergency_months",
                        "monthly_debt_payment", "savings_amount")
            if c in X.columns][:3]
    print(f"   {tag} · khoảng [p10, p90] theo từng nhóm")
    overlaps = {}
    for axis in axes:
        print(f"        {axis}")
        bounds = {}
        for cls in GROUPS:
            values = X.loc[y == cls, axis]
            if len(values) == 0:
                continue
            lo, hi = np.percentile(values, [10, 90])
            bounds[cls] = (lo, hi)
            print(f"          {cls:<13} [{lo:>10.3f} , {hi:>10.3f}]")
        pairs = list(bounds)
        hits = 0
        total = 0
        for i in range(len(pairs)):
            for j in range(i + 1, len(pairs)):
                a, b = bounds[pairs[i]], bounds[pairs[j]]
                total += 1
                if min(a[1], b[1]) > max(a[0], b[0]):
                    hits += 1
        overlaps[axis] = hits / total if total else 0.0
        print(f"          → {hits}/{total} cặp nhóm chồng lấn")
    return overlaps


def test_threshold(indicators: pd.DataFrame, y: pd.Series, tag: str) -> float:
    tests = {
        "net_cashflow < 0": indicators["net_cashflow"] < 0,
        "dti >= 0.40": indicators["dti"] >= 0.40,
        "emergency_months < 1": indicators["emergency_months"] < 1.0,
        "savings_rate < 0.10": indicators["savings_rate"] < 0.10,
    }
    worst = 0.0
    for name, cond in tests.items():
        if cond.sum() == 0:
            continue
        line = f"   {tag} · {name:<22}"
        for cls in GROUPS:
            p = ((y == cls) & cond).sum() / cond.sum()
            worst = max(worst, p)
            line += f" {cls[:5]}={p:.3f}"
        print(line)
    print(f"   {tag} · P(nhóm | ngưỡng) CAO NHẤT = {worst:.3f}")
    return worst


def main() -> int:
    print("Đang dựng hai tập dữ liệu…")
    X_old, y_old, df_old = build_old()
    X_new, y_new, ds = build_new()

    ind_old = rule_indicators.compute_frame(df_old)
    ind_new = rule_indicators.compute_frame(ds.truth)

    _rule("PHÂN BỐ LỚP")
    for tag, y in (("CŨ ", y_old), ("MỚI", y_new)):
        share = y.value_counts(normalize=True)
        print(f"   {tag} " + " · ".join(
            f"{k}={v:.1%}" for k, v in share.sort_index().items())
            + f"   (lớp nhỏ nhất {share.min():.1%})")

    _rule("TEST 1 — CÂY NÔNG: nhãn có phải một luật đơn giản không")
    r1_old = test_shallow_tree(X_old, y_old, "CŨ ")
    print()
    r1_new = test_shallow_tree(X_new, y_new, "MỚI")

    _rule("TEST 2 — CẮT FEATURE: ba cột có thay được cả bộ không")
    r2_old = test_ablation(X_old, y_old, "CŨ ")
    print()
    r2_new = test_ablation(X_new, y_new, "MỚI")

    _rule("TEST 3 — PERMUTATION IMPORTANCE")
    test_permutation(X_old, y_old, "CŨ ")
    print()
    share_new = test_permutation(X_new, y_new, "MỚI")

    _rule("TEST 4 — CHỒNG LẤN GIỮA CÁC NHÓM")
    test_overlap(X_old, y_old, "CŨ ")
    print()
    test_overlap(X_new, y_new, "MỚI")

    _rule("TEST 5 — MỘT NGƯỠNG ĐƠN LẺ CÓ ĐỊNH ĐOẠT NHÓM KHÔNG")
    w_old = test_threshold(ind_old, y_old, "CŨ ")
    print()
    w_new = test_threshold(ind_new, y_new, "MỚI")

    _rule("KẾT LUẬN")
    checks = [
        ("Cây sâu 5 KHÔNG đạt accuracy ≈ 1", r1_new[5] < 0.97,
         f"{r1_new[5]:.4f} (cũ {r1_old[5]:.4f})"),
        ("3 feature KHÔNG thay được cả bộ", r2_new["gap"] > 0.02,
         f"chênh {r2_new['gap']:+.4f} (cũ {r2_old['gap']:+.4f})"),
        ("Không feature nào chiếm > 40% đóng góp", share_new.max() < 0.40,
         f"cao nhất {share_new.max():.1%}"),
        ("Không ngưỡng nào cho P(nhóm) > 0,90", w_new < 0.90,
         f"cao nhất {w_new:.3f} (cũ {w_old:.3f})"),
        ("Mọi lớp chiếm ≥ 10% dân số",
         y_new.value_counts(normalize=True).min() >= 0.10,
         f"nhỏ nhất {y_new.value_counts(normalize=True).min():.1%}"),
    ]
    passed = 0
    for name, ok, detail in checks:
        print(f"   {'✅' if ok else '❌'} {name:<42} {detail}")
        passed += bool(ok)

    print(f"\n   {passed}/{len(checks)} tiêu chí đạt.")
    if passed == len(checks):
        print("   → Tập dữ liệu đủ điều kiện để huấn luyện.")
        return 0
    print("   → CHƯA đủ điều kiện. Không train trước khi xử lý các mục ❌.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
