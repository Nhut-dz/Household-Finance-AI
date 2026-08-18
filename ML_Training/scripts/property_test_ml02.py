r"""PHẦN 7 — Property test đơn điệu cho ML02 (F06).

    .venv\Scripts\python.exe scripts/property_test_ml02.py
    .venv\Scripts\python.exe scripts/property_test_ml02.py --json

Ba mệnh đề, tất cả đều đi qua ĐÚNG đường inference thật
(`orchestrator.analyze`), vì đó mới là hành vi người dùng gặp:

    1. Số khoản quá hạn tăng   0 → 1 → 2 → 3   ⟹ P(vỡ nợ) không được giảm
    2. Số tiền quá hạn tăng                     ⟹ P(vỡ nợ) không được giảm
    3. DTI tăng                                 ⟹ P(vỡ nợ) không được giảm

Vì sao quét NHIỀU hồ sơ nền chứ không phải một
------------------------------------------------
Một cặp lẻ đi sai chiều có thể chỉ là cách cây chẻ nhánh ở đúng điểm đó — với
model phi tuyến, đơn điệu từng điểm không phải tính chất bắt buộc. Thứ phải
loại trừ là sai chiều CÓ HỆ THỐNG. Vì vậy mỗi mệnh đề chạy trên một lưới hồ
sơ nền và báo cáo tỉ lệ vi phạm, chứ không kết luận từ một ca.

Ngưỡng đánh giá đặt ở tỉ lệ vi phạm, không ở từng ca: một chuỗi đơn điệu chặt
trên >= 90% hồ sơ nền là hành vi lành mạnh; dưới 50% là sai chiều có hệ thống.
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

from hfml.config import CONFIG
from hfml.pipeline.normalizer import normalize_input
from hfml.pipeline.adapters import to_ml02_frame
from hfml.pipeline.predictor import get_ml02

#: Chênh lệch nhỏ hơn mức này coi như bằng nhau, không tính là vi phạm.
SLACK = 1e-6
#: Tỉ lệ hồ sơ nền phải giữ đúng chiều để mệnh đề được coi là ĐẠT.
PASS_RATE = 0.90


def profile(income: float, *, savings: float = 150e6,
            total_debt: float = 200e6) -> dict:
    return {
        "representative_name": "Hồ sơ property test",
        "birth_year": 1988,
        "residence": "TP. Hồ Chí Minh",
        "household_size": 4,
        "children_count": 2,
        "has_dependents": False,
        "average_monthly_income": income,
        "average_monthly_expense": income * 0.45,
        "has_debt": True,
        "total_current_debt": total_debt,
        "monthly_debt_payment": 0,
        "has_savings": savings > 0,
        "savings_amount": savings,
        "assets": ["cash", "real_estate"],
        "financial_needs": ["home_loan"],
    }


def loan(*, amount: float, monthly_payment: float, asset_price: float,
         borrower_age: int, employment_years: float,
         previous_loan_count: int, late_payment_count: int,
         has_overdue_loan: bool, total_overdue_amount: float) -> dict:
    return {
        "borrower_age": borrower_age, "gender": "male",
        "marital_status": "married", "children_count": 2,
        "education_level": "higher", "occupation": "office_staff",
        "employment_years": employment_years, "loan_amount": amount,
        "loan_term_months": 240, "monthly_payment": monthly_payment,
        "asset_price": asset_price, "loan_purpose": "buy_house",
        "previous_loan_count": previous_loan_count,
        "late_payment_count": late_payment_count,
        "has_overdue_loan": has_overdue_loan,
        "total_overdue_amount": total_overdue_amount,
    }


_MODEL = None


def risk(payload: dict) -> float | None:
    """P(vỡ nợ) qua đúng adapter inference. `None` nếu hồ sơ không hợp lệ."""
    global _MODEL
    if _MODEL is None:
        _MODEL = get_ml02()
    norm = normalize_input(payload)
    if norm.loan is None:
        return None
    frame = to_ml02_frame(norm.profile, norm.loan)
    return float(_MODEL.risk_probability(frame)[0])


@dataclass
class Property:
    name: str
    levels: list[str]
    series: list[list[float]] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.series)

    @property
    def n_monotone(self) -> int:
        return sum(all(b >= a - SLACK for a, b in zip(s, s[1:]))
                   for s in self.series)

    @property
    def n_strict_down(self) -> int:
        """Chuỗi đi xuống ở MỌI bước — dấu hiệu sai chiều có hệ thống."""
        return sum(all(b < a - SLACK for a, b in zip(s, s[1:]))
                   for s in self.series)

    @property
    def rate(self) -> float:
        return self.n_monotone / self.n if self.n else 0.0

    @property
    def passed(self) -> bool:
        return self.rate >= PASS_RATE

    def mean_series(self) -> list[float]:
        frame = pd.DataFrame(self.series)
        return [float(v) for v in frame.mean()]


#: Lưới hồ sơ nền — mỗi tổ hợp là một hộ khác nhau về thu nhập, khoản vay,
#: tuổi và thâm niên. Giữ nguyên trong mọi mệnh đề để so sánh được.
GRID = list(itertools.product(
    [15e6, 30e6, 60e6],                    # thu nhập/tháng
    [600e6, 1_200e6, 2_000e6],             # tiền vay
    [(38, 10.0), (28, 3.0), (50, 20.0)],   # (tuổi, thâm niên)
))


def prop_overdue_count() -> Property:
    p = Property("1 · Số khoản quá hạn tăng → rủi ro không giảm",
                 ["0 khoản", "1 khoản", "2 khoản", "3 khoản"])
    for income, amount, (age, emp) in GRID:
        series = []
        for level in (0, 1, 2, 3):
            payload = profile(income)
            payload["loan_application"] = loan(
                amount=amount, monthly_payment=income * 0.25,
                asset_price=amount * 1.4, borrower_age=age,
                employment_years=emp,
                # 6 khoản nền để `late_payment_count` không bị adapter kẹp
                previous_loan_count=6, late_payment_count=level,
                has_overdue_loan=level > 0,
                total_overdue_amount=30e6 * level)
            value = risk(payload)
            if value is None:
                series = []
                break
            series.append(value)
        if series:
            p.series.append(series)
    return p


def prop_overdue_amount() -> Property:
    p = Property("2 · Số tiền quá hạn tăng → rủi ro không giảm",
                 ["0đ", "20tr", "80tr", "200tr"])
    for income, amount, (age, emp) in GRID:
        series = []
        for over in (0, 20e6, 80e6, 200e6):
            payload = profile(income)
            payload["loan_application"] = loan(
                amount=amount, monthly_payment=income * 0.25,
                asset_price=amount * 1.4, borrower_age=age,
                employment_years=emp, previous_loan_count=6,
                # Giữ NGUYÊN số khoản quá hạn, chỉ đổi số tiền
                late_payment_count=1, has_overdue_loan=over > 0,
                total_overdue_amount=over)
            value = risk(payload)
            if value is None:
                series = []
                break
            series.append(value)
        if series:
            p.series.append(series)
    return p


def prop_dti() -> Property:
    """DTI tăng, `credit_term_implied` GIỮ NGUYÊN.

    ⚠️ Bản đầu của mệnh đề này SAI, và sai theo kiểu dễ mắc nhất: nó nâng DTI
    bằng cách tăng `monthly_payment` trong khi giữ `loan_amount`. Nhưng

        credit_term_implied = loan_amount / (monthly_payment × 12)

    nên tăng `monthly_payment` cũng KÉO `credit_term_implied` xuống — mà đó
    lại là feature mạnh nhất của model (SHAP 0,373, gấp 4 lần `dti`). Kết quả
    "rủi ro giảm khi DTI tăng" thật ra là "rủi ro giảm khi kỳ hạn ngắn lại",
    một kết luận hoàn toàn hợp lý về mặt tín dụng.

    Cách đúng: hạ THU NHẬP, giữ nguyên `monthly_payment` và `loan_amount`.
    Khi đó `dti = payment / income` tăng còn `credit_term_implied` bất động,
    nên chênh lệch xác suất quy được về đúng một biến.
    """
    p = Property("3 · DTI tăng → rủi ro không giảm (kỳ hạn giữ nguyên)",
                 ["DTI thấp", "DTI vừa", "DTI cao", "DTI rất cao"])
    for _, amount, (age, emp) in GRID:
        series = []
        # Giữ payment và amount cố định; chỉ hạ thu nhập để đẩy DTI lên.
        payment = amount / 120.0
        for income in (payment * 12, payment * 8, payment * 5, payment * 3):
            payload = profile(income)
            payload["loan_application"] = loan(
                amount=amount, monthly_payment=payment,
                asset_price=amount * 1.4, borrower_age=age,
                employment_years=emp, previous_loan_count=2,
                late_payment_count=0, has_overdue_loan=False,
                total_overdue_amount=0)
            value = risk(payload)
            if value is None:
                series = []
                break
            series.append(value)
        if series:
            p.series.append(series)
    return p


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    logging.getLogger("hfml").setLevel(logging.WARNING)

    props = [prop_overdue_count(), prop_overdue_amount(), prop_dti()]

    if args.json:
        print(json.dumps([{
            "property": p.name, "levels": p.levels, "n_profiles": p.n,
            "n_monotone": p.n_monotone, "rate": p.rate,
            "mean_series": p.mean_series(), "result": "PASS" if p.passed else "FAIL",
        } for p in props], ensure_ascii=False, indent=2))
    else:
        print("\n" + "=" * 92)
        print(f"  PHẦN 7 · PROPERTY TEST ĐƠN ĐIỆU — ML02 qua đường inference thật")
        print(f"  {len(GRID)} hồ sơ nền mỗi mệnh đề · ngưỡng đạt {PASS_RATE:.0%}")
        print("=" * 92)
        for p in props:
            mark = "✅ PASS" if p.passed else "❌ FAIL"
            print(f"\n{p.name}")
            print(f"  Giữ đúng chiều : {p.n_monotone}/{p.n} hồ sơ nền "
                  f"({p.rate:.1%})   {mark}")
            print(f"  Đi xuống mọi bước (sai chiều hệ thống): "
                  f"{p.n_strict_down}/{p.n}")
            means = p.mean_series()
            print(f"  P trung bình theo mức:")
            for lvl, val, prev in zip(p.levels, means, [None] + means[:-1]):
                delta = "" if prev is None else f"   ({val - prev:+.5f})"
                print(f"      {lvl:<12}{val:.5f}{delta}")

        n_ok = sum(p.passed for p in props)
        print("\n" + "=" * 92)
        print(f"  TỔNG: {n_ok}/{len(props)} mệnh đề ĐẠT")
        print("=" * 92)

    out = CONFIG.paths.runs / "testcases"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "property": p.name, "n_profiles": p.n, "n_monotone": p.n_monotone,
        "rate": p.rate, "result": "PASS" if p.passed else "FAIL",
        "mean_series": " → ".join(f"{v:.5f}" for v in p.mean_series()),
    } for p in props]).to_csv(out / "property_ml02.csv", index=False,
                              encoding="utf-8-sig")

    return 0 if all(p.passed for p in props) else 1


if __name__ == "__main__":
    raise SystemExit(main())
