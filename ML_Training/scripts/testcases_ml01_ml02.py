r"""Bộ 30 test case kiểm thử hai model ML01 và ML02 (F06 — Model Evaluation).

    .venv\Scripts\python.exe scripts/testcases_ml01_ml02.py
    .venv\Scripts\python.exe scripts/testcases_ml01_ml02.py --json

Chạy qua đúng điểm vào công khai `hfml.pipeline.orchestrator.analyze` — cùng
đường mà backend gọi — nên cái được kiểm là hồ sơ form → kết quả model, không
phải một lát cắt nội bộ nào đó.

Bốn loại kiểm chứng, vì hai model có bản chất khác nhau
--------------------------------------------------------
ML01 có ground truth thật: `g(·)` (`ml01_recommendation.labeler.label_frame`)
là hàm xác định sinh ra chính bộ nhãn mà model được huấn luyện trên đó. Nên
nhãn kỳ vọng KHÔNG hardcode — nó được tính lại từ hồ sơ, và test hỏi đúng một
câu: model có tái tạo được `g(·)` từ 17 biến THÔ hay không (bản thân `g(·)`
dùng ba tỉ lệ bị cấm khỏi `X`, xem §6.1c).

ML02 thì KHÔNG có ground truth cho một hồ sơ đơn lẻ — không ai biết hộ này có
vỡ nợ hay không. Chấm `PASS/FAIL` bằng một nhãn tự nghĩ ra là tự chấm điểm
chính mình. Vì vậy ML02 được kiểm bằng ba loại mệnh đề kiểm chứng được:

    HỢP ĐỒNG    xác suất ∈ [0,1] · tổng 2 lớp = 1 · ngưỡng 0,1303 được áp
                (KHÔNG phải 0,5) · thiếu đầu vào thì báo mã lý do đúng
    ĐƠN ĐIỆU    xấu đi một chiều (DTI ↑, nợ quá hạn, thâm niên ↓) thì xác
                suất rủi ro KHÔNG được giảm
    BẤT BIẾN    nhân toàn bộ số tiền ×1000 phải cho ĐÚNG một kết quả — đây là
                mệnh đề §2.1 của PLAN, kiểm bằng mã chứ không bằng lời hứa

Hai case biên độ (`CLEAR_LOW` / `CLEAR_HIGH`) vẫn có nhãn kỳ vọng, nhưng chỉ
ở hai đầu mút mà mọi mô hình rủi ro đều phải xếp đúng — không phải ở vùng
giữa, nơi nhãn kỳ vọng chỉ là ý kiến.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

import numpy as np

from hfml.config import CONFIG
from hfml.ml.ml01_recommendation.labeler import (
    DEFAULT_THRESHOLDS,
    RecommendationGroup,
    label_frame,
)
from hfml.pipeline.orchestrator import analyze
from hfml.rules import indicators as rule_indicators

#: Ngưỡng nghiệp vụ ML02 KHÔNG được ghim cứng ở đây.
#:
#: Bản đầu ghim `0.13029100000858307` rồi khẳng định artifact phải bằng đúng
#: số đó. Sai về mặt thiết kế test: ngưỡng là thứ task 14 suy ra từ dữ liệu,
#: nên **mỗi lần train lại hợp lệ nó sẽ đổi** — và test đỏ lên dù không có gì
#: hỏng. Ghim nó là biến một tham số được học thành một hằng số kỳ vọng.
#:
#: Điều thật sự cần kiểm là hai mệnh đề bất biến qua mọi lần train:
#:     1. `predict()` cắt tại ngưỡng ĐÃ CHỐT TRONG ARTIFACT, không phải 0,5
#:     2. ngưỡng đó khác 0,5 (tỉ lệ nền 8,07% thì 0,5 là vô nghĩa)
ML02_THRESHOLD_MUST_NOT_BE = 0.5

#: Sai số cho phép khi so hai xác suất phải bằng nhau (bất biến, tái lập).
EPS = 1e-9
#: Nới cho các mệnh đề đơn điệu: chênh nhỏ hơn mức này coi như không giảm.
MONOTONE_SLACK = 1e-6


# ==========================================================================
# Hồ sơ mẫu
# ==========================================================================
def household(
    *,
    income: float,
    expense: float,
    debt_payment: float = 0.0,
    total_debt: float = 0.0,
    savings: float = 0.0,
    birth_year: int | None = 1988,
    household_size: int = 4,
    children_count: int = 2,
    has_dependents: bool = False,
    assets: list[str] | None = None,
    needs: list[str] | None = None,
) -> dict:
    """Hồ sơ hộ gia đình — phần dùng chung cho cả ML01 lẫn ML02."""
    payload: dict[str, Any] = {
        "representative_name": "Hộ kiểm thử",
        "residence": "TP. Hồ Chí Minh",
        "household_size": household_size,
        "children_count": children_count,
        "has_dependents": has_dependents,
        "average_monthly_income": income,
        "average_monthly_expense": expense,
        "has_debt": debt_payment > 0 or total_debt > 0,
        "total_current_debt": total_debt,
        "monthly_debt_payment": debt_payment,
        "has_savings": savings > 0,
        "savings_amount": savings,
        "assets": assets if assets is not None else ["cash"],
        "financial_needs": needs if needs is not None else ["saving"],
    }
    if birth_year is not None:
        payload["birth_year"] = birth_year
    return payload


def loan(
    *,
    amount: float,
    monthly_payment: float,
    asset_price: float,
    term_months: int = 240,
    borrower_age: int = 38,
    employment_years: float = 10.0,
    previous_loan_count: int = 2,
    late_payment_count: int = 0,
    has_overdue_loan: bool = False,
    total_overdue_amount: float = 0.0,
    occupation: str = "office_staff",
    education_level: str = "higher",
    children_count: int = 2,
) -> dict:
    """Khối "Thông tin khoản vay" — đầu vào riêng của ML02."""
    return {
        "borrower_age": borrower_age,
        "gender": "male",
        "marital_status": "married",
        "children_count": children_count,
        "education_level": education_level,
        "occupation": occupation,
        "employment_years": employment_years,
        "loan_amount": amount,
        "loan_term_months": term_months,
        "monthly_payment": monthly_payment,
        "asset_price": asset_price,
        "loan_purpose": "buy_house",
        "previous_loan_count": previous_loan_count,
        "late_payment_count": late_payment_count,
        "has_overdue_loan": has_overdue_loan,
        "total_overdue_amount": total_overdue_amount,
    }


def with_loan(profile: dict, loan_block: dict) -> dict:
    """Gắn khối vay + bật `home_loan` — điều kiện để ML02 được chạy (§4.1)."""
    out = copy.deepcopy(profile)
    out["financial_needs"] = ["home_loan"]
    out["loan_application"] = loan_block
    return out


def scale_money(payload: dict, factor: float) -> dict:
    """Nhân MỌI trường tiền lên `factor` lần — dùng cho case bất biến đơn vị."""
    money_profile = ("average_monthly_income", "average_monthly_expense",
                     "total_current_debt", "monthly_debt_payment",
                     "savings_amount")
    money_loan = ("loan_amount", "monthly_payment", "asset_price",
                  "total_overdue_amount")
    out = copy.deepcopy(payload)
    for key in money_profile:
        if out.get(key) is not None:
            out[key] = out[key] * factor
    if "loan_application" in out:
        for key in money_loan:
            if out["loan_application"].get(key) is not None:
                out["loan_application"][key] = out["loan_application"][key] * factor
    return out


# ==========================================================================
# Chạy một hồ sơ
# ==========================================================================
def run(payload: dict) -> dict:
    """Chạy hồ sơ qua orchestrator, trả về `structured result` dạng dict."""
    return analyze(payload).to_dict()


def indicators_of(payload: dict) -> dict:
    """Ba chỉ số mà `g(·)` đặt ngưỡng lên — để in ra cho người đọc đối chiếu."""
    ind = rule_indicators.compute(payload)
    return {
        "savings_months": ind.emergency_months,
        "dti": ind.dti,
        "savings_rate": ind.savings_rate,
    }


def _label_frame_input(payload: dict) -> pd.DataFrame:
    return pd.DataFrame([{
        "average_monthly_income": payload["average_monthly_income"],
        "average_monthly_expense": payload["average_monthly_expense"],
        "monthly_debt_payment": payload.get("monthly_debt_payment") or 0.0,
        "savings_amount": payload.get("savings_amount") or 0.0,
    }])


def expected_ml01(payload: dict) -> str:
    """Nhãn đúng theo `g(·)` HIỆN HÀNH — tính lại từ hồ sơ, KHÔNG hardcode.

    Dùng chính `label_frame` của `labeler.py`: nếu định nghĩa 4 nhóm đổi thì
    kỳ vọng của test đổi theo, không có bản sao thứ hai bị trôi.
    """
    return str(label_frame(_label_frame_input(payload)).iloc[0])


def expected_ml01_legacy(payload: dict) -> str:
    """Nhãn theo `g(·)` CŨ — bản `savings_rate` BỎ QUÊN khoản trả nợ.

    Đây không phải định nghĩa hợp lệ, và cũng không phải kỳ vọng của test.
    Nó tồn tại để CHẨN ĐOÁN: artifact `ml01_xgboost_vfinal` được train ngày
    14/08/2026, tức TRƯỚC khi `compute_indicators` được gộp về
    `hfml.rules.indicators`. Một case FAIL mà nhãn model khớp bản cũ này thì
    nguyên nhân là artifact CŨ, không phải model đoán sai — hai kết luận dẫn
    tới hai hành động hoàn toàn khác nhau.

        cũ   `savings_rate = (thu − chi) / thu`
        mới  `savings_rate = (thu − chi − trả nợ) / thu`

    Đo trên dân số seed 42 (20.000 hộ): 26,2% số hộ đổi nhãn giữa hai bản.
    """
    df = _label_frame_input(payload)
    income = df["average_monthly_income"].astype(float)
    expense = df["average_monthly_expense"].astype(float)
    savings = df["savings_amount"].astype(float).fillna(0.0)
    payment = df["monthly_debt_payment"].astype(float).fillna(0.0)
    safe_income = income.where(income > 0)
    safe_expense = expense.where(expense > 0)

    sm = (savings / safe_expense).replace(np.nan, np.inf)
    dti = (payment / safe_income).fillna(0.0)
    sr = ((income - expense) / safe_income).fillna(0.0)   # ← bản CŨ

    t = DEFAULT_THRESHOLDS
    conditions = [
        (sr < 0) | (sm < t.emergency_savings_months),
        dti >= t.debt_focus_dti,
        (sm < t.buffer_savings_months) | (sr < t.buffer_savings_rate),
    ]
    choices = [RecommendationGroup.EMERGENCY.value,
               RecommendationGroup.DEBT_FOCUS.value,
               RecommendationGroup.BUILD_BUFFER.value]
    return str(np.select(conditions, choices,
                         default=RecommendationGroup.GROWTH.value)[0])


# ==========================================================================
# Khung test case
# ==========================================================================
@dataclass
class Result:
    id: str
    model: str
    name: str
    category: str
    inputs: dict
    expected: str
    actual: str
    passed: bool
    note: str = ""

    @property
    def status(self) -> str:
        return "PASS" if self.passed else "FAIL"


@dataclass
class Case:
    id: str
    model: str
    name: str
    category: str
    run: Callable[[], Result]


CASES: list[Case] = []


def case(id: str, model: str, name: str, category: str):
    def wrap(fn: Callable[[], Result]):
        CASES.append(Case(id, model, name, category, fn))
        return fn
    return wrap


def ml01_label_case(
    cid: str, name: str, category: str, profile: dict, note: str = "",
) -> Result:
    """Khuôn chung cho case ML01 kiểu "model có khớp `g(·)` không"."""
    ind = indicators_of(profile)
    want = expected_ml01(profile)
    data = run(profile)
    part = data["ml01"]

    if not part.get("available"):
        return Result(cid, "ml01", name, category, _ml01_inputs(profile, ind),
                      want, f"KHÔNG CÓ KẾT QUẢ [{part.get('reason_code')}]",
                      False, part.get("error", ""))

    got = part["label"]
    passed = got == want
    detail = (note or f"g(·): sm={ind['savings_months']:.2f} · "
                      f"dti={ind['dti']:.3f} · sr={ind['savings_rate']:.3f}")

    # Chẩn đoán FAIL: model khớp bản `g(·)` CŨ nghĩa là artifact chưa train lại
    # sau khi định nghĩa `savings_rate` đổi — không phải model đoán sai.
    if not passed:
        legacy = expected_ml01_legacy(profile)
        detail += (f" ⟶ NGUYÊN NHÂN: nhãn model khớp `g(·)` CŨ ({legacy}) — "
                   f"artifact train 14/08 chưa cập nhật định nghĩa savings_rate"
                   if legacy == got else
                   f" ⟶ NGUYÊN NHÂN: sai lệch sát biên (g(·) cũ cũng cho "
                   f"{legacy}, model vẫn trượt)")
    return Result(cid, "ml01", name, category, _ml01_inputs(profile, ind),
                  want, f"{got} (p={part['probability']:.3f})", passed, detail)


def _ml01_inputs(profile: dict, ind: dict) -> dict:
    return {
        "thu nhập/tháng": f"{profile['average_monthly_income']:,.0f}",
        "chi tiêu/tháng": f"{profile['average_monthly_expense']:,.0f}",
        "trả nợ/tháng": f"{(profile.get('monthly_debt_payment') or 0):,.0f}",
        "tiết kiệm": f"{(profile.get('savings_amount') or 0):,.0f}",
        "tuổi": 2026 - profile["birth_year"] if profile.get("birth_year") else None,
        "hộ/con": f"{profile['household_size']}/{profile['children_count']}",
        "savings_months": round(ind["savings_months"], 3),
        "dti": round(ind["dti"], 4),
        "savings_rate": round(ind["savings_rate"], 4),
    }


def _ml02_inputs(profile: dict, extra: dict | None = None) -> dict:
    lo = profile.get("loan_application", {})
    out = {
        "thu nhập/tháng": f"{profile['average_monthly_income']:,.0f}",
        "tiền vay": f"{lo.get('loan_amount', 0):,.0f}",
        "trả nợ vay/tháng": f"{lo.get('monthly_payment', 0):,.0f}",
        "giá tài sản": f"{lo.get('asset_price', 0):,.0f}",
        "tuổi/thâm niên": f"{lo.get('borrower_age')}/{lo.get('employment_years')}",
        "số khoản đã vay": lo.get("previous_loan_count"),
        "trả chậm / quá hạn": f"{lo.get('late_payment_count')} / "
                              f"{'có' if lo.get('has_overdue_loan') else 'không'}",
    }
    if extra:
        out.update(extra)
    return out


# ==========================================================================
# ML01 — 15 case
# ==========================================================================
@case("TC-ML01-01", "ml01", "Hộ dư dả, đệm dày → GROWTH", "Nhãn cơ bản")
def _():
    return ml01_label_case(
        "TC-ML01-01", "Hộ dư dả, đệm dày → GROWTH", "Nhãn cơ bản",
        household(income=40e6, expense=18e6, debt_payment=2e6,
                  total_debt=100e6, savings=200e6,
                  assets=["cash", "real_estate", "gold"]))


@case("TC-ML01-02", "ml01", "Dòng tiền âm → EMERGENCY", "Nhãn cơ bản")
def _():
    return ml01_label_case(
        "TC-ML01-02", "Dòng tiền âm → EMERGENCY", "Nhãn cơ bản",
        household(income=15e6, expense=16e6, debt_payment=2e6,
                  total_debt=80e6, savings=5e6))


@case("TC-ML01-03", "ml01", "Dư tiền nhưng không có đệm → EMERGENCY", "Nhãn cơ bản")
def _():
    return ml01_label_case(
        "TC-ML01-03", "Dư tiền nhưng không có đệm → EMERGENCY", "Nhãn cơ bản",
        household(income=30e6, expense=20e6, savings=10e6))


@case("TC-ML01-04", "ml01", "Gánh nặng trả nợ nặng → DEBT_FOCUS", "Nhãn cơ bản")
def _():
    return ml01_label_case(
        "TC-ML01-04", "Gánh nặng trả nợ nặng → DEBT_FOCUS", "Nhãn cơ bản",
        household(income=30e6, expense=12e6, debt_payment=13e6,
                  total_debt=600e6, savings=100e6))


@case("TC-ML01-05", "ml01", "Đệm mỏng (2 tháng) → BUILD_BUFFER", "Nhãn cơ bản")
def _():
    return ml01_label_case(
        "TC-ML01-05", "Đệm mỏng (2 tháng) → BUILD_BUFFER", "Nhãn cơ bản",
        household(income=25e6, expense=15e6, debt_payment=1e6,
                  total_debt=50e6, savings=30e6))


@case("TC-ML01-06", "ml01", "Tỉ lệ tiết kiệm 7,5% < 10% → BUILD_BUFFER", "Nhãn cơ bản")
def _():
    return ml01_label_case(
        "TC-ML01-06", "Tỉ lệ tiết kiệm 7,5% < 10% → BUILD_BUFFER", "Nhãn cơ bản",
        household(income=20e6, expense=17e6, debt_payment=1.5e6,
                  total_debt=40e6, savings=100e6))


@case("TC-ML01-07", "ml01", "Biên DTI = 0,400 đúng bằng ngưỡng", "Ca biên")
def _():
    return ml01_label_case(
        "TC-ML01-07", "Biên DTI = 0,400 đúng bằng ngưỡng", "Ca biên",
        household(income=25e6, expense=8e6, debt_payment=10e6,
                  total_debt=500e6, savings=60e6),
        note="dti = 0,400 — điều kiện là `>=` nên phải rơi vào DEBT_FOCUS")


@case("TC-ML01-08", "ml01", "Biên DTI = 0,396 ngay dưới ngưỡng", "Ca biên")
def _():
    return ml01_label_case(
        "TC-ML01-08", "Biên DTI = 0,396 ngay dưới ngưỡng", "Ca biên",
        household(income=25e6, expense=8e6, debt_payment=9.9e6,
                  total_debt=500e6, savings=60e6),
        note="Lệch 0,004 so với TC-07 — hai case này phải ra hai nhãn khác nhau")


@case("TC-ML01-09", "ml01", "Biên savings_months = 1,0 đúng bằng ngưỡng", "Ca biên")
def _():
    return ml01_label_case(
        "TC-ML01-09", "Biên savings_months = 1,0 đúng bằng ngưỡng", "Ca biên",
        household(income=20e6, expense=10e6, savings=10e6),
        note="`< 1` là sai → không EMERGENCY; nhưng `< 3` đúng → BUILD_BUFFER")


@case("TC-ML01-10", "ml01", "Biên savings_months = 0,99 ngay dưới ngưỡng", "Ca biên")
def _():
    return ml01_label_case(
        "TC-ML01-10", "Biên savings_months = 0,99 ngay dưới ngưỡng", "Ca biên",
        household(income=20e6, expense=10e6, savings=9.9e6),
        note="Lệch 100.000đ tiết kiệm so với TC-09 — đổi nhãn sang EMERGENCY")


@case("TC-ML01-11", "ml01", "Biên savings_months = 3,0 đúng bằng ngưỡng", "Ca biên")
def _():
    return ml01_label_case(
        "TC-ML01-11", "Biên savings_months = 3,0 đúng bằng ngưỡng", "Ca biên",
        household(income=20e6, expense=10e6, savings=30e6),
        note="`< 3` sai và sr = 0,5 ≥ 0,10 → GROWTH")


@case("TC-ML01-12", "ml01", "Vừa thâm hụt vừa DTI 0,45 → lấy nhãn nặng nhất",
      "Quy tắc phân xử")
def _():
    return ml01_label_case(
        "TC-ML01-12", "Vừa thâm hụt vừa DTI 0,45 → lấy nhãn nặng nhất",
        "Quy tắc phân xử",
        household(income=20e6, expense=12e6, debt_payment=9e6,
                  total_debt=400e6, savings=50e6),
        note="Thỏa cả EMERGENCY lẫn DEBT_FOCUS → `g(·)` đơn trị, chọn EMERGENCY")


@case("TC-ML01-13", "ml01", "Không nợ, không tiết kiệm, thu ≈ chi", "Ca cực trị")
def _():
    return ml01_label_case(
        "TC-ML01-13", "Không nợ, không tiết kiệm, thu ≈ chi", "Ca cực trị",
        household(income=12e6, expense=11e6, savings=0, total_debt=0,
                  household_size=3, children_count=1, assets=[]),
        note="savings_months = 0 → EMERGENCY dù dòng tiền vẫn dương")


@case("TC-ML01-14", "ml01", "Thiếu năm sinh → từ chối dự đoán, không đoán bừa",
      "Hợp đồng")
def _():
    profile = household(income=30e6, expense=15e6, savings=90e6,
                        birth_year=None)
    data = run(profile)
    part = data["ml01"]
    got_code = part.get("reason_code")
    ok = (not part.get("available")) and got_code == "missing_input"
    return Result(
        "TC-ML01-14", "ml01", "Thiếu năm sinh → từ chối dự đoán, không đoán bừa",
        "Hợp đồng",
        {"thu nhập/tháng": "30,000,000", "chi tiêu/tháng": "15,000,000",
         "tiết kiệm": "90,000,000", "birth_year": "KHÔNG KHAI"},
        "available=False · reason_code=missing_input",
        f"available={part.get('available')} · reason_code={got_code}", ok,
        "Điền tuổi mặc định thì model vẫn trả một nhóm trông hợp lý — §6.1c cấm")


@case("TC-ML01-15", "ml01", "Hợp đồng xác suất: 4 lớp, tổng = 1, nhãn = argmax",
      "Hợp đồng")
def _():
    profile = household(income=28e6, expense=16e6, debt_payment=3e6,
                        total_debt=200e6, savings=60e6)
    data = run(profile)
    part = data["ml01"]
    probs = part.get("probabilities", [])
    total = sum(p["probability"] for p in probs)
    top = max(probs, key=lambda p: p["probability"])["label"] if probs else None
    ok = (len(probs) == 4 and abs(total - 1.0) < 1e-6
          and top == part.get("label")
          and all(0.0 <= p["probability"] <= 1.0 for p in probs))
    return Result(
        "TC-ML01-15", "ml01",
        "Hợp đồng xác suất: 4 lớp, tổng = 1, nhãn = argmax", "Hợp đồng",
        _ml01_inputs(profile, indicators_of(profile)),
        "n_lớp=4 · tổng=1,000000 · nhãn=argmax · mọi p ∈ [0,1]",
        f"n_lớp={len(probs)} · tổng={total:.6f} · nhãn={part.get('label')} "
        f"· argmax={top}", ok)


# ==========================================================================
# ML02 — 15 case
# ==========================================================================
#: Hồ sơ vay dùng làm gốc cho các case đơn điệu — mọi biến thể chỉ đổi ĐÚNG
#: một trường so với nó, để chênh lệch xác suất quy được về trường đó.
BASE_PROFILE = household(income=40e6, expense=18e6, debt_payment=0,
                         total_debt=200e6, savings=150e6,
                         assets=["cash", "real_estate"])
BASE_LOAN = dict(amount=1_200_000_000, monthly_payment=10e6,
                 asset_price=1_800_000_000)


def ml02_of(data: dict) -> dict:
    return data["ml02"]


def ml02_pair_case(
    cid: str, name: str, category: str,
    worse: dict, better: dict,
    worse_desc: str, better_desc: str,
    note: str,
) -> Result:
    """Case đơn điệu: hồ sơ `worse` phải có xác suất rủi ro ≥ `better`."""
    p_worse = ml02_of(run(worse))
    p_better = ml02_of(run(better))
    if not (p_worse.get("available") and p_better.get("available")):
        return Result(cid, "ml02", name, category,
                      {"A": worse_desc, "B": better_desc},
                      "cả hai biến thể chạy được",
                      f"A={p_worse.get('reason_code')} · "
                      f"B={p_better.get('reason_code')}", False, note)
    a, b = p_worse["probability"], p_better["probability"]
    ok = a >= b - MONOTONE_SLACK
    return Result(
        cid, "ml02", name, category,
        {"A (xấu hơn)": worse_desc, "B (tốt hơn)": better_desc},
        "P(A) ≥ P(B)",
        f"P(A)={a:.4f} [{p_worse['label']}] · P(B)={b:.4f} [{p_better['label']}]"
        f" · chênh {a - b:+.4f}", ok, note)


@case("TC-ML02-01", "ml02", "Hồ sơ vay an toàn rõ rệt → LOW_RISK", "Biên độ")
def _():
    payload = with_loan(
        household(income=60e6, expense=20e6, debt_payment=0, total_debt=0,
                  savings=400e6, assets=["cash", "real_estate", "gold"]),
        loan(amount=800_000_000, monthly_payment=6e6,
             asset_price=2_500_000_000, borrower_age=42,
             employment_years=15.0, previous_loan_count=2,
             late_payment_count=0, has_overdue_loan=False))
    part = ml02_of(run(payload))
    got = part.get("label")
    return Result(
        "TC-ML02-01", "ml02", "Hồ sơ vay an toàn rõ rệt → LOW_RISK", "Biên độ",
        _ml02_inputs(payload, {"DTI khoản vay": "6/60 = 0,10",
                               "LTV": "800/2.500 = 0,32"}),
        "LOW_RISK",
        f"{got} (P(vỡ nợ)={part.get('probability', float('nan')):.4f})",
        got == "LOW_RISK",
        "Thu nhập cao · DTI 0,10 · thâm niên 15 năm · không quá hạn")


@case("TC-ML02-02", "ml02", "Hồ sơ vay rủi ro rõ rệt → HIGH_RISK", "Biên độ")
def _():
    payload = with_loan(
        household(income=12e6, expense=10e6, debt_payment=0, total_debt=350e6,
                  savings=2e6, assets=[]),
        loan(amount=1_500_000_000, monthly_payment=12e6,
             asset_price=1_600_000_000, borrower_age=23,
             employment_years=0.5, previous_loan_count=6,
             late_payment_count=4, has_overdue_loan=True,
             total_overdue_amount=80e6, occupation="laborer",
             education_level="secondary"))
    part = ml02_of(run(payload))
    got = part.get("label")
    return Result(
        "TC-ML02-02", "ml02", "Hồ sơ vay rủi ro rõ rệt → HIGH_RISK", "Biên độ",
        _ml02_inputs(payload, {"DTI khoản vay": "12/12 = 1,00",
                               "nợ quá hạn": "80.000.000"}),
        "HIGH_RISK",
        f"{got} (P(vỡ nợ)={part.get('probability', float('nan')):.4f})",
        got == "HIGH_RISK",
        "DTI 1,00 · thâm niên 6 tháng · 4 khoản trả chậm · đang có nợ quá hạn")


@case("TC-ML02-03", "ml02", "Hợp đồng xác suất: 2 lớp, tổng = 1, p ∈ [0,1]",
      "Hợp đồng")
def _():
    payload = with_loan(BASE_PROFILE, loan(**BASE_LOAN))
    part = ml02_of(run(payload))
    probs = part.get("probabilities", [])
    total = sum(p["probability"] for p in probs)
    ok = (len(probs) == 2 and abs(total - 1.0) < 1e-6
          and all(0.0 <= p["probability"] <= 1.0 for p in probs)
          and [p["label"] for p in probs] == ["LOW_RISK", "HIGH_RISK"])
    return Result(
        "TC-ML02-03", "ml02",
        "Hợp đồng xác suất: 2 lớp, tổng = 1, p ∈ [0,1]", "Hợp đồng",
        _ml02_inputs(payload),
        "2 lớp [LOW_RISK, HIGH_RISK] · tổng=1,000000 · p ∈ [0,1]",
        f"{len(probs)} lớp {[p['label'] for p in probs]} · tổng={total:.6f}", ok,
        "Cột 1 phải là HIGH_RISK — đảo cột là lỗi im lặng, model vẫn chạy")


@case("TC-ML02-04", "ml02", "Ngưỡng nghiệp vụ 0,1303 được áp (KHÔNG dùng 0,5)",
      "Hợp đồng")
def _():
    from hfml.inference.lifecycle import MANAGER
    from hfml.inference.settings import ML02

    thr = MANAGER.get(ML02).model.threshold
    payload = with_loan(
        household(income=18e6, expense=11e6, total_debt=300e6, savings=20e6),
        loan(amount=1_000_000_000, monthly_payment=9e6,
             asset_price=1_300_000_000, borrower_age=29,
             employment_years=2.0, previous_loan_count=3,
             late_payment_count=1))
    part = ml02_of(run(payload))
    p = part.get("probability")
    want = "HIGH_RISK" if p >= thr else "LOW_RISK"
    naive = "HIGH_RISK" if p >= 0.5 else "LOW_RISK"
    ok = (part.get("label") == want
          and thr != ML02_THRESHOLD_MUST_NOT_BE
          and 0.0 < thr < 0.5)
    return Result(
        "TC-ML02-04", "ml02",
        "Ngưỡng nghiệp vụ của artifact được áp (KHÔNG dùng 0,5)", "Hợp đồng",
        _ml02_inputs(payload, {"ngưỡng artifact": f"{thr:.6f}"}),
        "nhãn cắt tại ngưỡng của artifact · ngưỡng ∈ (0 · 0,5)",
        f"ngưỡng={thr:.6f} · P={p:.4f} → {part.get('label')} "
        f"(nếu dùng 0,5 thì ra {naive})", ok,
        "Không ghim giá trị ngưỡng: task 14 suy nó từ dữ liệu nên mỗi lần "
        "train lại hợp lệ nó sẽ đổi")


@case("TC-ML02-05", "ml02", "Bất biến đơn vị tiền: nhân ×1000 không đổi kết quả",
      "Bất biến")
def _():
    payload = with_loan(BASE_PROFILE, loan(**BASE_LOAN))
    a = ml02_of(run(payload))
    b = ml02_of(run(scale_money(payload, 1000.0)))
    diff = abs(a["probability"] - b["probability"])
    ok = diff < EPS and a["label"] == b["label"]
    return Result(
        "TC-ML02-05", "ml02",
        "Bất biến đơn vị tiền: nhân ×1000 không đổi kết quả", "Bất biến",
        {"gốc": "thu 40tr · vay 1,2 tỷ · trả 10tr/tháng",
         "×1000": "thu 40 tỷ · vay 1.200 tỷ · trả 10 tỷ/tháng"},
        "P(gốc) = P(×1000), cùng nhãn",
        f"P(gốc)={a['probability']:.10f} · P(×1000)={b['probability']:.10f} "
        f"· |Δ|={diff:.2e}", ok,
        "Mệnh đề §2.1: mọi feature tiền của ML02 là TỈ LỆ nên bất biến đơn vị")


@case("TC-ML02-06", "ml02", "Đơn điệu theo DTI (kỳ hạn giữ nguyên)", "Đơn điệu")
def _():
    # ⚠️ Bản đầu nâng DTI bằng cách tăng `monthly_payment` và giữ `loan_amount`
    # — nhưng `credit_term_implied = loan_amount / (monthly_payment × 12)` nên
    # phép đó đồng thời rút ngắn kỳ hạn, mà kỳ hạn là feature SHAP hạng 1
    # (0,373) còn `dti` chỉ hạng 7 (0,093). Case cũ đo lẫn hai biến.
    #
    # Nay hạ THU NHẬP thay vì tăng khoản trả: `dti = payment / income` tăng,
    # `credit_term_implied` bất động.
    worse = with_loan(
        household(income=15e6, expense=7e6, debt_payment=0, total_debt=200e6,
                  savings=150e6, assets=["cash", "real_estate"]),
        loan(**BASE_LOAN))
    better = with_loan(
        household(income=60e6, expense=27e6, debt_payment=0, total_debt=200e6,
                  savings=150e6, assets=["cash", "real_estate"]),
        loan(**BASE_LOAN))
    return ml02_pair_case(
        "TC-ML02-06", "Đơn điệu theo DTI (kỳ hạn giữ nguyên)", "Đơn điệu",
        worse, better,
        "thu 15tr, trả 10tr/tháng (DTI 0,67)",
        "thu 60tr, trả 10tr/tháng (DTI 0,17)",
        "Chỉ đổi thu nhập nên `credit_term_implied` không đổi — chênh lệch quy "
        "được về đúng `dti`")


@case("TC-ML02-07", "ml02", "Đơn điệu theo số tiền vay: 500tr → 2 tỷ", "Đơn điệu")
def _():
    worse = with_loan(BASE_PROFILE, loan(**{**BASE_LOAN, "amount": 2_000_000_000}))
    better = with_loan(BASE_PROFILE, loan(**{**BASE_LOAN, "amount": 500_000_000}))
    return ml02_pair_case(
        "TC-ML02-07", "Đơn điệu theo số tiền vay: 500tr → 2 tỷ", "Đơn điệu",
        worse, better,
        "vay 2.000.000.000 (credit_income_ratio 4,17)",
        "vay 500.000.000 (credit_income_ratio 1,04)",
        "Chỉ đổi `loan_amount` → `credit_income_ratio`: số năm thu nhập để trả hết")


@case("TC-ML02-08", "ml02", "Nợ quá hạn làm tăng rủi ro", "Đơn điệu")
def _():
    worse = with_loan(BASE_PROFILE, loan(
        **BASE_LOAN, has_overdue_loan=True, total_overdue_amount=60e6,
        late_payment_count=2))
    better = with_loan(BASE_PROFILE, loan(
        **BASE_LOAN, has_overdue_loan=False, total_overdue_amount=0,
        late_payment_count=0))
    return ml02_pair_case(
        "TC-ML02-08", "Nợ quá hạn làm tăng rủi ro", "Đơn điệu",
        worse, better,
        "có nợ quá hạn 60.000.000, 2 khoản trả chậm",
        "không nợ quá hạn, 0 khoản trả chậm",
        "Mục C của form → nhóm feature bureau, 3/5 feature mạnh nhất (task 13)")


@case("TC-ML02-09", "ml02", "Thâm niên làm việc: 0,5 năm vs 15 năm", "Đơn điệu")
def _():
    worse = with_loan(BASE_PROFILE, loan(**BASE_LOAN, employment_years=0.5))
    better = with_loan(BASE_PROFILE, loan(**BASE_LOAN, employment_years=15.0))
    return ml02_pair_case(
        "TC-ML02-09", "Thâm niên làm việc: 0,5 năm vs 15 năm", "Đơn điệu",
        worse, better, "đi làm 0,5 năm", "đi làm 15 năm",
        "`employment_years` + `employment_ratio` — 2/7 feature bộ rút gọn")


@case("TC-ML02-10", "ml02", "Không khai khoản vay → không trả xác suất vỡ nợ",
      "Hợp đồng")
def _():
    profile = household(income=30e6, expense=15e6, savings=90e6,
                        needs=["saving"])
    part = ml02_of(run(profile))
    ok = (not part.get("available")) and part.get("reason_code") == "missing_input"
    return Result(
        "TC-ML02-10", "ml02",
        "Không khai khoản vay → không trả xác suất vỡ nợ", "Hợp đồng",
        {"financial_needs": "saving (không phải home_loan)",
         "loan_application": "KHÔNG CÓ"},
        "available=False · reason_code=missing_input",
        f"available={part.get('available')} · "
        f"reason_code={part.get('reason_code')}", ok,
        "§4.1: đưa 'xác suất vỡ nợ 8%' cho người không định vay là số vô nghĩa")


@case("TC-ML02-11", "ml02", "Chưa từng vay bao giờ (bureau_no_record = 1)",
      "Ca cực trị")
def _():
    payload = with_loan(BASE_PROFILE, loan(
        **BASE_LOAN, previous_loan_count=0, late_payment_count=0,
        has_overdue_loan=False))
    part = ml02_of(run(payload))
    p = part.get("probability")
    ok = part.get("available") and p is not None and 0.0 <= p <= 1.0
    return Result(
        "TC-ML02-11", "ml02", "Chưa từng vay bao giờ (bureau_no_record = 1)",
        "Ca cực trị",
        _ml02_inputs(payload, {"previous_loan_count": 0}),
        "chạy được · p ∈ [0,1] (không rơi vào nhánh thiếu dữ liệu)",
        f"available={part.get('available')} · P={p if p is None else f'{p:.4f}'} "
        f"→ {part.get('label')}", ok,
        "Không có bản ghi bureau nghĩa là CHƯA TỪNG VAY → điền 0, không NaN")


@case("TC-ML02-12", "ml02", "Trả chậm (5) nhiều hơn số khoản vay (2) → bị kẹp",
      "Ca cực trị")
def _():
    payload = with_loan(BASE_PROFILE, loan(
        **BASE_LOAN, previous_loan_count=2, late_payment_count=5,
        has_overdue_loan=True, total_overdue_amount=30e6))
    part = ml02_of(run(payload))
    p = part.get("probability")
    ok = part.get("available") and p is not None and 0.0 <= p <= 1.0
    return Result(
        "TC-ML02-12", "ml02",
        "Trả chậm (5) nhiều hơn số khoản vay (2) → bị kẹp", "Ca cực trị",
        _ml02_inputs(payload),
        "không lỗi · p ∈ [0,1] (adapter kẹp về min(5, 2) = 2)",
        f"available={part.get('available')} · P={p if p is None else f'{p:.4f}'} "
        f"→ {part.get('label')}", ok,
        "Bureau không thể đếm số khoản quá hạn nhiều hơn tổng số khoản")


@case("TC-ML02-13", "ml02", "Thu nhập ngoại lai 900tr/tháng → không vỡ, có kẹp biên",
      "Ca cực trị")
def _():
    payload = with_loan(
        household(income=900e6, expense=50e6, total_debt=0, savings=2e9,
                  assets=["cash", "real_estate", "investment"]),
        loan(**BASE_LOAN))
    part = ml02_of(run(payload))
    p = part.get("probability")
    ok = part.get("available") and p is not None and 0.0 <= p <= 1.0
    return Result(
        "TC-ML02-13", "ml02",
        "Thu nhập ngoại lai 900tr/tháng → không vỡ, có kẹp biên", "Ca cực trị",
        _ml02_inputs(payload),
        "không crash · p ∈ [0,1]",
        f"available={part.get('available')} · P={p if p is None else f'{p:.4f}'} "
        f"→ {part.get('label')}", ok,
        "`OutlierClipper` kẹp về phân vị 99,9% của train thay vì đẩy model ra "
        "ngoài phân phối")


@case("TC-ML02-14", "ml02", "Tái lập: chạy 2 lần cùng hồ sơ → cùng xác suất",
      "Hợp đồng")
def _():
    payload = with_loan(BASE_PROFILE, loan(**BASE_LOAN))
    a = ml02_of(run(payload))
    b = ml02_of(run(copy.deepcopy(payload)))
    diff = abs(a["probability"] - b["probability"])
    ok = diff == 0.0 and a["label"] == b["label"]
    return Result(
        "TC-ML02-14", "ml02", "Tái lập: chạy 2 lần cùng hồ sơ → cùng xác suất",
        "Hợp đồng", _ml02_inputs(payload),
        "P(lần 1) = P(lần 2), chênh đúng 0",
        f"P₁={a['probability']:.12f} · P₂={b['probability']:.12f} · Δ={diff:.2e}",
        ok, "Điều kiện của F06 task 6 — inference phải xác định, seed 42")


@case("TC-ML02-15", "ml02", "Số khoản trả chậm: 0 vs 5 (cùng 6 khoản đã vay)",
      "Đơn điệu")
def _():
    worse = with_loan(BASE_PROFILE, loan(
        **BASE_LOAN, previous_loan_count=6, late_payment_count=5,
        has_overdue_loan=True, total_overdue_amount=50e6))
    better = with_loan(BASE_PROFILE, loan(
        **BASE_LOAN, previous_loan_count=6, late_payment_count=0,
        has_overdue_loan=False, total_overdue_amount=0))
    return ml02_pair_case(
        "TC-ML02-15", "Số khoản trả chậm: 0 vs 5 (cùng 6 khoản đã vay)",
        "Đơn điệu", worse, better,
        "6 khoản đã vay, 5 khoản trả chậm", "6 khoản đã vay, 0 khoản trả chậm",
        "`bureau_overdue_loan_share` đi từ 0,83 xuống 0 — cùng số khoản nền")


# ==========================================================================
# Chạy và xuất kết quả
# ==========================================================================
def execute() -> list[Result]:
    results: list[Result] = []
    for c in CASES:
        try:
            results.append(c.run())
        except Exception as exc:  # noqa: BLE001 — case lỗi là FAIL, không dừng cả bộ
            results.append(Result(c.id, c.model, c.name, c.category, {},
                                  "chạy được", f"NGOẠI LỆ: {exc}", False))
    return results


def to_frame(results: list[Result]) -> pd.DataFrame:
    return pd.DataFrame([{
        "id": r.id,
        "model": r.model.upper(),
        "loại": r.category,
        "tên case": r.name,
        "input": " · ".join(f"{k}={v}" for k, v in r.inputs.items()),
        "expected": r.expected,
        "actual": r.actual,
        "kết quả": r.status,
        "ghi chú": r.note,
    } for r in results])


def print_report(results: list[Result]) -> None:
    for model in ("ml01", "ml02"):
        subset = [r for r in results if r.model == model]
        title = ("ML01 · Financial Recommendation Group (4 nhóm)" if model == "ml01"
                 else "ML02 · Home Credit Risk (LOW_RISK / HIGH_RISK)")
        print("\n" + "=" * 100)
        print(f"  {title}   —   {sum(r.passed for r in subset)}/{len(subset)} PASS")
        print("=" * 100)
        for r in subset:
            mark = "✅ PASS" if r.passed else "❌ FAIL"
            print(f"\n[{r.id}] {r.name}   ({r.category})")
            for k, v in r.inputs.items():
                print(f"      · {k}: {v}")
            print(f"    Expected : {r.expected}")
            print(f"    Actual   : {r.actual}")
            print(f"    Kết quả  : {mark}")
            if r.note:
                print(f"    Ghi chú  : {r.note}")

    n_pass = sum(r.passed for r in results)
    print("\n" + "=" * 100)
    print(f"  TỔNG: {n_pass}/{len(results)} PASS · {len(results) - n_pass} FAIL "
          f"({n_pass / len(results):.1%})")
    for model in ("ml01", "ml02"):
        subset = [r for r in results if r.model == model]
        print(f"    {model.upper()}: {sum(r.passed for r in subset)}/{len(subset)}")
    failed = [r for r in results if not r.passed]
    if failed:
        print("\n  Case FAIL:")
        for r in failed:
            print(f"    ❌ {r.id} — {r.name}")
            print(f"       expected: {r.expected}")
            print(f"       actual  : {r.actual}")
    print("=" * 100)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="in JSON thô")
    parser.add_argument("--out", type=Path, default=None,
                        help="thư mục ghi kết quả (mặc định runs/testcases)")
    args = parser.parse_args()

    logging.getLogger("hfml").setLevel(logging.WARNING)
    results = execute()

    if args.json:
        print(json.dumps([{
            "id": r.id, "model": r.model, "category": r.category,
            "name": r.name, "inputs": r.inputs, "expected": r.expected,
            "actual": r.actual, "result": r.status, "note": r.note,
        } for r in results], ensure_ascii=False, indent=2))
    else:
        print_report(results)

    out_dir = args.out or (CONFIG.paths.runs / "testcases")
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = to_frame(results)
    frame.to_csv(out_dir / "testcases_ml01_ml02.csv", index=False,
                 encoding="utf-8-sig")
    (out_dir / "testcases_ml01_ml02.json").write_text(
        frame.to_json(orient="records", force_ascii=False, indent=2),
        encoding="utf-8")
    if not args.json:
        print(f"\nĐã ghi: {out_dir / 'testcases_ml01_ml02.csv'}")

    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
