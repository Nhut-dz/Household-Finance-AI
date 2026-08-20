"""ML02 task 1 — Khám phá Home Credit Dataset (F04 · M04 · Tuần 4).

Vì sao cần bước này khi F01 task 6 đã kiểm chất lượng dữ liệu
--------------------------------------------------------------
`hfml.data.quality` trả lời *"dữ liệu sạch tới đâu"* — thiếu bao nhiêu, có
sentinel không, trùng lặp không. Nó **không** trả lời câu hỏi của F04:
*"cột nào dự báo được vỡ nợ, và cột nào trong số đó form của mình lấy được?"*

Đó mới là câu quyết định kiến trúc ML02: bảng so sánh **Full vs Rút gọn**
(PLAN.md §7.2) chỉ có nội dung khi biết chính xác mình đang bỏ đi bao nhiêu
sức mạnh dự báo để đổi lấy khả năng triển khai.

Thước đo: Information Value (IV), không phải tương quan
-------------------------------------------------------
Ba lý do, theo thứ tự quan trọng:

1. **Tương quan tuyến tính vô dụng ở dataset này.** PLAN.md §4.3e đã đo:
   |r| cao nhất trên toàn bộ 110 cột số chỉ **0,179** (`EXT_SOURCE_3`). Tin
   vào Pearson thì kết luận "chẳng cột nào có ích" — sai hoàn toàn.
2. **IV so được giữa cột số và cột hạng mục.** Muốn xếp `AMT_CREDIT` cạnh
   `NAME_EDUCATION_TYPE` trên cùng một bảng thì cần một thang chung.
3. **IV là thước đo chuẩn của ngành chấm điểm tín dụng**, kèm thang diễn giải
   đã được dùng rộng rãi (xem `IV_BANDS`). Trước hội đồng, "IV = 0,32, mức
   mạnh" nói được nhiều hơn "mutual information = 0,0041".

IV đi cùng WoE (Weight of Evidence) theo từng khoảng giá trị, nên ngoài một
con số tổng còn xem được **quan hệ có đơn điệu không** — chỗ mà cây quyết định
sẽ chẻ nhánh.

Một điều module này CỐ Ý làm: **coi NaN là một khoảng riêng**, không bỏ đi.
Ở Home Credit việc thiếu dữ liệu tự nó dự báo được vỡ nợ (PLAN.md §4.3b: 6 cờ
`_MISSING`, lift từ 0,624 tới 1,339). Bỏ NaN khỏi bảng WoE là vứt mất đúng
phần thông tin đó.

Module này chỉ ĐO và BÁO CÁO. Không sinh feature (thuộc `features.py`),
không train (thuộc `train.py`), không sửa dữ liệu (thuộc `preprocessing`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import numpy as np
import pandas as pd

from hfml.config import CONFIG
from hfml.data import loader
from hfml.logger import get_logger

log = get_logger(__name__)

# --------------------------------------------------------------------------
# Thang diễn giải IV
# --------------------------------------------------------------------------
#: Mốc dưới → tên mức. Thang quy ước của ngành chấm điểm tín dụng (Siddiqi,
#: *Credit Risk Scorecards*), dùng rộng rãi tới mức hội đồng nào cũng nhận ra.
#:
#: Mức cuối KHÔNG phải "càng cao càng tốt": IV > 0,5 trên dữ liệu thật gần như
#: luôn là dấu hiệu rò rỉ nhãn — một cột chứa sẵn câu trả lời. Gắn nhãn
#: "đáng ngờ" ngay tại đây để không ai kịp mừng.
IV_BANDS: Final[tuple[tuple[float, str], ...]] = (
    (0.50, "đáng ngờ (nghi rò rỉ nhãn)"),
    (0.30, "mạnh"),
    (0.10, "trung bình"),
    (0.02, "yếu"),
    (0.00, "gần như vô dụng"),
)


def iv_band(iv: float) -> str:
    """Xếp một giá trị IV vào mức diễn giải."""
    for threshold, name in IV_BANDS:
        if iv >= threshold:
            return name
    return IV_BANDS[-1][1]


#: Cộng vào tử/mẫu của WoE để khoảng không có hồ sơ vỡ nợ không cho ra `inf`.
#: 0,5 là hiệu chỉnh Haldane–Anscombe — nhỏ hơn một quan sát nên không bịa ra
#: tín hiệu, nhưng đủ để giữ con số hữu hạn.
WOE_SMOOTHING: Final[float] = 0.5

#: Khoảng chiếm dưới ngưỡng này bị gộp vào nhóm `__RARE__`. Một hạng mục 20
#: hồ sơ có thể cho WoE rất lớn hoàn toàn do ngẫu nhiên, và IV cộng dồn nó vào
#: sẽ đẩy một cột vô dụng lên đầu bảng.
MIN_BIN_SHARE: Final[float] = 0.01

#: Số khoảng khi rời rạc hoá cột số. 10 (thập phân vị) là mặc định quen thuộc
#: của ngành, đủ mịn để thấy dạng quan hệ mà không vụn.
DEFAULT_BINS: Final[int] = 10

#: Một GIÁ TRỊ đơn lẻ chiếm từ ngần này dân số trở lên được tách thành khoảng
#: riêng, phần còn lại mới đem chia phân vị.
#:
#: Không có bước này thì `pd.qcut` hỏng âm thầm trên các cột dồn khối. Đã sập
#: đúng bẫy đó khi chạy lần đầu: `BUREAU_TOTAL_OVERDUE` có **98,92% giá trị
#: bằng 0**, nên cả 10 mốc phân vị đều rơi vào 0, `duplicates="drop"` gộp hết
#: lại còn MỘT khoảng, và IV ra đúng 0,0000 — trông y như "cột không có tín
#: hiệu" trong khi thật ra là phép đo không chạy. Cùng cơ chế đó sẽ nuốt mất
#: sentinel `DAYS_EMPLOYED = 365243` (18,01%) vào chung thập phân vị cuối.
MASS_POINT_SHARE: Final[float] = 0.05

#: Nhãn của khoảng chứa giá trị thiếu và nhóm hạng mục hiếm.
MISSING_BIN: Final[str] = "__MISSING__"
RARE_BIN: Final[str] = "__RARE__"

#: Cột không bao giờ là feature — khoá hồ sơ và chính cái nhãn.
NON_FEATURE_COLUMNS: Final[frozenset[str]] = frozenset({
    loader.ID_COLUMN, loader.TARGET_COLUMN,
})


# --------------------------------------------------------------------------
# WoE / IV
# --------------------------------------------------------------------------
def _binned(series: pd.Series, bins: int, min_share: float) -> pd.Series:
    """Rời rạc hoá một cột thành các khoảng có tên đọc được.

    Ba loại khoảng, theo thứ tự ưu tiên:

        __MISSING__   NaN — luôn đứng riêng, không bao giờ bị loại
        =<giá trị>    một giá trị đơn lẻ chiếm ≥ `MASS_POINT_SHARE` dân số
        (a, b]        phân vị của phần dân số còn lại

    Tách khối trước rồi mới chia phân vị là chỗ then chốt. Ở Home Credit rất
    nhiều cột dồn về một giá trị: `BUREAU_TOTAL_OVERDUE` 98,92% bằng 0,
    `DAYS_EMPLOYED` 18,01% bằng sentinel 365243, `CNT_CHILDREN` ~70% bằng 0.
    Đem nguyên các cột đó cho `pd.qcut` thì mọi mốc phân vị trùng nhau, tất cả
    gộp về một khoảng, và IV ra 0 — không phải vì cột vô dụng mà vì phép đo
    không chạy.
    """
    if not pd.api.types.is_numeric_dtype(series) or series.nunique(dropna=True) <= bins:
        return series.astype(str).where(series.notna(), MISSING_BIN)

    share = series.value_counts(normalize=True, dropna=True)
    mass_values = set(share[share >= MASS_POINT_SHARE].index)

    labels = pd.Series(MISSING_BIN, index=series.index, dtype=object)
    is_mass = series.isin(mass_values) & series.notna()
    labels[is_mass] = series[is_mass].map(lambda v: f"={v:g}")

    rest = series[series.notna() & ~is_mass]
    if not rest.empty:
        # Số khoảng phải co lại theo phần dân số CÒN LẠI, không giữ cứng 10.
        # `BUREAU_TOTAL_OVERDUE` sau khi tách khối 0 chỉ còn 1,08% dân số; chia
        # tiếp thành 10 thì mỗi khoảng ~0,1%, nhỏ hơn `min_share` nên bị loại
        # hết khỏi phép tính lift — cột có tín hiệu thật lại báo lift 0,99.
        q = max(1, min(bins, int(len(rest) / len(series) / min_share)))
        # `duplicates="drop"` vẫn cần: phần còn lại có thể vẫn còn trùng mốc,
        # chỉ là không còn trùng tới mức sập về một khoảng.
        labels.loc[rest.index] = pd.qcut(
            rest, q=q, duplicates="drop").astype(str)

    return labels


def _group_rare(labels: pd.Series, min_share: float) -> pd.Series:
    """Gộp các khoảng quá nhỏ lại thành một nhóm `__RARE__`.

    `__MISSING__` được miễn: nó có thể hiếm nhưng vẫn là một tín hiệu có ý
    nghĩa riêng, gộp nó vào nhóm tạp là làm mất đúng thứ đang muốn đo.
    """
    share = labels.value_counts(normalize=True)
    rare = {v for v, s in share.items() if s < min_share and v != MISSING_BIN}
    return labels.where(~labels.isin(rare), RARE_BIN) if rare else labels


def woe_table(
    feature: pd.Series,
    target: pd.Series,
    bins: int = DEFAULT_BINS,
    min_share: float = MIN_BIN_SHARE,
) -> pd.DataFrame:
    """Bảng WoE/IV của một cột: mỗi dòng một khoảng giá trị.

    Cột trả về:

        bin           tên khoảng (`__MISSING__` / `__RARE__` là hai tên riêng)
        n             số hồ sơ
        share         tỉ trọng trên toàn tập
        n_bad         số hồ sơ vỡ nợ (TARGET = 1)
        bad_rate      tỉ lệ vỡ nợ trong khoảng
        lift          bad_rate ÷ tỉ lệ vỡ nợ chung — đọc nhanh hơn WoE
        woe           ln(tỉ trọng tốt ÷ tỉ trọng xấu)
        iv_part       phần đóng góp vào IV tổng

    `woe` âm nghĩa là khoảng đó rủi ro CAO hơn trung bình (nhiều "xấu" hơn tỉ
    trọng của nó). Quy ước dấu này là quy ước phổ biến của ngành; điều quan
    trọng là dùng nhất quán, và ở đây chỉ định nghĩa một lần.
    """
    frame = pd.DataFrame({"bin": _binned(feature, bins, min_share),
                          "y": target.astype(int)})
    frame["bin"] = _group_rare(frame["bin"], min_share)

    grouped = frame.groupby("bin", observed=True)["y"].agg(n="size", n_bad="sum")
    grouped["n_good"] = grouped["n"] - grouped["n_bad"]

    total_bad = float(grouped["n_bad"].sum())
    total_good = float(grouped["n_good"].sum())
    if total_bad == 0 or total_good == 0:
        raise ValueError(
            "Nhãn chỉ có một lớp trong tập được truyền vào — không tính được WoE.")

    # Hiệu chỉnh cộng 0,5 vào cả tử và mẫu: khoảng có 0 hồ sơ vỡ nợ vẫn cho ra
    # số hữu hạn thay vì -inf, mà không tạo ra tín hiệu không có thật.
    bad_share = (grouped["n_bad"] + WOE_SMOOTHING) / (total_bad + WOE_SMOOTHING * len(grouped))
    good_share = (grouped["n_good"] + WOE_SMOOTHING) / (total_good + WOE_SMOOTHING * len(grouped))

    base_rate = total_bad / (total_bad + total_good)
    table = pd.DataFrame({
        "bin": grouped.index.astype(str),
        "n": grouped["n"].astype(int).to_numpy(),
        "share": (grouped["n"] / len(frame)).to_numpy(),
        "n_bad": grouped["n_bad"].astype(int).to_numpy(),
        "bad_rate": (grouped["n_bad"] / grouped["n"]).to_numpy(),
        "lift": (grouped["n_bad"] / grouped["n"] / base_rate).to_numpy(),
        "woe": np.log(good_share / bad_share).to_numpy(),
    })
    table["iv_part"] = (good_share - bad_share).to_numpy() * table["woe"]
    return table.sort_values("bin", ignore_index=True)


def information_value(
    feature: pd.Series,
    target: pd.Series,
    bins: int = DEFAULT_BINS,
    min_share: float = MIN_BIN_SHARE,
) -> float:
    """IV tổng của một cột — tổng `iv_part` của bảng WoE."""
    return float(woe_table(feature, target, bins, min_share)["iv_part"].sum())


def rank_by_information_value(
    df: pd.DataFrame,
    target: str = loader.TARGET_COLUMN,
    bins: int = DEFAULT_BINS,
    min_share: float = MIN_BIN_SHARE,
    exclude: frozenset[str] = NON_FEATURE_COLUMNS,
) -> pd.DataFrame:
    """Xếp hạng MỌI cột theo sức mạnh dự báo. Thành phẩm chính của task 1.

    Ngoài `iv` còn trả về `max_lift` — tỉ lệ vỡ nợ của khoảng xấu nhất chia
    cho tỉ lệ chung. Hai con số này KHÔNG thay thế được cho nhau, và chỗ chúng
    lệch nhau là chỗ dễ kết luận sai nhất:

        IV       cân theo tỉ trọng dân số → đo giá trị cột đó với TOÀN BỘ
                 danh mục, tức "cột này giúp phân biệt được bao nhiêu hồ sơ"
        max_lift KHÔNG cân theo tỉ trọng → đo mức rủi ro của nhóm xấu nhất,
                 tức "khi cột này bật, hồ sơ đó nguy hiểm tới đâu"

    Ví dụ có thật trong dataset: `BUREAU_HAS_OVERDUE` có IV chỉ 0,0089 (mức
    "gần như vô dụng") nhưng `max_lift` = 1,97 — nhóm đang có khoản quá hạn vỡ
    nợ **gần gấp đôi** trung bình. IV thấp chỉ vì nhóm đó có 1,1% dân số. Đọc
    mỗi cột IV rồi loại nó đi là loại mất một tín hiệu rất mạnh trên đúng
    nhóm mà hệ thống cần cảnh báo nhất.

    Cột nào tính không được (chỉ một giá trị, toàn NaN) vẫn xuất hiện trong
    bảng với `iv = 0` và ghi lý do — im lặng bỏ qua thì lúc đối chiếu số cột
    sẽ không ai biết mấy cột kia đi đâu mất.
    """
    y = df[target]
    rows: list[dict] = []

    for column in df.columns:
        if column in exclude:
            continue

        series = df[column]
        note = ""
        max_lift, worst_bin = float("nan"), ""
        try:
            table = woe_table(series, y, bins=bins, min_share=min_share)
        except ValueError as exc:                      # nhãn một lớp
            iv, note = 0.0, str(exc)
        else:
            iv = float(table["iv_part"].sum())
            # Chỉ xét khoảng ĐỌC ĐƯỢC THÀNH MỘT NHÓM và đủ lớn để tin.
            # `__RARE__` bị loại vì nó là túi gom nhiều hạng mục chẳng liên
            # quan gì nhau — lift của nó không mô tả nhóm người nào cả. Bỏ
            # bước lọc này thì `FLAG_DOCUMENT_2` (bật ở 13/307.511 hồ sơ) leo
            # lên đầu bảng với lift 3,81, thuần tuý do ngẫu nhiên.
            usable = table[(table["bin"] != RARE_BIN)
                           & (table["share"] >= min_share)]
            if not usable.empty:
                worst = usable.loc[usable["lift"].idxmax()]
                max_lift, worst_bin = float(worst["lift"]), str(worst["bin"])
            if series.nunique(dropna=True) <= 1:
                note = "cột hằng số — IV không có ý nghĩa"

        rows.append({
            "column": column,
            "dtype": "categorical"
                     if not pd.api.types.is_numeric_dtype(series) else "numeric",
            "missing_rate": float(series.isna().mean()),
            "n_unique": int(series.nunique(dropna=True)),
            "iv": iv,
            "band": iv_band(iv),
            "max_lift": max_lift,
            "worst_bin": worst_bin,
            "note": note,
        })

    return (pd.DataFrame(rows)
            .sort_values("iv", ascending=False, ignore_index=True))


# --------------------------------------------------------------------------
# Ánh xạ form ↔ Home Credit — căn cứ của bảng Full vs Rút gọn (§7.2)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class FormField:
    """Một trường form và cột Home Credit tương ứng.

    `source` cho biết trường đó nằm ở màn nào — quan trọng vì hai màn có tỉ lệ
    điền khác hẳn nhau: mọi người dùng đều qua "Nhập thông tin", chỉ người
    đang tính vay mới điền "Thông tin khoản vay".
    """
    field: str
    source: str                    # "household" | "loan" | "derived"
    home_credit: str | None        # None = Home Credit không có cột tương ứng
    note: str = ""


#: 20 trường form ↔ Home Credit, chốt sau khi màn "Thông tin khoản vay" lên
#: (15/08/2026). Đây là danh sách quyết định bộ RÚT GỌN của ML02 gồm những gì.
#:
#: `home_credit = None` không phải thiếu sót: đó là những trường form thu được
#: mà Home Credit KHÔNG có (tiết kiệm, chi tiêu tháng), nên chúng phục vụ tầng
#: rule chứ không vào ML02. Chiều ngược lại — Home Credit có mà form không thu
#: được — mới là phần làm bộ Rút gọn yếu đi, và `EXT_SOURCE_1/2/3` đứng đầu
#: danh sách đó.
FORM_FIELDS: Final[tuple[FormField, ...]] = (
    # -- Màn "Nhập thông tin" ---------------------------------------------
    FormField("household_size", "household", "CNT_FAM_MEMBERS"),
    FormField("children_count", "household", "CNT_CHILDREN"),
    FormField("birth_year", "household", "DAYS_BIRTH", "đổi sang tuổi"),
    FormField("average_monthly_income", "household", "AMT_INCOME_TOTAL",
              "CHỈ dùng làm mẫu số của feature tỉ lệ, không vào X (§2.1)"),
    FormField("average_monthly_expense", "household", None,
              "Home Credit không có cột chi tiêu — chỉ phục vụ tầng rule"),
    FormField("savings_amount", "household", None,
              "Home Credit không có cột tiết kiệm"),
    FormField("total_current_debt", "household", None,
              "gần nhất là bureau.AMT_CREDIT_SUM_DEBT, không phải cùng một thứ"),
    FormField("monthly_debt_payment", "household", None,
              "AMT_ANNUITY là kỳ trả của khoản ĐANG XIN, không phải nợ đang có"),

    # -- Màn "Thông tin khoản vay" ----------------------------------------
    FormField("borrower_age", "loan", "DAYS_BIRTH"),
    FormField("gender", "loan", "CODE_GENDER"),
    FormField("marital_status", "loan", "NAME_FAMILY_STATUS"),
    FormField("education_level", "loan", "NAME_EDUCATION_TYPE"),
    FormField("occupation", "loan", "OCCUPATION_TYPE"),
    FormField("employment_years", "loan", "DAYS_EMPLOYED"),
    FormField("loan_amount", "loan", "AMT_CREDIT"),
    FormField("monthly_payment", "loan", "AMT_ANNUITY"),
    FormField("asset_price", "loan", "AMT_GOODS_PRICE"),
    FormField("loan_term_months", "loan", None, "suy ra từ AMT_CREDIT/AMT_ANNUITY"),
    FormField("loan_purpose", "loan", None,
              "chỉ có ở previous_application, 95,8% là XAP/XNA"),

    # -- Mục C: lịch sử tín dụng, tổng hợp từ bureau.csv -------------------
    FormField("previous_loan_count", "loan", "BUREAU_LOAN_COUNT"),
    FormField("late_payment_count", "loan", "BUREAU_OVERDUE_LOAN_COUNT"),
    FormField("has_overdue_loan", "loan", "BUREAU_HAS_OVERDUE"),
    FormField("total_overdue_amount", "loan", "BUREAU_TOTAL_OVERDUE"),
)


def form_coverage(iv_ranking: pd.DataFrame) -> pd.DataFrame:
    """Ghép danh sách trường form với sức mạnh dự báo đo được của cột tương ứng.

    Bảng này trả lời thẳng câu hỏi của §7.2: form lấy được bao nhiêu phần sức
    mạnh dự báo, và mất những cột nào.
    """
    strength = iv_ranking.set_index("column")["iv"]
    band = iv_ranking.set_index("column")["band"]

    rows = [{
        "field": f.field,
        "source": f.source,
        "home_credit": f.home_credit or "—",
        "iv": float(strength.get(f.home_credit, np.nan))
              if f.home_credit else np.nan,
        "band": str(band.get(f.home_credit, "—")) if f.home_credit else "—",
        "note": f.note,
    } for f in FORM_FIELDS]

    return pd.DataFrame(rows)


def unreachable_columns(iv_ranking: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Những cột mạnh nhất mà form KHÔNG lấy được — cái giá của bộ Rút gọn.

    Đây là bảng phải đưa vào phần "phân tích tính khả thi triển khai" của báo
    cáo. Nói "bộ rút gọn kém hơn" mà không chỉ ra kém vì mất cột nào thì hội
    đồng sẽ hỏi ngay.
    """
    reachable = {f.home_credit for f in FORM_FIELDS if f.home_credit}
    return (iv_ranking[~iv_ranking["column"].isin(reachable)]
            .head(top_n)
            .reset_index(drop=True))


# --------------------------------------------------------------------------
# bureau.csv — nguồn của mục C trên form
# --------------------------------------------------------------------------
# Phép gộp bureau nay do `features.py` (task 3) sở hữu, module này chỉ dùng
# lại. Trước 15/08/2026 hai file có hai bản cài đặt riêng cho cùng một phép
# gộp — mà hai bản cài đặt cho cùng một câu hỏi thì sớm muộn cho hai câu trả
# lời khác nhau về cùng một khách hàng.
#
# `aggregate_bureau` và `merge_bureau` giữ nguyên tên ở đây để phần khảo sát
# của task 1 và bộ test của nó không phải sửa theo.
from hfml.ml.ml02_credit_risk.features import (  # noqa: E402
    BUREAU_COLUMNS,
    aggregate_bureau,
    merge_bureau,
)


# --------------------------------------------------------------------------
# Khoảng cách miền VNĐ ↔ Home Credit (§2.1)
# --------------------------------------------------------------------------
#: Feature tỉ lệ dựng được từ chính `application_train.csv`, kèm công thức để
#: bảng phân phối tự giải thích được mà không phải tra chỗ khác.
RATIO_FEATURES: Final[dict[str, str]] = {
    "dti": "AMT_ANNUITY / AMT_INCOME_TOTAL",
    "credit_income_ratio": "AMT_CREDIT / AMT_INCOME_TOTAL",
    "credit_goods_markup": "AMT_CREDIT / AMT_GOODS_PRICE",
    "children_ratio": "CNT_CHILDREN / CNT_FAM_MEMBERS",
    "employment_ratio": "DAYS_EMPLOYED / DAYS_BIRTH",
}


def build_ratio_features(app: pd.DataFrame) -> pd.DataFrame:
    """Dựng các feature tỉ lệ để đo phân phối của chúng.

    Bản rút gọn của `hfml.data.features.builder`, cố ý đặt riêng: ở đây chỉ
    cần đo phân phối để đối chiếu với hồ sơ Việt Nam, không cần toàn bộ hợp
    đồng feature của tầng train.

    `DAYS_EMPLOYED = 365243` (sentinel, 18,01% dữ liệu) được loại trước khi
    tính `employment_ratio` — để nguyên thì tỉ lệ ra -19,4 và cả bảng phân
    phối thành vô nghĩa.
    """
    employed = app["DAYS_EMPLOYED"].where(app["DAYS_EMPLOYED"] != 365243)
    income = app["AMT_INCOME_TOTAL"].replace(0, np.nan)

    return pd.DataFrame({
        "dti": app["AMT_ANNUITY"] / income,
        "credit_income_ratio": app["AMT_CREDIT"] / income,
        "credit_goods_markup": app["AMT_CREDIT"] / app["AMT_GOODS_PRICE"].replace(0, np.nan),
        "children_ratio": app["CNT_CHILDREN"] / app["CNT_FAM_MEMBERS"].replace(0, np.nan),
        "employment_ratio": employed / app["DAYS_BIRTH"],
    })


def distribution_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Phân vị của từng cột — dạng bảng đưa thẳng vào báo cáo được."""
    quantiles = (0.01, 0.25, 0.50, 0.75, 0.99)
    rows = [{
        "feature": column,
        "formula": RATIO_FEATURES.get(column, ""),
        "missing_rate": float(frame[column].isna().mean()),
        **{f"p{int(q * 100)}": float(frame[column].quantile(q)) for q in quantiles},
    } for column in frame.columns]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Chạy toàn bộ
# --------------------------------------------------------------------------
@dataclass
class ExplorationReport:
    """Toàn bộ thành phẩm của task 1. Mỗi thuộc tính ghi ra một file CSV."""
    target: dict = field(default_factory=dict)
    iv_ranking: pd.DataFrame = field(default_factory=pd.DataFrame)
    form_coverage: pd.DataFrame = field(default_factory=pd.DataFrame)
    unreachable: pd.DataFrame = field(default_factory=pd.DataFrame)
    bureau_iv: pd.DataFrame = field(default_factory=pd.DataFrame)
    bureau_coverage: dict = field(default_factory=dict)
    ratio_distribution: pd.DataFrame = field(default_factory=pd.DataFrame)
    woe_details: dict[str, pd.DataFrame] = field(default_factory=dict)

    def tables(self) -> dict[str, pd.DataFrame]:
        """Tên file (không đuôi) → bảng, dùng khi ghi ra đĩa."""
        return {
            "iv_ranking": self.iv_ranking,
            "form_coverage": self.form_coverage,
            "unreachable_columns": self.unreachable,
            "bureau_iv": self.bureau_iv,
            "ratio_distribution": self.ratio_distribution,
            **{f"woe_{name}": table for name, table in self.woe_details.items()},
        }


#: Các cột được xuất bảng WoE chi tiết. Năm cột đầu chính là năm dropdown của
#: màn "Thông tin khoản vay" — cần biết từng lựa chọn trên form ứng với mức
#: rủi ro nào, nếu không thì tầng LLM không diễn giải kết quả được.
WOE_DETAIL_COLUMNS: Final[tuple[str, ...]] = (
    "CODE_GENDER", "NAME_FAMILY_STATUS", "NAME_EDUCATION_TYPE",
    "OCCUPATION_TYPE", "NAME_CONTRACT_TYPE",
    "DAYS_BIRTH", "DAYS_EMPLOYED", "EXT_SOURCE_2",
)


def explore(nrows: int | None = None, with_bureau: bool = True) -> ExplorationReport:
    """Chạy đủ task 1 và trả về mọi bảng đo được.

    `nrows` giới hạn số dòng để chạy thử nhanh — bảng sẽ khác bản đầy đủ, nên
    chỉ dùng lúc phát triển, đừng lấy số đó đưa vào báo cáo.
    """
    app = loader.load_application_train(nrows=nrows)
    y = app[loader.TARGET_COLUMN]
    log.info("Xếp hạng IV cho %d cột…", app.shape[1] - len(NON_FEATURE_COLUMNS))

    report = ExplorationReport(
        target=loader.target_distribution(nrows=nrows),
        iv_ranking=rank_by_information_value(app),
    )
    report.form_coverage = form_coverage(report.iv_ranking)
    report.unreachable = unreachable_columns(report.iv_ranking)
    report.ratio_distribution = distribution_table(build_ratio_features(app))
    report.woe_details = {
        column: woe_table(app[column], y)
        for column in WOE_DETAIL_COLUMNS if column in app.columns
    }

    if with_bureau:
        log.info("Đọc và tổng hợp bureau.csv…")
        # KHÔNG truyền `nrows` xuống bureau: file này là quan hệ một-nhiều,
        # cắt 20.000 DÒNG ĐẦU chỉ ra 212 khách hàng, và mọi hồ sơ còn lại bị
        # coi là "chưa từng vay". Đọc đủ (6 cột, ~10s) rồi để bước merge tự
        # giới hạn theo tập hồ sơ đang xét — chạy thử vẫn ra số đúng.
        bureau = loader.load_raw("bureau", columns=list(BUREAU_COLUMNS))
        merged = merge_bureau(app, aggregate_bureau(bureau))

        bureau_cols = [c for c in merged.columns if c.startswith("BUREAU_")]
        report.bureau_iv = rank_by_information_value(
            merged[bureau_cols + [loader.TARGET_COLUMN]])
        no_record = merged["BUREAU_NO_RECORD"].astype(bool)
        report.bureau_coverage = {
            "n_bureau_rows": len(bureau),
            "n_customers_with_record": int((~no_record).sum()),
            "n_customers_without_record": int(no_record.sum()),
            "share_without_record": float(no_record.mean()),
            "default_rate_without_record": float(y[no_record].mean()),
            "default_rate_with_record": float(y[~no_record].mean()),
        }
        for column in ("BUREAU_LOAN_COUNT", "BUREAU_HAS_OVERDUE"):
            report.woe_details[column] = woe_table(merged[column], y)

    return report


def write_tables(report: ExplorationReport, subdir: str = "ml02_eda") -> list:
    """Ghi mọi bảng ra `src/training/runs/ml02_eda/*.csv`."""
    out_dir = CONFIG.paths.runs / subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for name, table in report.tables().items():
        if table.empty:
            continue
        path = out_dir / f"{name}.csv"
        table.to_csv(path, index=False, encoding="utf-8")
        written.append(path)

    log.info("Đã ghi %d bảng → %s", len(written), out_dir)
    return written
