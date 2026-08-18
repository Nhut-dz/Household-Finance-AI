"""ML02 task 2 — Data Cleaning (F04 · M04 · Tuần 4).

Nối tiếp task 1 ([explore.py](explore.py)), KHÔNG làm lại EDA. Task này biến
dữ liệu thô thành một bộ dữ liệu sạch mà task 3 dùng được ngay, và ghi lại
đúng những gì đã làm để việc đó tái lập được.

Ranh giới quan trọng nhất của file này
---------------------------------------
Làm sạch chia làm hai loại, và **chỉ loại thứ nhất được ghi ra đĩa**:

    KHÔNG HỌC GÌ      Biến đổi theo từng dòng, `fit()` rỗng. Kết quả của một
    (làm ở đây)       dòng không phụ thuộc dòng nào khác → chạy trước khi
                      chia train/test cũng không rò rỉ.
                      · sentinel → NaN + cờ `_MISSING`
                      · chuỗi giả ('XNA'/'XAP'/'Unknown') → NaN
                      · chuẩn hoá kiểu dữ liệu
                      · bỏ dòng trùng · gắn cờ dòng bất hợp lệ

    CÓ HỌC            Cần thống kê của cả tập: trung vị, phân vị, danh sách
    (KHÔNG làm ở đây) hạng mục, tỉ lệ thiếu theo cột.
                      · `HighMissingDropper` · `OutlierClipper`
                      · `SimpleImputer` · encoder · khử tương quan

Loại thứ hai bắt buộc nằm trong `sklearn.Pipeline` và `fit` **chỉ trên tập
train** (PLAN.md §4.4). Đem chúng ra chạy trước rồi lưu kết quả là rò rỉ dữ
liệu: trung vị dùng để điền thiếu sẽ được tính trên cả những dòng sau này là
tập test, và metric thu được sẽ lạc quan hơn thực tế mà không có dấu hiệu gì.

Vì vậy `CLEANED` không phải "dữ liệu đã sẵn sàng cho model" mà là "dữ liệu đã
hết bẩn ở mức từng dòng". Phần còn lại là việc của Pipeline, và
`PIPELINE_STEPS_REMAINING` liệt kê ra để không ai tưởng đã làm xong.

Bỏ dòng trùng và gắn cờ dòng bất hợp lệ: KHÁC nhau, có lý do
-------------------------------------------------------------
    Dòng trùng `SK_ID_CURR`  → BỎ, và phải bỏ TRƯỚC khi chia tập.
                               Cùng một khách hàng nằm ở cả train lẫn test là
                               rò rỉ theo đúng nghĩa đen: model đã thấy đáp án.
    Dòng bất hợp lệ          → chỉ GẮN CỜ, không bỏ.
                               Bỏ trước khi chia thì tập test cũng sạch theo,
                               và chỉ số đo được sẽ đẹp hơn thực tế lúc chạy
                               thật — nơi hồ sơ bất hợp lệ vẫn cứ đến. Cờ
                               `INVALID_ROW` để task 3 bỏ chúng khỏi RIÊNG tập
                               train, giữ nguyên tập test.

Không làm ở task này: feature engineering (gộp bureau, dựng tỉ lệ) và train.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from hfml.config import CONFIG
from hfml.data import loader
from hfml.data.preprocessing.cleaner import (
    FLAG_SUFFIX,
    MISSING_FLAGS,
    normalize_missing,
)
from hfml.data.preprocessing.validator import (
    INVALID_RULES,
    duplicate_id_mask,
    invalid_mask,
    validation_report,
)
from hfml.logger import get_logger

log = get_logger(__name__)

# --------------------------------------------------------------------------
# Cột không bao giờ là feature
# --------------------------------------------------------------------------
#: Nhãn. Để lọt vào feature set là model học thuộc đáp án — dạng rò rỉ thô
#: sơ nhất và cũng dễ xảy ra nhất khi người ta `df.drop(columns=['TARGET'])`
#: ở một chỗ rồi quên ở chỗ khác.
TARGET_COLUMN: Final[str] = loader.TARGET_COLUMN

#: Khoá hồ sơ. Không phải nhãn, nhưng vẫn phải loại: mã định danh thường mang
#: thông tin thời gian (cấp tăng dần theo ngày nộp đơn), và model bắt được
#: xu hướng theo thời gian là học một thứ không tồn tại lúc chạy thật.
#:
#: Đã ĐO trên dữ liệu thật chứ không suy đoán: `corr(SK_ID_CURR, TARGET)` =
#: −0,0021, tỉ lệ vỡ nợ theo thập phân vị ID nằm trong khoảng 7,91%–8,29%
#: quanh mức chung 8,07%. Tức mã này KHÔNG mang tín hiệu — nhưng vẫn loại,
#: vì lý do loại nó là nguyên tắc chứ không phải kết quả đo.
ID_COLUMN: Final[str] = loader.ID_COLUMN

#: Cờ kỹ thuật do chính task này sinh ra. Là siêu dữ liệu về dòng, không phải
#: thuộc tính của khách hàng.
INVALID_ROW_FLAG: Final[str] = "INVALID_ROW"

#: Ba cột trên, gom lại. `feature_columns()` dùng nó để trả về đúng phần được
#: phép làm feature.
NON_FEATURE_COLUMNS: Final[frozenset[str]] = frozenset({
    TARGET_COLUMN, ID_COLUMN, INVALID_ROW_FLAG,
})

# --------------------------------------------------------------------------
# Dữ liệu tương lai
# --------------------------------------------------------------------------
#: Cột thời gian của `bureau.csv` phải ≤ 0 — số ngày TRƯỚC khi nộp đơn. Giá
#: trị dương nghĩa là thông tin đến SAU thời điểm nộp đơn, tức thứ mà lúc
#: chạy thật hệ thống không thể biết.
#:
#: Đo được **17 dòng** `DAYS_CREDIT_UPDATE > 0` (0,00099%, giá trị 10→372
#: ngày), thuộc 17 khách hàng khác nhau.
BUREAU_PAST_ONLY_COLUMNS: Final[tuple[str, ...]] = (
    "DAYS_CREDIT",
    "DAYS_ENDDATE_FACT",
    "DAYS_CREDIT_UPDATE",
)

#: ⚠️ `DAYS_CREDIT_ENDDATE` CỐ TÌNH không nằm trong danh sách trên, dù 35,11%
#: giá trị của nó là số dương. Đó là *ngày kết thúc dự kiến* của khoản vay còn
#: hiệu lực — một con số đã biết ngay lúc ký hợp đồng, nên biết nó tại thời
#: điểm nộp đơn là hoàn toàn hợp lệ.
#:
#: Ghi hằng số này ra để lần sau có người thấy "35% giá trị dương" rồi tưởng
#: là lỗi và đi "sửa" nó — sửa là mất một feature hợp lệ.
BUREAU_FUTURE_LOOKING_OK: Final[tuple[str, ...]] = ("DAYS_CREDIT_ENDDATE",)

#: Ngưỡng |r| với nhãn để coi một cột là đáng ngờ. Trên `application_train`,
#: |r| cao nhất chỉ 0,179 (`EXT_SOURCE_3`) nên còn rất xa ngưỡng này; nó tồn
#: tại để bắt cột rò rỉ được thêm vào sau này, không phải để bắt cột hiện có.
LEAKAGE_CORRELATION_THRESHOLD: Final[float] = 0.50

#: Các bước CÓ HỌC còn lại, phải nằm trong Pipeline và `fit` chỉ trên train.
#: Liệt kê ra để không ai đọc xong task 2 rồi tưởng dữ liệu đã sẵn sàng.
PIPELINE_STEPS_REMAINING: Final[tuple[tuple[str, str], ...]] = (
    ("HighMissingDropper", "học danh sách cột thiếu quá ngưỡng"),
    ("OutlierClipper", "học phân vị 0,1%–99,9% để kẹp biên"),
    ("SimpleImputer", "học trung vị / mode để điền thiếu"),
    ("OrdinalEncoder", "học bảng hạng mục → mã"),
    ("NearZeroVarianceRemover", "học cột nào gần như hằng số"),
    ("CorrelatedFeatureRemover", "học cặp cột tương quan cao"),
    ("SupervisedFeatureSelector", "NHÌN NHÃN — bắt buộc trong fold"),
)


# --------------------------------------------------------------------------
# Nhật ký từng bước
# --------------------------------------------------------------------------
@dataclass
class CleaningStep:
    """Một bước làm sạch, kèm số đo trước/sau.

    Ghi cả bước KHÔNG thay đổi gì (`rows_before == rows_after`) — biết chắc
    một bước đã chạy và không tìm thấy gì khác hẳn với việc không biết nó có
    chạy hay không.
    """

    name: str
    description: str
    rows_before: int
    rows_after: int
    cols_before: int
    cols_after: int
    detail: str = ""

    @property
    def rows_removed(self) -> int:
        return self.rows_before - self.rows_after

    @property
    def cols_added(self) -> int:
        return self.cols_after - self.cols_before


@dataclass
class CleaningReport:
    """Toàn bộ thành phẩm của task 2."""

    table: str
    steps: list[CleaningStep] = field(default_factory=list)

    #: Bảng sáu phép kiểm của `leakage_audit()`. CHỈ bảng đó — nó có cột
    #: `passed` mà `passed_leakage_audit` dựa vào.
    leakage: pd.DataFrame = field(default_factory=pd.DataFrame)

    #: Bảng đếm thông tin tương lai của bureau. Cấu trúc KHÁC HẲN `leakage`
    #: nên để riêng: nhét chung một ô từng làm `passed_leakage_audit` ném
    #: KeyError vì bảng này không có cột `passed`.
    future_information: pd.DataFrame = field(default_factory=pd.DataFrame)

    dtypes: pd.DataFrame = field(default_factory=pd.DataFrame)
    validation: pd.DataFrame = field(default_factory=pd.DataFrame)
    missing: pd.DataFrame = field(default_factory=pd.DataFrame)

    def steps_frame(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(s) for s in self.steps])

    @property
    def passed_leakage_audit(self) -> bool:
        """Bảng rỗng = báo cáo này không chạy kiểm toán, coi như không cản trở.

        Chỉ `application` chạy sáu phép kiểm; `bureau` dùng
        `future_information` với ràng buộc riêng.
        """
        if self.leakage.empty or "passed" not in self.leakage.columns:
            return True
        return bool(self.leakage["passed"].all())


def _step(name: str, description: str, before: pd.DataFrame,
          after: pd.DataFrame, detail: str = "") -> CleaningStep:
    return CleaningStep(
        name=name, description=description,
        rows_before=len(before), rows_after=len(after),
        cols_before=before.shape[1], cols_after=after.shape[1],
        detail=detail,
    )


# --------------------------------------------------------------------------
# Kiểu dữ liệu
# --------------------------------------------------------------------------
def dtype_report(df: pd.DataFrame) -> pd.DataFrame:
    """Phân loại từng cột thành số / hạng mục, kèm chỗ hai thứ không khớp nhau.

    `semantic` khác `dtype` ở một nhóm đáng chú ý: **33 cột kiểu số nhưng chỉ
    nhận {0, 1}** (`FLAG_OWN_CAR` đã mã hoá, 20 cột `FLAG_DOCUMENT_*`…).
    Chúng là biến nhị phân đội lốt số.

    Với bốn thuật toán cây thì điều đó **vô hại** — cây chẻ theo ngưỡng, và
    `x ≤ 0,5` trên cột nhị phân cho đúng một phép phân hoạch. Ghi ra vì hai
    lý do: bảng feature importance đọc dễ hơn khi biết cột nào là cờ, và nếu
    sau này có ai thêm một baseline tuyến tính thì đây là chỗ phải xử lý.
    """
    rows = []
    for column in df.columns:
        series = df[column]
        is_numeric = pd.api.types.is_numeric_dtype(series)
        values = set(series.dropna().unique()) if is_numeric else set()

        if not is_numeric:
            semantic = "categorical"
        elif values <= {0, 1}:
            semantic = "binary"
        elif series.nunique(dropna=True) <= 2:
            semantic = "binary"
        else:
            semantic = "numeric"

        rows.append({
            "column": column,
            "dtype": str(series.dtype),
            "semantic": semantic,
            "n_unique": int(series.nunique(dropna=True)),
            "missing_rate": float(series.isna().mean()),
        })
    return pd.DataFrame(rows)


def normalize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Ép cột chuỗi sang `category`.

    Không ép cột số sang kiểu hẹp hơn: `int8` tiết kiệm RAM nhưng lại làm
    `NaN` không biểu diễn được, mà bước điền thiếu còn nằm ở phía sau trong
    Pipeline.

    Đã kiểm trên dữ liệu thật: **không có cột chuỗi nào thực chất là số**
    (ép `to_numeric` trên cả 16 cột chuỗi ra 100% `NaN`), nên không có bước
    "sửa kiểu sai" nào phải làm. Kiểm vẫn chạy trong `dtype_issues()` để lần
    sau dữ liệu đổi thì biết.
    """
    out = df.copy()
    for column in out.columns:
        if pd.api.types.is_object_dtype(out[column]) or pd.api.types.is_string_dtype(out[column]):
            out[column] = out[column].astype("category")
    return out


def dtype_issues(df: pd.DataFrame) -> pd.DataFrame:
    """Cột chuỗi nhưng thực chất chứa số — tức kiểu bị đọc sai lúc nạp."""
    rows = []
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            continue
        converted = pd.to_numeric(df[column], errors="coerce")
        n_numeric = int(converted.notna().sum())
        if n_numeric:
            rows.append({
                "column": column,
                "dtype": str(df[column].dtype),
                "n_parsable_as_number": n_numeric,
                "rate": n_numeric / len(df) if len(df) else 0.0,
            })
    return pd.DataFrame(rows, columns=["column", "dtype", "n_parsable_as_number", "rate"])


# --------------------------------------------------------------------------
# Kiểm toán rò rỉ dữ liệu
# --------------------------------------------------------------------------
def _audit_row(check: str, passed: bool, measured: str, note: str) -> dict:
    return {"check": check, "passed": passed, "measured": measured, "note": note}


def leakage_audit(
    df: pd.DataFrame,
    target: str = TARGET_COLUMN,
    id_column: str = ID_COLUMN,
) -> pd.DataFrame:
    """Sáu phép kiểm rò rỉ, mỗi phép trả về một dòng có ĐO ĐƯỢC.

    Bảng này là thứ đem ra trả lời câu hỏi *"làm sao bạn biết không có rò
    rỉ?"*. Trả lời bằng "tôi đã cẩn thận" thì không kiểm chứng được; trả lời
    bằng sáu con số thì được.

    Cố ý KHÔNG ném lỗi khi có phép kiểm trượt: nhiệm vụ của hàm này là ĐO và
    BÁO. Việc quyết định dừng hay đi tiếp thuộc về người chạy — và
    `build_clean_dataset()` sẽ chặn nếu có phép kiểm trượt.
    """
    rows: list[dict] = []
    features = feature_columns(df)

    # 1. Nhãn không nằm trong feature set.
    rows.append(_audit_row(
        "target_excluded_from_features",
        target not in features,
        f"{target} {'KHÔNG' if target not in features else 'CÓ'} trong {len(features)} feature",
        "Nhãn lọt vào X là model học thuộc đáp án — accuracy ~100%, vô nghĩa.",
    ))

    # 2. Khoá hồ sơ không nằm trong feature set.
    rows.append(_audit_row(
        "id_excluded_from_features",
        id_column not in features,
        f"{id_column} {'KHÔNG' if id_column not in features else 'CÓ'} trong feature set",
        "Mã định danh thường mang thông tin thời gian cấp mã.",
    ))

    # 3. Khoá hồ sơ có mang tín hiệu thời gian không — ĐO chứ không suy đoán.
    if id_column in df.columns and target in df.columns:
        deciles = pd.qcut(df[id_column], 10, duplicates="drop")
        rates = df.groupby(deciles, observed=True)[target].mean()
        spread = float(rates.max() - rates.min())

        # Ngưỡng phải theo CỠ MẪU, không được là một con số cố định. Mười nhóm
        # chia từ 200 dòng thì mỗi nhóm 20 dòng, và chênh lệch 15 điểm phần
        # trăm giữa nhóm cao nhất với thấp nhất là nhiễu lấy mẫu thuần tuý —
        # một ngưỡng cứng kiểu "0,02" sẽ báo động giả ở mọi tập nhỏ.
        #
        # So với 3 lần sai số chuẩn của một tỉ lệ trong nhóm: vượt qua đó thì
        # chênh lệch mới lớn hơn mức ngẫu nhiên giải thích được.
        base_rate = float(df[target].mean())
        per_decile = max(len(df) / max(len(rates), 1), 1)
        std_error = float(np.sqrt(base_rate * (1 - base_rate) / per_decile))
        threshold = 3 * std_error

        rows.append(_audit_row(
            "id_carries_no_time_signal",
            spread <= threshold,
            f"tỉ lệ vỡ nợ theo thập phân vị ID: {rates.min():.4f}–{rates.max():.4f} "
            f"(chênh {spread:.4f}, ngưỡng 3σ = {threshold:.4f})",
            "Chênh lệch lớn hơn nhiễu lấy mẫu nghĩa là ID là biến thời gian "
            "trá hình — model sẽ học một xu hướng không tồn tại lúc chạy thật.",
        ))

    # 4. Không cột nào tương quan bất thường với nhãn.
    if target in df.columns:
        numeric = df[features].select_dtypes(include=[np.number])
        # Bỏ cột hằng số trước khi tính tương quan: độ lệch chuẩn 0 làm phép
        # chia ra `nan` kèm RuntimeWarning. `FLAG_MOBIL` (307.510/307.511 dòng
        # bằng 1) và 3 cột `FLAG_DOCUMENT_*` rơi vào đây. Tương quan của một
        # cột hằng số không định nghĩa được, nên loại là đúng chứ không phải
        # để cho đỡ ồn.
        varying = numeric.loc[:, numeric.std(numeric_only=True) > 0]
        correlations = varying.corrwith(df[target]).abs().dropna()
        worst = correlations.idxmax() if len(correlations) else "—"
        highest = float(correlations.max()) if len(correlations) else 0.0
        rows.append(_audit_row(
            "no_feature_correlates_with_target",
            highest < LEAKAGE_CORRELATION_THRESHOLD,
            f"|r| cao nhất {highest:.4f} ở `{worst}` "
            f"(ngưỡng {LEAKAGE_CORRELATION_THRESHOLD})",
            "Một cột tương quan gần tuyệt đối với nhãn gần như luôn là nhãn "
            "trá hình. Task 1 cũng đã kiểm bằng IV: không cột nào > 0,5.",
        ))

    # 5. Không có cột nào trùng nội dung với nhãn.
    if target in df.columns:
        identical = [c for c in features
                     if df[c].dtype == df[target].dtype and df[c].equals(df[target])]
        rows.append(_audit_row(
            "no_column_duplicates_the_target",
            not identical,
            f"{len(identical)} cột trùng khít nhãn {identical if identical else ''}",
            "Bản sao của nhãn dưới một cái tên khác.",
        ))

    # 6. Không còn dòng trùng khoá hồ sơ.
    if id_column in df.columns:
        duplicates = int(duplicate_id_mask(df, id_column).sum())
        rows.append(_audit_row(
            "no_duplicate_customers",
            duplicates == 0,
            f"{duplicates} dòng trùng {id_column}",
            "Cùng một khách nằm ở cả train lẫn test là rò rỉ theo nghĩa đen.",
        ))

    return pd.DataFrame(rows)


def bureau_future_information(bureau: pd.DataFrame) -> pd.DataFrame:
    """Đếm giá trị mang thông tin SAU thời điểm nộp đơn ở `bureau.csv`.

    Chỉ soi các cột trong `BUREAU_PAST_ONLY_COLUMNS`. `DAYS_CREDIT_ENDDATE`
    được miễn có chủ ý — xem `BUREAU_FUTURE_LOOKING_OK`.
    """
    rows = []
    for column in BUREAU_PAST_ONLY_COLUMNS:
        if column not in bureau.columns:
            continue
        series = bureau[column]
        hits = int((series > 0).sum())
        rows.append({
            "column": column,
            "n_future_rows": hits,
            "rate": hits / len(bureau) if len(bureau) else 0.0,
            "max_days_after": float(series.max()) if hits else 0.0,
            "n_customers": int(bureau.loc[series > 0, "SK_ID_CURR"].nunique()) if hits else 0,
        })
    return pd.DataFrame(rows)


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Các cột được phép làm feature — mọi cột trừ `NON_FEATURE_COLUMNS`.

    Một nơi duy nhất trả lời câu "cột nào được vào X". Mỗi chỗ tự viết
    `drop(columns=['TARGET'])` là kiểu chỗ nhớ chỗ quên.
    """
    return [c for c in df.columns if c not in NON_FEATURE_COLUMNS]


# --------------------------------------------------------------------------
# Làm sạch
# --------------------------------------------------------------------------
def clean_application(df: pd.DataFrame) -> tuple[pd.DataFrame, list[CleaningStep]]:
    """Làm sạch `application_train.csv` ở mức từng dòng.

    Bốn bước, tất cả đều KHÔNG học gì từ dữ liệu nên chạy trước khi chia tập
    là an toàn. Xem docstring đầu file để biết những bước nào cố ý bị hoãn.
    """
    steps: list[CleaningStep] = []

    # 1. Bỏ dòng trùng khoá hồ sơ — BẮT BUỘC trước khi chia tập.
    before = df
    duplicates = duplicate_id_mask(df, ID_COLUMN)
    out = df.loc[~duplicates].reset_index(drop=True)
    steps.append(_step(
        "drop_duplicate_customers",
        f"Bỏ dòng trùng {ID_COLUMN}",
        before, out,
        f"{int(duplicates.sum())} dòng trùng. Phải bỏ TRƯỚC khi chia tập: cùng "
        "một khách ở cả train lẫn test là rò rỉ theo nghĩa đen.",
    ))

    # 2. Sentinel + chuỗi giả → NaN, kèm 6 cờ `_MISSING`.
    before = out
    out = normalize_missing(out, add_flags=True)
    flags = [f.name for f in MISSING_FLAGS if f.name in out.columns]
    steps.append(_step(
        "normalize_missing",
        "Sentinel và chuỗi giả → NaN, sinh cờ nhị phân",
        before, out,
        f"DAYS_EMPLOYED=365243 và 'XNA'/'XAP'/'Unknown' → NaN. Sinh {len(flags)} "
        f"cờ: {', '.join(flags)}. Giữ cờ vì ở dataset này việc THIẾU dữ liệu tự "
        "nó dự báo được vỡ nợ (lift 0,624–1,339).",
    ))

    # 3. Chuẩn hoá kiểu.
    before = out
    issues = dtype_issues(out)
    out = normalize_dtypes(out)
    steps.append(_step(
        "normalize_dtypes",
        "Cột chuỗi → category",
        before, out,
        f"{len(issues)} cột chuỗi thực chất chứa số (0 = không có cột nào bị "
        "đọc sai kiểu).",
    ))

    # 4. Gắn cờ dòng bất hợp lệ — KHÔNG bỏ. Xem docstring đầu file.
    before = out
    invalid = invalid_mask(out, INVALID_RULES)
    out = out.assign(**{INVALID_ROW_FLAG: invalid.astype(int).to_numpy()})
    steps.append(_step(
        "flag_invalid_rows",
        f"Gắn cờ `{INVALID_ROW_FLAG}`, không bỏ dòng",
        before, out,
        f"{int(invalid.sum())} dòng vi phạm ít nhất một quy tắc. Chỉ gắn cờ: bỏ "
        "trước khi chia tập thì tập test cũng sạch theo, và chỉ số sẽ đẹp hơn "
        "thực tế. Task 3 bỏ chúng khỏi RIÊNG tập train.",
    ))

    return out, steps


def clean_bureau(df: pd.DataFrame) -> tuple[pd.DataFrame, list[CleaningStep]]:
    """Làm sạch `bureau.csv` ở mức từng dòng.

    ⚠️ **Không bỏ dòng trùng nội dung.** Có 2.059 dòng giống hệt nhau ở 16/17
    cột (khác mỗi `SK_ID_BUREAU`), thuộc 1.865 khách hàng. Chúng KHÔNG phải
    lỗi: `SK_ID_BUREAU` không trùng dòng nào, nên đó là những khoản vay riêng
    biệt tình cờ cùng số tiền và cùng ngày — chuyện bình thường với hai khoản
    tiêu dùng nhỏ mở cùng lúc.

    Bỏ chúng sẽ làm `previous_loan_count` (ô "Số khoản vay trước đây" của
    form) đếm thiếu. Đây đúng là cái bẫy PLAN.md §4.3c đã ghi: khử trùng theo
    toàn bộ dòng trên một bộ cột rút gọn thì xoá nhầm bản ghi hợp lệ.
    """
    steps: list[CleaningStep] = []

    # 1. Bỏ dòng có thông tin đến SAU thời điểm nộp đơn.
    before = df
    future = pd.Series(False, index=df.index)
    for column in BUREAU_PAST_ONLY_COLUMNS:
        if column in df.columns:
            future |= df[column] > 0
    out = df.loc[~future].reset_index(drop=True)
    steps.append(_step(
        "drop_future_information",
        "Bỏ bản ghi có cột thời gian > 0",
        before, out,
        f"{int(future.sum())} dòng có thông tin cập nhật SAU ngày nộp đơn — "
        "thứ mà lúc chạy thật hệ thống không thể biết. "
        "`DAYS_CREDIT_ENDDATE` được miễn: ngày kết thúc dự kiến của khoản vay "
        "còn hiệu lực đã biết ngay lúc ký, dương là bình thường.",
    ))

    # 2. Chuẩn hoá kiểu.
    before = out
    out = normalize_dtypes(out)
    steps.append(_step(
        "normalize_dtypes",
        "Cột chuỗi → category",
        before, out,
    ))

    # 3. Kiểm trùng, KHÔNG bỏ — xem docstring.
    before = out
    content_duplicates = int(
        out.drop(columns=["SK_ID_BUREAU"], errors="ignore").duplicated().sum())
    steps.append(_step(
        "keep_content_duplicates",
        "Giữ nguyên bản ghi trùng nội dung",
        before, out,
        f"{content_duplicates} dòng trùng ở mọi cột trừ SK_ID_BUREAU. Giữ lại: "
        "chúng là khoản vay riêng biệt, bỏ đi sẽ đếm thiếu số khoản vay trước đây.",
    ))

    return out, steps


# --------------------------------------------------------------------------
# Chạy toàn bộ + xuất kết quả
# --------------------------------------------------------------------------
class LeakageAuditFailed(RuntimeError):
    """Có phép kiểm rò rỉ trượt — dừng lại, không ghi dữ liệu bẩn ra đĩa."""


#: Nơi đặt dữ liệu đã làm sạch. `data/interim` đúng nghĩa: đã hết bẩn nhưng
#: chưa dựng feature. Không commit vào git.
OUTPUT_SUBDIR: Final[str] = "ml02"

#: Nén gzip thay vì parquet để khỏi thêm phụ thuộc `pyarrow`. Đọc lại chậm
#: hơn nhưng mở bằng bất cứ công cụ nào cũng được, và task 3 chỉ đọc một lần.
APPLICATION_FILE: Final[str] = "application_clean.csv.gz"
BUREAU_FILE: Final[str] = "bureau_clean.csv.gz"
METADATA_FILE: Final[str] = "cleaning_metadata.json"


def output_dir() -> Path:
    return CONFIG.paths.data_interim / OUTPUT_SUBDIR


def build_clean_dataset(
    nrows: int | None = None,
    with_bureau: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame | None, dict[str, CleaningReport]]:
    """Chạy đủ task 2 và trả về `(application sạch, bureau sạch, báo cáo)`.

    Ném `LeakageAuditFailed` nếu bất kỳ phép kiểm rò rỉ nào trượt. Ghi dữ liệu
    đã biết là rò rỉ ra đĩa còn tệ hơn không ghi gì: task 3 sẽ dùng nó mà
    không hỏi lại, và chỉ số cuối cùng sẽ đẹp một cách vô nghĩa.
    """
    app_raw = loader.load_application_train(nrows=nrows)
    app, app_steps = clean_application(app_raw)

    app_report = CleaningReport(
        table="application_train.csv",
        steps=app_steps,
        leakage=leakage_audit(app),
        dtypes=dtype_report(app),
        validation=validation_report(app, INVALID_RULES),
        missing=(app.isna().mean().rename("missing_rate")
                 .reset_index().rename(columns={"index": "column"})
                 .sort_values("missing_rate", ascending=False, ignore_index=True)),
    )

    if not app_report.passed_leakage_audit:
        failed = app_report.leakage[~app_report.leakage["passed"]]
        raise LeakageAuditFailed(
            "Kiểm toán rò rỉ trượt, không ghi dữ liệu:\n"
            + failed.to_string(index=False))

    reports = {"application": app_report}
    bureau: pd.DataFrame | None = None

    if with_bureau:
        # KHÔNG truyền `nrows` xuống bureau: file là quan hệ một-nhiều, cắt
        # theo dòng đầu chỉ ra một nhúm khách hàng và mọi hồ sơ còn lại bị coi
        # như chưa từng vay.
        bureau_raw = loader.load_raw("bureau")
        bureau, bureau_steps = clean_bureau(bureau_raw)
        reports["bureau"] = CleaningReport(
            table="bureau.csv",
            steps=bureau_steps,
            future_information=bureau_future_information(bureau_raw),
            dtypes=dtype_report(bureau),
        )

    return app, bureau, reports


def build_metadata(
    app: pd.DataFrame,
    bureau: pd.DataFrame | None,
    reports: dict[str, CleaningReport],
) -> dict:
    """Metadata để task 3 biết chính xác nó đang nhận cái gì.

    Có `feature_columns` tường minh: task 3 đọc danh sách này thay vì tự suy
    ra bằng cách bỏ vài cột nó nhớ được — nhớ sót một cột là rò rỉ.
    """
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "task": "ML02 task 2 — Data Cleaning",
        "random_seed": CONFIG.random_seed,
        "application": {
            "file": APPLICATION_FILE,
            "n_rows": len(app),
            "n_cols": app.shape[1],
            "target_column": TARGET_COLUMN,
            "id_column": ID_COLUMN,
            "non_feature_columns": sorted(NON_FEATURE_COLUMNS),
            "feature_columns": feature_columns(app),
            "missing_flags": [f.name for f in MISSING_FLAGS if f.name in app.columns],
            "n_invalid_rows_flagged": int(app[INVALID_ROW_FLAG].sum()),
            "positive_rate": float(app[TARGET_COLUMN].mean()),
        },
        "bureau": None if bureau is None else {
            "file": BUREAU_FILE,
            "n_rows": len(bureau),
            "n_cols": bureau.shape[1],
        },
        "steps": {
            name: [asdict(s) for s in report.steps]
            for name, report in reports.items()
        },
        "leakage_audit_passed": all(r.passed_leakage_audit for r in reports.values()),
        # Nhắc lại phần CHƯA làm, ngay trong metadata: ai đọc file này để dùng
        # dữ liệu thì cũng đọc luôn được là còn thiếu gì.
        "pipeline_steps_remaining": [
            {"step": name, "learns": what} for name, what in PIPELINE_STEPS_REMAINING
        ],
    }


def write_outputs(
    app: pd.DataFrame,
    bureau: pd.DataFrame | None,
    reports: dict[str, CleaningReport],
) -> dict[str, Path]:
    """Ghi dữ liệu sạch + metadata + các bảng báo cáo."""
    out_dir = output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = CONFIG.paths.runs / "ml02_cleaning"
    runs_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}

    app_path = out_dir / APPLICATION_FILE
    app.to_csv(app_path, index=False, compression="gzip", encoding="utf-8")
    written["application"] = app_path

    if bureau is not None:
        bureau_path = out_dir / BUREAU_FILE
        bureau.to_csv(bureau_path, index=False, compression="gzip", encoding="utf-8")
        written["bureau"] = bureau_path

    metadata_path = out_dir / METADATA_FILE
    metadata_path.write_text(
        json.dumps(build_metadata(app, bureau, reports), ensure_ascii=False, indent=2),
        encoding="utf-8")
    written["metadata"] = metadata_path

    for name, report in reports.items():
        for label, table in (("steps", report.steps_frame()),
                             ("leakage", report.leakage),
                             ("future_information", report.future_information),
                             ("dtypes", report.dtypes),
                             ("validation", report.validation),
                             ("missing", report.missing)):
            if table is None or table.empty:
                continue
            path = runs_dir / f"{name}_{label}.csv"
            table.to_csv(path, index=False, encoding="utf-8")
            written[f"{name}_{label}"] = path

    log.info("Đã ghi %d file → %s và %s", len(written), out_dir, runs_dir)
    return written


def load_clean_application() -> pd.DataFrame:
    """Nạp lại `application` đã làm sạch — điểm vào của task 3."""
    path = output_dir() / APPLICATION_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"Chưa có {path}. Chạy `python scripts/clean_ml02.py` trước.")
    return pd.read_csv(path, compression="gzip", low_memory=False)


def load_clean_bureau() -> pd.DataFrame:
    path = output_dir() / BUREAU_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"Chưa có {path}. Chạy `python scripts/clean_ml02.py` trước.")
    return pd.read_csv(path, compression="gzip", low_memory=False)


def load_metadata() -> dict:
    path = output_dir() / METADATA_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"Chưa có {path}. Chạy `python scripts/clean_ml02.py` trước.")
    return json.loads(path.read_text(encoding="utf-8"))
