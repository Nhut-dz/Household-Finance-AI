"""Kết xuất báo cáo khám phá ML02 ra `docs/ml02_eda.md` (F04 task 1).

Tách khỏi `explore.py` theo đúng ranh giới của ML01: `explore` ĐO, `report`
DIỄN ĐẠT. Nhờ vậy sửa câu chữ báo cáo không đụng vào phép đo, và ngược lại —
chỗ dễ vô tình làm số liệu đổi theo cách viết.

File sinh ra CÓ commit vào git (khác dataset), vì nó là căn cứ cho mục "phân
tích tính khả thi triển khai" của báo cáo cuối và cho bảng Full vs Rút gọn.
"""
from __future__ import annotations

import pandas as pd

from hfml.config import CONFIG
from hfml.logger import get_logger
from hfml.ml.ml02_credit_risk.explore import ExplorationReport, IV_BANDS

log = get_logger(__name__)

DOC_FILENAME = "ml02_eda.md"

#: Số dòng hiện trong bảng xếp hạng IV của báo cáo. Bảng đầy đủ 121 dòng nằm
#: ở `iv_ranking.csv`; nhồi cả vào markdown thì không ai đọc hết.
TOP_N = 20


def _table(frame: pd.DataFrame, columns: dict[str, str], digits: int = 4) -> str:
    """Đổi DataFrame thành bảng markdown. `columns` là {tên cột: nhãn hiển thị}."""
    present = {c: label for c, label in columns.items() if c in frame.columns}
    lines = [
        "| " + " | ".join(present.values()) + " |",
        "|" + "|".join("---" for _ in present) + "|",
    ]
    for _, row in frame.iterrows():
        cells = []
        for column in present:
            value = row[column]
            if isinstance(value, float):
                cells.append("—" if pd.isna(value) else f"{value:,.{digits}f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _target_section(target: dict) -> list[str]:
    return [
        "## 1. Nhãn `TARGET` — điểm xuất phát của mọi quyết định sau đó",
        "",
        "| | |",
        "|---|---:|",
        f"| Số hồ sơ | {target['n_rows']:,} |",
        f"| `TARGET = 1` (khó khăn trả nợ) | {target['n_positive']:,} |",
        f"| Tỉ lệ dương | {target['positive_rate']:.4%} |",
        f"| `scale_pos_weight` | {target['scale_pos_weight']:.2f} |",
        f"| Accuracy của model đoán toàn `0` | {target['majority_class_accuracy']:.4%} |",
        "",
        "Dòng cuối là lý do **chọn model bằng PR-AUC chứ không phải accuracy**: một "
        "model không học gì đã đạt hơn 91%. Accuracy vẫn báo cáo cho đủ bộ chỉ số, "
        "nhưng không được dùng để kết luận (PLAN.md §7.3).",
        "",
    ]


def _iv_section(report: ExplorationReport) -> list[str]:
    ranking = report.iv_ranking
    bands = ranking["band"].value_counts()

    lines = [
        "## 2. Sức mạnh dự báo của từng cột — Information Value",
        "",
        "Thước đo là **IV (Information Value)** chứ không phải tương quan Pearson. "
        "Lý do đã đo được: trên chính dataset này, |r| tuyến tính cao nhất của mọi "
        "cột số chỉ **0,179** — tin vào Pearson thì kết luận nhầm là không cột nào "
        "có ích. IV còn so được cột số với cột hạng mục trên cùng một thang, và "
        "coi `NaN` là một khoảng riêng nên không vứt mất tín hiệu của việc thiếu "
        "dữ liệu.",
        "",
        "Thang diễn giải (quy ước ngành chấm điểm tín dụng):",
        "",
        "| IV | Mức | Số cột |",
        "|---|---|---:|",
    ]
    for threshold, name in IV_BANDS:
        lines.append(f"| ≥ {threshold:.2f} | {name} | {int(bands.get(name, 0))} |")

    lines += [
        "",
        f"### {TOP_N} cột mạnh nhất",
        "",
        _table(ranking.head(TOP_N), {
            "column": "Cột",
            "dtype": "Kiểu",
            "missing_rate": "Thiếu",
            "iv": "IV",
            "band": "Mức",
            "max_lift": "lift xấu nhất",
        }),
        "",
        f"Bảng đầy đủ {len(ranking)} cột: `src/training/runs/ml02_eda/iv_ranking.csv`.",
        "",
    ]
    lines += _lift_section(ranking)
    return lines


def _lift_section(ranking: pd.DataFrame) -> list[str]:
    """Cột có nhóm rủi ro rất cao nhưng IV thấp vì nhóm đó nhỏ.

    Đây là phần dễ đọc sai nhất của cả bảng, nên tách hẳn ra một mục thay vì
    để lẫn trong bảng xếp hạng.
    """
    sneaky = (ranking[(ranking["iv"] < 0.02) & (ranking["max_lift"] >= 1.5)]
              .sort_values("max_lift", ascending=False)
              .head(10))
    if sneaky.empty:
        return []

    return [
        "### ⚠️ IV thấp **không** đồng nghĩa vô dụng",
        "",
        "IV cân theo tỉ trọng dân số, nên một cột chỉ bật ở nhóm nhỏ sẽ có IV "
        "thấp dù nhóm đó rủi ro rất cao. Hai con số đo hai thứ khác nhau:",
        "",
        "| | Trả lời câu hỏi |",
        "|---|---|",
        "| **IV** | cột này giúp phân biệt được bao nhiêu hồ sơ trong TOÀN BỘ danh mục |",
        "| **lift xấu nhất** | khi cột này bật, hồ sơ đó nguy hiểm tới đâu |",
        "",
        "Các cột bị xếp mức thấp theo IV nhưng có nhóm rủi ro gấp rưỡi trở lên:",
        "",
        _table(sneaky, {
            "column": "Cột",
            "worst_bin": "Khoảng xấu nhất",
            "iv": "IV",
            "band": "Mức theo IV",
            "max_lift": "lift",
        }),
        "",
        "Loại các cột này chỉ vì IV thấp là loại đúng tín hiệu mạnh nhất trên "
        "nhóm mà hệ thống cần cảnh báo nhất. Quyết định giữ/bỏ để cho bước "
        "feature selection có giám sát trong Pipeline làm (F01 task 13), không "
        "cắt tay theo một ngưỡng IV.",
        "",
    ]


def _coverage_section(report: ExplorationReport) -> list[str]:
    coverage = report.form_coverage
    mapped = coverage[coverage["home_credit"] != "—"]
    unmapped = coverage[coverage["home_credit"] == "—"]

    total_iv = float(report.iv_ranking["iv"].sum())
    reachable_iv = float(mapped["iv"].sum(skipna=True))

    return [
        "## 3. Form lấy được bao nhiêu — căn cứ của bảng Full vs Rút gọn (§7.2)",
        "",
        f"Sau khi màn **Thông tin khoản vay** lên (15/08/2026), form thu được "
        f"**{len(coverage)} trường**, trong đó **{len(mapped)}** ánh xạ được sang "
        f"một cột Home Credit và **{len(unmapped)}** thì không.",
        "",
        _table(coverage, {
            "field": "Trường form",
            "source": "Màn",
            "home_credit": "Cột Home Credit",
            "iv": "IV",
            "band": "Mức",
            "note": "Ghi chú",
        }),
        "",
        f"Tổng IV của các cột form lấy được: **{reachable_iv:.4f}** trên tổng "
        f"**{total_iv:.4f}** của toàn bộ dataset "
        f"(**{reachable_iv / total_iv:.1%}**).",
        "",
        "> Con số đó là **chỉ dấu, không phải kết luận**. IV cộng dồn giữa các cột "
        "> tương quan với nhau sẽ đếm trùng cùng một lượng thông tin, nên "
        "> \"giữ được 40% IV\" không có nghĩa là \"giữ được 40% năng lực dự báo\". "
        "> Con số dùng để kết luận là **PR-AUC của hai model train thật**, và đó "
        "> chính là việc của task tiếp theo.",
        "",
        "### Những cột mạnh nhất mà form KHÔNG lấy được",
        "",
        "Đây là cái giá của bộ Rút gọn, phải nêu đích danh trong báo cáo — nói "
        "\"bộ rút gọn kém hơn\" mà không chỉ ra kém vì mất gì thì hội đồng hỏi ngay.",
        "",
        _table(report.unreachable.head(10), {
            "column": "Cột",
            "dtype": "Kiểu",
            "missing_rate": "Thiếu",
            "iv": "IV",
            "band": "Mức",
        }),
        "",
    ]


def _bureau_section(report: ExplorationReport) -> list[str]:
    if not report.bureau_coverage:
        return []

    cov = report.bureau_coverage
    return [
        "## 4. `bureau.csv` — nguồn của mục C trên form",
        "",
        f"`bureau.csv` có **{cov['n_bureau_rows']:,} khoản vay** của "
        f"**{cov['n_customers_with_record']:,} khách hàng**. Gộp về một dòng mỗi "
        "khách thì ra đúng bốn ô mục C mà form đang hỏi.",
        "",
        "| | |",
        "|---|---:|",
        f"| Hồ sơ KHÔNG có bản ghi bureau | {cov['n_customers_without_record']:,} "
        f"({cov['share_without_record']:.2%}) |",
        f"| Vỡ nợ khi KHÔNG có bản ghi | {cov['default_rate_without_record']:.4%} |",
        f"| Vỡ nợ khi CÓ bản ghi | {cov['default_rate_with_record']:.4%} |",
        "",
        "Nhóm không có bản ghi được điền **0 chứ không phải NaN**: không tìm thấy "
        "gì ở trung tâm tín dụng nghĩa là *chưa từng vay*, đúng bằng câu trả lời "
        "`previous_loan_count = 0` mà form cho phép chọn. Impute trung vị vào đây "
        "là gán cho người chưa từng vay một lịch sử tín dụng trung bình mà họ "
        "không có. Riêng `BUREAU_HISTORY_YEARS` giữ `NaN` — số năm có lịch sử tín "
        "dụng của người chưa từng vay không phải 0 năm, nó **không tồn tại**.",
        "",
        "### IV của phần tổng hợp bureau",
        "",
        _table(report.bureau_iv, {
            "column": "Cột tổng hợp",
            "missing_rate": "Thiếu",
            "iv": "IV",
            "band": "Mức",
            "max_lift": "lift xấu nhất",
        }),
        "",
        *_overdue_note(report),
        "⚠️ **Một chỗ lệch định nghĩa phải ghi vào `model_card.md`:** form hỏi "
        "*\"số lần trả chậm\"*, nhưng `bureau.csv` chỉ ghi trạng thái quá hạn "
        "**hiện tại** của từng khoản (`CREDIT_DAY_OVERDUE`), không ghi lịch sử "
        "từng kỳ. `BUREAU_OVERDUE_LOAN_COUNT` vì vậy là số **khoản** đang quá "
        "hạn, không phải số **lần** trả chậm. Hai đại lượng gần nhau nhưng không "
        "bằng nhau — đừng để nó thành một phép đồng nhất ngầm.",
        "",
    ]


def _overdue_note(report: ExplorationReport) -> list[str]:
    """Ví dụ cụ thể cho luận điểm "IV thấp ≠ vô dụng", đọc thẳng từ số đo.

    Cố tình KHÔNG chép tay ba con số vào câu văn: dataset đổi mà câu văn giữ
    nguyên thì báo cáo nói sai mà không ai biết — đúng loại lỗi mà file sinh
    tự động sinh ra để tránh.
    """
    table = report.woe_details.get("BUREAU_HAS_OVERDUE")
    if table is None or len(table) < 2:
        return []

    clean = table.iloc[table["bin"].astype(str).tolist().index("0.0")]
    overdue = table.iloc[table["bin"].astype(str).tolist().index("1.0")]

    return [
        "Đọc bảng này phải kèm mục 2.1: `BUREAU_HAS_OVERDUE` có IV rất thấp, "
        f"nhưng đó là vì chỉ **{overdue['share']:.2%}** hồ sơ đang có khoản quá "
        f"hạn — còn trong nhóm đó thì tỉ lệ vỡ nợ **{overdue['bad_rate']:.2%} so "
        f"với {clean['bad_rate']:.2%}**, tức gấp **{overdue['lift']:.2f} lần** "
        "trung bình. Đây đúng là loại tín hiệu mà mục C của form sinh ra để bắt.",
        "",
    ]


def _domain_section(report: ExplorationReport) -> list[str]:
    return [
        "## 5. Khoảng cách miền VNĐ ↔ Home Credit (§2.1)",
        "",
        "Home Credit không dùng VNĐ: `AMT_INCOME_TOTAL` trung vị ≈ 147.150, người "
        "dùng Việt Nam nhập 50.000.000 — lệch ~340 lần. Model gặp giá trị ngoài "
        "phân phối huấn luyện sẽ **trả về số vô nghĩa mà không báo lỗi**. Cách xử "
        "lý là bỏ hết giá trị tiền tuyệt đối, chỉ giữ feature **tỉ lệ** — và tỉ lệ "
        "thì bất biến với đơn vị tiền tệ.",
        "",
        "Phân phối các tỉ lệ đó trên chính `application_train.csv`:",
        "",
        _table(report.ratio_distribution, {
            "feature": "Feature",
            "formula": "Công thức",
            "missing_rate": "Thiếu",
            "p1": "p1",
            "p25": "p25",
            "p50": "trung vị",
            "p75": "p75",
            "p99": "p99",
        }),
        "",
        "Hai điều bảng này xác nhận, cả hai đều đã ghi ở PLAN.md §2.1b:",
        "",
        "1. **`credit_goods_markup` luôn ≥ 1,0** — Home Credit cộng phí và bảo hiểm "
        "vào `AMT_CREDIT`, nên tỉ lệ này đo **mức đội giá**, KHÔNG phải tỉ lệ vay "
        "trên tài sản. Nó không cùng đại lượng với `loan_amount / asset_price` của "
        "form (vay 70%, tự có 30%), nên hai thứ phải tách riêng.",
        "2. **`dti` trung vị ≈ 0,16** xác nhận `AMT_INCOME_TOTAL` và `AMT_ANNUITY` "
        "cùng kỳ. Nếu thu nhập theo NĂM mà kỳ trả theo THÁNG thì DTI sẽ là "
        "0,16 × 12 ≈ 196% — bất khả.",
        "",
    ]


def render(report: ExplorationReport) -> str:
    """Dựng nội dung markdown của `docs/ml02_eda.md`."""
    lines = [
        "# ML02 — Khám phá Home Credit Dataset (F04 task 1)",
        "",
        "> File này sinh tự động bởi `scripts/explore_ml02.py`. Đừng sửa tay.",
        "",
        "Bước này KHÁC với kiểm tra chất lượng dữ liệu ở F01 task 6. "
        "`docs/dataset.md` trả lời *\"dữ liệu sạch tới đâu\"*; file này trả lời "
        "*\"cột nào dự báo được vỡ nợ, và cột nào trong số đó form của mình lấy "
        "được\"* — câu hỏi quyết định kiến trúc hai phiên bản model của §7.2.",
        "",
    ]
    lines += _target_section(report.target)
    lines += _iv_section(report)
    lines += _coverage_section(report)
    lines += _bureau_section(report)
    lines += _domain_section(report)
    lines += [
        "## 6. Kết luận rút ra cho các task sau",
        "",
        "| # | Kết luận | Ảnh hưởng tới |",
        "|---|---|---|",
        "| 1 | Chọn model bằng PR-AUC, `scale_pos_weight` ≈ 11,39 | task 4, 11 |",
        "| 2 | `EXT_SOURCE_*` mạnh nhất nhưng form không lấy được | task 3 — bộ Full vs Rút gọn |",
        "| 3 | Phần tổng hợp bureau bổ sung được nhóm tín hiệu lịch sử tín dụng | task 2 — feature engineering |",
        "| 4 | Thiếu dữ liệu tự nó là tín hiệu → giữ cờ `_MISSING` | task 2 |",
        "| 5 | Chỉ dùng feature tỉ lệ, bỏ mọi giá trị tiền tuyệt đối | task 1, 2 |",
        "",
    ]
    return "\n".join(lines)


def write_doc(report: ExplorationReport):
    """Ghi `docs/ml02_eda.md`."""
    path = CONFIG.paths.docs / DOC_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(report), encoding="utf-8")
    log.info("Đã ghi báo cáo khám phá → %s", path)
    return path
