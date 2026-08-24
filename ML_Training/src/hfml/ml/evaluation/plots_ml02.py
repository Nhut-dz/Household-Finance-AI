"""Biểu đồ cho báo cáo ML02 (F04).

Sáu hình, mỗi hình đọc từ kết quả ĐÃ CÓ trong `src/training/runs/ml02_*/` —
không train lại, không tính lại chỉ số, không chạm tập test:

    ml02_evaluation/precision_recall_curve.png   curves.csv · metrics.csv   (task 11)
    ml02_evaluation/roc_curve.png                curves.csv · metrics.csv   (task 11)
    ml02_evaluation/threshold_analysis.png       threshold_sweep.csv        (task 11)
    ml02_comparison/model_comparison.png         comparison.csv
                                                 confidence_interval.csv    (task 12)
    ml02_importance/feature_importance.png       builtin/permutation/shap   (task 13)
    ml02_selection/confusion_matrix_test.png     test_confusion.csv         (task 14)

Đối xứng với `evaluation/plots.py` (ML01) và dùng chung bộ style của nó. Khác
ở chỗ ML02 chia kết quả theo thư mục con từng task, nên mỗi hàm vẽ ghi vào
ĐÚNG thư mục chứa CSV đã sinh ra nó — hình và bảng số của nó luôn nằm cạnh
nhau, không có chuyện đọc hình của task này bên cạnh bảng của task khác.

Vì sao mỗi hình chọn dạng đó
-----------------------------
**PR curve** — đường, hai panel (rút gọn · full). PR-AUC là chỉ số CHỌN MODEL
của ML02 (§7.3) nên đây là hình chính. Đường ngang mốc = tỉ lệ nền 8,07%, tức
mức của model đoán bừa; không có mốc đó thì PR-AUC 0,17 trông như thảm hoạ,
trong khi nó là gấp 2,1 lần mức bừa.

**ROC curve** — cũng đường, hai panel, kèm đường chéo may rủi. Có mặt vì
người đọc quen ROC, NHƯNG tiêu đề phụ nói thẳng nó lạc quan với dữ liệu lệch
8/92: phần lớn diện tích dưới ROC đến từ vùng ngưỡng không ai vận hành.

**Threshold analysis** — hai hàng. Hàng trên: F1 lớp dương theo ngưỡng cho cả
4 thuật toán, chấm đánh dấu đỉnh. Hàng dưới: precision và recall của model
dẫn đầu, để thấy cái giá của việc dịch ngưỡng. Tách hai hàng chứ không chồng
lên một trục: F1 và cặp precision/recall trả lời hai câu khác nhau, gộp vào
một panel thì 12 đường không đọc được.

**So sánh model** — chấm kèm thanh khoảng tin cậy, KHÔNG phải cột. Task 12 đo
bootstrap chính là để trả lời "chênh lệch này có thật không", nên hình phải
vẽ được khoảng tin cậy — cột thì không có chỗ đặt nó. Hai panel dùng CHUNG
dải trục x: khoảng cách rút gọn ↔ full là cái giá của việc form không thu
được EXT_SOURCE_1/2/3 (§7.2), và chỉ chung trục mới nhìn ra.

**Feature importance** — ba panel small multiples, mỗi cách đo một panel với
TRỤC X RIÊNG theo đơn vị gốc của nó. Không quy về một thang: built-in là tỉ
trọng impurity (tổng = 1), permutation là mức PR-AUC TỤT khi xáo cột, SHAP là
trung bình |giá trị| — ba đại lượng khác nhau, vẽ chung một trục là so sánh
sai. Thứ tự hàng dùng chung, nên chỗ ba cách đo BẤT ĐỒNG hiện ra ngay ở việc
cột dài ngắn lệch nhau giữa các panel.

**Confusion matrix test** — heatmap một màu, tô theo tỉ lệ HÀNG, ghi số đếm
thô. Một panel thôi vì tập test chỉ được mở ĐÚNG MỘT LẦN cho MỘT model đã
chốt (task 14) — vẽ 8 model trên test như ML01 là chuyện ML02 cố ý không làm.
Ô `false_negative` được chú thích riêng: đó là ca vỡ nợ bị bỏ lọt, đắt nhất
trong bài toán này.

Màu
---
`SERIES` của ML01 chỉ có 3 slot, mà ML02 có 4 thuật toán. Slot 4 của bảng màu
gốc là yellow — KHÔNG dùng được: đặt cạnh orange thì rớt sàn thị lực thường
(ΔE 13,7 < 15) ở chế độ `--pairs all`. Đã chạy validator trên cả bốn ứng viên
còn lại và chốt violet:

    node scripts/validate_palette.js "#2a78d6,#eb6834,#1baf7a,#4a3aa7"
         --mode light --surface "#fcfcfb" --pairs all
    → ALL CHECKS PASS (CVD ΔE 9,2 · thị lực thường ΔE 16,3)

Dùng `--pairs all` chứ không phải pairlist kề nhau vì các đường PR/ROC CẮT
NHAU: cặp nào cũng có thể thành cặp phải phân biệt, không chỉ cặp đứng cạnh.

Aqua và violet có contrast dưới 3:1 trên nền `#fcfcfb`, nên theo *relief rule*
phải kèm nhãn hiện hoặc table view — table view chính là các file CSV nằm
cùng thư mục với hình.

Màu gắn theo THUẬT TOÁN, không theo thứ hạng (`ALGO_COLOR`). Lọc bớt model
hay đổi bảng xếp hạng đều không được phép sơn lại các model còn lại.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import matplotlib

# Backend không cửa sổ — script và test chạy không có màn hình.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.transforms as mtransforms  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from hfml.config import CONFIG  # noqa: E402
from hfml.logger import get_logger  # noqa: E402

# Dùng lại nguyên style của ML01 thay vì chép sang đây. `_style` và `_figure`
# có gạch dưới nhưng cùng package `hfml.ml.evaluation`, và chép 20 dòng hằng
# số màu sang file thứ hai là cách chắc chắn để hai bản trôi khỏi nhau — đúng
# lỗi mà chính docstring của `plots.py` cảnh báo.
from hfml.ml.evaluation.plots import (  # noqa: E402
    AXIS,
    BLUES,
    DPI,
    GRID,
    INK,
    INK_SECONDARY,
    MUTED,
    SURFACE,
    _figure,
    _style,
)

log = get_logger(__name__)

#: 4 slot categorical, đã qua `validate_palette.js --pairs all` (xem docstring).
SERIES_ML02: Final[tuple[str, ...]] = ("#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7")

#: Thứ tự thuật toán theo task 7 → 10. Quyết định cả màu lẫn thứ tự legend.
ALGO_ORDER: Final[tuple[str, ...]] = (
    "decision_tree", "bagging", "random_forest", "xgboost")

#: Thuật toán → màu, CỐ ĐỊNH. Không sinh theo vòng lặp trên dữ liệu đang vẽ:
#: làm vậy thì bỏ một model khỏi bảng là ba model còn lại đổi màu hết.
ALGO_COLOR: Final[dict[str, str]] = dict(zip(ALGO_ORDER, SERIES_ML02))

#: Tên hiển thị — `decision_tree` trong hình đọc gãy hơn `Decision Tree`.
ALGO_LABEL: Final[dict[str, str]] = {
    "decision_tree": "Decision Tree",
    "bagging": "Bagging",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}

#: Bộ feature nào vẽ trước. Rút gọn đứng trước vì nó là bộ TRIỂN KHAI ĐƯỢC
#: (§7.2); full chỉ để tham chiếu, nên đứng sau.
FEATURE_SET_ORDER: Final[tuple[str, ...]] = ("reduced", "full")

FEATURE_SET_LABEL: Final[dict[str, str]] = {
    "reduced": "bộ RÚT GỌN — triển khai được",
    "full": "bộ FULL — tham chiếu, form không thu được EXT_SOURCE",
}

EVAL_SUBDIR: Final[str] = "ml02_evaluation"
COMPARE_SUBDIR: Final[str] = "ml02_comparison"
IMPORTANCE_SUBDIR: Final[str] = "ml02_importance"
SELECTION_SUBDIR: Final[str] = "ml02_selection"

#: Bộ feature mặc định khi hình chỉ vẽ được một model. Không phải "bộ thắng"
#: mà là bộ DUY NHẤT deploy được — quyết định này thuộc §7.2, không phụ thuộc
#: lần train nào thắng.
DEPLOY_FEATURE_SET: Final[str] = "reduced"


def _resolve(runs_dir, subdir: str, name: str) -> tuple[Path, Path]:
    """`(thư mục con, đường dẫn file)` — báo lỗi rõ nếu chưa chạy task sinh ra nó."""
    root = CONFIG.paths.runs if runs_dir is None else Path(runs_dir)
    out_dir = root / subdir
    path = out_dir / name
    if not path.exists():
        raise FileNotFoundError(
            f"Chưa có {path}. Biểu đồ đọc từ kết quả đã lưu — chạy task sinh "
            "ra file này trước, hàm vẽ không tự tính lại.")
    return out_dir, path


def _optional(runs_dir, subdir: str, name: str) -> Path | None:
    """Như `_resolve` nhưng trả `None` khi thiếu — cho phần không bắt buộc."""
    root = CONFIG.paths.runs if runs_dir is None else Path(runs_dir)
    path = root / subdir / name
    return path if path.exists() else None


def _algos_in(frame: pd.DataFrame) -> list[str]:
    """Thuật toán có trong bảng, theo `ALGO_ORDER`; tên lạ xếp cuối."""
    present = set(frame["algo"])
    known = [a for a in ALGO_ORDER if a in present]
    return known + sorted(present - set(known))


def _feature_sets_in(frame: pd.DataFrame) -> list[str]:
    present = set(frame["feature_set"])
    known = [f for f in FEATURE_SET_ORDER if f in present]
    return known + sorted(present - set(known))


def _color(algo: str) -> str:
    """Màu của một thuật toán. Tên ngoài danh sách thì lấy xám trung tính."""
    return ALGO_COLOR.get(algo, MUTED)


def _label(algo: str) -> str:
    return ALGO_LABEL.get(algo, algo)


def _base_rate(metrics: pd.DataFrame) -> float | None:
    """Tỉ lệ nền, suy từ `pr_auc / pr_auc_lift` — không cần đọc thêm file nào.

    `pr_auc_lift` được định nghĩa là PR-AUC chia cho tỉ lệ lớp dương, nên phép
    chia ngược lại trả đúng tỉ lệ đó. Lấy trung vị của 8 model để một dòng
    hỏng không kéo lệch đường mốc.
    """
    if not {"pr_auc", "pr_auc_lift"} <= set(metrics.columns):
        return None
    lift = metrics["pr_auc_lift"].replace(0, np.nan)
    rates = (metrics["pr_auc"] / lift).dropna()
    return float(rates.median()) if len(rates) else None


def _deployed_threshold(runs_dir) -> float | None:
    """Ngưỡng đã chốt ở task 14, nếu đã chạy task đó.

    Trả `None` khi chưa có `decision.json` — hình threshold vẫn vẽ được, chỉ
    thiếu đường dọc đánh dấu. Vẽ hình task 11 mà bắt buộc phải có kết quả
    task 14 là dựng ngược thứ tự phụ thuộc của F04.
    """
    path = _optional(runs_dir, SELECTION_SUBDIR, "decision.json")
    if path is None:
        return None
    try:
        decision = json.loads(path.read_text(encoding="utf-8"))
        return float(decision["threshold"]["value"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        # File hỏng hoặc đổi cấu trúc thì bỏ phần trang trí, không làm hỏng hình.
        log.warning("Không đọc được ngưỡng từ %s — bỏ qua đường đánh dấu", path)
        return None


def _save(fig, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    log.info("Ghi biểu đồ → %s", out)
    return out


def _legend(ax, **kwargs):
    """Legend không khung, chữ dùng mực phụ chứ không mượn màu của series."""
    legend = ax.legend(frameon=False, **kwargs)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)
    return legend


def _vi(value: float, digits: int = 4) -> str:
    """Số theo kiểu Việt — dấu phẩy thập phân. Dùng cho câu chữ, không cho bảng.

    Bảng và legend giữ dấu chấm để khớp với `plots.py` của ML01; riêng phần
    đọc thành câu ("tỉ lệ nền 8,07%") thì dấu chấm đọc sai nghĩa.
    """
    return f"{value:.{digits}f}".replace(".", ",")


def _vi_pct(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f}".replace(".", ",") + "%"


def _titles(fig, title: str, subtitle: str | None = None,
            note: str | None = None, note_color: str = MUTED,
            x: float = 0.005) -> None:
    """Xếp tiêu đề · phụ đề · ghi chú thành một khối trên đỉnh hình.

    Không dùng `fig.suptitle`: `suptitle` nhận `y` theo TỈ LỆ chiều cao hình,
    nên cùng một khoảng cách 0,045 là 16pt ở hình cao 5 inch và 27pt ở hình
    cao 8 inch. Hình cao thì thưa, hình thấp thì tiêu đề ĐÈ LÊN phụ đề — đúng
    lỗi đã thấy ở cả sáu hình lần vẽ đầu.

    Ở đây khoảng cách tính bằng POINT rồi mới quy ra tỉ lệ, nên mọi hình cách
    nhau đúng một lượng như nhau bất kể cao bao nhiêu. Xếp từ dưới lên vì
    `va="bottom"` neo theo đáy dòng, và dòng dưới cùng phải nằm sát mép hình.
    """
    height = fig.get_figheight()

    def to_fraction(points: float) -> float:
        return points / 72.0 / height

    def stack(text: str, size: float, color: str, y: float) -> float:
        fig.text(x, y, text, ha="left", va="bottom", fontsize=size, color=color)
        lines = text.count("\n") + 1
        return y + to_fraction(size * 1.32 * lines + 5)

    y = 1.0 + to_fraction(4)
    if note:
        y = stack(note, 8.5, note_color, y)
    if subtitle:
        y = stack(subtitle, 8.5, MUTED, y)
    fig.text(x, y, title, ha="left", va="bottom", fontsize=13, color=INK)


def _panel_title(ax, feature_set: str) -> None:
    ax.set_title(FEATURE_SET_LABEL.get(feature_set, feature_set),
                 color=INK, fontsize=10.5, pad=10, loc="left")


# ------------------------------------------------------------ PR curve

def plot_precision_recall_curve(runs_dir=None, out=None):
    """Precision–Recall của 8 model trên validation (task 11).

    Hình CHÍNH của ML02: PR-AUC là chỉ số chọn model (§7.3). Đường ngang là
    tỉ lệ nền — mức mà một model đoán bừa đạt được — nên khoảng cách từ đường
    cong xuống đường đó mới là phần model thực sự đóng góp.
    """
    out_dir, path = _resolve(runs_dir, EVAL_SUBDIR, "curves.csv")
    _, metrics_path = _resolve(runs_dir, EVAL_SUBDIR, "metrics.csv")
    curves = pd.read_csv(path)
    curves = curves[curves["curve"] == "pr"]
    if curves.empty:
        raise ValueError(f"{path} không có dòng nào với curve == 'pr'.")
    metrics = pd.read_csv(metrics_path).set_index(["algo", "feature_set"])
    base_rate = _base_rate(pd.read_csv(metrics_path))

    feature_sets = _feature_sets_in(curves)

    # Trục y cắt trên. Ở recall → 0 precision vọt lên 1,0 vì mẫu số chỉ còn
    # vài hồ sơ — giữ nguyên dải 0–1 thì toàn bộ phần đọc được (0,08–0,35) bị
    # ép vào một phần ba dưới của panel. Cắt PHẦN TRÊN chứ không cắt gốc 0:
    # gốc 0 và đường tỉ lệ nền là hai mốc phải giữ để so độ lớn cho đúng.
    settled = curves[curves["x"] >= 0.05]["y"]
    ceiling = min(1.0, float(settled.max()) * 1.15) if len(settled) else 1.0

    fig, axes = plt.subplots(1, len(feature_sets),
                             figsize=(6.2 * len(feature_sets), 5.0), dpi=DPI,
                             sharey=True)
    fig.patch.set_facecolor(SURFACE)
    axes = np.atleast_1d(axes).ravel()

    for ax, feature_set in zip(axes, feature_sets):
        block = curves[curves["feature_set"] == feature_set]
        for algo in _algos_in(block):
            line = block[block["algo"] == algo].sort_values("x")
            score = metrics.loc[(algo, feature_set), "pr_auc"] \
                if (algo, feature_set) in metrics.index else float("nan")
            # Giá trị PR-AUC nằm ngay trong legend: 8 đường cắt nhau nên nhãn
            # dán vào thân đường sẽ chồng, còn legend thì vừa là chú giải vừa
            # là bảng tra — nhãn không bao giờ chỉ dựa vào màu.
            suffix = "" if pd.isna(score) else f"  ·  {score:.4f}"
            ax.plot(line["x"], line["y"], color=_color(algo), linewidth=2,
                    zorder=3, solid_capstyle="round",
                    label=f"{_label(algo)}{suffix}")

        if base_rate is not None:
            ax.axhline(base_rate, color=AXIS, linewidth=1.4, zorder=2)
            # Nhãn dán bên TRÁI: ở recall cao mọi đường cong đều tụt về sát
            # đường này, dán bên phải là chữ nằm đè lên bốn đường một lúc.
            ax.annotate(f"đoán bừa = tỉ lệ nền {_vi_pct(base_rate)}",
                        xy=(0.015, base_rate), xycoords=("axes fraction", "data"),
                        textcoords="offset points", xytext=(0, 6),
                        ha="left", va="bottom", fontsize=8, color=MUTED)

        _style(ax)
        _panel_title(ax, feature_set)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, ceiling)
        ax.set_xlabel("recall — bắt được bao nhiêu ca vỡ nợ",
                      color=INK_SECONDARY, fontsize=9.5)
        _legend(ax, loc="upper right", fontsize=8.5, title="PR-AUC")
    axes[0].set_ylabel("precision — báo động đúng bao nhiêu phần",
                       color=INK_SECONDARY, fontsize=9.5)

    subtitle = ("PR-AUC là chỉ số chọn model (§7.3): với 8% lớp dương, "
                "accuracy và ROC đều không cầm lái được.")
    if ceiling < 1.0:
        subtitle += (f"\nTrục y cắt ở {_vi(ceiling, 2)} — đoạn recall → 0 vọt "
                     "lên 1,0 do mẫu số chỉ còn vài hồ sơ, không phải model tốt.")
    _titles(fig, "ML02 — Precision–Recall trên tập validation", subtitle)

    out = Path(out) if out else out_dir / "precision_recall_curve.png"
    return _save(fig, out)


# ----------------------------------------------------------- ROC curve

def plot_roc_curve(runs_dir=None, out=None):
    """ROC của 8 model trên validation (task 11).

    Kèm đường chéo may rủi. Tiêu đề phụ nói rõ ROC LẠC QUAN trên dữ liệu lệch
    8/92 — đưa hình này ra mà không kèm câu đó thì người đọc thấy AUC 0,76 và
    tưởng model đã tốt, trong khi PR-AUC cùng model chỉ 0,25.
    """
    out_dir, path = _resolve(runs_dir, EVAL_SUBDIR, "curves.csv")
    _, metrics_path = _resolve(runs_dir, EVAL_SUBDIR, "metrics.csv")
    curves = pd.read_csv(path)
    curves = curves[curves["curve"] == "roc"]
    if curves.empty:
        raise ValueError(f"{path} không có dòng nào với curve == 'roc'.")
    metrics = pd.read_csv(metrics_path).set_index(["algo", "feature_set"])

    feature_sets = _feature_sets_in(curves)
    fig, axes = plt.subplots(1, len(feature_sets),
                             figsize=(5.8 * len(feature_sets), 5.0), dpi=DPI,
                             sharey=True)
    fig.patch.set_facecolor(SURFACE)
    axes = np.atleast_1d(axes).ravel()

    for ax, feature_set in zip(axes, feature_sets):
        block = curves[curves["feature_set"] == feature_set]
        ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1.4, zorder=2)
        ax.annotate("đoán ngẫu nhiên", xy=(0.72, 0.72), rotation=45,
                    rotation_mode="anchor", ha="left", va="bottom",
                    fontsize=8, color=MUTED)

        for algo in _algos_in(block):
            line = block[block["algo"] == algo].sort_values("x")
            score = metrics.loc[(algo, feature_set), "roc_auc"] \
                if (algo, feature_set) in metrics.index else float("nan")
            suffix = "" if pd.isna(score) else f"  ·  {score:.4f}"
            ax.plot(line["x"], line["y"], color=_color(algo), linewidth=2,
                    zorder=3, solid_capstyle="round",
                    label=f"{_label(algo)}{suffix}")

        _style(ax)
        _panel_title(ax, feature_set)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        # Panel VUÔNG. Cả hai trục là tỉ lệ 0–1 nên khung méo làm đường chéo
        # may rủi không còn nghiêng 45°, và nhãn xoay 45° dán trên nó lệch hẳn
        # khỏi đường.
        ax.set_box_aspect(1)
        ax.set_xlabel("tỉ lệ báo động giả (FPR)", color=INK_SECONDARY, fontsize=9.5)
        _legend(ax, loc="lower right", fontsize=8.5, title="ROC-AUC")
    axes[0].set_ylabel("tỉ lệ bắt đúng (TPR)", color=INK_SECONDARY, fontsize=9.5)

    _titles(fig, "ML02 — ROC trên tập validation",
            "⚠️ ROC LẠC QUAN với dữ liệu lệch 8/92: phần lớn diện tích đến từ "
            "vùng ngưỡng không ai vận hành.\nChọn model thì xem "
            "precision_recall_curve.png, không xem hình này.")

    out = Path(out) if out else out_dir / "roc_curve.png"
    return _save(fig, out)


# -------------------------------------------------- phân tích ngưỡng

def plot_threshold_analysis(runs_dir=None, out=None):
    """Quét ngưỡng: F1 theo ngưỡng cho 4 model, và cái giá precision/recall.

    Hàng trên trả lời "ngưỡng nào cho F1 cao nhất" — đúng quy tắc mà task 14
    dùng để chốt ngưỡng. Hàng dưới trả lời "trả giá bằng gì": đẩy ngưỡng xuống
    thì recall lên và precision xuống, và hai đường cắt nhau ở đâu là chuyện
    một cột F1 không nói ra.

    Đường dọc là ngưỡng đã chốt ở task 14 — chỉ vẽ khi `decision.json` đã có.
    """
    out_dir, path = _resolve(runs_dir, EVAL_SUBDIR, "threshold_sweep.csv")
    sweep = pd.read_csv(path)
    feature_sets = _feature_sets_in(sweep)
    deployed = _deployed_threshold(runs_dir)

    fig, axes = plt.subplots(2, len(feature_sets),
                             figsize=(6.0 * len(feature_sets), 8.2), dpi=DPI,
                             sharex=True)
    fig.patch.set_facecolor(SURFACE)
    axes = np.atleast_2d(axes)
    if axes.shape[0] != 2:                      # một bộ feature → cột đơn
        axes = axes.reshape(2, -1)
    fig.subplots_adjust(hspace=0.28, wspace=0.22)

    for column, feature_set in enumerate(feature_sets):
        block = sweep[sweep["feature_set"] == feature_set]

        # ---- hàng trên: F1 lớp dương của cả 4 thuật toán ----
        top = axes[0, column]
        best_algo, best_f1 = None, -np.inf
        for algo in _algos_in(block):
            line = block[block["algo"] == algo].sort_values("threshold")
            top.plot(line["threshold"], line["f1_positive"], color=_color(algo),
                     linewidth=2, zorder=3, solid_capstyle="round",
                     label=_label(algo))
            peak = line.loc[line["f1_positive"].idxmax()]
            # Chấm đỉnh: ngưỡng tối ưu của TỪNG model nằm ở chỗ khác nhau, và
            # đó chính là thứ hình này phải hiện ra.
            top.scatter([peak["threshold"]], [peak["f1_positive"]], s=46,
                        color=_color(algo), zorder=4,
                        edgecolor=SURFACE, linewidth=1.6)
            if peak["f1_positive"] > best_f1:
                best_algo, best_f1 = algo, float(peak["f1_positive"])

        _style(top)
        _panel_title(top, feature_set)
        top.set_ylabel("F1 lớp dương", color=INK_SECONDARY, fontsize=9.5)
        # Góc dưới trái trống: F1 đi lên rồi mới đổ, còn đỉnh nằm bên phải.
        # Đặt legend ở "upper right" thì nó nằm đúng trên đỉnh của cả 4 đường.
        _legend(top, loc="lower left", fontsize=8.5)

        # ---- hàng dưới: precision và recall của model dẫn đầu ----
        bottom = axes[1, column]
        leader = block[block["algo"] == best_algo].sort_values("threshold")
        # Hai đường cùng thang 0–1 nên dùng CHUNG một trục y. Không bao giờ
        # dựng trục y thứ hai: hai thang trên một panel là hình nói dối.
        bottom.plot(leader["threshold"], leader["precision_positive"],
                    color=SERIES_ML02[0], linewidth=2, zorder=3,
                    solid_capstyle="round", label="precision")
        bottom.plot(leader["threshold"], leader["recall_positive"],
                    color=SERIES_ML02[1], linewidth=2, zorder=3,
                    solid_capstyle="round", label="recall")
        _style(bottom)
        bottom.set_title(f"cái giá của ngưỡng — {_label(best_algo)} (F1 cao nhất)",
                         color=INK, fontsize=10.5, pad=10, loc="left")
        bottom.set_xlabel("ngưỡng phân loại HIGH_RISK",
                          color=INK_SECONDARY, fontsize=9.5)
        bottom.set_ylabel("tỉ lệ", color=INK_SECONDARY, fontsize=9.5)
        # Góc trên phải trống: precision còn thấp ở đó, recall thì đã đổ xuống.
        _legend(bottom, loc="upper right", fontsize=8.5)

        if deployed is not None:
            for ax in (top, bottom):
                ax.axvline(deployed, color=INK_SECONDARY, linewidth=1.2, zorder=2)
            top.annotate(f"ngưỡng đã chốt {deployed:.4f}",
                         xy=(deployed, 1.0), xycoords=("data", "axes fraction"),
                         textcoords="offset points", xytext=(5, -12),
                         ha="left", va="top", fontsize=8, color=INK_SECONDARY)

    subtitle = ("0,5 KHÔNG phải ngưỡng vận hành: với tỉ lệ nền 8%, nó xếp gần "
                "như mọi hồ sơ vào LOW_RISK.")
    if deployed is None:
        subtitle += "\n(Chạy task 14 để hình có đường ngưỡng đã chốt.)"
    _titles(fig, "ML02 — Phân tích ngưỡng trên tập validation", subtitle)

    out = Path(out) if out else out_dir / "threshold_analysis.png"
    return _save(fig, out)


# ------------------------------------------------------- so sánh model

def plot_model_comparison(runs_dir=None, out=None):
    """Xếp hạng 8 model kèm khoảng tin cậy bootstrap (task 12).

    Chấm là PR-AUC đo được, thanh ngang là khoảng tin cậy 95% từ bootstrap.
    Hai khoảng CHỒNG nhau nghĩa là chênh lệch chưa phân biệt được với nhiễu —
    đó là câu hỏi task 12 sinh ra để trả lời, và một biểu đồ cột không có chỗ
    nào đặt được thông tin đó.

    Hai panel dùng chung dải trục x: khoảng cách rút gọn ↔ full chính là cái
    giá của việc form không thu được EXT_SOURCE_1/2/3 (§7.2).
    """
    out_dir, path = _resolve(runs_dir, COMPARE_SUBDIR, "comparison.csv")
    comparison = pd.read_csv(path)

    interval_path = _optional(runs_dir, COMPARE_SUBDIR, "confidence_interval.csv")
    if interval_path is not None:
        intervals = pd.read_csv(interval_path)
        comparison = comparison.merge(
            intervals[["algo", "feature_set", "ci_low", "ci_high"]],
            on=["algo", "feature_set"], how="left")
    else:
        comparison["ci_low"] = np.nan
        comparison["ci_high"] = np.nan

    base_rate = _base_rate(comparison)
    feature_sets = _feature_sets_in(comparison)

    # Dải x chung cho mọi panel, tính trên cả điểm lẫn hai đầu khoảng tin cậy.
    spread = pd.concat([comparison["pr_auc"], comparison["ci_low"],
                        comparison["ci_high"]]).dropna()
    low, high = float(spread.min()), float(spread.max())
    if base_rate is not None:
        low = min(low, base_rate)
    margin = (high - low) * 0.18 or 0.01

    fig, axes = plt.subplots(len(feature_sets), 1,
                             figsize=(9.6, 2.6 * len(feature_sets) + 1.4),
                             dpi=DPI, sharex=True)
    fig.patch.set_facecolor(SURFACE)
    axes = np.atleast_1d(axes).ravel()
    fig.subplots_adjust(hspace=0.42)

    for ax, feature_set in zip(axes, feature_sets):
        block = (comparison[comparison["feature_set"] == feature_set]
                 .sort_values("pr_auc").reset_index(drop=True))
        y = np.arange(len(block))

        for position, row in zip(y, block.itertuples()):
            colour = _color(row.algo)
            if not (pd.isna(row.ci_low) or pd.isna(row.ci_high)):
                ax.plot([row.ci_low, row.ci_high], [position, position],
                        color=colour, linewidth=2.4, alpha=0.42, zorder=2,
                        solid_capstyle="round")
            # Vòng nền quanh chấm để chấm không chìm vào thanh khoảng tin cậy.
            ax.scatter([row.pr_auc], [position], s=104, color=colour, zorder=4,
                       edgecolor=SURFACE, linewidth=2)
            leading = getattr(row, "rank", None) == 1
            ax.annotate(f"{row.pr_auc:.4f}", (row.pr_auc, position),
                        textcoords="offset points", xytext=(0, 13),
                        ha="center", fontsize=8.5,
                        color=INK if leading else INK_SECONDARY,
                        fontweight="bold" if leading else "normal")

        if base_rate is not None:
            ax.axvline(base_rate, color=AXIS, linewidth=1.4, zorder=1)

        _style(ax)
        ax.set_yticks(y, [_label(a) for a in block["algo"]])
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
        ax.yaxis.grid(False)
        ax.set_ylim(-0.8, len(block) - 0.1)
        ax.set_xlim(low - margin, high + margin)
        _panel_title(ax, feature_set)

    axes[-1].set_xlabel("PR-AUC trên validation (chấm) · khoảng tin cậy 95% bootstrap (thanh)",
                        color=INK_SECONDARY, fontsize=9.5)
    if base_rate is not None:
        axes[-1].annotate(f"đoán bừa {_vi_pct(base_rate)}",
                          xy=(base_rate, 0), xycoords=("data", "axes fraction"),
                          textcoords="offset points", xytext=(5, 6),
                          ha="left", va="bottom", fontsize=8, color=MUTED)

    _titles(fig, "ML02 — Xếp hạng model theo PR-AUC",
            "Hai khoảng tin cậy CHỒNG nhau = chênh lệch chưa phân biệt được "
            "với nhiễu.\nXếp riêng từng bộ feature vì đó là hai bài toán "
            "triển khai khác nhau (§7.2).")

    out = Path(out) if out else out_dir / "model_comparison.png"
    return _save(fig, out)


# -------------------------------------------------- feature importance

#: Cách đo → (tên file, nhãn panel, đơn vị trục x). Thứ tự này là thứ tự panel.
_IMPORTANCE_METHODS: Final[tuple[tuple[str, str, str], ...]] = (
    ("builtin", "Built-in (impurity)", "tỉ trọng impurity — tổng mỗi model = 1"),
    ("permutation", "Permutation", "PR-AUC TỤT bao nhiêu khi xáo cột"),
    ("shap", "SHAP", "trung bình |giá trị SHAP|"),
)


def _leading_model(runs_dir, feature_set: str) -> tuple[str, str]:
    """`(algo, feature_set)` của model dẫn đầu bộ đã cho, theo task 12.

    Đọc `comparison.csv` chứ KHÔNG đọc `decision.json`: task 13 chạy TRƯỚC
    task 14, nên trong cùng một lần chạy pipeline thì `decision.json` (nếu có)
    là của lần chạy TRƯỚC, còn `comparison.csv` mới là kết quả vừa sinh.
    """
    _, path = _resolve(runs_dir, COMPARE_SUBDIR, "comparison.csv")
    comparison = pd.read_csv(path)
    block = comparison[comparison["feature_set"] == feature_set]
    if block.empty:
        raise ValueError(
            f"{path} không có model nào ở bộ feature {feature_set!r}.")
    leader = block.loc[block["pr_auc"].idxmax()]
    return str(leader["algo"]), feature_set


def plot_feature_importance(runs_dir=None, out=None, top_n: int = 12,
                            algo: str | None = None,
                            feature_set: str | None = None):
    """Ba cách đo importance của model dẫn đầu, small multiples (task 13).

    Mỗi panel một cách đo với TRỤC X RIÊNG. Gộp ba cách lên một trục là so
    sánh ba đại lượng khác đơn vị — built-in là tỉ trọng cộng lại bằng 1,
    permutation là mức PR-AUC tụt, SHAP là trung bình |giá trị|.

    Thứ tự hàng dùng chung cho cả ba panel (theo thứ hạng trung bình), nên
    chỗ ba cách đo BẤT ĐỒNG lộ ra ngay: cùng một hàng mà cột dài ở panel này,
    ngắn ở panel kia.

    Permutation có thể ÂM — xáo cột đó làm model tốt lên, tức cột chỉ đang
    thêm nhiễu. Giữ nguyên dấu âm, không cắt về 0.
    """
    feature_set = feature_set or DEPLOY_FEATURE_SET
    if algo is None:
        algo, feature_set = _leading_model(runs_dir, feature_set)

    out_dir, _ = _resolve(runs_dir, IMPORTANCE_SUBDIR, "builtin.csv")

    # `--builtin-only` khiến permutation.csv và shap.csv không được ghi. Vẽ
    # đúng số panel có dữ liệu chứ không dựng panel rỗng.
    available: list[tuple[str, str, str, pd.DataFrame]] = []
    for name, title, unit in _IMPORTANCE_METHODS:
        path = _optional(runs_dir, IMPORTANCE_SUBDIR, f"{name}.csv")
        if path is None:
            continue
        table = pd.read_csv(path)
        block = table[(table["algo"] == algo) &
                      (table["feature_set"] == feature_set)]
        if not block.empty:
            available.append((name, title, unit, block))

    if not available:
        raise ValueError(
            f"Không có bảng importance nào cho {algo} · bộ {feature_set} trong "
            f"{out_dir}. Chạy `scripts/importance_ml02.py` trước.")

    # Thứ tự hàng: thứ hạng trung bình của các cách đo CÓ MẶT. Không dùng
    # rank_comparison.csv — bảng đó gộp cả cách đo có thể đang vắng mặt.
    ranks = pd.concat(
        [block.set_index("feature")["rank"].rename(name)
         for name, _, _, block in available], axis=1)
    order = ranks.mean(axis=1).nsmallest(top_n).index[::-1]   # tốt nhất lên trên

    fig, axes = plt.subplots(1, len(available),
                             figsize=(4.6 * len(available),
                                      0.42 * len(order) + 2.2),
                             dpi=DPI, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    axes = np.atleast_1d(axes).ravel()
    fig.subplots_adjust(wspace=0.12)

    base = np.arange(len(order))
    for index, (ax, (name, title, unit, block)) in enumerate(zip(axes, available)):
        indexed = block.set_index("feature")
        values = indexed["importance"].reindex(order)
        colour = SERIES_ML02[index % len(SERIES_ML02)]

        # Permutation kèm độ lệch giữa các lần lặp — thiếu nó thì hai cột
        # chênh nhau chút đỉnh bị đọc thành "cột này quan trọng hơn".
        error = None
        if name == "permutation" and "std" in indexed.columns:
            error = indexed["std"].reindex(order).to_numpy()

        ax.barh(base, values.to_numpy(), height=0.66, color=colour, zorder=3,
                xerr=error, error_kw={"ecolor": INK_SECONDARY, "elinewidth": 1,
                                      "capsize": 2, "alpha": 0.7})
        _style(ax)
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
        ax.yaxis.grid(False)
        ax.set_title(title, color=INK, fontsize=10.5, pad=10, loc="left")
        ax.set_xlabel(unit, color=INK_SECONDARY, fontsize=8.5)
        if values.min() < 0:
            # Vạch 0 hiện rõ để phần âm đọc được là âm.
            ax.axvline(0, color=AXIS, linewidth=1)

    axes[0].set_yticks(base, order)
    axes[0].tick_params(labelsize=8.5)

    _titles(fig, f"ML02 — Feature importance · {_label(algo)} · bộ {feature_set}",
            f"Top {len(order)} theo thứ hạng trung bình, đo trên validation. "
            "Mỗi panel MỘT trục x riêng — ba cách đo khác đơn vị, không so "
            "trực tiếp được.\n⚠️ Đây là CHẨN ĐOÁN, không phải bước chọn feature.")

    out = Path(out) if out else out_dir / "feature_importance.png"
    return _save(fig, out)


# ------------------------------------------ confusion matrix trên test

def plot_confusion_matrix_test(runs_dir=None, out=None):
    """Confusion matrix của model đã chốt, trên tập test (task 14).

    Một panel, không phải bốn như ML01: tập test của ML02 chỉ được mở ĐÚNG
    MỘT LẦN, cho ĐÚNG MỘT model, sau khi task 14 đã chốt. Vẽ 8 model trên
    test nghĩa là đã chấm test 8 lần — chính là điều F04 cấm.

    Tô theo tỉ lệ HÀNG (hàng = lớp thật, nên đường chéo chính là recall) và
    ghi số đếm thô. Với 92/8 mà tô theo số đếm thì hàng lớp âm luôn đậm đặc
    còn hàng lớp dương luôn trắng, bất kể model đúng hay sai.
    """
    out_dir, path = _resolve(runs_dir, SELECTION_SUBDIR, "test_confusion.csv")
    matrix = pd.read_csv(path, index_col=0)
    counts = matrix.to_numpy(dtype=float)
    if counts.shape[0] != counts.shape[1]:
        raise ValueError(f"{path} không phải ma trận vuông: {counts.shape}.")

    row_totals = counts.sum(axis=1, keepdims=True)
    share = np.divide(counts, row_totals, out=np.zeros_like(counts),
                      where=row_totals != 0)

    labels = [str(v) for v in matrix.index]
    columns = [str(c) for c in matrix.columns]

    fig, ax = _figure(figsize=(7.6, 5.6), dpi=DPI)
    mesh = ax.imshow(share, cmap=BLUES, vmin=0.0, vmax=1.0)

    ax.set_xticks(range(len(columns)), columns, fontsize=9)
    ax.set_yticks(range(len(labels)), labels, fontsize=9)
    ax.set_xlabel("dự đoán", color=INK_SECONDARY, fontsize=10)
    ax.set_ylabel("thật", color=INK_SECONDARY, fontsize=10)
    ax.tick_params(colors=MUTED, length=0)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            # Ô đậm thì chữ trắng, ô nhạt thì chữ đen — nếu không, một nửa số
            # trên hình không đọc được.
            ink = "#ffffff" if share[i, j] > 0.55 else INK
            # Dòng dưới lệch xuống 0,22 inch so với số đếm — dịch bằng
            # `ScaledTranslation` chứ không cộng vào toạ độ dữ liệu, để khoảng
            # cách giữa hai dòng không đổi khi ô to nhỏ khác nhau.
            shift = mtransforms.ScaledTranslation(0, -0.22, fig.dpi_scale_trans)
            ax.text(j, i, f"{int(counts[i, j]):,}".replace(",", "."),
                    ha="center", va="center", fontsize=13, color=ink)
            ax.text(j, i, f"{_vi_pct(share[i, j], 1)} của hàng",
                    ha="center", va="center", fontsize=8.5, color=ink,
                    alpha=0.85, transform=ax.transData + shift)

    # Ô bỏ lọt = hàng lớp dương, cột dự đoán âm. Viền cam vì đây là ô đắt nhất
    # của bài toán và heatmap không tự nói ra điều đó. Lời chú thích đi lên
    # khối tiêu đề chứ không dán cạnh ô: dán dưới ô thì nó đè lên nhãn trục x.
    highlighted = counts.shape == (2, 2)
    if highlighted:
        ax.add_patch(plt.Rectangle((-0.5, 0.5), 1, 1, fill=False,
                                   edgecolor="#eb6834", linewidth=2.4, zorder=5))

    bar = fig.colorbar(mesh, ax=ax, shrink=0.82, pad=0.03)
    bar.set_label("tỉ lệ theo hàng (đường chéo = recall)",
                  color=INK_SECONDARY, fontsize=9)
    bar.ax.tick_params(colors=MUTED, labelsize=8, length=0)
    bar.outline.set_visible(False)

    total = int(counts.sum())
    subtitle_parts = [f"{total:,}".replace(",", ".") + " hồ sơ chưa từng được chạm ở task 1–13"]
    threshold = _deployed_threshold(runs_dir)
    if threshold is not None:
        subtitle_parts.append(f"ngưỡng {_vi(threshold)}")

    _titles(fig, "ML02 — Confusion matrix của model đã chốt, tập TEST",
            " · ".join(subtitle_parts),
            note=("Ô viền cam = ca vỡ nợ BỊ BỎ LỌT, ô đắt nhất của bài toán."
                  if highlighted else None),
            note_color="#eb6834")

    out = Path(out) if out else out_dir / "confusion_matrix_test.png"
    return _save(fig, out)


# --------------------------------------------------------- điều phối

def generate_evaluation_plots(runs_dir=None) -> dict[str, Path]:
    """Ba hình của task 11 → `ml02_evaluation/`."""
    return {
        "precision_recall_curve": plot_precision_recall_curve(runs_dir),
        "roc_curve": plot_roc_curve(runs_dir),
        "threshold_analysis": plot_threshold_analysis(runs_dir),
    }


def generate_comparison_plots(runs_dir=None) -> dict[str, Path]:
    """Hình của task 12 → `ml02_comparison/`."""
    return {"model_comparison": plot_model_comparison(runs_dir)}


def generate_importance_plots(runs_dir=None, top_n: int = 12) -> dict[str, Path]:
    """Hình của task 13 → `ml02_importance/`."""
    return {"feature_importance": plot_feature_importance(runs_dir, top_n=top_n)}


def generate_selection_plots(runs_dir=None) -> dict[str, Path]:
    """Hình của task 14 → `ml02_selection/`."""
    return {"confusion_matrix_test": plot_confusion_matrix_test(runs_dir)}


def generate_ml02_plots(runs_dir=None, top_n: int = 12) -> dict[str, Path]:
    """Cả sáu hình. Trả `dict` tên → đường dẫn.

    Dùng khi muốn vẽ lại toàn bộ mà không chạy lại task nào. Trong pipeline
    thì mỗi script tự gọi nhóm hình của nó ngay sau khi ghi CSV, nên hình
    không bao giờ được dựng từ số của lần chạy trước.
    """
    figures: dict[str, Path] = {}
    figures.update(generate_evaluation_plots(runs_dir))
    figures.update(generate_comparison_plots(runs_dir))
    figures.update(generate_importance_plots(runs_dir, top_n=top_n))
    figures.update(generate_selection_plots(runs_dir))
    return figures
