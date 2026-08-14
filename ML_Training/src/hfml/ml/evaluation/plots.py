"""Biểu đồ cho báo cáo ML01 (F03).

Ba hình, mỗi hình đọc từ kết quả ĐÃ CÓ trong `src/training/runs/` — không
train lại, không tính lại chỉ số:

    confusion_matrix_test.png   test_confusion.csv          (task 11)
    model_comparison.png        model_comparison.csv        (task 12)
    feature_importance.png      feature_importance.csv      (task 13)

Vì sao mỗi hình chọn dạng đó
-----------------------------
**Confusion matrix** — ma trận đếm, nên là heatmap một màu light→dark. Bốn
model xếp thành small multiples để so được *kiểu lỗi*, không chỉ số lỗi. Màu
mã hoá tỉ lệ THEO HÀNG chứ không phải số đếm thô: hàng là lớp thật, nên tỉ lệ
hàng chính là recall, và bốn lớp có cỡ rất khác nhau (593 → 1.285) — tô theo
số đếm thì hai lớp lớn lúc nào cũng đậm hơn bất kể model đúng hay sai.

**So sánh model** — dumbbell (hai chấm nối bằng một đoạn), không phải cột.
Macro-F1 của bốn model nằm trong dải 0,84–0,92; cột thì bắt buộc gốc 0 nên
mọi khác biệt bị nén thành vô hình, mà cắt trục gốc của cột là bóp méo. Chấm
không mã hoá độ lớn bằng chiều dài nên được phép dùng dải trục hẹp, và đoạn
nối chính là `gap` CV → test mà task 12 đo.

**Feature importance** — cột ngang, gốc 0 (importance là độ lớn, gốc 0 có
nghĩa). Ba model cạnh nhau thay vì gộp trung bình: chỗ chúng BẤT ĐỒNG mới là
thông tin, ví dụ `has_savings` chạy từ 0,0001 (cây đơn) tới 0,2460 (XGBoost).

Màu
---
Bảng màu categorical dùng 3 slot đầu (blue · orange · aqua), đã chạy qua
validator ở chế độ `--pairs all` trên nền `#fcfcfb`: đạt cả dải sáng, sàn
chroma, tách CVD (ΔE 9,2) và sàn thị lực thường (ΔE 24,0). Aqua có contrast
2,74 < 3:1 nên theo *relief rule* phải kèm nhãn hiện hoặc table view — table
view chính là các file CSV nằm cùng thư mục.

Heatmap dùng thang MỘT MÀU (blue 100→700). Không dùng rainbow: nhiều màu cho
một đại lượng liên tục làm người đọc thấy ranh giới ở chỗ không có ranh giới.
"""
from __future__ import annotations

from typing import Final

import matplotlib

# Backend không cửa sổ — script và test chạy không có màn hình.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from hfml.config import CONFIG  # noqa: E402
from hfml.logger import get_logger  # noqa: E402

log = get_logger(__name__)

#: 3 slot categorical đầu tiên — bộ duy nhất qua được `--pairs all`.
SERIES: Final[tuple[str, ...]] = ("#2a78d6", "#eb6834", "#1baf7a")

SURFACE: Final[str] = "#fcfcfb"
INK: Final[str] = "#0b0b0b"
INK_SECONDARY: Final[str] = "#52514e"
MUTED: Final[str] = "#898781"
GRID: Final[str] = "#e1e0d9"
AXIS: Final[str] = "#c3c2b7"

#: Thang tuần tự một màu, blue 100 → 700.
SEQUENTIAL_STEPS: Final[tuple[str, ...]] = (
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
)
BLUES = LinearSegmentedColormap.from_list("hfml_blue", SEQUENTIAL_STEPS)

DPI: Final[int] = 150


def _style(ax) -> None:
    """Trục và lưới lùi về sau: hairline liền nét, một tông so với nền.

    Lưới nét đứt bị loại — nó thêm nhiễu thị giác và bị đọc nhầm thành
    "ngưỡng" trong khi nó chỉ là lưới.
    """
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.grid(True, color=GRID, linewidth=0.8, linestyle="-", zorder=0)
    ax.set_axisbelow(True)


def _figure(*args, **kwargs):
    fig, ax = plt.subplots(*args, **kwargs)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def _resolve(runs_dir, name: str):
    runs_dir = CONFIG.paths.runs if runs_dir is None else runs_dir
    path = runs_dir / name
    if not path.exists():
        raise FileNotFoundError(
            f"Chưa có {path}. Biểu đồ đọc từ kết quả đã lưu — chạy task sinh "
            "ra file này trước, hàm vẽ không tự tính lại.")
    return runs_dir, path


# ------------------------------------------------------ confusion matrix

def plot_confusion_matrix(runs_dir=None, out=None, algos=None):
    """Confusion matrix trên tập test, small multiples 4 model (task 11).

    Ô được TÔ theo tỉ lệ hàng nhưng GHI số đếm thô: màu để so giữa các lớp và
    giữa các model, số để đọc chính xác. Thiếu số thì hình vi phạm đúng điều
    cấm "mã hoá bằng màu mà không có bảng đối chiếu".
    """
    runs_dir, path = _resolve(runs_dir, "test_confusion.csv")
    data = pd.read_csv(path)
    label_column = data.columns[0]
    labels = list(dict.fromkeys(data[label_column]))
    algos = list(algos or dict.fromkeys(data["algo"]))

    rows = int(np.ceil(len(algos) / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(11, 4.6 * rows), dpi=DPI)
    fig.patch.set_facecolor(SURFACE)
    axes = np.atleast_1d(axes).ravel()
    # Nới khoảng cách giữa các panel: sát nhau thì nhãn trục của panel phải
    # đè lên ô của panel trái.
    fig.subplots_adjust(wspace=0.35, hspace=0.45)

    mesh = None
    for index, (ax, algo) in enumerate(zip(axes, algos)):
        block = data[data["algo"] == algo].set_index(label_column).loc[labels, labels]
        counts = block.to_numpy(dtype=float)
        # Chia theo tổng HÀNG = tỉ lệ trong từng lớp thật = recall trên đường chéo.
        share = counts / counts.sum(axis=1, keepdims=True)

        mesh = ax.imshow(share, cmap=BLUES, vmin=0.0, vmax=1.0)
        ax.set_title(algo, color=INK, fontsize=11, pad=10)

        # Small multiples: chỉ cột trái ghi nhãn hàng, chỉ hàng dưới ghi nhãn
        # cột. Lặp nhãn ở cả bốn panel vừa thừa vừa gây chồng chữ.
        first_column = index % 2 == 0
        last_row = index // 2 == rows - 1
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        if last_row:
            ax.set_xticklabels(labels, rotation=30, ha="right")
            ax.set_xlabel("dự đoán", color=INK_SECONDARY, fontsize=9)
        else:
            ax.set_xticklabels([])
        if first_column:
            ax.set_yticklabels(labels)
            ax.set_ylabel("thật", color=INK_SECONDARY, fontsize=9)
        else:
            ax.set_yticklabels([])
        ax.tick_params(colors=MUTED, labelsize=8, length=0)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)

        for i in range(len(labels)):
            for j in range(len(labels)):
                # Ô đậm thì chữ trắng, ô nhạt thì chữ đen — nếu không, nửa số
                # trên hình không đọc được.
                ax.text(j, i, f"{int(counts[i, j]):,}", ha="center", va="center",
                        fontsize=9, color="#ffffff" if share[i, j] > 0.55 else INK)

    for ax in axes[len(algos):]:
        ax.set_visible(False)

    bar = fig.colorbar(mesh, ax=axes[:len(algos)].tolist(), shrink=0.75, pad=0.02)
    bar.set_label("tỉ lệ theo hàng (recall trên đường chéo)",
                  color=INK_SECONDARY, fontsize=9)
    bar.ax.tick_params(colors=MUTED, labelsize=8, length=0)
    bar.outline.set_visible(False)

    # Cỡ tập test tính từ chính dữ liệu, KHÔNG hard-code: tổng một ma trận là
    # số hồ sơ đã chấm. Trước đây ghi cứng "4.000 hồ sơ", và khi task 5 đổi
    # cách chia thành 70/15/15 thì tiêu đề nói sai mà hình vẫn vẽ ra bình thường.
    n_test = int(data[data["algo"] == algos[0]][labels].to_numpy(dtype=float).sum())
    # Đổi dấu nghìn sang kiểu Việt trên RIÊNG con số. Gọi `.replace(",", ".")`
    # lên cả câu thì dấu phẩy ngăn vế trong tiêu đề cũng bị đổi thành dấu chấm.
    n_test_vi = f"{n_test:,}".replace(",", ".")
    fig.suptitle(
        f"ML01 — Confusion matrix nhóm định hướng tài chính, tập test ({n_test_vi} hồ sơ)",
        color=INK, fontsize=13, y=0.98)
    out = out or runs_dir / "confusion_matrix_test.png"
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    log.info("Ghi biểu đồ → %s", out)
    return out


# ----------------------------------------------------- so sánh model

def plot_model_comparison(runs_dir=None, out=None):
    """CV → test macro-F1 của 4 model, dạng dumbbell (task 12).

    Đoạn nối giữa hai chấm chính là `gap`. Trục x cắt hẹp quanh vùng dữ liệu —
    hợp lệ vì chấm không mã hoá độ lớn bằng chiều dài; cột thì không được
    phép làm vậy.
    """
    runs_dir, path = _resolve(runs_dir, "model_comparison.csv")
    data = pd.read_csv(path).sort_values("cv_macro_f1").reset_index(drop=True)

    fig, ax = _figure(figsize=(9, 4.2), dpi=DPI)
    y = np.arange(len(data))

    for position, row in zip(y, data.itertuples()):
        ax.plot([row.cv_macro_f1, row.test_macro_f1], [position, position],
                color=AXIS, linewidth=2, zorder=2, solid_capstyle="round")
    ax.scatter(data["cv_macro_f1"], y, s=90, color=SERIES[0], zorder=3,
               label="CV trên train (chọn model)")
    ax.scatter(data["test_macro_f1"], y, s=90, color=SERIES[1], zorder=3,
               label="Test (báo cáo)")

    for position, row in zip(y, data.itertuples()):
        ax.annotate(f"{row.cv_macro_f1:.4f}", (row.cv_macro_f1, position),
                    textcoords="offset points", xytext=(0, 11), ha="center",
                    fontsize=8, color=INK_SECONDARY)
        ax.annotate(f"{row.test_macro_f1:.4f}", (row.test_macro_f1, position),
                    textcoords="offset points", xytext=(0, -16), ha="center",
                    fontsize=8, color=INK_SECONDARY)

    _style(ax)
    ax.set_yticks(y, data["algo"])
    # Chừa lề trên/dưới, nếu không nhãn giá trị của hàng ngoài cùng đè lên trục.
    ax.set_ylim(-0.75, len(data) - 0.25)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.yaxis.grid(False)
    ax.set_xlabel("macro-F1", color=INK_SECONDARY, fontsize=10)
    # numpy 2 bỏ phương thức `ndarray.ptp()`, chỉ còn hàm `np.ptp()`.
    span = float(np.ptp(data[["cv_macro_f1", "test_macro_f1"]].to_numpy()))
    ax.set_xlim(float(data["test_macro_f1"].min()) - span * 0.18,
                float(data["cv_macro_f1"].max()) + span * 0.18)
    ax.set_title("ML01 — macro-F1: CV trên train so với tập test",
                 color=INK, fontsize=13, pad=14, loc="left")
    legend = ax.legend(frameon=False, loc="lower right", fontsize=9)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    out = out or runs_dir / "model_comparison.png"
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    log.info("Ghi biểu đồ → %s", out)
    return out


# ------------------------------------------------- feature importance

def plot_feature_importance(runs_dir=None, out=None, top_n: int = 10):
    """Feature importance của các model có cung cấp nó (task 13).

    Cột ngang gốc 0, ba model cạnh nhau. Bagging vắng mặt vì
    `BaggingClassifier` không phơi ra `feature_importances_` — đó là dữ kiện,
    không phải thiếu sót của hình.

    Không ghi số lên từng cột: 30 nhãn là nhiễu. Giá trị chính xác nằm ở
    `feature_importance.csv` cùng thư mục — đó là table view của hình này.
    """
    runs_dir, path = _resolve(runs_dir, "feature_importance.csv")
    long = pd.read_csv(path)
    pivot = long.pivot_table(index="feature", columns="algo", values="importance")
    pivot = pivot.assign(mean=pivot.mean(axis=1)).nlargest(top_n, "mean")
    algos = [column for column in pivot.columns if column != "mean"]
    order = pivot.index[::-1]                       # lớn nhất lên trên cùng

    # Chiều cao theo SỐ FEATURE, không nhân với số model: nhân vào thì hình
    # cao 17 inch và các cột dày thành khối đặc, đọc rất ồn.
    fig, ax = _figure(figsize=(9, 0.46 * len(order) + 1.8), dpi=DPI)
    height = 0.72 / len(algos)
    base = np.arange(len(order))

    #: Dưới ngưỡng này cột mảnh tới mức nhìn như KHÔNG CÓ. Đúng chỗ đó lại là
    #: thông tin: `has_savings` được cây đơn chấm 0,0001 vì một lát cắt trên
    #: `savings_amount` đã bắt trọn thông tin đó. Ghi số ra để "gần 0" không
    #: bị đọc thành "thiếu dữ liệu".
    invisible = 0.005

    for index, algo in enumerate(algos):
        offset = (index - (len(algos) - 1) / 2) * height
        values = pivot.loc[order, algo]
        ax.barh(base + offset, values, height=height * 0.9,
                color=SERIES[index % len(SERIES)], label=algo, zorder=3)

        for position, value in zip(base + offset, values):
            if value < invisible:
                ax.annotate(f"{value:.4f}", (value, position),
                            textcoords="offset points", xytext=(5, 0),
                            va="center", fontsize=7,
                            color=SERIES[index % len(SERIES)])

    _style(ax)
    ax.set_yticks(base, order)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.yaxis.grid(False)
    ax.set_xlabel("mức độ quan trọng (tổng mỗi model = 1)",
                  color=INK_SECONDARY, fontsize=10)
    ax.set_title(f"ML01 — {top_n} feature quan trọng nhất, theo từng model",
                 color=INK, fontsize=13, pad=14, loc="left")
    legend = ax.legend(frameon=False, loc="lower right", fontsize=9)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    out = out or runs_dir / "feature_importance.png"
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    log.info("Ghi biểu đồ → %s", out)
    return out


# ------------------------------------------------------- bảng kết quả

#: Cột của bảng: (nhãn hiển thị, mép phải tính theo tỉ lệ chiều ngang).
#: Số căn phải nên toạ độ là mép PHẢI; riêng cột đầu căn trái, xử lý riêng.
_TABLE_COLUMNS: Final[tuple[tuple[str, float], ...]] = (
    # `n CV` đứng ngay sau tên thuật toán vì nó quyết định mấy cột sau có so
    # được với nhau không: `results.csv` giữ dòng MỚI NHẤT của từng split, nên
    # train lại một model với `--rows` khác sẽ ghép CV của cỡ này với test của
    # cỡ kia, và `chênh CV→test` thành con số vô nghĩa. Hiện n ra thì chỗ lệch
    # tự lộ; giấu đi thì người đọc tin vào một phép trừ sai.
    ("n CV", 0.200),
    ("macro-F1 CV", 0.330),
    ("sd fold", 0.400),
    ("macro-F1 test", 0.520),
    ("accuracy test", 0.640),
    ("chênh CV→test", 0.768),
    ("fit (s)", 0.845),
)

#: Nền dòng dẫn đầu. Xanh rất nhạt — đủ để mắt bắt được hàng, không đủ để
#: cạnh tranh với chữ.
_ROW_HIGHLIGHT: Final[str] = "#eaf2fd"


def _latest_by_algo(results: pd.DataFrame, split: str) -> pd.DataFrame:
    """Bản ghi mới nhất của từng thuật toán ở một `split`.

    `results.csv` ghi NỐI: mỗi lần train thêm một dòng chứ không đè lên dòng
    cũ. Lấy bản mới nhất chứ không lấy trung bình — trung bình của các lần
    chạy khác cấu hình là con số không mô tả lần chạy nào cả.
    """
    rows = results[results["split"] == split]
    if rows.empty:
        return rows.set_index("algo") if "algo" in rows else rows
    return rows.sort_values("run_at").groupby("algo").tail(1).set_index("algo")


def _cell(value, digits: int = 4, plus: bool = False) -> str:
    """Số đã format, hoặc `—` khi thiếu. Ô trống phải nhìn ra là trống."""
    if value is None or pd.isna(value):
        return "—"
    return f"{value:+.{digits}f}" if plus else f"{value:.{digits}f}"


def plot_results_table(runs_dir=None, out=None):
    """Bảng kết quả các model, vẽ thẳng từ `results.csv`.

    Khác ba hình trên ở nguồn đọc: chúng cần CSV của task 11–13, hình này chỉ
    cần `results.csv` — file mà MỌI lần train đều ghi. Nhờ vậy nó sinh được
    ngay sau `train_bagging.py`, không phải chạy hết chuỗi đánh giá trước.

    Cột `test` để `—` khi chưa chấm test lần nào. Đó là trạng thái thật sau
    một lần train đơn lẻ, và hiện nó ra tốt hơn là bỏ trống cả cột cho người
    đọc tự đoán.

    Là BẢNG chứ không phải biểu đồ, vì việc ở đây là tra số chính xác trên
    nhiều chỉ số cùng lúc — so sánh hình dạng đã có `model_comparison.png`.
    Thanh ngang cuối mỗi dòng chỉ để xếp hạng bằng mắt, gốc 0 và thang 0–1.
    """
    runs_dir, path = _resolve(runs_dir, "results.csv")
    results = pd.read_csv(path)

    cv = _latest_by_algo(results, "cv_train")
    test = _latest_by_algo(results, "test")
    algos = list(dict.fromkeys(list(cv.index) + list(test.index)))
    if not algos:
        raise ValueError(f"{path} không có dòng nào ở split cv_train hoặc test.")

    def _get(frame, algo, column):
        if algo not in frame.index or column not in frame.columns:
            return float("nan")
        return frame.loc[algo, column]

    table = pd.DataFrame({
        "cv_n": [_get(cv, a, "n_rows") for a in algos],
        "cv_f1": [_get(cv, a, "macro_f1") for a in algos],
        "cv_sd": [_get(cv, a, "macro_f1_std") for a in algos],
        "test_f1": [_get(test, a, "macro_f1") for a in algos],
        "test_acc": [_get(test, a, "accuracy") for a in algos],
        "fit": [_get(cv, a, "fit_seconds") for a in algos],
    }, index=algos)
    table["gap"] = table["cv_f1"] - table["test_f1"]

    # Xếp theo test nếu đã chấm test, ngược lại theo CV. Trộn hai thang vào
    # một cột sắp xếp thì thứ hạng không còn nghĩa gì.
    rank_column = "test_f1" if table["test_f1"].notna().any() else "cv_f1"
    table = table.sort_values(rank_column, ascending=False)
    best = table[rank_column].idxmax()

    row_height = 0.42
    header_height = 1.55
    fig, ax = _figure(figsize=(11, header_height + row_height * (len(table) + 1)),
                      dpi=DPI)
    ax.set_xlim(0, 1)
    # Mép dưới nới xuống -0.4: nền của dòng cuối kéo tới y-0.34, dừng ylim ở 0
    # thì đúng dòng đó bị cắt mất một nửa.
    ax.set_ylim(-0.4, len(table) + 1.15)
    ax.axis("off")

    top = len(table) + 0.15

    for index, (algo, row) in enumerate(table.iterrows()):
        y = top - 1 - index
        if algo == best:
            ax.add_patch(plt.Rectangle((0, y - 0.34), 1, 0.78,
                                       facecolor=_ROW_HIGHLIGHT,
                                       edgecolor="none", zorder=0))

        weight = "bold" if algo == best else "normal"
        ax.text(0.005, y, algo, va="center", ha="left", fontsize=10,
                color=INK, fontweight=weight)

        n_cv = ("—" if pd.isna(row["cv_n"]) else f"{int(row['cv_n']):,}")
        values = (n_cv, _cell(row["cv_f1"]), _cell(row["cv_sd"]),
                  _cell(row["test_f1"]), _cell(row["test_acc"]),
                  _cell(row["gap"], plus=True), _cell(row["fit"], digits=1))
        for (_, x), text in zip(_TABLE_COLUMNS, values):
            ax.text(x, y, text, va="center", ha="right", fontsize=9.5,
                    color=INK if algo == best else INK_SECONDARY,
                    family="monospace", fontweight=weight)

        score = row[rank_column]
        if pd.notna(score):
            ax.add_patch(plt.Rectangle((0.865, y - 0.12), 0.13, 0.24,
                                       facecolor=GRID, edgecolor="none", zorder=1))
            ax.add_patch(plt.Rectangle((0.865, y - 0.12), 0.13 * float(score), 0.24,
                                       facecolor=SERIES[0] if algo == best else AXIS,
                                       edgecolor="none", zorder=2))

    header_y = top - 0.15
    ax.text(0.005, header_y, "thuật toán", va="center", ha="left", fontsize=8.5,
            color=MUTED, fontweight="bold")
    for label, x in _TABLE_COLUMNS:
        ax.text(x, header_y, label, va="center", ha="right", fontsize=8.5,
                color=MUTED, fontweight="bold")
    ax.text(0.865, header_y, f"{rank_column.replace('_f1', '')} (0→1)",
            va="center", ha="left", fontsize=8.5, color=MUTED, fontweight="bold")
    ax.plot([0, 1], [header_y - 0.42] * 2, color=AXIS, linewidth=0.8)

    seeds = sorted(set(results["random_seed"].dropna().astype(int)))
    latest = str(results["run_at"].max())[:16].replace("T", " ")
    subtitle = (f"nguồn results.csv · {len(results)} bản ghi · "
                f"seed {', '.join(map(str, seeds))} · "
                f"bản ghi mới nhất {latest} UTC")
    ax.set_title("ML01 — kết quả theo thuật toán",
                 color=INK, fontsize=13, pad=26, loc="left")
    ax.text(0, 1.0, subtitle, transform=ax.transAxes, va="bottom", ha="left",
            fontsize=8.5, color=MUTED)

    out = out or runs_dir / "results_table.png"
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    log.info("Ghi bảng kết quả → %s", out)
    return out


def generate_ml01_plots(runs_dir=None, top_n: int = 10) -> dict:
    """Sinh cả ba hình. Trả về `dict` tên → đường dẫn."""
    return {
        "confusion_matrix": plot_confusion_matrix(runs_dir),
        "model_comparison": plot_model_comparison(runs_dir),
        "feature_importance": plot_feature_importance(runs_dir, top_n=top_n),
    }
