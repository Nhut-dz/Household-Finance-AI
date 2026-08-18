"""ML02 task 13 — Phân tích feature importance (F04 · M04 · Tuần 4).

Nối tiếp task 12. Task này trả lời *"cái gì đang dẫn dắt dự đoán rủi ro"*.
Chốt model là task 14, export là task 15.

Ba cách đo, và vì sao cần cả ba
--------------------------------
Một cột "importance" duy nhất rất dễ bị đọc như chân lý, trong khi ba phương
pháp phổ biến đo ba thứ khác nhau và **thường xếp hạng khác nhau**:

    built-in (impurity)   Cây giảm được bao nhiêu tạp chất khi chẻ theo cột
                          này. Rẻ, có sẵn. **Thiên vị cột nhiều giá trị**:
                          một cột liên tục có hàng nghìn điểm cắt khả dĩ nên
                          gần như luôn tìm được lát cắt "có vẻ tốt", còn cột
                          nhị phân chỉ có một. Đây là thiên vị đã biết rõ, và
                          nó KHÔNG tự lộ ra ở bảng.

    permutation           Xáo trộn một cột rồi đo PR-AUC tụt bao nhiêu. Đo
                          đúng **đóng góp vào năng lực dự báo**, không phải
                          số lần được chọn. Đắt hơn nhiều nhưng không thiên vị
                          theo số giá trị.

    SHAP                  Phân bổ phần đóng góp cho từng dự đoán rồi lấy trung
                          bình |giá trị|. Cho cả hướng tác động ở mức từng hồ
                          sơ — thứ mà tầng `llm` cần để giải thích một kết quả
                          cụ thể (§7.4, SHAP local top-5).

Chỗ ba bảng bất đồng ý nhau chính là chỗ đáng đọc nhất.

Đo permutation trên MA TRẬN ĐÃ BIẾN ĐỔI, không qua cả Pipeline
----------------------------------------------------------------
`sklearn.inspection.permutation_importance` chạy `n_features × n_repeats` lần
dự đoán. Cho cả Pipeline chạy lại từ đầu mỗi lần nghĩa là gộp bureau, dựng tỉ
lệ, điền thiếu, mã hoá… lặp lại hàng trăm lần — chậm gấp bội mà không đo thêm
gì. Biến đổi MỘT lần rồi hoán vị trên ma trận kết quả đo đúng thứ cần đo:
đóng góp của từng **feature đầu vào model**.

⚠️ Kết quả ở đây là CHẨN ĐOÁN, không phải bước chọn feature
--------------------------------------------------------------
Permutation và SHAP được đo trên tập **validation**. Dùng chúng để bỏ bớt
feature rồi train lại và chấm lại trên chính tập đó là **rò rỉ**: tập
validation khi ấy đã tham gia quyết định feature nào tồn tại, nên chỉ số thu
được sẽ lạc quan mà không có dấu hiệu gì.

Việc chọn feature có giám sát đã có chỗ của nó — `SupervisedFeatureSelector`
nằm TRONG Pipeline nên nó chỉ nhìn thấy tập train (§4.3e). Bảng ở đây để ĐỌC
và để giải trình, không để quay lại sửa feature set.

Tập test không bị chạm ở task này.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from hfml.config import CONFIG
from hfml.logger import get_logger
from hfml.ml.ml02_credit_risk.evaluate import artifact_path, available_models
from hfml.ml.ml02_credit_risk.train import TrainingData

log = get_logger(__name__)

#: Chỉ số dùng cho permutation — đúng chỉ số CHỌN MODEL của ML02.
#:
#: Đo bằng accuracy thì cột nào cũng "không quan trọng": bỏ hết feature vẫn
#: được 91,93% nhờ đoán toàn lớp âm.
PERMUTATION_SCORING: Final[str] = "average_precision"

#: Số lần hoán vị mỗi cột. 5 đủ để thứ hạng ổn định; con số này chỉ ảnh hưởng
#: độ rộng của `std`, không ảnh hưởng thứ tự các cột đầu bảng.
N_REPEATS: Final[int] = 5

#: Số hồ sơ lấy mẫu từ validation cho permutation và SHAP.
#:
#: Permutation chạy `n_features × n_repeats` lần dự đoán — với 82 feature là
#: 410 lần. Trên đủ 46.127 hồ sơ thì mỗi model mất vài phút; 10.000 hồ sơ đã
#: đủ để thứ hạng ổn định, vì thứ hạng phụ thuộc mức chênh giữa các cột chứ
#: không phụ thuộc chữ số thập phân thứ tư.
SAMPLE_SIZE: Final[int] = 10_000

#: Số feature giữ lại trong bảng tóm tắt của mỗi model.
TOP_N: Final[int] = 15

IMPORTANCE_SUBDIR: Final[str] = "ml02_importance"


@dataclass
class ImportanceResult:
    """Kết quả phân tích của MỘT model."""

    algo: str
    feature_set: str
    builtin: pd.DataFrame = field(default_factory=pd.DataFrame)
    permutation: pd.DataFrame = field(default_factory=pd.DataFrame)
    shap: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def slug(self) -> str:
        return f"ml02_{self.algo}_{self.feature_set}"


# --------------------------------------------------------------------------
# Lấy feature name và ma trận đã biến đổi
# --------------------------------------------------------------------------
def transformed_matrix(pipeline: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Áp phần feature của Pipeline, trả về ma trận model thật sự nhìn thấy.

    Tên cột ở đây là tên feature CUỐI CÙNG — sau khi bureau đã gộp, tỉ lệ đã
    dựng, cột thiếu quá ngưỡng đã bỏ và cột tương quan đã khử. Đó mới là bộ
    tên khớp với `feature_importances_` của model.
    """
    return pipeline.named_steps["features"].transform(X)


def _estimator(pipeline: Pipeline):
    return pipeline.named_steps["model"]


# --------------------------------------------------------------------------
# 1. Built-in importance
# --------------------------------------------------------------------------
def builtin_importance(pipeline: Pipeline, feature_names: list[str]) -> pd.DataFrame:
    """Importance có sẵn của model.

    `BaggingClassifier` không có `feature_importances_` — phải trung bình qua
    các cây con, và phải ánh xạ lại qua `estimators_features_` vì mỗi cây con
    có thể chỉ nhìn thấy một tập con cột. Bỏ bước ánh xạ thì với
    `max_features < 1.0` các con số sẽ gán nhầm cột — mà bảng vẫn trông bình
    thường.
    """
    model = _estimator(pipeline)

    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "estimators_"):
        values = np.zeros(len(feature_names), dtype=float)
        columns = getattr(model, "estimators_features_",
                          [np.arange(len(feature_names))] * len(model.estimators_))
        for child, used in zip(model.estimators_, columns):
            values[np.asarray(used)] += child.feature_importances_
        values /= len(model.estimators_)
    else:
        raise TypeError(
            f"{type(model).__name__} không có importance nội tại — "
            "dùng permutation thay thế.")

    return (pd.DataFrame({"feature": feature_names, "importance": values})
            .sort_values("importance", ascending=False, ignore_index=True))


# --------------------------------------------------------------------------
# 2. Permutation importance
# --------------------------------------------------------------------------
def permutation_table(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    n_repeats: int = N_REPEATS,
    seed: int | None = None,
) -> pd.DataFrame:
    """Xáo trộn từng cột rồi đo PR-AUC tụt bao nhiêu.

    `importance` là mức TỤT trung bình. Giá trị âm nghĩa là xáo trộn cột đó
    làm model **tốt lên** — dấu hiệu cột đó chỉ đang thêm nhiễu.
    """
    seed = CONFIG.random_seed if seed is None else seed
    matrix = transformed_matrix(pipeline, X)

    result = permutation_importance(
        _estimator(pipeline), matrix, np.asarray(y).astype(int),
        scoring=PERMUTATION_SCORING,
        n_repeats=n_repeats, random_state=seed, n_jobs=-1)

    return (pd.DataFrame({
        "feature": list(matrix.columns),
        "importance": result.importances_mean,
        "std": result.importances_std,
    }).sort_values("importance", ascending=False, ignore_index=True))


# --------------------------------------------------------------------------
# 3. SHAP (global)
# --------------------------------------------------------------------------
def shap_table(
    pipeline: Pipeline,
    X: pd.DataFrame,
    max_rows: int = SAMPLE_SIZE,
) -> pd.DataFrame:
    """Trung bình |giá trị SHAP| của từng feature — mức quan trọng toàn cục.

    Dùng `TreeExplainer`: cả bốn thuật toán đều là cây, nên nó tính chính xác
    chứ không xấp xỉ như `KernelExplainer`.

    Trả về bảng RỖNG nếu không tính được, thay vì ném lỗi: SHAP là phần bổ
    sung cho hai cách đo kia, mất nó không làm hỏng cả task 13.
    """
    try:
        import shap
    except ImportError:
        log.warning("Chưa cài shap — bỏ qua phần SHAP.")
        return pd.DataFrame(columns=["feature", "importance"])

    matrix = transformed_matrix(pipeline, X).head(max_rows)
    try:
        explainer = shap.TreeExplainer(_estimator(pipeline))
        values = explainer.shap_values(matrix, check_additivity=False)
    except Exception as exc:  # noqa: BLE001 — SHAP nhạy với phiên bản model
        log.warning("Không tính được SHAP: %s", exc)
        return pd.DataFrame(columns=["feature", "importance"])

    values = np.asarray(values)
    # Model nhị phân của sklearn trả về (n, features, 2); XGBoost trả (n, features).
    if values.ndim == 3:
        values = values[:, :, 1]

    return (pd.DataFrame({
        "feature": list(matrix.columns),
        "importance": np.abs(values).mean(axis=0),
    }).sort_values("importance", ascending=False, ignore_index=True))


# --------------------------------------------------------------------------
# Gộp ba cách đo
# --------------------------------------------------------------------------
def rank_comparison(result: ImportanceResult, top_n: int = TOP_N) -> pd.DataFrame:
    """Đặt ba bảng cạnh nhau theo THỨ HẠNG, không theo giá trị.

    Ba phương pháp có thang đo khác nhau hoàn toàn — impurity cộng lại bằng 1,
    permutation tính bằng mức tụt PR-AUC, SHAP tính bằng đơn vị log-odds. So
    giá trị là so hai thứ không cùng đơn vị; so thứ hạng thì được.
    """
    frames = []
    for name, table in (("builtin", result.builtin),
                        ("permutation", result.permutation),
                        ("shap", result.shap)):
        if table.empty:
            continue
        ranked = table.copy()
        ranked[f"rank_{name}"] = range(1, len(ranked) + 1)
        frames.append(ranked.set_index("feature")[[f"rank_{name}"]])

    if not frames:
        return pd.DataFrame()

    merged = pd.concat(frames, axis=1)
    rank_columns = [c for c in merged.columns if c.startswith("rank_")]
    merged["rank_mean"] = merged[rank_columns].mean(axis=1)
    # Chênh lệch thứ hạng lớn nhất giữa ba cách đo — chỗ ba bảng bất đồng.
    merged["rank_spread"] = (merged[rank_columns].max(axis=1)
                             - merged[rank_columns].min(axis=1))
    return (merged.sort_values("rank_mean")
            .head(top_n).reset_index().rename(columns={"index": "feature"}))


def analyse_model(
    algo: str,
    feature_set: str,
    data: TrainingData,
    sample_size: int = SAMPLE_SIZE,
    n_repeats: int = N_REPEATS,
    with_permutation: bool = True,
    with_shap: bool = True,
) -> ImportanceResult:
    """Chạy đủ ba cách đo cho một model."""
    pipeline = joblib.load(artifact_path(algo, feature_set))

    sample = data.X_validation.head(sample_size)
    y_sample = data.y_validation.head(sample_size)
    names = list(transformed_matrix(pipeline, sample.head(1)).columns)

    result = ImportanceResult(
        algo=algo, feature_set=feature_set,
        builtin=builtin_importance(pipeline, names))

    if with_permutation:
        log.info("Permutation %s · %s (%d feature × %d lần)",
                 algo, feature_set, len(names), n_repeats)
        result.permutation = permutation_table(
            pipeline, sample, y_sample, n_repeats=n_repeats)

    if with_shap:
        log.info("SHAP %s · %s", algo, feature_set)
        result.shap = shap_table(pipeline, sample)

    return result


def analyse_all(
    data: TrainingData,
    only: list[tuple[str, str]] | None = None,
    **kwargs,
) -> list[ImportanceResult]:
    """Phân tích mọi model có artifact, hoặc riêng danh sách được chỉ định."""
    pairs = only if only is not None else available_models()
    return [analyse_model(algo, feature_set, data, **kwargs)
            for algo, feature_set in pairs]


# --------------------------------------------------------------------------
# Ghi kết quả
# --------------------------------------------------------------------------
def output_dir() -> Path:
    return CONFIG.paths.runs / IMPORTANCE_SUBDIR


def _long(results: list[ImportanceResult], attribute: str) -> pd.DataFrame:
    frames = []
    for result in results:
        table = getattr(result, attribute)
        if table.empty:
            continue
        tagged = table.copy()
        tagged.insert(0, "feature_set", result.feature_set)
        tagged.insert(0, "algo", result.algo)
        tagged["rank"] = range(1, len(tagged) + 1)
        frames.append(tagged)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def write_importance(results: list[ImportanceResult]) -> dict[str, Path]:
    """Ghi ba bảng dài + bảng đối chiếu thứ hạng + metadata."""
    out_dir = output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    tables = {
        "builtin": _long(results, "builtin"),
        "permutation": _long(results, "permutation"),
        "shap": _long(results, "shap"),
    }
    comparisons = [rank_comparison(r).assign(algo=r.algo, feature_set=r.feature_set)
                   for r in results]
    comparisons = [c for c in comparisons if not c.empty]
    tables["rank_comparison"] = (pd.concat(comparisons, ignore_index=True)
                                 if comparisons else pd.DataFrame())

    written: dict[str, Path] = {}
    for name, table in tables.items():
        if table.empty:
            continue
        path = out_dir / f"{name}.csv"
        table.to_csv(path, index=False, encoding="utf-8")
        written[name] = path

    metadata = {
        "task": "ML02 task 13 — Phân tích feature importance",
        "measured_on": "validation",
        "test_set_touched": False,
        "sample_size": SAMPLE_SIZE,
        "permutation_scoring": PERMUTATION_SCORING,
        "n_repeats": N_REPEATS,
        "methods": ["builtin_impurity", "permutation", "shap"],
        "purpose": "diagnostic",
        "leakage_note": "Permutation và SHAP đo trên tập validation. Dùng chúng "
                        "để bỏ bớt feature rồi train lại và chấm lại trên chính "
                        "tập đó là RÒ RỈ. Việc chọn feature có giám sát nằm "
                        "TRONG Pipeline nên chỉ nhìn thấy tập train (§4.3e).",
        "feature_selection_changed": False,
        "models": [r.slug for r in results],
    }
    path = out_dir / "importance_metadata.json"
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    written["metadata"] = path

    log.info("Đã ghi %d file → %s", len(written), out_dir)
    return written
