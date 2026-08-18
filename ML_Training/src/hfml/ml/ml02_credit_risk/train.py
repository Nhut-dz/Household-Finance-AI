"""ML02 task 7–10 — Train bốn thuật toán (F04 · M04 · Tuần 4).

File này giữ phần khung dùng chung cho cả bốn task train, và **task 7 chỉ cài
đặt Decision Tree**. Ba thuật toán còn lại được thêm ở task 8, 9, 10.

Phần khung phải dùng chung, không phải cho gọn mà vì so sánh
--------------------------------------------------------------
Bảng ở task 12 chỉ có nghĩa khi bốn thuật toán được đối xử y hệt nhau: cùng
phép chia (task 5), cùng Pipeline feature (task 3), cùng tỉ số phạt (task 4),
cùng seed, cùng bộ chỉ số. Mỗi task tự viết vòng train riêng thì chỉ cần một
tham số khác nhau ở một chỗ là bảng so sánh so nhầm — mà không có gì trong
bảng để lộ ra điều đó. `fit_and_evaluate()` là nơi duy nhất quyết định những
thứ đó, nên bốn thuật toán không thể lệch nhau.

Pipeline được fit LẠI trong từng lần train
-------------------------------------------
Không nạp `.joblib` mà task 3 đã ghi. Hai lý do:

1. Artifact của task 3 được fit trên một phép chia 85/15 khác, dựng trước khi
   task 5 chốt phép chia chính thức.
2. Ngay cả khi trùng, fit sẵn một lần rồi dùng chung vẫn là cách sai về
   nguyên tắc: Pipeline **là một phần của quá trình huấn luyện**, nó học trung
   vị, phân vị, bảng hạng mục từ tập train. Tách nó ra ngoài thì mỗi lần đổi
   tập train phải nhớ fit lại, và quên một lần là rò rỉ.

Hai bộ feature, hai model
--------------------------
Mỗi thuật toán được train hai lần — bộ FULL và bộ RÚT GỌN (§7.2). Chênh lệch
PR-AUC giữa hai bộ chính là mục *"phân tích tính khả thi triển khai"*: bộ FULL
có `EXT_SOURCE_1/2/3` mà form không thu được.

Chấm trên validation, KHÔNG chạm test
--------------------------------------
Tập test khoá tới task 14. Mọi con số ở task 7–12 đều đo trên validation.
`fit_and_evaluate()` không nhận tập test — muốn chạm vào nó phải sửa chữ ký
hàm, tức phải cố ý.

Chỉ số train cũng được ghi, và ghi có mục đích: khoảng cách train − validation
là thứ duy nhất cho thấy model có học thuộc hay không. Với cây đơn thì đó
chính là vai trò của nó trong báo cáo (§6.3).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Final

import joblib
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from hfml.config import CONFIG
from hfml.logger import get_logger
from hfml.ml.evaluation.metrics import binary_confusion, binary_metrics
from hfml.ml.ml02_credit_risk.clean import (
    TARGET_COLUMN,
    load_clean_application,
    load_clean_bureau,
)
from hfml.ml.ml02_credit_risk.features import (
    aggregate_bureau,
    build_feature_pipeline,
    split_features_and_target,
)
from hfml.ml.ml02_credit_risk.imbalance import estimator_params, imbalance_params
from hfml.ml.ml02_credit_risk.split import load_split

log = get_logger(__name__)

#: Hai bộ feature của §7.2. Thứ tự cố định để bảng báo cáo ổn định.
FEATURE_SETS: Final[tuple[str, ...]] = ("reduced", "full")

#: Bốn thuật toán của F04, đúng thứ tự task 7 → 10.
DECISION_TREE: Final[str] = "decision_tree"
BAGGING: Final[str] = "bagging"
RANDOM_FOREST: Final[str] = "random_forest"
XGBOOST: Final[str] = "xgboost"

#: Thứ tự cố định cho bảng so sánh ở task 12.
ALGORITHMS: Final[tuple[str, ...]] = (
    DECISION_TREE, BAGGING, RANDOM_FOREST, XGBOOST,
)

#: Nơi để artifact TRUNG GIAN của task 7–10.
#:
#: Đây KHÔNG phải export — export là task 15, và nó ghi vào chỗ khác kèm đủ
#: metadata cho inference. Thư mục này chỉ để task 11–14 khỏi phải train lại
#: từ đầu, và train lại thì có nguy cơ ra số khác nếu một tham số nào đó lệch.
MODELS_SUBDIR: Final[str] = "ml02_models"


# --------------------------------------------------------------------------
# Siêu tham số
# --------------------------------------------------------------------------
#: Một lá phải đại diện ít nhất ngần này phần dân số train.
#:
#: Đây là quy tắc suy từ CỠ DỮ LIỆU chứ không phải một con số dò được: với
#: 215.257 dòng train thì 0,1% ≈ 215 hồ sơ, và ở tỉ lệ dương 8,07% thì một lá
#: như vậy chứa ~17 ca vỡ nợ. Dưới mức đó, xác suất của lá là ước lượng từ vài
#: quan sát — con số vô nghĩa, mà `predict_proba` vẫn trả về đều đặn.
#:
#: Cây không giới hạn gì sẽ chẻ tới lá một phần tử, đạt PR-AUC ~1,0 trên train
#: và sụp trên validation. Ràng buộc theo tỉ lệ dân số giữ cho quy tắc này
#: đúng cả khi cỡ dữ liệu đổi.
MIN_LEAF_SHARE: Final[float] = 0.001


#: Số cây của các thuật toán tập hợp (Bagging ở task 8, Random Forest ở task 9).
#:
#: Phải là MỘT con số dùng chung: Bagging và Random Forest khác nhau đúng ở
#: chỗ RF lấy mẫu thêm feature tại mỗi lát cắt. Cho hai bên số cây khác nhau
#: thì chênh lệch giữa chúng lẫn cả phần "nhiều cây hơn", và §6.3 không còn
#: đối chiếu được hai cơ chế giảm phương sai nữa.
#:
#: 50 chứ không phải 10 (mặc định sklearn) hay 500: phương sai của trung bình
#: giảm theo 1/n nên phần lợi lớn nhất nằm ở vài chục cây đầu, còn chi phí thì
#: tăng tuyến tính. Đây là lập luận về dạng đường cong, không phải một con số
#: dò được từ tập validation — dò trên validation là việc của bước tinh chỉnh.
N_ESTIMATORS: Final[int] = 50


def decision_tree_params(n_train: int) -> dict:
    """Siêu tham số Decision Tree, suy từ cỡ tập train.

    `max_depth=None` là có chủ ý: để `min_samples_leaf` làm việc điều tiết.
    Đặt thêm một `max_depth` cụ thể sẽ phải chọn con số, mà chọn con số nào
    thì cần thử — và thử là việc của bước tinh chỉnh, không phải task 7.
    """
    return {
        "max_depth": None,
        "min_samples_leaf": max(1, int(n_train * MIN_LEAF_SHARE)),
        "random_state": CONFIG.random_seed,
    }


# --------------------------------------------------------------------------
# Kết quả một lần train
# --------------------------------------------------------------------------
@dataclass
class TrainedModel:
    """Một lần train: thuật toán × bộ feature."""

    algo: str
    feature_set: str
    pipeline: Pipeline
    params: dict = field(default_factory=dict)
    metrics_train: dict[str, float] = field(default_factory=dict)
    metrics_validation: dict[str, float] = field(default_factory=dict)
    confusion_validation: pd.DataFrame = field(default_factory=pd.DataFrame)
    n_train: int = 0
    n_validation: int = 0
    n_features: int = 0

    @property
    def slug(self) -> str:
        return f"ml02_{self.algo}_{self.feature_set}"

    @property
    def pr_auc(self) -> float:
        return self.metrics_validation.get("pr_auc", float("nan"))

    @property
    def overfit_gap(self) -> float:
        """PR-AUC train − validation. Lớn = model học thuộc tập train."""
        return (self.metrics_train.get("pr_auc", float("nan"))
                - self.metrics_validation.get("pr_auc", float("nan")))

    def row(self) -> dict:
        """Một dòng cho bảng so sánh và cho `results.csv`."""
        return {
            "task": "ml02",
            "algo": self.algo,
            "feature_set": self.feature_set,
            "random_seed": CONFIG.random_seed,
            "n_rows": self.n_train,
            "n_splits": 1,              # holdout, KHÔNG K-Fold
            "split": "validation",
            "n_features": self.n_features,
            "note": f"train PR-AUC {self.metrics_train.get('pr_auc', float('nan')):.4f}",
            "overfit_gap": self.overfit_gap,
            **self.metrics_validation,
        }


# --------------------------------------------------------------------------
# Nạp dữ liệu theo đúng phép chia của task 5
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class TrainingData:
    """Dữ liệu train/validation, đã cắt theo phép chia đã lưu.

    KHÔNG có tập test — cố ý. Task 7–12 không có lý do gì chạm vào nó, và
    không có thuộc tính nào để chạm thì cũng không có cách nào lỡ tay.
    """

    X_train: pd.DataFrame
    y_train: pd.Series
    X_validation: pd.DataFrame
    y_validation: pd.Series
    bureau_aggregates: pd.DataFrame


def load_training_data(nrows: int | None = None) -> TrainingData:
    """Nạp dữ liệu sạch (task 2), gộp bureau (task 3), cắt theo split (task 5)."""
    df = load_clean_application()
    split = load_split()
    aggregates = aggregate_bureau(load_clean_bureau())

    train = split.apply(df, "train")
    validation = split.apply(df, "validation")
    if nrows:
        train, validation = train.head(nrows), validation.head(max(nrows // 4, 100))

    X_train, y_train = split_features_and_target(train)
    X_validation, y_validation = split_features_and_target(validation)

    log.info("Dữ liệu train %d · validation %d", len(X_train), len(X_validation))
    return TrainingData(X_train, y_train, X_validation, y_validation, aggregates)


# --------------------------------------------------------------------------
# Vòng train dùng chung
# --------------------------------------------------------------------------
def fit_and_evaluate(
    algo: str,
    data: TrainingData,
    make_estimator: Callable[[int, pd.Series], BaseEstimator],
    feature_set: str = "reduced",
) -> TrainedModel:
    """Fit một thuật toán trên train, chấm trên validation.

    `make_estimator(n_train, y_train)` trả về estimator đã gắn tham số cân
    bằng lớp. Nhận `y_train` chứ không nhận sẵn một con số: `scale_pos_weight`
    phải tính từ **riêng tập train** (task 4), và bắt hàm tạo nhận `y_train`
    làm điều đó thành bắt buộc chứ không phải thói quen tốt.

    Pipeline được `fit` trên train rồi mới `transform` validation. Không có
    đường nào để thống kê của validation lọt vào phép biến đổi.
    """
    pipeline = build_feature_pipeline(
        feature_set=feature_set, bureau_aggregates=data.bureau_aggregates)
    estimator = make_estimator(len(data.X_train), data.y_train)

    full = Pipeline([("features", pipeline), ("model", estimator)])
    log.info("── Train %s · bộ %s ──", algo, feature_set)
    full.fit(data.X_train, data.y_train)

    proba_train = full.predict_proba(data.X_train)[:, 1]
    proba_validation = full.predict_proba(data.X_validation)[:, 1]

    n_features = len(pipeline.named_steps["preprocess"].get_feature_names_out())
    result = TrainedModel(
        algo=algo,
        feature_set=feature_set,
        pipeline=full,
        params=estimator.get_params(deep=False),
        metrics_train=binary_metrics(data.y_train, proba_train),
        metrics_validation=binary_metrics(data.y_validation, proba_validation),
        confusion_validation=binary_confusion(data.y_validation, proba_validation),
        n_train=len(data.X_train),
        n_validation=len(data.X_validation),
        n_features=n_features,
    )
    log.info("%s · %s: validation PR-AUC %.4f · ROC-AUC %.4f · recall lớp 1 %.4f "
             "(gap train−val %.4f)",
             algo, feature_set, result.pr_auc,
             result.metrics_validation["roc_auc"],
             result.metrics_validation["recall_positive"], result.overfit_gap)
    return result


# --------------------------------------------------------------------------
# Task 7 — Decision Tree
# --------------------------------------------------------------------------
def train_decision_tree(
    data: TrainingData | None = None,
    feature_sets: tuple[str, ...] = FEATURE_SETS,
) -> list[TrainedModel]:
    """Task 7 — train Decision Tree trên cả hai bộ feature.

    `class_weight='balanced'` đến từ task 4, không hardcode ở đây: một chỗ
    duy nhất quyết định tỉ số phạt thì bốn thuật toán không thể lệch nhau.
    """
    data = data if data is not None else load_training_data()

    def make(n_train: int, y_train: pd.Series) -> BaseEstimator:
        return DecisionTreeClassifier(
            **decision_tree_params(n_train),
            **imbalance_params(DECISION_TREE, y_train),
        )

    return [fit_and_evaluate(DECISION_TREE, data, make, feature_set=fs)
            for fs in feature_sets]


# --------------------------------------------------------------------------
# Task 8 — Bagging Classifier
# --------------------------------------------------------------------------
def train_bagging(
    data: TrainingData | None = None,
    feature_sets: tuple[str, ...] = FEATURE_SETS,
) -> list[TrainedModel]:
    """Task 8 — train Bagging trên cả hai bộ feature.

    **Cây con dùng ĐÚNG siêu tham số của task 7.** Đó là điều kiện để đọc được
    bảng so sánh: Bagging chỉ khác Decision Tree ở chỗ lấy 50 mẫu bootstrap rồi
    trung bình lại, nên chênh lệch PR-AUC giữa hai dòng **chính là** phần do
    giảm phương sai đem lại. Cho cây con một `min_samples_leaf` khác thì con số
    đó lẫn cả phần "cây được điều tiết khác đi", và §6.3 mất chỗ đối chiếu.

    `class_weight` đặt trên **cây con**, không phải trên `BaggingClassifier` —
    lớp đó không có tham số này (task 4). Truyền nhầm lên ngoài sẽ `TypeError`;
    tệ hơn là nếu bị nuốt trong `**kwargs` thì model train mất cân bằng trong
    khi bảng cấu hình vẫn ghi là đã cân bằng.

    Giữ `max_features=1.0` (mặc định) một cách có chủ ý: đó là ranh giới với
    Random Forest ở task 9, vốn khác đúng ở chỗ lấy mẫu feature tại mỗi lát cắt.
    """
    data = data if data is not None else load_training_data()

    def make(n_train: int, y_train: pd.Series) -> BaseEstimator:
        base = DecisionTreeClassifier(
            **decision_tree_params(n_train),
            **estimator_params(BAGGING),        # class_weight cho CÂY CON
        )
        return BaggingClassifier(
            estimator=base,
            n_estimators=N_ESTIMATORS,
            random_state=CONFIG.random_seed,
            n_jobs=-1,
            **imbalance_params(BAGGING, y_train),   # rỗng — xem docstring
        )

    return [fit_and_evaluate(BAGGING, data, make, feature_set=fs)
            for fs in feature_sets]


# --------------------------------------------------------------------------
# Task 9 — Random Forest
# --------------------------------------------------------------------------
#: Số feature xét tại MỖI lát cắt. `"sqrt"` là mặc định của sklearn cho phân
#: loại, và ở đây nó là **thứ duy nhất** phân biệt Random Forest với Bagging:
#:
#:     Bagging         lấy mẫu bootstrap DÒNG, mỗi lát cắt xét ĐỦ cột
#:     Random Forest   lấy mẫu bootstrap DÒNG + mỗi lát cắt chỉ xét √p cột
#:
#: √17 ≈ 4 cột ở bộ rút gọn, √82 ≈ 9 cột ở bộ full. Việc ép mỗi lát cắt bỏ qua
#: phần lớn cột làm các cây bớt giống nhau — đó là cơ chế giảm phương sai thứ
#: hai mà Bagging không có, và chênh lệch giữa hai dòng trong bảng đo đúng nó.
RF_MAX_FEATURES: Final[str] = "sqrt"


def train_random_forest(
    data: TrainingData | None = None,
    feature_sets: tuple[str, ...] = FEATURE_SETS,
) -> list[TrainedModel]:
    """Task 9 — train Random Forest trên cả hai bộ feature.

    Giữ nguyên mọi thứ của task 8 trừ `max_features`: cùng 50 cây, cùng
    `min_samples_leaf` suy từ cỡ dữ liệu, cùng tỉ số phạt, cùng seed. Nhờ vậy
    chênh lệch PR-AUC so với Bagging đọc được **chính xác** là đóng góp của
    việc lấy mẫu feature, không lẫn thứ gì khác.

    `class_weight='balanced'` (không phải `'balanced_subsample'`) đến từ task 4.
    `'balanced_subsample'` tính lại trọng số trên từng mẫu bootstrap nên tỉ số
    phạt dao động quanh 11,39 thay vì đúng bằng nó — và khi đó Random Forest
    không còn nhận cùng một mức phạt với ba thuật toán kia, tức bảng so sánh ở
    task 12 so trên hai sân khác nhau.
    """
    data = data if data is not None else load_training_data()

    def make(n_train: int, y_train: pd.Series) -> BaseEstimator:
        params = decision_tree_params(n_train)
        return RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features=RF_MAX_FEATURES,
            random_state=CONFIG.random_seed,
            n_jobs=-1,
            **imbalance_params(RANDOM_FOREST, y_train),
        )

    return [fit_and_evaluate(RANDOM_FOREST, data, make, feature_set=fs)
            for fs in feature_sets]


# --------------------------------------------------------------------------
# Task 10 — XGBoost
# --------------------------------------------------------------------------
#: Siêu tham số XGBoost — cấu hình CỐ ĐỊNH, chọn trước khi nhìn validation.
#:
#: Vì sao KHÔNG ép `n_estimators = 50` cho "công bằng" với task 8, 9: boosting
#: và bagging dùng cây theo hai cách khác hẳn nhau. Bagging trung bình 50 cây
#: ĐỘC LẬP mỗi cây đã là một model đủ mạnh; boosting cộng dồn hàng trăm cây NÔNG,
#: mỗi cây chỉ sửa phần dư của các cây trước. Bắt hai bên cùng số cây là so số
#: lượng của hai thứ không cùng đơn vị. Cái phải bằng nhau là **điều kiện thí
#: nghiệm** — phép chia, Pipeline, tỉ số phạt, seed, bộ chỉ số — chứ không phải
#: siêu tham số nội bộ của từng họ thuật toán.
#:
#: `learning_rate=0.1` + `n_estimators=200` thay cho mặc định `0.3` + `100`:
#: bước nhỏ hơn và nhiều bước hơn là cách khắc phục sách vở cho việc boosting
#: học thuộc, giữ tổng năng lực xấp xỉ như cũ. Đây là lựa chọn ĐẶT TRƯỚC, không
#: dò trên validation — dò trên chính tập dùng để báo cáo thì con số báo cáo
#: thành lạc quan.
XGBOOST_PARAMS: Final[dict] = {
    "n_estimators": 200,
    "learning_rate": 0.1,
    "max_depth": 6,              # mặc định của thư viện
    # Lấy mẫu ngẫu nhiên dòng và cột — điều tiết chuẩn của boosting ngẫu nhiên.
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    # Chỉ số theo dõi khớp với chỉ số CHỌN MODEL của ML02 (§7.3). Để mặc định
    # `logloss` thì thứ XGBoost tối ưu hoá bên trong lệch khỏi thứ đem đi so.
    "eval_metric": "aucpr",
    # `hist` cho 215.257 dòng — cùng kết quả, nhanh hơn nhiều so với `exact`.
    "tree_method": "hist",
}


def train_xgboost(
    data: TrainingData | None = None,
    feature_sets: tuple[str, ...] = FEATURE_SETS,
) -> list[TrainedModel]:
    """Task 10 — train XGBoost trên cả hai bộ feature.

    Cân bằng lớp bằng `scale_pos_weight`, KHÔNG phải `class_weight` — XGBoost
    không có tham số đó. Con số lấy từ `imbalance_params(XGBOOST, y_train)` của
    task 4, và nó được tính từ **riêng tập train**: đã kiểm ở task 4 rằng tỉ số
    này trùng khít tỉ số trọng số mà `class_weight='balanced'` sinh ra cho ba
    thuật toán kia (11,387150, sáu chữ số). Nhờ vậy cả bốn nhận đúng cùng một
    mức phạt dù dùng hai cơ chế khác tên.

    KHÔNG dùng early stopping. Early stopping cần một tập để dừng, mà tập đó ở
    đây chỉ có thể là validation — tức chính tập dùng để báo cáo và để chọn
    model ở task 12. Dừng theo nó rồi lại chấm trên nó là chọn tham số trên
    chính tập đánh giá, và con số báo cáo sẽ lạc quan hơn thực tế.
    """
    data = data if data is not None else load_training_data()

    def make(n_train: int, y_train: pd.Series) -> BaseEstimator:
        return XGBClassifier(
            **XGBOOST_PARAMS,
            random_state=CONFIG.random_seed,
            n_jobs=-1,
            **imbalance_params(XGBOOST, y_train),
        )

    return [fit_and_evaluate(XGBOOST, data, make, feature_set=fs)
            for fs in feature_sets]


# --------------------------------------------------------------------------
# Lưu kết quả
# --------------------------------------------------------------------------
def models_dir() -> Path:
    return CONFIG.paths.runs / MODELS_SUBDIR


def save_run(model: TrainedModel) -> dict[str, Path]:
    """Ghi artifact trung gian + chỉ số của một lần train.

    ⚠️ KHÔNG phải export. Task 15 mới export model được chọn, kèm feature list,
    label mapping và metadata cho inference. File ở đây chỉ để task 11–14 khỏi
    phải train lại — train lại thì có nguy cơ ra số khác nếu một tham số lệch.
    """
    out_dir = models_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    path = out_dir / f"{model.slug}.joblib"
    joblib.dump(model.pipeline, path)
    written["pipeline"] = path

    path = out_dir / f"{model.slug}.confusion.csv"
    model.confusion_validation.to_csv(path, encoding="utf-8")
    written["confusion"] = path

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "task": "ML02 task 7–10 — train",
        "artifact_kind": "intermediate",   # export là task 15
        "algo": model.algo,
        "feature_set": model.feature_set,
        "random_seed": CONFIG.random_seed,
        "n_train": model.n_train,
        "n_validation": model.n_validation,
        "n_features": model.n_features,
        "params": {k: str(v) for k, v in model.params.items()},
        "metrics_train": model.metrics_train,
        "metrics_validation": model.metrics_validation,
        "overfit_gap_pr_auc": model.overfit_gap,
        "evaluated_on": "validation",
        "test_set_touched": False,
    }
    path = out_dir / f"{model.slug}.metadata.json"
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    written["metadata"] = path
    return written


def results_frame(models: list[TrainedModel]) -> pd.DataFrame:
    return pd.DataFrame([m.row() for m in models])
