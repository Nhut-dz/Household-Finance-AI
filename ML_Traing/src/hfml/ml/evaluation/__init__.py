"""Đánh giá model — dùng chung cho ML01 và ML02 (PLAN.md §11).

    metrics.py   classification_metrics(): accuracy, macro-F1, per-class P/R,
                 PR-AUC, ROC-AUC, Brier score
    plots.py     confusion matrix, PR curve, calibration curve, SHAP summary

Giao thức đánh giá — bảng đối chiếu:

                        ML01                        ML02
    Nguồn nhãn          Synthetic, hàm g(·)         TARGET (nhãn thật)
    Số lớp              4                           2
    Cân bằng            mỗi lớp ≥ 10%               8,07% dương
    Chia dữ liệu        StratifiedKFold             StratifiedKFold
    CHỌN MODEL          Macro-F1                    PR-AUC
    Báo cáo kèm         Accuracy, CM, per-class     ROC-AUC, Macro-F1, CM, Accuracy
    Baseline            DummyClassifier(stratified) DummyClassifier(stratified)
    Calibration         —                           CalibratedClassifierCV + curve + Brier
    Giải thích          Feature importance          Feature importance + SHAP

Accuracy được báo cáo đầy đủ trong bảng cho đủ yêu cầu môn học, nhưng KHÔNG
được dùng để kết luận ở ML02 — xem docstring của `hfml.ml`.

Mọi kết quả ghi thêm một dòng vào `experiments/results.csv`: model, feature
set, seed, toàn bộ metric, ngày chạy (F07 task 1).
"""
