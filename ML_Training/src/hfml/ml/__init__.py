"""Tầng ml — hai bài toán phân loại (F03, F04 · M03, M04 · Tuần 3–4).

Tầng này chỉ trả NHÃN + XÁC SUẤT. Không tự sinh khuyến nghị — việc diễn giải
kết quả thành lời khuyên thuộc về `hfml.rules` và `hfml.llm`.

    base.py                  Contract chung: fit / predict / predict_proba
    registry.py              Lưu & tải artifact + metadata
    ml01_recommendation/     ML01 — 4 nhóm khuyến nghị (synthetic, F03)
    ml02_credit_risk/        ML02 — rủi ro tín dụng Home Credit (nhãn thật, F04)
    evaluation/              metrics + plots dùng chung cho cả hai

Cả hai bài toán chạy CÙNG một giao thức để bảng so sánh có nghĩa
(PLAN.md §11): cùng `random_seed = 42`, cùng `StratifiedKFold`, cùng
`DummyClassifier(strategy='stratified')` làm baseline, và cùng 4 thuật toán
đại diện 4 nhóm:

    DecisionTreeClassifier   Trees     — baseline diễn giải được
    BaggingClassifier        Bagging   — giảm variance
    RandomForestClassifier   Forests   — bagging + lấy mẫu feature
    XGBoost                  Boosting  — giảm bias

Chỉ số chọn model khác nhau giữa hai bài toán, và đây là điểm dễ bị hỏi nhất:

    ML01 → Macro-F1   (4 lớp không cân bằng)
    ML02 → PR-AUC     (8,07% dương; đoán "không ai vỡ nợ" đã đạt 91,93%
                       accuracy, nên accuracy KHÔNG dùng để chọn model)
"""
