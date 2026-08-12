"""Tầng pipeline — inference & tổng hợp Rule + ML (F05 · M05 · Tuần 5).

Nơi một hồ sơ chạy xuyên suốt: form → validate → preprocessing → 5 rule +
ML01 + ML02 → một structured result JSON duy nhất.

    normalizer.py     task 1, 2   Chuẩn hóa input, áp lại Pipeline đã dump ở F01
    predictor.py      task 4, 5   Inference ML01 và ML02
    result.py                     Schema cố định của structured result
    orchestrator.py   task 6, 7   Gom kết quả Rule + ML, kiểm tra confidence
    analyze.py        M08 task 15 Một hàm analyze(payload) -> Result

Đầu ra cố định schema, chứa: kết quả 5 rule, nhãn + xác suất ML01,
P(vỡ nợ) + nhóm rủi ro ML02, SHAP top-5, cờ cảnh báo dữ liệu bất thường.

Quy tắc hạ cấp khi thiếu tin cậy (task 7) — quan trọng: nếu
`max(predict_proba) < ngưỡng tin cậy` thì hạ xuống kết luận của rule và đánh
dấu `low_confidence: true` trong JSON. Tầng `llm` phải diễn đạt điều này ra
thành lời, KHÔNG được im lặng bỏ qua.

Preprocessing lúc inference phải là ĐÚNG object đã `joblib.dump` cùng model
ở F01, không phải dựng lại bằng tay — dựng lại là lệch so với training.
"""
