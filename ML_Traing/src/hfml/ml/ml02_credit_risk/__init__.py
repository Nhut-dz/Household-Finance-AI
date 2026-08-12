"""ML02 — Home Credit Risk Classification (F04 · M04 · Tuần 4).

`application_train.csv`: 307.511 hồ sơ, nhãn `TARGET` (1 = khó khăn trả nợ).
Đây là NHÃN THẬT, thu thập độc lập — trụ cột ML mạnh nhất của đồ án.

    features.py   Ánh xạ form ↔ Home Credit, dựng 2 feature set
    train.py      Train 4 thuật toán × 2 phiên bản, calibrate, export

Train HAI phiên bản để so sánh (PLAN.md §7.2):

    Full      Toàn bộ feature, kể cả EXT_SOURCE_1/2/3 → chứng minh năng lực
              kỹ thuật, AUC cao
    Rút gọn   Chỉ feature ánh xạ được từ form → model THỰC SỰ DEPLOY

`EXT_SOURCE_1/2/3` là điểm tín dụng từ nguồn ngoài, nhóm feature mạnh nhất
của Home Credit nhưng KHÔNG thu được từ form. Bảng so sánh AUC hai phiên bản
chính là mục "phân tích tính khả thi triển khai" trong báo cáo.

Class imbalance: 8,07% dương (24.825 / 307.511). Xử lý bằng
`scale_pos_weight ≈ 11,4` (XGBoost) hoặc `class_weight='balanced'`
(DT / RF / Bagging).

Calibration là bắt buộc, không phải tùy chọn: hệ thống ra quyết định theo
ngưỡng xác suất, mà xác suất chưa calibrate thì ngưỡng vô nghĩa. Dùng
`CalibratedClassifierCV` + calibration curve + Brier score.

Giải thích: built-in importance + SHAP summary plot (global). SHAP local
top-5 cho một hồ sơ được đưa vào JSON ở tầng `pipeline` để `llm` diễn đạt.

Bài toán DUYỆT VAY nằm ngoài phạm vi — ML02 chỉ ước lượng xác suất tham
khảo, không phải cam kết và không thay thế quyết định của tổ chức tín dụng.

Phạm vi chạy (chốt 11/08/2026): ML02 CHỈ chạy khi
`HouseholdProfile.needs_loan_analysis` là True, tức người dùng có chọn một
nhu cầu trong `LOAN_TRIGGER_NEEDS`. Lý do gồm cả kỹ thuật lẫn sản phẩm:

    kỹ thuật  5 ô của khối vay (nghề nghiệp, số năm đi làm, giá tài sản,
              số tiền vay, kỳ hạn) chỉ được hỏi trong trường hợp đó, nên
              thiếu chúng thì phần lớn feature là NaN.
    sản phẩm  Đưa "xác suất vỡ nợ 8%" cho người không hề định vay là một
              con số vô nghĩa với họ và dễ bị hiểu nhầm.

Với hồ sơ không có nhu cầu vay, hệ thống trả về 4 rule + nhóm khuyến nghị
ML01, và tầng `llm` phải nói rõ là không có phần đánh giá rủi ro tín dụng
chứ không im lặng bỏ qua.
"""
