# BÁO CÁO CÁC FILE ĐÃ CHỈNH SỬA VÀ CẬP NHẬT (CONFIGFILE.MD)

---

## 📜 NGUYÊN TẮC LÀM VIỆC & QUY TRÌNH THAY ĐỔI (OPERATING PRINCIPLES)

1. **Ghi chép đầy đủ Timeline (Ngày & Giờ)**: Mọi thao tác kiểm tra, sửa đổi code hoặc thêm mới file đều phải được đánh dấu mốc thời gian rõ ràng (YYYY-MM-DD HH:mm:ss).
2. **Liệt kê chi tiết vị trí & nội dung**: Nêu rõ tên file, đường dẫn tuyệt đối, vị trí dòng bị sửa hoặc thêm mới, nguyên nhân và tác dụng của thay đổi.
3. **Quy tắc Kiểm tra trước - Hỏi ý kiến trước khi Sửa**:
   - **Bước 1**: Rà soát, phát hiện và phân tích vấn đề chi tiết trước.
   - **Bước 2**: Đưa danh sách vấn đề ra cho Người dùng (User) xem xét và hỏi ý kiến có muốn fix hay không.
   - **Bước 3**: **CHỈ thực hiện sửa code khi Người dùng đồng ý.**

---

## 🕒 TIMELINE NHẬT KÝ THAY ĐỔI NỘI DUNG CODE (NHẬT KÝ CHI TIẾT)

### 📌 [2026-08-17 09:48:00] - Fix lỗi gán nhãn ML01 (Thu nhập cao, chi phí thấp, tiết kiệm = 0 bị đánh nhãn sai thành EMERGENCY)

- **Vị trí sửa**: ML_Training/src/hfml/ml/ml01_recommendation/labeler.py (Dòng 123 – 130)
- **Đường dẫn**: d:\CS116\DAHP\Household-Finance-ML-Python\src\hfml\ml\ml01_recommendation\labeler.py và D:\CS116\Household-Finance-AI\ML_Training\src\hfml\ml\ml01_recommendation\labeler.py
- **Nội dung thay đổi**:
  - *Trước*: (ind["savings_rate"] < 0) | (ind["savings_months"] < t.emergency_savings_months)
  - *Sau*: (ind["savings_rate"] < 0) | ((ind["savings_months"] < t.emergency_savings_months) & (ind["savings_rate"] < t.buffer_savings_rate))
- **Tác dụng**: Giúp các hộ thu nhập cao, chi phí thấp (dư thặng dư lớn nhưng chưa có tiết kiệm) không bị gán nhãn sai thành Nguy cấp (EMERGENCY), mà được chuyển đúng về nhóm BUILD_BUFFER.

---

### 📌 [2026-08-17 11:02:00] - Huấn luyện lại và đồng bộ Mô hình ML01 cho thư mục Household-Finance-AI/ML_Training

- **Thao tác**: Chạy retrain mô hình XGBoost ML01 trên tập dữ liệu gán nhãn chuẩn mới tại D:\CS116\Household-Finance-AI\ML_Training.

---

### 📌 [2026-08-18 14:47:00] - Thực thi Toàn bộ Migration CSDL PostgreSQL bằng PHP 8.4 (C:\php84\php.exe)

- **Trạng thái**: Toàn bộ 14/14 file migration trong thư mục database/migrations đã đạt trạng thái 100% Ran trên CSDL PostgreSQL (household_finance). Tạo bảng 	blconversations và 	blloan_applications kết nối thông suốt.

---

### 📌 [2026-08-18 15:56:00] - Nâng cấp Bộ Diễn Đạt Văn Bản Động cho ML01 & Rule Base trong 
arrator.py

- **Vị trí sửa đổi**: D:\CS116\Household-Finance-AI\ML_Training\src\hfml\llm\narrator.py
- **Nội dung cập nhật**: Nâng cấp toàn bộ câu văn nhận xét và tư vấn cho 4 nhóm định hướng ML01 (EMERGENCY, DEBT_FOCUS, BUILD_BUFFER, GROWTH) thành văn phong tài chính cá nhân tự nhiên, thân thiện.

---

### 📌 [2026-08-18 16:30:00] - Fix Lỗi Dữ liệu Tài sản (ssets), Phụng dưỡng người già (supports_elderly) & Dòng tiền thâm hụt

- **Vị trí sửa đổi**:
  - D:\CS116\Household-Finance-AI\Back-end\app\Services\AdvisorClient.php
  - d:\CS116\DAHP\Household-Finance-BE-App\app\Services\AdvisorClient.php
  - D:\CS116\Household-Finance-AI\ML_Training\src\hfml\api\main.py
- **Nội dung cập nhật**:
  1. Thêm nạp mảng ssets (house, car, land) trong householdPayload() phía Laravel để truyền sang Python API. Kết quả hiển thị đúng "Tài sản: Bất động sản, Phương tiện (Xe)".
  2. Kết nối biến supports_elderly để ghi nhận mốc đệm dự phòng khẩn cấp 6 tháng cho hộ có người già.
  3. Xử lý câu hiển thị số dư thâm hụt khi dòng tiền bị âm: "Thâm hụt (bội chi) khoảng 25,000,000 VNĐ/tháng (DEFICIT)".

---

### 📌 [2026-08-18 17:00:00] - Nâng cấp Tầng LLM Context Builder & Chuyển đổi Mã Kỹ thuật sang Tiếng Việt Tự nhiên (Theo OKR F05)

- **Vị trí sửa đổi**:
  - D:\CS116\Household-Finance-AI\ML_Training\src\hfml\llm\client.py (Hàm _template_answer)
  - D:\CS116\Household-Finance-AI\ML_Training\src\hfml\api\main.py (Bảng status_map & ule_names)
- **Nội dung cập nhật**:
  - Ánh xạ toàn bộ mã kỹ thuật khô cứng (CRITICAL, DEFICIT, REJECTED, RB01, RB02, RB05) sang cụm từ tiếng Việt tự nhiên.

---

### 📌 [2026-08-18 17:21:00] - Hiển thị Tên Tài sản Chi tiết (Nhà ở, Đất đai, Phương tiện (Xe)) & Nâng cấp Render Markdown HTML (FormattedText)

- **Vị trí sửa đổi**:
  - D:\CS116\Household-Finance-AI\ML_Training\src\hfml\api\main.py
  - D:\CS116\Household-Finance-AI\Front-end\src\pages\ChatbotPage.tsx
- **Nội dung cập nhật**:
  1. Tách biệt house ➔ **"Nhà ở"**, land ➔ **"Đất đai"**, car ➔ **"Phương tiện (Xe)"**. Kết quả hiển thị chính xác từng loại tài sản: "Tài sản: Nhà ở, Đất đai, Phương tiện (Xe)".
  2. Đưa thành phần đánh giá tài sản tích lũy trực tiếp vào khối phân tích tổng quan ML01 / Chẩn đoán sức khỏe tài chính.
  3. Nhúng component FormattedText vào ChatbotPage.tsx chuyển toàn bộ ký tự **chữ in đậm** thành thẻ HTML <strong> sắc nét và dấu gạch đầu dòng màu tím cho giao diện Frontend.