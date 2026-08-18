# BÁO CÁO CÁC FILE ĐÃ CHỈNH SỬA VÀ CẬP NHẬT (CONFIGFILE.MD)

Ngày cập nhật: **17/08/2026**
Nội dung công việc: **Sửa lỗi đánh giá nhãn ML01 (Thu nhập cao, chi phí thấp, tiết kiệm = 0 bị đánh nhãn sai thành EMERGENCY).**

---

## 📋 DANH SÁCH FILE BỊ SỬA ĐỔI / CẬP NHẬT

### 1. File sửa đổi logic chính: `ML_Training/src/hfml/ml/ml01_recommendation/labeler.py`
- **Đường dẫn file**: `d:\CS116\DAHP\Household-Finance-ML-Python\src\hfml\ml\ml01_recommendation\labeler.py` (và `d:\CS116\Household-Finance-AI\ML_Training\src\hfml\ml\ml01_recommendation\labeler.py`)
- **Vị trí sửa**: Dòng `123 - 130` (trong hàm `label_frame`).
- **Nội dung sửa chi tiết**:
  - **Mã nguồn cũ (Trước khi sửa)**:
    ```python
    conditions = [
        (ind["savings_rate"] < 0) | (ind["savings_months"] < t.emergency_savings_months),
        ind["dti"] >= t.debt_focus_dti,
        (ind["savings_months"] < t.buffer_savings_months)
        | (ind["savings_rate"] < t.buffer_savings_rate),
    ]
    ```
  - **Mã nguồn mới (Sau khi sửa)**:
    ```python
    conditions = [
        (ind["savings_rate"] < 0)
        | ((ind["savings_months"] < t.emergency_savings_months) & (ind["savings_rate"] < t.buffer_savings_rate)),
        ind["dti"] >= t.debt_focus_dti,
        (ind["savings_months"] < t.buffer_savings_months)
        | (ind["savings_rate"] < t.buffer_savings_rate),
    ]
    ```
- **Giải thích tác dụng**:
  Yêu cầu kết hợp toán tử `&` kiểm tra `savings_rate < 0.10` khi `savings_months < 1.0`. Nếu một hộ có thu nhập 200M, chi 20M (tỷ lệ tiết kiệm/thặng dư `savings_rate = 90%`), hộ này sẽ **KHÔNG bị ép vào nhóm EMERGENCY (Nguy cấp)** nữa, mà được hạ xuống nhóm **`BUILD_BUFFER` (Cần xây dựng quỹ dự phòng)** đúng với bản chất tài chính.

---

### 2. Files Artifact mô hình ML được huấn luyện lại:
- **`src/training/runs/ml01_xgboost_v1.joblib`** (File artifact mô hình XGBoost mới được retrain).
- **`src/training/runs/ml01_xgboost_vfinal.joblib`** (File mô hình chạy chính thức cho API endpoint `/predict`).
- **`src/training/runs/results.csv`** (Ghi nhận nhật ký lượt train mới).

---

## ✅ KẾT QUẢ ĐẠT ĐƯỢC SAU KHI FIX
- Với số liệu test: **Thu nhập = 200.000.000 VNĐ**, **Chi phí = 20.000.000 VNĐ**, **Tiết kiệm = 0 VNĐ**, **Nợ = 0 VNĐ**:
  - **Trước khi sửa**: Dự đoán nhãn = `EMERGENCY` (Tài chính nguy cấp / Mức rủi ro cao).
  - **Sau khi sửa & Retrain**: Dự đoán nhãn = **`BUILD_BUFFER` (Cần xây dựng quỹ dự phòng)** với độ tin cậy **99.94%**.
