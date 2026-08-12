# Tích hợp ML01 vào hệ thống — FE → BE → ML → BE → FE

Model ML01 (`ml01_xgboost_vfinal`, chọn ở F03 task 14) được đưa vào luồng chạy
thật qua ba service độc lập.

```
┌──────────────┐   GET /api/households/{id}/prediction   ┌──────────────┐
│  FE  :5173   │ ─────────────────────────────────────►  │  BE  :8000   │
│ React + Vite │ ◄─────────────────────────────────────  │   Laravel    │
└──────────────┘        { label, probabilities }         └──────┬───────┘
                                                                │ POST /predict
                                                   17 feature   │ (JSON)
                                                                ▼
                                                         ┌──────────────┐
                                                         │  ML  :8001   │
                                                         │   FastAPI    │
                                                         │  XGBoost     │
                                                         └──────────────┘
```

## Chạy ba service

Mở ba terminal, theo thứ tự này — BE gọi ML lúc có request nên ML cần sẵn sàng
trước, còn FE chỉ cần BE.

### 1. ML service (cổng 8001)

```powershell
cd Projects\ML\Household-Finance-ML-Python
.venv\Scripts\python.exe -m uvicorn hfml.api.main:app --host 127.0.0.1 --port 8001
```

Kiểm tra:

```powershell
curl http://127.0.0.1:8001/health
```

`ml01.loaded` phải là `true`. Nếu `false`, thiếu artifact — chạy train và export
trước (xem [PLAN.md](../PLAN.md) §6, task 7–15):

```powershell
.venv\Scripts\python.exe scripts\train_ml01.py
```

> Dùng `.venv\Scripts\python.exe`, **không** dùng `python` — package `hfml` chỉ
> cài trong venv.

### 2. BE Laravel (cổng 8000)

```powershell
cd Projects\BE\Household-Finance-BE-App
php artisan serve --host=127.0.0.1 --port=8000
```

Cần trong `.env`:

```
PYTHON_ADVISOR_URL=http://127.0.0.1:8001
PYTHON_ADVISOR_TIMEOUT=30
PYTHON_ADVISOR_TOKEN=
```

Đổi `.env` thì phải `php artisan config:clear`.

### 3. FE React (cổng 5173)

```powershell
cd Projects\Front-End\Household-Finance-FE-App
npm run dev
```

FE đọc `VITE_API_BASE_URL`, mặc định `http://127.0.0.1:8000/api`.

## Luồng dữ liệu

1. Người dùng nhập hồ sơ ở màn **Nhập thông tin** → `POST /api/households`.
2. Sang màn **Phương án đề xuất**, `PredictionCard` gọi
   `GET /api/households/{id}/prediction`.
3. BE xét quyền sở hữu (`user_id` khi có Bearer token, `guest_session_id` khi
   không), quy đổi hồ sơ sang **17 feature** rồi `POST /predict` sang ML.
4. ML nạp artifact (lần đầu, sau đó giữ trong bộ nhớ), dự đoán, trả nhãn + xác
   suất 4 lớp.
5. FE hiển thị nhãn, độ tin cậy và thanh xác suất từng lớp.

## Hai chỗ hai bên không khớp, xử lý ở BE

Đây là phần dễ hỏng nhất của tích hợp, ghi lại để sau khỏi phải dò lại.

### `age` — DB chỉ có `birth_year`, và cho phép trống

Model bắt buộc có `age`. `AdvisorClient::predictionPayload()` tính
`năm hiện tại − birth_year`. Thiếu `birth_year` thì ném
`MissingBirthYearException` → **422** kèm lỗi ở field `birth_year`, để form biết
ô nào cần điền.

Cố ý không điền tuổi mặc định: điền bừa thì model vẫn trả về một nhãn trông hợp
lý, và không ai biết nó dựa trên tuổi bịa.

### Tài sản — hai bộ từ vựng cùng tồn tại

| Nguồn | Giá trị |
|---|---|
| `AssetTypeEnum` (BE) | `house` · `car` · `land` · `other` |
| ML01 được train trên | `cash` · `vehicle` · `real_estate` · `insurance` · `gold` · `investment` |
| **Thực tế trong DB** | cả hai bộ, **cộng** `savings_certificate` |

`tblassets.asset_type` là `string(50)` không ràng buộc nên mọi giá trị đều lọt
vào được. `AdvisorClient::ASSET_TO_FEATURE` map cả hai bộ; giá trị lạ
(`savings_certificate`, `other`) bị bỏ qua và cột tương ứng để `false`.

**Quan trọng:** payload đọc `$asset->getRawOriginal('asset_type')`, KHÔNG đọc
`$asset->asset_type`. Cast của model ép sang `AssetTypeEnum` (4 giá trị), nên
truy cập thuộc tính đã cast với một hàng `cash` sẽ ném `ValueError` và làm sập
cả lời gọi.

## Mã lỗi

| Mã | Khi nào | FE làm gì |
|---|---|---|
| 200 | Thành công | Hiện nhãn + xác suất |
| 403 | Hồ sơ không thuộc người gọi | Hiện thông báo |
| 404 | Không tìm thấy hồ sơ | Hiện thông báo |
| 422 | Hồ sơ thiếu năm sinh | Hiện lỗi, mời bổ sung |
| 503 | ML chưa cấu hình / không phản hồi | Hiện cảnh báo vàng, phần còn lại của trang vẫn dùng được |

Thẻ dự đoán tự gọi API riêng, không dùng chung lần tải với "phương án đề xuất" —
hồ sơ chưa được tính phương án (404) vẫn xem được nhóm của mình.

## Ngưỡng tin cậy

`confidence_threshold` trong [config/config.yaml](../config/config.yaml) hiện là
`0.60`. Xác suất cao nhất dưới ngưỡng thì response có `low_confidence: true`, và
FE hiện cảnh báo rằng kết quả chỉ là gợi ý tham khảo. Hiển thị nhãn như một kết
luận chắc chắn trong khi model đang phân vân là chỗ người dùng bị dẫn sai.

## Kiểm nhanh từng chặng

```powershell
# ML đứng một mình
curl -X POST http://127.0.0.1:8001/predict -H "Content-Type: application/json" -d "{\"average_monthly_income\":40000000,\"average_monthly_expense\":18000000,\"savings_amount\":300000000,\"total_current_debt\":0,\"monthly_debt_payment\":0,\"household_size\":3,\"children_count\":1,\"age\":40,\"has_debt\":false,\"has_savings\":true,\"has_dependents\":false}"

# BE → ML  (thay id và guest_session_id bằng hồ sơ có thật)
curl "http://127.0.0.1:8000/api/households/{id}/prediction?guest_session_id={token}"
```

Thiếu hoặc thừa field ở `/predict` đều trả **422** kèm tên field — `extra="forbid"`
trong schema cố ý bắt lỗi map tên thay vì nuốt im.
