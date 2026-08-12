# API mà Frontend cần

Tài liệu do phía FE (`Household-Finance-FE-App`) đề xuất, đối chiếu với backend
`Household-Finance-BE-App`. Phần tính toán tư vấn của 4 nút gợi ý trong Chatbot
(Gói tiết kiệm / Gói đầu tư / Gói vay mua nhà / Quy tắc 50-30-20) do nhóm Python
đảm nhiệm nên **không nằm trong tài liệu này**.

## Quy ước chung

- Base URL: `http://127.0.0.1:8000/api` (FE đọc từ biến `VITE_API_BASE_URL`).
- Mọi phản hồi giữ đúng khung của `ApiResponseTrait`:

```jsonc
// Thành công
{ "status": true, "message": "...", "result": { "data": { } } }

// Thất bại
{ "status": false, "message": "...", "result": { "errors": { "field": ["..."] } } }
```

- Xác thực: `Authorization: Bearer <token>` (Sanctum). Khi chưa đăng nhập, FE gửi
  `guest_session_id` — một UUID sinh ở lần đầu vào trang và lưu trong
  `localStorage`, giữ nguyên giữa các phiên.
- Enum dùng chung: `assets` ∈ `house | car | land | other`,
  `financial_needs` ∈ `buy_house | buy_car | buy_land | loan | other`.
- Tiền tệ gửi/nhận dạng số (VNĐ, không định dạng): `35000000`.

## Trạng thái

Cập nhật 06/08/2026: **backend đã làm xong toàn bộ**, FE đã nối vào các endpoint
bên dưới. Riêng nhóm xác thực (mục 3) mới có API, giao diện đăng nhập chưa dựng.

| API | Trạng thái |
| --- | --- |
| `POST /households` | Đã có, FE đã gọi |
| `GET /households/latest`, `GET/PUT/DELETE /households/{id}` | Đã có, FE đã gọi |
| `GET /households/{id}/proposal` | Đã có, FE đã gọi |
| `GET/POST /households/{id}/messages` | Đã có, FE đã gọi |
| `POST /auth/register`, `/auth/login`, `/auth/logout` | Đã có, **FE chưa dựng màn đăng nhập** |
| `GET /user` | Đã có |

---

## 1. Hồ sơ hộ gia đình

Nhóm quan trọng nhất. Hiện FE giữ hồ sơ trong bộ nhớ trình duyệt: refresh trang
là mất, và nút "Sửa hồ sơ" tạo thêm một bản ghi mới thay vì cập nhật.

### 1.1 `GET /households/latest`

Lấy hồ sơ gần nhất để khôi phục màn Chatbot và Phương án sau khi tải lại trang.

- Query: `guest_session_id=<uuid>` (bắt buộc khi chưa đăng nhập; có token thì bỏ qua
  và lấy theo `user_id`).
- `200` khi có dữ liệu, `404` khi chưa từng gửi hồ sơ.

```jsonc
{
  "status": true,
  "message": "Lấy hồ sơ thành công",
  "result": {
    "data": {
      "id": 12,
      "user_id": null,
      "guest_session_id": "9f1c2b7a-3d4e-4a55-8f0b-2c6d1e7a9b30",
      "representative_name": "Nguyễn Văn A",
      "birth_year": 1991,
      "household_size": 5,
      "children_count": 2,
      "residence": "TP. Hồ Chí Minh",
      "average_monthly_income": 35000000,
      "average_monthly_expense": 17000000,
      "has_debt": true,
      "total_current_debt": 500000000,
      "monthly_debt_payment": 5000000,
      "has_savings": true,
      "savings_amount": 150000000,
      "has_dependents": true,
      "assets": ["house", "land"],
      "financial_needs": ["buy_house"],
      "created_at": "2026-08-06T10:12:00.000000Z",
      "updated_at": "2026-08-06T10:12:00.000000Z"
    }
  }
}
```

### 1.2 `GET /households/{id}`

Cùng cấu trúc phản hồi như trên. Dùng khi FE đã biết `id` (vừa tạo xong).

- `403` nếu `id` không thuộc `user_id` hoặc `guest_session_id` của người gọi.

### 1.3 `PUT /households/{id}`

Cập nhật hồ sơ khi người dùng bấm "Sửa hồ sơ" / "Sửa".

- Body: **giống hệt** `POST /households` (cùng bộ field, cùng luật validate).
- Trả về bản ghi sau khi cập nhật, cấu trúc như 1.1.
- `200` thành công, `403` không phải chủ sở hữu, `422` dữ liệu không hợp lệ.
- Yêu cầu: cập nhật luôn `assets` và `financial_needs` theo danh sách mới
  (xoá cái không còn được chọn), không cộng dồn.

### 1.4 `DELETE /households/{id}`

Cho nút "Xóa dữ liệu" ở trang Phương án đề xuất. Xoá kèm `assets`,
`financial_goals` và lịch sử chat của hồ sơ.

- `200` hoặc `204`; `403` nếu không phải chủ sở hữu.

---

## 2. Phương án đề xuất

Trang "Phương án đề xuất" hiện đang hiển thị **số liệu cứng trong code FE**
(9.000.000đ/tháng, 24 tháng, 12%/năm, 40/40/20, hạn mức 1.2 tỷ, 32.000.000đ/tháng,
mục tiêu 1.8 tỷ, 3 bước lộ trình). Cần một endpoint duy nhất thay thế toàn bộ.

### 2.1 `GET /households/{id}/proposal`

Backend có thể tự tính hoặc gọi sang service Python rồi cache lại kết quả.

```jsonc
{
  "status": true,
  "message": "Lấy phương án thành công",
  "result": {
    "data": {
      "household_id": 12,
      "summary": "Gia đình bạn hoàn toàn có thể đạt mục tiêu tích lũy tài chính an toàn nếu duy trì tích lũy kỷ luật và tối ưu các khoản nợ hiện tại.",
      "overview": {
        "net_income": 18000000,          // Thu nhập ròng / tháng
        "current_debt": 500000000,       // Nợ hiện tại
        "current_savings": 150000000,    // Tiết kiệm hiện có
        "target_accumulation": 1800000000 // Nhu cầu tích lũy
      },
      "savings_plan": {
        "monthly_contribution": 9000000,
        "term_months": 24,
        "interest_rate": 12,             // %/năm
        "note": "Phù hợp giúp gia đình tích lũy đều đặn, an toàn và linh hoạt."
      },
      "investment_plan": {
        "allocation": "40% / 40% / 20%",
        "risk_level": "Rủi ro vừa phải",
        "note": "Danh mục cân bằng giúp tối ưu tăng trưởng và kiểm soát rủi ro."
      },
      "loan_plan": {
        "product": "Vay mua nhà trả góp",
        "credit_limit": 1200000000,
        "monthly_payment": 32000000,
        "note": "Phù hợp với khả năng tài chính hiện tại và kế hoạch dài hạn."
      },
      "roadmap": [
        { "step": 1, "text": "Tăng quỹ dự phòng rủi ro lên tối thiểu 3-6 tháng chi phí sinh hoạt." },
        { "step": 2, "text": "Giảm nợ ngắn hạn để cải thiện chỉ số DTI và tối ưu lãi suất vay." },
        { "step": 3, "text": "Duy trì tiết kiệm 10.000.000đ/tháng để đạt mục tiêu tích lũy đúng kế hoạch." }
      ],
      "generated_at": "2026-08-06T10:15:00.000000Z"
    }
  }
}
```

- `404` khi hồ sơ chưa được tính phương án; FE sẽ hiện màn "Chưa có dữ liệu
  phương án đề xuất" đã dựng sẵn.
- Ba khối `savings_plan` / `investment_plan` / `loan_plan` cho phép `null` nếu
  không áp dụng cho hộ đó — FE sẽ ẩn thẻ tương ứng.
- Số tiền trả về **luôn là số**, để FE tự định dạng theo vi-VN.

> Nút "Xuất PDF": FE có thể tự render từ dữ liệu trên. Chỉ cần backend làm
> `GET /households/{id}/proposal/pdf` nếu muốn file do server sinh.

---

## 3. Xác thực

Nút "Đăng nhập" trên banner trang chủ hiện chưa nối. `GET /user` đã có nhưng
chưa có đường lấy token, nên nhánh "User đăng nhập" trong `HouseholdController`
chưa bao giờ chạy được.

### 3.1 `POST /auth/register`

```jsonc
// Request
{ "name": "Nguyễn Văn A", "email": "a@example.com", "password": "secret123", "password_confirmation": "secret123" }
```

### 3.2 `POST /auth/login`

```jsonc
// Request
{ "email": "a@example.com", "password": "secret123" }

// Response 200
{
  "status": true,
  "message": "Đăng nhập thành công",
  "result": {
    "data": {
      "token": "1|xxxxxxxxxxxxxxxxxxxxx",
      "token_type": "Bearer",
      "user": { "id": 1, "name": "Nguyễn Văn A", "email": "a@example.com" }
    }
  }
}
```

- `422` khi sai thông tin đăng nhập.
- Mong muốn: cho phép gửi kèm `guest_session_id` khi đăng nhập/đăng ký để
  backend gán các hồ sơ khách trước đó về `user_id` vừa tạo.

### 3.3 `POST /auth/logout`

Thu hồi token hiện tại. `200`, không cần body.

---

## 4. Lịch sử hội thoại Chatbot

Phần sinh câu trả lời do Python lo; FE chỉ cần chỗ lưu và đọc lại hội thoại.

### 4.1 `GET /households/{id}/messages`

```jsonc
{
  "status": true,
  "message": "Lấy hội thoại thành công",
  "result": {
    "data": [
      { "id": 1, "role": "ai",   "content": "Cảm ơn bạn đã chia sẻ thông tin!...", "created_at": "2026-08-06T10:13:00.000000Z" },
      { "id": 2, "role": "user", "content": "Gợi ý chi tiết gói tiết kiệm tối ưu.", "created_at": "2026-08-06T10:13:20.000000Z" }
    ]
  }
}
```

- `role` ∈ `ai | user`, sắp xếp theo `created_at` tăng dần.

### 4.2 `POST /households/{id}/messages`

Gửi câu hỏi tự do của người dùng, backend chuyển sang service Python và trả về
câu trả lời.

```jsonc
// Request
{ "content": "Gia đình tôi nên trả nợ trước hay tiết kiệm trước?" }

// Response 201 — trả cả tin nhắn người dùng và câu trả lời để FE render một lần
{
  "status": true,
  "message": "Gửi thành công",
  "result": {
    "data": {
      "user_message": { "id": 3, "role": "user", "content": "...", "created_at": "..." },
      "ai_message":   { "id": 4, "role": "ai",   "content": "...", "created_at": "..." }
    }
  }
}
```

---

## Không cần API

- Danh sách 34 tỉnh/thành và danh sách năm sinh: để cứng ở FE
  (`src/data/locations.ts`), dữ liệu gần như không đổi.
- Nhãn tiếng Việt của `assets` / `financial_needs`: hiện có ở cả hai phía
  (`ASSET_LABELS` bên FE, `AssetTypeEnum::label()` bên BE) — giữ nguyên, chỉ cần
  nhớ sửa thì sửa cả hai.
- Nút "Đọc bằng giọng nói": dùng `SpeechSynthesis` của trình duyệt.
- Hai thẻ nhu cầu ở trang chủ: chỉ điều hướng nội bộ.

## Thứ tự đề xuất triển khai

1. **Mục 1** — `GET latest` / `GET {id}` / `PUT` / `DELETE`: chặn được lỗi mất hồ
   sơ khi refresh và lỗi sinh bản ghi rác mỗi lần sửa.
2. **Mục 2** — `GET .../proposal`: gỡ toàn bộ số liệu hard-code khỏi FE.
3. **Mục 4** — lịch sử chat.
4. **Mục 3** — xác thực.
