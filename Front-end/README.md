# Household Finance — Frontend

Bộ source frontend cho ứng dụng quản lý tài chính gia đình.

## Tech stack

- **React 18** + **TypeScript**
- **Vite 5** (dev server + build)
- **Tailwind CSS 3**
- **Lucide Icons** (`lucide-react`)

## Tính năng layout

Responsive layout hỗ trợ **2 chế độ hiển thị**, chuyển đổi qua nút toggle nổi ở góc dưới phải:

| Chế độ | Mô tả |
| --- | --- |
| **Web App (Fullscreen)** | App trải toàn màn hình. Trên màn hình rộng (≥ `lg`) hiện sidebar; màn hình hẹp tự thu về layout mobile với bottom nav. |
| **Smartphone** | App được render trong khung điện thoại (390 × 844) giả lập giữa màn hình — luôn dùng mobile layout để phản ánh đúng trải nghiệm trên điện thoại. |

Trạng thái chế độ được lưu ở `localStorage` (`hf.viewMode`).

> Lưu ý: mobile layout được điều khiển theo `mode` (không chỉ dựa vào media query của Tailwind), vì media query đo theo cửa sổ trình duyệt chứ không theo khung 390px của emulator.

## Cấu trúc

```
src/
├─ main.tsx                 # Entry, bọc ViewModeProvider
├─ App.tsx                  # Shell: sidebar / bottom nav / topbar + hook layout
├─ index.css                # Tailwind directives + utilities
├─ context/
│  └─ ViewModeContext.tsx   # State chế độ hiển thị (fullscreen | phone)
└─ components/
   ├─ DeviceFrame.tsx       # Khung giả lập smartphone / fullscreen
   ├─ ViewModeToggle.tsx    # Nút chuyển chế độ
   └─ Dashboard.tsx         # Nội dung demo: số dư, ngân sách, giao dịch
```

## Chạy dự án

```bash
npm install
npm run dev       # http://localhost:5173
npm run build     # type-check + build production vào dist/
npm run preview   # preview bản build
```
