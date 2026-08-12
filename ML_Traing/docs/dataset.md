# Dataset — Home Credit Default Risk

> File này sinh tự động bởi `scripts/build_dataset_docs.py`. Đừng sửa tay.

- **Nguồn:** https://www.kaggle.com/c/home-credit-default-risk/data
- **License:** theo điều khoản cuộc thi Kaggle — dùng cho mục đích học tập, không phân phối lại dữ liệu
- **Thư mục:** `dataset/home-credit-default-risk` (**không commit vào git**)
- **Chốt phiên bản lúc:** 2026-08-10T09:27:32.143598+00:00

## Phiên bản file (SHA-256)

| File | Kích thước | SHA-256 |
|---|---:|---|
| `application_train.csv` | 158.4 MB | `52e96b895b1112e1…` |
| `previous_application.csv` | 386.2 MB | `5046cd657ee04df2…` |
| `bureau.csv` | 162.1 MB | `9d799143423f2807…` |
| `installments_payments.csv` | 689.6 MB | `428c2e2496e4d6d6…` |
| `HomeCredit_columns_description.csv` | 37 KB | `eef7665398228a80…` |

## Nhãn — `application_train.csv`

| | |
|---|---:|
| Số hồ sơ | 307,511 |
| `TARGET = 1` (khó khăn trả nợ) | 24,825 |
| `TARGET = 0` | 282,686 |
| Tỉ lệ dương | 8.0729% |
| `scale_pos_weight` (XGBoost) | 11.39 |
| Accuracy của model đoán toàn `0` | 91.9271% |

Con số cuối là lý do **không dùng accuracy để chọn model** ở ML02: một model không học gì đã đạt hơn 91%.

## Dữ liệu synthetic của ML01

ML01 không dùng dataset này. Cách sinh dân số hộ gia đình và hàm sinh nhãn `g(·)` mô tả trong `hfml.ml.ml01_recommendation` (F03).
