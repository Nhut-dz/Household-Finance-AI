# Dataset — Home Credit Default Risk

> File này sinh tự động bởi `scripts/build_dataset_docs.py`. Đừng sửa tay.

- **Nguồn:** https://www.kaggle.com/c/home-credit-default-risk/data
- **License:** theo điều khoản cuộc thi Kaggle — dùng cho mục đích học tập, không phân phối lại dữ liệu
- **Thư mục:** `dataset/home-credit-default-risk` (**không commit vào git**)
- **Chốt phiên bản lúc:** 2026-08-24T03:04:50.266737+00:00

## Phiên bản file (SHA-256)

| File | Kích thước | SHA-256 |
|---|---:|---|
| `application_train.csv` | 158.4 MB | `52e96b895b1112e1…` |
| `bureau.csv` | 162.1 MB | `9d799143423f2807…` |
| `HomeCredit_columns_description.csv` | 37 KB | `eef7665398228a80…` |

Không lưu trên đĩa (ngoài phạm vi F04, **không cần tải**): previous_application, installments_payments

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

## Chất lượng dữ liệu

### `application_train.csv` — 307,511 dòng × 122 cột

| Mức | Mã | Mô tả |
|---|---|---|
| error | `sentinel_value` | `DAYS_EMPLOYED` có 55,374 dòng (18.01%) mang giá trị canh gác 365,243 — phải chuyển NaN VÀ giữ cờ nhị phân, vì nhóm này vỡ nợ ít hơn hẳn |
| warning | `placeholder_as_category` | Giá trị Unknown/XAP/XNA đang nằm như một hạng mục thật ở: CODE_GENDER (4), NAME_FAMILY_STATUS (2), ORGANIZATION_TYPE (55,374) |
| warning | `high_missing` | 41 cột thiếu trên 50% dữ liệu (cao nhất `COMMONAREA_AVG` 69.9%) — impute cũng chỉ là bịa số |
| info | `high_cardinality` | Categorical nhiều hạng mục (>30): `ORGANIZATION_TYPE` (58) — one-hot sẽ nổ chiều, cân nhắc gộp nhóm |
| info | `class_imbalance` | `TARGET` mất cân bằng: 8.07% dương. Model đoán toàn lớp đa số đã đạt 91.93% accuracy — chọn model bằng PR-AUC, không dùng accuracy |

## Dữ liệu synthetic của ML01

ML01 không dùng dataset này. Cách sinh dân số hộ gia đình và hàm sinh nhãn `g(·)` mô tả trong `hfml.ml.ml01_recommendation` (F03).
