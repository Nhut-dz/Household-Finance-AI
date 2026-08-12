# PLAN v4 — AI tư vấn tài chính hộ gia đình

> Bản này viết lại theo **Kế hoạch thực hiện (docx)** và **Bảng đầu mục công việc (xlsx)**.
> Khung F01–F07 / M01–M08 / 8 tuần được giữ nguyên; phần bổ sung là chi tiết kỹ thuật
> và giao thức đánh giá để từng đầu mục có thể thi công và bảo vệ được.

**Phạm vi:** Python / Rule-Based / Machine Learning / LLM
**Tuần 1 bắt đầu:** 10/08/2026 · **Kết thúc:** 04/10/2026 · `random_seed = 42`

---

## 1. Mục tiêu

Xây dựng pipeline Python cho hệ thống AI tư vấn tài chính hộ gia đình, gồm **5 bài toán Rule-Based**, **2 bài toán Machine Learning**, và **LLM đóng vai giải thích kết quả** — cung cấp khuyến nghị tài chính tham khảo và hỗ trợ hội thoại.

**Ranh giới cứng của hệ thống:**

| Tầng | Trách nhiệm | Không được làm |
|---|---|---|
| **Rule-Based** | Tính toán tài chính xác định | Không "học" gì cả |
| **Machine Learning** | Phân loại — trả về nhãn + xác suất | Không tự sinh khuyến nghị |
| **LLM** | Diễn đạt kết quả thành tiếng Việt | **Tuyệt đối không tính toán số** |

Nguyên tắc xuyên suốt: *AI là công cụ hỗ trợ phân tích và khuyến nghị tham khảo — không khẳng định đưa ra giải pháp tài chính tối ưu, không quyết định thay người dùng.*

**Ngoài phạm vi:** fine-tuning / retraining theo chu kỳ; bài toán ML thứ ba; FastAPI + database chỉ ở mức đủ demo.

---

## 2. Ràng buộc dữ liệu — điều kiện chi phối toàn bộ kiến trúc

Hệ thống **chỉ thu thập dữ liệu một lần**: form onboarding. Không theo dõi giao dịch, không có lịch sử nhiều tháng.

- ❌ Không có `lag`, `rolling`, xu hướng → **không làm dự báo chuỗi thời gian**
- ✅ Mọi mô hình ML là **cross-sectional** — một dòng hồ sơ vào, một kết quả ra
- ✅ Home Credit khớp hoàn hảo: mỗi hồ sơ là một dòng độc lập

**Định vị sản phẩm (ghi rõ trong báo cáo):** hệ thống **đánh giá tại thời điểm và so sánh với hồ sơ tương đồng**, không phải dự báo tương lai.

### 2.1 Feature tỉ lệ — xử lý khác đơn vị tiền tệ

Home Credit không dùng VNĐ: `AMT_INCOME_TOTAL` trung vị ≈ 147.150, người dùng VN nhập 50.000.000 — lệch ~340 lần. Model gặp giá trị ngoài phân phối huấn luyện sẽ **trả về số vô nghĩa mà không báo lỗi**.

**Quy tắc: loại bỏ mọi giá trị tiền tuyệt đối khỏi feature set của ML02.**

| Bỏ | Dùng thay | Ý nghĩa |
|---|---|---|
| `AMT_ANNUITY` | `dti` = trả nợ tháng ÷ thu nhập tháng | Gánh nặng trả nợ |
| `AMT_CREDIT` | `ltv` = tiền vay ÷ giá nhà | Tỉ lệ vay trên tài sản |
| `AMT_INCOME_TOTAL` | `credit_income_ratio` = tiền vay ÷ thu nhập năm | Số năm thu nhập để trả hết |
| `AMT_GOODS_PRICE` | `income_per_capita` = thu nhập ÷ nhân khẩu | Mức sống |
| — | `savings_months` = tiết kiệm ÷ chi tiêu tháng | Số tháng sống sót |
| — | `debt_income_ratio` = dư nợ ÷ thu nhập năm | Đòn bẩy hiện tại |

Feature phi tiền tệ giữ nguyên: tuổi, số con, số nhân khẩu, số năm đi làm, nghề nghiệp, sở hữu nhà/xe/đất, vùng.

> Trong báo cáo, trình bày mục này dưới tên **"xử lý bài toán chuyển miền (domain transfer) giữa dataset nghiên cứu và dữ liệu người dùng Việt Nam"**.

### 2.1b Ba đính chính sau khi thi công (task 12 — [builder.py](src/hfml/data/features/builder.py))

**1. `income_per_capita` KHÔNG phải feature tỉ lệ.** Tiền ÷ người vẫn là tiền: hộ VN
50.000.000 ÷ 4 = 12.500.000 so với Home Credit 147.150 ÷ 2 = 73.575 — vẫn lệch 170 lần.
Đưa nó vào feature set của ML02 là tái tạo đúng vấn đề mục này muốn diệt. Thay bằng
`income_per_capita_ratio` = thu nhập đầu người ÷ **mức tham chiếu của chính quần thể đó**
(Home Credit dùng trung vị tập train, VN dùng số GSO). Chưa có mức tham chiếu thì để
`NaN`, không bịa số.

**2. `ltv` của form và `AMT_CREDIT/AMT_GOODS_PRICE` của Home Credit là HAI đại lượng
khác nhau.** Home Credit cộng phí và bảo hiểm vào `AMT_CREDIT` nên tỉ lệ này **luôn ≥ 1,0**
(p1 = 1,000 · trung vị 1,119) — nó đo **mức đội giá**, không đo tỉ lệ vay trên tài sản.
Còn `loan_amount / asset_price` = 0,70 nghĩa là "vay 70%, tự có 30%". Gộp hai thứ này lại
là đẩy hồ sơ VN ra ngoài hẳn phân phối huấn luyện. Đã tách: `ltv` (chỉ form, phục vụ RB05)
và `credit_goods_markup` (chỉ Home Credit, chỉ dùng cho bộ Full).

**3. Kỳ thu nhập của Home Credit.** Kaggle chỉ ghi *"Income of the client"*. Loại trừ được
bằng số: nếu thu nhập theo NĂM mà `AMT_ANNUITY` theo THÁNG thì DTI = 0,163 × 12 = **196%**,
bất khả → hai cột cùng kỳ. Thêm bằng chứng: `credit_income_ratio` trung vị Home Credit
**3,27** khớp với **3,89** của một hồ sơ mua nhà VN tính theo **thu nhập năm**.

**Ba nhóm feature** — thay cho cách chia hai nhóm trước đây:

| Nhóm | Nội dung | Dùng cho |
|---|---|---|
| `BOTH` (7) | `dti` · `credit_income_ratio` · `children_ratio` · `age_years` · `employment_years` · `employment_ratio` · `income_per_capita_ratio` | **Bộ rút gọn ML02** — deploy được |
| `HOME_CREDIT_ONLY` (1) | `credit_goods_markup` | Chỉ bộ Full của ML02 |
| `FORM_ONLY` (5) | `ltv` · `savings_months` · `debt_income_ratio` · `savings_rate` · `expense_income_ratio` | ML01 + tầng rule |

`savings_months` và `debt_income_ratio` chỉ có ở phía form vì `application_train.csv`
**không có cột nào về tiết kiệm hay dư nợ hiện tại** (đã kiểm: chỉ 4 cột `AMT_*` là
`INCOME_TOTAL`, `CREDIT`, `ANNUITY`, `GOODS_PRICE`).

> **Còn một giới hạn phải ghi vào `docs/model_card.md`:** `dti` của Home Credit là kỳ trả
> của khoản **đang xin vay**, còn của form là khoản nợ **đang có**. Cùng là "phần thu nhập
> dành trả nợ" và phân phối khớp nhau (0,163 vs 0,20) nên model không bị lệch miền, nhưng
> đây là hai khoản nợ khác nhau — không được lờ đi khi bảo vệ.

---

## 3. Kiến trúc pipeline

```
                        User Input (form)
                              ↓
                    Validation & sanity check
                              ↓
                      Data Preprocessing
                              ↓
            ┌──────────────────────────────┐
            │  Rule-Based Engine           │
            │  RB01 · RB02 luôn chạy       │
            │  RB03 · RB04 · RB05 theo      │
            │  goal_type người dùng chọn    │
            └──────────────┬───────────────┘
                           │
              ┌────────────┴────────────┐
              ↓                         ↓
            ML01                      ML02
   Recommendation Group        Home Credit Risk
   (nhóm khuyến nghị)          (P vỡ nợ)
     — mọi hồ sơ —          — CHỈ khi needs_loan_analysis —
              │                         │
              └────────────┬────────────┘
                           ↓
                  Tổng hợp kết quả
              (structured result JSON)
                           ↓
                     ┌─────┴─────┐
                     │    LLM    │  ← chỉ nhận JSON đã tính sẵn
                     └─────┬─────┘
                           ↓
              Validate: mọi con số trong câu trả lời
                   phải khớp JSON đầu vào
                           ↓
                Giải thích + Khuyến nghị
```

---

## 4. F01 — Data Preparation & Preprocessing (14 task · M01 · Tuần 1)

| # | Task | Ưu tiên | File |
|---|---|---|---|
| 1 | Thiết lập cấu trúc project Python | TB | ✅ 5 tầng — xem [README.md](README.md), test [test_structure.py](tests/test_structure.py) |
| 2 | Cấu hình virtual environment và requirements | TB | ✅ `.venv` + [requirements.txt](requirements.txt) |
| 3 | Thiết lập config và logging | TB | ✅ [config.py](src/hfml/config.py), [logger.py](src/hfml/logger.py) |
| 4 | **Xác định schema dữ liệu user input** | **Cao** | ✅ [schema.py](src/hfml/data/schema.py) · [test_schema.py](tests/test_schema.py) |
| 5 | Thu thập và import dữ liệu Home Credit | TB | ✅ [loader.py](src/hfml/data/loader.py) · [test_loader.py](tests/test_loader.py) |
| 6 | Kiểm tra chất lượng dữ liệu | TB | ✅ [quality.py](src/hfml/data/quality.py) · [test_quality.py](tests/test_quality.py) |
| 7 | **Quản lý phiên bản dataset** | **Cao** | ✅ [dataset.md](docs/dataset.md) · `docs/dataset_manifest.json` · [build_dataset_docs.py](scripts/build_dataset_docs.py) |
| 8 | Xử lý missing values | TB | ✅ [cleaner.py](src/hfml/data/preprocessing/cleaner.py) · [test_cleaner.py](tests/test_cleaner.py) |
| 9 | Xử lý duplicate và dữ liệu bất hợp lệ | TB | ✅ [validator.py](src/hfml/data/preprocessing/validator.py) · [test_validator.py](tests/test_validator.py) |
| 10 | Encoding biến categorical | TB | ✅ [encoders.py](src/hfml/data/preprocessing/encoders.py) · [test_encoders.py](tests/test_encoders.py) |
| 11 | Scaling numerical features | TB | ✅ [encoders.py](src/hfml/data/preprocessing/encoders.py) — mặc định tắt, xem 4.3d |
| 12 | Feature engineering dữ liệu tài chính | TB | ✅ [builder.py](src/hfml/data/features/builder.py) · [test_builder.py](tests/test_builder.py) |
| 13 | Feature selection | TB | ✅ [selection.py](src/hfml/data/features/selection.py) · [test_selection.py](tests/test_selection.py) |
| 14 | **Đóng gói preprocessing bằng Pipeline** | **Cao** | ✅ [pipeline.py](src/hfml/data/preprocessing/pipeline.py) · [test](tests/test_pipeline_preprocessing.py) · [show_preprocessing.py](scripts/show_preprocessing.py) |

> **Ghi chú task 1 (đã xong):** scaffold v3 cũ (dự báo chi tiêu chuỗi thời gian —
> `expense_forecaster`, feature `lag`/`rolling`, panel tháng, endpoint `/forecast/expense`)
> đã bị xóa vì trái với ràng buộc ở mục 2. Code cũ vẫn tra được ở commit `6b46090`.
> Cấu trúc mới gom theo **5 tầng**: `data` · `rules` · `ml` · `pipeline` · `llm` (+ `api`).

### 4.1 Schema form đầu vào (task 4) — ✅ [schema.py](src/hfml/data/schema.py)

**16 trường đã có** (theo bảng API của backend). Tên trường trong schema lấy theo cột
`Field input (API)`, **không** theo tên cột DB — 7 chỗ hai bên lệch nhau
(`residence`/`location`, `average_monthly_income`/`monthly_income`,
`average_monthly_expense`/`monthly_living_cost`, `has_dependents`/`supports_elderly`,
`total_current_debt`/`total_debt`, `savings_amount`/`current_savings`,
`guest_session_id`/`session_token`) ghi trong hằng `API_TO_DB_COLUMN`.

`monthly_debt_payment` **đã có sẵn** trong form — PLAN cũ xếp nhầm vào diện bổ sung.

**Bổ sung 5 ô — tất cả nằm trong KHỐI VAY**, chỉ hiện khi người dùng chọn một nhu cầu
trong `LOAN_TRIGGER_NEEDS` (chốt 11/08/2026):

| Ô mới | Kiểu | Ánh xạ Home Credit | Dùng cho |
|---|---|---|---|
| `occupation` | Select 16 giá trị | `OCCUPATION_TYPE` | ML02 |
| `employment_years` | Select dải / Number | `DAYS_EMPLOYED` | ML02 (2/7 feature bộ rút gọn) |
| `asset_price` | Number | `AMT_GOODS_PRICE` | RB05 (LTV) |
| `loan_amount` | Number | `AMT_CREDIT` | RB05, `credit_income_ratio` |
| `loan_term_months` | Select 8 mốc | → `AMT_ANNUITY` | RB05 (hạn mức vay) |

**Ma sát người dùng — lý do dồn cả 5 ô vào khối vay:**

| Người dùng | Trước | Sau |
|---|---:|---:|
| Chỉ xem sức khỏe tài chính | ~12 ô | **~12 ô — không đổi** |
| Muốn phân tích khoản vay | 16 ô | 21 ô |

`occupation` và `employment_years` phục vụ **ML02 — dự báo rủi ro tín dụng**, mà ML02 chỉ
có ý nghĩa với người đang tính vay. Đặt chúng ở phần nhân thân thì mọi người dùng phải
nhập thêm 2 ô cho một tính năng họ không dùng. Nguyên tắc: **ma sát chịu được khi nó gắn
với thứ người dùng chủ động muốn** — ai đang tính vay 1,4 tỷ thì 5 ô là bình thường, bất
kỳ hồ sơ tín dụng nào cũng hỏi nhiều hơn thế.

> **Hệ quả về phạm vi: ML02 chỉ chạy khi `needs_loan_analysis = True`.** Với người dùng
> còn lại, hệ thống trả về 4 rule + nhóm khuyến nghị ML01, **không có xác suất vỡ nợ**.
> Đây là chủ ý chứ không phải thiếu sót: đưa "xác suất vỡ nợ 8%" cho người không hề định
> vay là một con số vô nghĩa với họ và dễ bị hiểu nhầm. Sơ đồ mục 3 phản ánh điều này.

> **Đã bỏ `region_type`.** Sau khi RB04 ra khỏi phạm vi, trường này chỉ còn phục vụ
> `income_per_capita_ratio` — mà feature đó lại cần một số liệu GSO chưa có. Bộ rút gọn
> của ML02 còn **6/7 feature**, `income_per_capita_ratio` để `NaN` và ghi vào phần giới
> hạn của báo cáo.

**Quyết định đã chốt:**

1. **Khối vay hiện với mọi `financial_needs` trừ `other`** — tức `buy_house`,
   `buy_car`, `buy_land`, `loan` (hằng `LOAN_TRIGGER_NEEDS`). Mua xe/đất ở VN phần
   lớn cũng có vay, và RB05 cần LTV cho cả ba loại tài sản. *(10/08/2026)*
2. **`monthly_debt_payment` bắt buộc khi `has_debt = true`** (backend đang để Tùy
   chọn). DTI là trục chính của RB02, RB05, ML02 — thiếu là ba thứ đó không chạy.
   *(10/08/2026)*
3. **Bỏ RB04 và 3 trường phân rã chi tiêu** khỏi phạm vi. *(11/08/2026)*
4. **Bỏ `region_type`; dồn `occupation` và `employment_years` vào khối vay.** Người
   chỉ xem sức khỏe tài chính không phải nhập thêm ô nào. Kéo theo: **ML02 chỉ chạy
   khi có nhu cầu vay**. *(11/08/2026)*
5. **`residence` giữ nguyên text tự do** — không dùng để suy ra vùng nữa. *(11/08/2026)*

**Chưa chốt:** `average_monthly_expense` hiện vẫn Tùy chọn ở backend. Thiếu nó thì
RB01, RB02 và nhãn ML01 đều không tính được — schema chấp nhận `None` nhưng gắn cờ
`MISSING_EXPENSE`. Nên chuyển thành bắt buộc ở form.

### 4.2 Kiểm tra đầu vào (task 6, 9)

**Phía form người dùng** — đã cài trong `HouseholdProfile.data_quality_flags()`:

- Cảnh báo khi `tỉ lệ tiết kiệm > 60%` hoặc `chi tiêu > thu nhập` — **cảnh báo, không chặn**:
  chi > thu chính là nhóm `EMERGENCY` của ML01, chặn nó là chặn đúng đối tượng cần tư vấn nhất
- **Sửa lỗi hiện có trong UI:** dư nợ 20.000.000đ đang hiển thị *"trả nợ ước tính 200.000đ/tháng"*
  (1%/tháng → 8,3 năm chưa tính lãi). Thay bằng trường người dùng tự nhập.

**Phía Home Credit** — `quality.check_application_train()` phát hiện 5 vấn đề, ghi vào
[docs/dataset.md](docs/dataset.md):

| Mức | Vấn đề | Xử lý ở task 8 |
|---|---|---|
| error | `DAYS_EMPLOYED = 365243` — 55.374 dòng (18,01%) | → `NaN` **+ giữ cờ nhị phân** (xem dưới) |
| warning | `XNA`/`Unknown` làm hạng mục thật: `ORGANIZATION_TYPE` (55.374), `CODE_GENDER` (4), `NAME_FAMILY_STATUS` (2) | → coi như missing |
| warning | 41/122 cột thiếu > 50% (cao nhất `COMMONAREA_AVG` 69,9%) | → loại khỏi feature set |
| info | `ORGANIZATION_TYPE` có 58 hạng mục | → gộp nhóm, đừng one-hot thẳng |
| info | `TARGET` 8,07% dương | → `scale_pos_weight`, chọn model bằng PR-AUC |

> **Vì sao sentinel `DAYS_EMPLOYED` phải giữ cờ, không được xóa** — đã kiểm chứng trên
> dữ liệu thật: 55.374 dòng đó trùng khít với `ORGANIZATION_TYPE = 'XNA'` và
> `OCCUPATION_TYPE` rỗng 100%, và `NAME_INCOME_TYPE` cho thấy đó là **55.352 người nghỉ
> hưu + 22 người thất nghiệp**. Nhóm này vỡ nợ **5,40%** so với **8,66%** của nhóm có việc
> làm. Tức "thiếu dữ liệu việc làm" tự nó là tín hiệu dự báo mạnh — impute bằng trung vị
> rồi vứt cờ đi là xóa mất thông tin, và mất luôn một câu trả lời hay trước hội đồng.

**Quản lý phiên bản (task 7):** `docs/dataset_manifest.json` giữ SHA-256 + kích thước của
cả 5 file. Chạy `python scripts/build_dataset_docs.py --verify` trước mỗi lần train lại —
dữ liệu đổi mà metric cũng đổi thì biết ngay nguyên nhân nằm ở đâu. Đây là điều kiện để
F06 task 6 (tái lập với seed 42) có căn cứ, vì dataset 1,4 GB không commit vào git được.

### 4.3 Quản lý phiên bản dataset (task 7)

```
dataset/home-credit-default-risk/
├── application_train.csv       166 MB — 307.511 hồ sơ, TARGET 8,07% dương   ✅
├── previous_application.csv    405 MB                                        ✅
├── bureau.csv                  170 MB — để dành                              ✅
├── installments_payments.csv   723 MB — để dành                              ✅
└── HomeCredit_columns_description.csv                                        ✅
```

Ghi `docs/dataset.md`: nguồn Kaggle, ngày tải, SHA-256, số dòng/cột, phân bố `TARGET`. Dataset **không commit vào git**.

**Đã kiểm chứng trên đĩa (10/08/2026)** — mọi con số trong PLAN khớp dữ liệu thật:

| | PLAN | Thực tế |
|---|---|---|
| Số hồ sơ | 307.511 | 307.511 ✅ |
| `TARGET = 1` | 8,07% | 8,0729% (24.825) ✅ |
| `AMT_INCOME_TOTAL` trung vị | ≈147.150 | 147.150 ✅ |
| Số cột | — | 122 |
| `SK_ID_CURR` trùng lặp | — | 0 |
| `DAYS_EMPLOYED == 365243` | "có sentinel" | **55.374 dòng = 18,01%** |
| `scale_pos_weight` | ≈11,4 | 11,39 ✅ |
| Accuracy của model đoán toàn 0 | 91,93% | 91,93% ✅ |

⚠️ Sentinel `DAYS_EMPLOYED` chiếm **18% dữ liệu**, không phải vài ca lẻ — gần như chắc chắn
là nhóm nghỉ hưu/không đi làm. Chuyển `NaN` là đúng, nhưng phải **thêm cờ
`DAYS_EMPLOYED_MISSING`**: bản thân việc không có số năm đi làm là tín hiệu dự báo,
vứt đi là mất thông tin (task 8).

### 4.3b Missing values (task 8) — ✅ [cleaner.py](src/hfml/data/preprocessing/cleaner.py)

Chia việc theo tiêu chí **có học gì từ dữ liệu hay không**:

| | Việc | Vì sao |
|---|---|---|
| `MissingNormalizer` | sentinel → NaN · chuỗi giả → NaN · sinh cờ `_MISSING` | Biến đổi theo từng dòng, `fit()` rỗng → chạy trước split cũng không rò rỉ |
| `HighMissingDropper` | bỏ cột thiếu > ngưỡng | **Có học** → bắt buộc trong Pipeline, `fit` chỉ trên train |

**Sáu cờ `_MISSING`, mỗi cờ có số đo biện minh** (đo trên 307.511 hồ sơ, vỡ nợ chung 8,07%):

| Cờ | Thiếu | Vỡ nợ khi thiếu / khi có | Lift |
|---|---:|---|---:|
| `DAYS_EMPLOYED_MISSING` | 18,01% | 5,40% / 8,66% | **0,624** |
| `OCCUPATION_TYPE_MISSING` | 31,35% | 6,51% / 8,79% | 0,741 |
| `CREDIT_BUREAU_REQ_MISSING` | 13,50% | 10,34% / 7,72% | **1,339** |
| `BUILDING_INFO_MISSING` | 48,27% | 9,23% / 6,99% | 1,321 |
| `EXT_SOURCE_3_MISSING` | 19,83% | 9,31% / 7,77% | 1,199 |
| `EXT_SOURCE_1_MISSING` | 56,38% | 8,52% / 7,50% | 1,137 |

**Sáu cờ chứ không phải 63:** 63 cột có missing chỉ tạo ra **28 mẫu thiếu phân biệt**, và
21 trong đó là cùng một khối "không có thông tin nhà ở" (lift 1,14–1,32). Tạo cờ cho từng
cột là nhân bản một thông tin hàng chục lần, làm loãng feature importance.

**Ba nhóm bị loại, có lý do:** `EXT_SOURCE_2` (lift 0,976 — không tín hiệu),
`SOCIAL_CIRCLE` (lift 0,436 nhưng chỉ 0,33% ≈ 1.000 dòng), `NAME_TYPE_SUITE` (0,42%).

> Câu trả lời cho *"bạn xử lý missing value thế nào?"*: điền trung vị trong Pipeline
> (fit trên train), **nhưng giữ cờ nhị phân** — vì ở dataset này việc thiếu dữ liệu tự nó
> dự báo được vỡ nợ, điền rồi bỏ cờ là xóa mất tín hiệu.

### 4.3c Duplicate & dữ liệu bất hợp lệ (task 9) — ✅ [validator.py](src/hfml/data/preprocessing/validator.py)

**Bất đối xứng train ↔ inference — điểm thiết kế chính:**

| | Được làm gì | Vì sao |
|---|---|---|
| Train | `clean_for_training()` — bỏ dòng trùng, bỏ dòng bất hợp lệ | 307.511 hồ sơ, bỏ vài chục dòng không ảnh hưởng |
| Inference | **CHỈ** `OutlierClipper` (biên đã học từ train) + gắn cờ | Người dùng đang chờ kết quả; trả về "hồ sơ bị loại" là hệ thống hỏng, không phải dữ liệu hỏng |

Cố ý **không** viết hàm `clean_for_inference()` để không ai gọi nhầm hàm bỏ dòng vào
đường inference.

**Kiểm chứng: Home Credit SẠCH** — dòng trùng 0 · `SK_ID_CURR` trùng 0 · cột trùng nội
dung 0 · đi làm trước khi sinh 0 · số con ≥ số nhân khẩu 0.

Phần bất hợp lệ còn lại đều là **ngoại lai, không phải sai logic**:
`AMT_INCOME_TOTAL` cao nhất 117.000.000 (**247× phân vị 99**), `CNT_CHILDREN` tới 19,
`OBS_30_CNT_SOCIAL_CIRCLE` tới 348 (35× p99). Xử lý bằng **kẹp biên (winsorize)** chứ
không bỏ dòng — giữ được hồ sơ mà vẫn không cho một giá trị kéo lệch scaler.

`OutlierClipper` học phân vị 0,1%–99,9% lúc `fit` (train), rồi áp y hệt lúc inference.
Nhờ vậy người dùng khai thu nhập 900 triệu/tháng chỉ bị kẹp về biên chứ không đẩy model
ra ngoài phân phối huấn luyện, và `clipped_mask()` gắn cờ để tầng `llm` nói ra.

> **Hai bẫy đã sập trong lúc làm, đều do test bắt được:**
> 1. Khử trùng theo *toàn bộ dòng* trên **bộ feature rút gọn** sẽ xóa nhầm khách hàng
>    hợp lệ — đủ 122 cột thì 0 dòng trùng, nhưng chỉ 7 cột thì đã có. Nay mặc định chỉ
>    khử theo `SK_ID_CURR`; khử toàn dòng phải bật `full_row=True` một cách có ý thức.
> 2. Cờ `DAYS_EMPLOYED_MISSING` cũng bắt đầu bằng `DAYS_`, nên quy tắc "`DAYS_*` phải ≤ 0"
>    bắt nhầm đúng 55.374 dòng — bằng số cờ đang bật. Quy tắc quét theo tiền tố nay loại
>    trừ hậu tố `_MISSING`.

### 4.3d Encoding & Scaling (task 10, 11) — ✅ [encoders.py](src/hfml/data/preprocessing/encoders.py)

Cả hai quyết định bị chi phối bởi một sự thật: **bốn thuật toán đều là cây**. Cây bất
biến với biến đổi đơn điệu, và chẻ nhánh theo ngưỡng từng cột.

**Task 10 — ordinal, không one-hot.** Đo trên `application_train.csv`: 16 cột
categorical, one-hot làm số cột nhảy **128 → 249**; riêng `ORGANIZATION_TYPE` có 57 hạng
mục → 57 cột nhị phân cực thưa, xé độ quan trọng của một biến thành 57 mảnh và đẩy cây
sâu vô ích. One-hot vẫn cài sẵn (`strategy="onehot"`, có gộp hạng mục hiếm) cho trường
hợp muốn so với baseline tuyến tính.

| Tình huống | Mã | Vì sao tách riêng |
|---|---|---|
| Hạng mục thiếu (NaN) | `MISSING_CODE = -2` | Không điền mode — điền mode là gán cho người ta một nghề họ không làm. Task 8 đã đo: `OCCUPATION_TYPE` thiếu → vỡ nợ 6,51% vs 8,79%, tức bản thân việc thiếu là tín hiệu |
| Hạng mục lạ lúc inference | `UNKNOWN_CODE = -1` | Người dùng gửi nghề chưa từng có trong train — **không được crash** (F06 task 1) |

**Task 11 — mặc định KHÔNG scale.** Dải giá trị giữa các cột lệch tới 10⁹ lần
(`AMT_INCOME_TOTAL` tới 117.000.000 vs `REGION_POPULATION_RELATIVE` 0,0003–0,07). Với
hồi quy tuyến tính/SVM đó là tai họa; với cây thì vô hại — ngưỡng `x ≤ 147.150` và
`x_scaled ≤ 0,31` cho ra **đúng một phép phân hoạch**.

> Câu trả lời cho *"sao không chuẩn hóa dữ liệu?"* không phải là lý thuyết suông:
> `test_scaling_does_not_change_tree_predictions` train hai Decision Tree cùng seed trên
> dữ liệu có/không scale và khẳng định **dự đoán trùng khít từng dòng**.

Scaling vẫn cài đủ (`standard` / `minmax` / `robust`) để bật khi cần baseline tuyến tính.
Nếu bật thì dùng **`robust`**: có test cho thấy `StandardScaler` bị một ngoại lai
117.000.000 nén 4 giá trị bình thường lại sát nhau, còn `RobustScaler` thì không.

**Điền thiếu cho cột số** đặt ở đây, dùng **trung vị** (cùng lý do với `RobustScaler`).
`SimpleImputer` học từ dữ liệu nên bắt buộc nằm trong Pipeline — đây chính là mắt xích
cuối của task 8 mà `cleaner.py` cố tình không làm.

### 4.3e Feature selection (task 13) — ✅ [selection.py](src/hfml/data/features/selection.py)

**Hai loại chọn feature, khác hẳn nhau về mức nguy hiểm:**

| Loại | Bước | Nhìn nhãn? | Đặt ở đâu |
|---|---|---|---|
| Không giám sát | `NearZeroVarianceRemover` · `CorrelatedFeatureRemover` | Không | Đầu Pipeline; `fit` trên train |
| **Có giám sát** | `SupervisedFeatureSelector` (mutual information) | **Có** | **Bắt buộc trong Pipeline** để mỗi CV fold fit lại |

> Chọn feature bằng nhãn **trên toàn bộ tập train** rồi mới chạy cross-validation là để
> phần validation của mỗi fold góp phần quyết định feature nào tồn tại → **metric CV lạc
> quan giả**. `test_supervised_selection_is_data_dependent` chứng minh: fit trên hai nửa
> dữ liệu khác nhau cho ra hai tập feature khác nhau. Cái gì phụ thuộc dữ liệu thì phải
> nằm trong fold.

**Đo trên 110 cột số (sau task 8):**

| | Số lượng | Ví dụ |
|---|---:|---|
| Cột gần hằng số (một giá trị > 99%) | **18** | `FLAG_MOBIL` 100% = 1 · `FLAG_DOCUMENT_2/10/12` 100% = 0 · 9/20 cột `FLAG_DOCUMENT_*` có tỉ lệ bật < 0,1% |
| Cặp \|r\| > 0,95 | **46** | bộ ba `AVG`/`MODE`/`MEDI` của khối thông tin nhà ở |
| Cặp \|r\| > 0,99 | 16 | |
| Bỏ tham lam mỗi cặp > 0,95 | **32/110 cột** | |

`FLAG_MOBIL` có 307.510/307.511 dòng bằng 1 nên `nunique() == 2` — kiểm tra "hằng số" thông
thường **bỏ sót**, phải dùng ngưỡng tỉ lệ.

> **Phát hiện đáng chú ý:** `FLAG_EMP_PHONE` ~ `DAYS_EMPLOYED_MISSING` có **r = 0,9999**.
> Cờ sinh ở task 8 gần như trùng khít một cột có sẵn của Home Credit — hợp lý, vì người
> nghỉ hưu không có điện thoại cơ quan. Vừa xác nhận cờ đó đúng ý nghĩa, vừa là lý do phải
> khử trùng lặp.

**Vì sao dùng mutual information chứ không phải tương quan Pearson:** quan hệ giữa feature
và rủi ro vỡ nợ thường phi tuyến. Trên `application_train.csv`, tương quan tuyến tính
tuyệt đối **cao nhất chỉ 0,179** (`EXT_SOURCE_3`) — tin vào Pearson thì sẽ kết luận nhầm
là "chẳng feature nào có ích".

**Tính tái lập:** thứ tự duyệt cặp tương quan được sắp xếp tường minh (r giảm dần, rồi theo
tên), quy tắc giữ/bỏ cố định (bảo vệ → thiếu ít hơn → xuất hiện trước). Chạy lại ra đúng
cùng danh sách cột — điều kiện cho F06 task 6.

### 4.4 Đóng gói Pipeline (task 14) — ✅ [pipeline.py](src/hfml/data/preprocessing/pipeline.py)

Toàn bộ impute → encode → scale nằm trong một `sklearn.Pipeline` / `ColumnTransformer` được `fit` **chỉ trên tập train** và `joblib.dump` cùng model. Preprocessing chạy rời trước khi split → **rò rỉ dữ liệu**, và inference sẽ lệch so với training.

**Bảy bước, theo đúng thứ tự:**

```
MissingNormalizer → HighMissingDropper → OutlierClipper → ColumnTransformer
   → NearZeroVarianceRemover → CorrelatedFeatureRemover → [SupervisedFeatureSelector]
```

Hai chỗ thứ tự có lý do, đảo là sai:
- **Kẹp biên TRƯỚC điền thiếu** — làm ngược lại thì trung vị vừa điền tham gia tính phân vị, mà ngoại lai đã kịp kéo lệch chính trung vị đó.
- **Khử hằng số / tương quan SAU encode** — trước đó cột categorical còn là chuỗi, không tính tương quan được.

**Phễu đo trên toàn bộ dữ liệu** (fit trên 246.008 dòng train, `python scripts/show_preprocessing.py`):

| Bước | Cột còn lại | Thay đổi |
|---|---:|---:|
| đầu vào | 121 | |
| `missing` | 127 | **+6** (cờ `_MISSING`) |
| `drop_high_missing` | 110 | −17 |
| `clip` | 110 | 0 |
| `encode` | 109 | −1 (`SK_ID_CURR`) |
| `nzv` | 90 | −19 |
| `decorrelate` | **67** | −23 |

Áp lên 61.503 dòng test: 67 cột, **0 ô thiếu**, thứ tự feature train ≡ test.

**Ba điều bộ test chứng minh** (25 test), mỗi điều ứng với một cách hệ thống hỏng âm thầm:

| | Test |
|---|---|
| `fit` chỉ dùng thống kê train | `test_fitting_on_all_data_differs_from_fitting_on_train` — nếu hai cách cho kết quả y hệt thì "fit trên train" chỉ là hình thức; khác nhau nghĩa là để test lọt vào fit là có hại thật |
| `joblib` giữ nguyên hành vi | `test_trained_model_survives_dump_and_load` — dump cả pipeline + Decision Tree, load lại `predict_proba` trùng khít |
| Thứ tự feature ổn định | `test_feature_order_is_stable_across_transforms` — thứ tự sai là lỗi im lặng, model vẫn trả xác suất nhưng vô nghĩa |

> **Phát hiện khi chạy thật:** chỉ **4/6 cờ `_MISSING`** sống sót tới cuối. Hai cờ bị bỏ ở
> bước khử tương quan vì Home Credit đã có sẵn cột mang đúng thông tin đó —
> `DAYS_EMPLOYED_MISSING` trùng `FLAG_EMP_PHONE` (|r| = 0,9999) và `BUILDING_INFO_MISSING`
> trùng `EMERGENCYSTATE_MODE` (|r| = 0,9783). **Không mất thông tin.** Nhưng khi cần bảng
> feature importance và SHAP đọc được bằng tiếng Việt (SHAP top-5 đưa sang tầng `llm`),
> truyền `protect=INTERPRETABLE_FLAGS` để giữ tên dễ hiểu — số feature không đổi, chỉ đổi
> cột nào bị bỏ. Với bộ **rút gọn** thì không cần: hai cột gốc kia không lấy được từ form.

---

## 5. F02 — Rule-Based Financial Analysis Engine (7 task · M02 · Tuần 2)

| # | Rule | Công thức | `goal_type` kích hoạt | Ưu tiên |
|---|---|---|---|---|
| **RB01** | Thu nhập, chi tiêu, số dư | `số dư = thu − chi` | (luôn chạy) | TB |
| **RB02** | Sức khỏe tài chính | `dti`, `savings_months`, `savings_rate` | (luôn chạy) | **Cao** |
| **RB03** | Tiến độ mục tiêu tiết kiệm | `cần/tháng = (mục tiêu − tích lũy) ÷ số tháng còn lại` | `saving` | TB |
| **RB04** | Phân bổ 50/30/20 | `needs/wants/savings = 50%/30%/20% × thu nhập` | `budget_50_30_20` | **Cao** |
| **RB05** | Khả năng đáp ứng khoản vay | `DTI ≤ ngưỡng` và `LTV ≤ ngưỡng` → hạn mức vay | `home_loan` | **Cao** |

Bốn `goal_type` trong `tblfinancial_goals` ánh xạ 1-1 với bốn rule. `investment` (87 hộ)
hiện chưa có rule — thuộc nhóm `GROWTH` của ML01.

> **RB04 bị loại rồi khôi phục (11/08/2026).** Ban đầu bỏ vì tưởng phải hỏi thêm 3 ô
> Needs/Wants/Savings. Đọc dữ liệu thật (600 dòng `tblcalculation_results`) mới thấy
> backend đang tính RB04 là **rule KÊ ĐƠN chứ không phải rule CHẨN ĐOÁN**:
>
> ```
> budget_needs   = 50% × monthly_income
> budget_wants   = 30% × monthly_income     tổng = đúng 100% thu nhập
> budget_savings = 20% × monthly_income     allocation_rule = "50/30/20" (600/600 dòng)
> ```
>
> Nó trả về **mức phân bổ đề xuất từ thu nhập**, không so với chi tiêu thực tế theo nhóm.
> Vì vậy **không cần thêm ô nhập nào** — 3 trường `needs_expense`/`wants_expense`/
> `savings_expense` đã bỏ và **không khôi phục**.
>
> Phần chẩn đoán vẫn làm được một nửa từ dữ liệu sẵn có: so `monthly_living_cost` với mức
> đề xuất cho needs+wants (80% thu nhập — thực tế trung vị chỉ 61%), và so tỉ lệ tiết kiệm
> thực tế với mốc 20%. **Không tách được** chi thực tế thành needs và wants — phải ghi rõ
> giới hạn này, đừng phát biểu như thể tách được.

### 5.1 Hợp đồng đầu ra — dựng lại từ DB thật (11/08/2026)

`tblcalculation_results` có **600 dòng đầy đủ dữ liệu**, cho biết chính xác backend kỳ vọng
tầng rule trả về gì. Đây là ràng buộc tích hợp, không phải gợi ý:

| Cột DB | Rule | Ghi chú quan trọng |
|---|---|---|
| `dti_ratio` | RB02 | **Lưu dạng PHẦN TRĂM (0–100), không phải tỉ lệ.** Đã kiểm: sai lệch so với `100 × trả nợ ÷ thu nhập` chỉ 0,002 |
| `dti_status` | RB02 | Chỉ **3 mức**: `LOW` (0–19,42) · `MEDIUM` (20,00–39,29) · `HIGH` (41,54–59,69) |
| `safe_loan_limit` | RB05 | Là **số tiền gốc vay**, không phải trả/tháng. Trung vị ≈ 18,5 × thu nhập tháng |
| `recommended_monthly_saving` | RB03 | ≈ 26,67% thu nhập, tức `4/3 × budget_savings` |
| `budget_needs/wants/savings` | RB04 | 50/30/20 của thu nhập |
| `allocation_rule` | RB04 | `"50/30/20"` ở cả 600 dòng |
| `raw_json` | tất cả | `{scope, intent, calculation{dti_percent, monthly_surplus, safe_new_monthly_payment}, profile_summary{...}}` |

**Ba điểm phải xử lý:**

1. **Ngưỡng `dti_status` HIGH = 40% khớp đúng `g(·)` của ML01** (`dti ≥ 0.40 → DEBT_FOCUS`).
   May mắn — hai tầng không mâu thuẫn. Giữ nguyên cả hai.
2. **Backend có 3 mức `dti_status`, PLAN ghi RB02 → 4 mức.** Phải chọn: hoặc RB02 trả 3 mức
   cho khớp DB, hoặc mở rộng cột `dti_status` lên 4 giá trị. Nếu muốn RB02 ánh xạ 1-1 với
   4 nhóm ML01 (§6.1) thì phải là 4.
3. ⚠️ **`safe_new_monthly_payment` = `monthly_surplus` = toàn bộ thu − chi.** Backend đang
   coi 100% số dư là khả năng trả nợ an toàn — quá mạnh tay, vì nó không chừa gì cho tiết
   kiệm và rủi ro. RB05 nên dùng **giới hạn DTI 36–40% thu nhập trừ nợ hiện có**, và nêu
   rõ điểm cải thiện này trong báo cáo.

**Task 6 — Tách cấu hình rule và threshold:** toàn bộ hệ số và ngưỡng đặt trong `config/rules.yaml`, **không hardcode**. Nạp qua `rules/thresholds.py`, mỗi ngưỡng kèm cột nguồn trích dẫn.

**Task 7 — Unit test cho 5 rule:** đây là hình thức "đánh giá" đúng cho tầng rule.

> ⚠️ **Sửa so với bảng xlsx:** RB02 / RB04 / RB05 đang được gắn thành phẩm *"Evaluation report — Accuracy, Macro-F1, Confusion Matrix"*.
> Rule là **hàm xác định**, không có ground truth độc lập — chấm accuracy cho rule là tự chấm điểm chính mình, và hội đồng sẽ hỏi ngay *"nhãn đúng lấy ở đâu ra?"*.
> Thành phẩm đúng cho hai rule này là: **bảng ngưỡng có trích dẫn nguồn** (DTI ≤ 36–40%, LTV ≤ 70–80%) + **ma trận case biên** (mỗi ngưỡng test 3 điểm: dưới / đúng bằng / trên) + **unit test pass 100%**.
> Cụm chỉ số phân loại được giữ nguyên và báo cáo đầy đủ ở F03 và F04 — nơi có nhãn thật.

Mỗi rule trả về `dict` có cấu trúc thống nhất: `{code, status, value, threshold, message_key}` — để tầng LLM chỉ việc diễn đạt, không phải suy luận.

---

## 6. F03 — ML01: Financial Recommendation Group Classification (15 task · M03 · Tuần 3)

**Tiến độ: 15/15 task — hoàn thành 12/08/2026.**

Hiện thực nằm ở [labeler.py](src/hfml/ml/ml01_recommendation/labeler.py),
[synthetic.py](src/hfml/data/synthetic.py) và
[train.py](src/hfml/ml/ml01_recommendation/train.py); test ở
[test_labeler.py](tests/test_labeler.py), [test_synthetic.py](tests/test_synthetic.py),
[test_train_ml01.py](tests/test_train_ml01.py). Output của các lần chạy nằm ở
`src/training/runs/`.

| # | Task | Trạng thái | Thành phẩm chính |
|---|---|---|---|
| 1 | Xác định feature đầu vào | ✅ | `RAW_FEATURES` — **17 cột thô**; `EXCLUDED_FROM_X` ghi lý do loại 9 trường (xem §6.1c) |
| 2 | Thiết kế phương pháp xây dựng label | ✅ | `g(·)` = `label_frame()` + `add_label_noise()` (nhiễu 3%, chỉ đảo nhóm liền kề) |
| 3 | Tạo synthetic dataset | ✅ | `generate_households()` — 20.000 hộ, seed 42 |
| 4 | Kiểm tra phân bố class | ✅ | `class_distribution()` + cổng 1 của `check_gates()` — lớp nhỏ nhất **14,8%** ≥ 10% |
| 5 | Chia train/validation/test | ✅ | `split_train_test()` — train 80% (16.000) · test 20% (4.000), CV 5-fold trong train (xem §6.3) |
| 6 | Thiết lập baseline | ✅ | `DummyClassifier(strategy='stratified')` — CV macro-F1 **0,2483** ≈ 1/k |
| 7 | Train Decision Tree | ✅ | `train_decision_tree()` → `ml01_decision_tree_v1.joblib` — CV macro-F1 **0,8423** |
| 8 | Train Bagging Classifier | ✅ | `train_bagging()` → `ml01_bagging_v1.joblib` — CV macro-F1 **0,9067** |
| 9 | Train Random Forest | ✅ | `train_random_forest()` → `ml01_random_forest_v1.joblib` — CV macro-F1 **0,8520** |
| 10 | Train XGBoost | ✅ | `train_xgboost()` → `ml01_xgboost_v1.joblib` — CV macro-F1 **0,9190** |
| 11 | Model Evaluation | ✅ | `evaluate_on_test()` → `test_per_class.csv`, `test_confusion.csv`, dòng `split=test` trong `results.csv` |
| 12 | Model Comparison | ✅ | `compare_models()` → `model_comparison.csv`, `model_comparison_per_class.csv` |
| 13 | Feature Importance Analysis | ✅ | `feature_importance_report()` → `feature_importance.csv`, `feature_importance_pivot.csv` |
| 14 | Select Best Model | ✅ | `record_model_selection()` → `model_selection.json` — chọn **XGBoost** theo CV macro-F1 |
| 15 | Export Model | ✅ | `export_final_model()` → `ml01_xgboost_vfinal.joblib` + metadata (seed, feature order, config, chỉ số CV/test, sha256) |

Entry-point chạy phần train (task 7–10): `.venv\Scripts\python.exe scripts/train_ml01.py`.
Từng thuật toán có script riêng: [train_bagging.py](scripts/train_bagging.py),
[train_random_forest.py](scripts/train_random_forest.py),
[train_xgboost.py](scripts/train_xgboost.py).

**Ba cổng kiểm chứng §6.2 — đều đạt:**

| Cổng | Điều kiện | Kết quả |
|---|---|---|
| Cân bằng lớp | mọi lớp ≥ 10% | ✅ nhỏ nhất 14,8% (`DEBT_FOCUS`) |
| Ranh giới không quá sạch | accuracy tốt nhất ≤ 0,98 | ✅ 0,9248 (đo trên test) |
| Thắng baseline rõ rệt | macro-F1 hơn baseline ≥ 0,05 và > 2σ | ✅ baseline 0,2483; model kém nhất 0,8423 |

**Kết quả cuối — CV (chọn model) đối chiếu test (báo cáo):**

| Thuật toán | CV macro-F1 | σ giữa fold | Test macro-F1 | gap (CV − test) |
|---|---|---|---|---|
| decision_tree | 0,8423 | 0,0076 | 0,8443 | −0,0020 |
| bagging | 0,9067 | 0,0046 | 0,9012 | +0,0054 |
| random_forest | 0,8520 | 0,0019 | 0,8388 | +0,0132 |
| **xgboost** ← chọn | **0,9190** | 0,0061 | **0,9136** | +0,0054 |

Khoảng cách tới hạng nhì là 0,0123 = **2,0×σ** — đủ để phân biệt với dao động giữa các
fold, nhưng không phải cách biệt lớn; nên ghi đúng như vậy trong báo cáo.

### 6.1 Bài toán — 4 nhóm, **chốt 11/08/2026**

Phân loại hồ sơ hộ gia đình vào **4 nhóm khuyến nghị tài chính**, xếp theo **mức độ
nghiêm trọng giảm dần**:

| # | Nhãn | Ý nghĩa | Mức độ |
|---|---|---|---|
| 1 | `EMERGENCY` | Tài chính nguy cấp | 🔴 Rất rủi ro |
| 2 | `DEBT_FOCUS` | Cần tập trung xử lý nợ | 🟠 Rủi ro |
| 3 | `BUILD_BUFFER` | Cần xây dựng quỹ dự phòng | 🟡 Trung bình |
| 4 | `GROWTH` | Tài chính tương đối tốt, có thể tăng trưởng | 🟢 Tốt |

Thứ tự này **không phải để trình bày** — nó là quy tắc phân xử: một hộ thỏa nhiều điều
kiện cùng lúc thì nhận nhãn **nặng nhất**. Nhờ vậy `g(·)` là một hàm đơn trị, không cần
luật phá hòa riêng.

### 6.1b Hàm sinh nhãn `g(·)` — bản chốt

```
g(hộ):
    savings_months = tiết kiệm ÷ chi tiêu/tháng
    dti            = trả nợ/tháng ÷ thu nhập/tháng
    savings_rate   = (thu nhập − chi tiêu) ÷ thu nhập

    nếu   savings_rate < 0  HOẶC  savings_months < 1     → EMERGENCY      🔴
    nếu   dti ≥ 0.40                                     → DEBT_FOCUS     🟠
    nếu   savings_months < 3  HOẶC  savings_rate < 0.10  → BUILD_BUFFER   🟡
    ngược lại                                            → GROWTH         🟢
```

**Nguồn của từng ngưỡng** (ghi vào báo cáo, cùng dạng với `config/rules.yaml` của F02):

| Ngưỡng | Giá trị | Căn cứ |
|---|---|---|
| `savings_rate < 0` | chi ≥ thu | Dòng tiền âm — không cần viện dẫn |
| `savings_months` | 1 · 3 | Khuyến nghị quỹ dự phòng phổ biến **3–6 tháng chi tiêu**; dưới 1 tháng là không có đệm |
| `dti` | 0.40 | Quy tắc **28/36** (DTI back-end ≤ 36%), nới lên 40% làm ngưỡng "cần xử lý" |
| `savings_rate` | 0.10 | Mức tiết kiệm tối thiểu thường được khuyến nghị |

> Các ngưỡng này định nghĩa **ranh giới quyết định của một thí nghiệm**, không phải một
> khẳng định về thế giới. Chúng cần *hợp lý và có dẫn nguồn*, không cần *đúng tuyệt đối* —
> xem định vị ở §6.2.

### 6.1c Feature set `X` — chỉ biến thô

⚠️ **Đính chính §6.1 bản cũ:** bản cũ ghi feature gồm trường thô *"cộng các tỉ lệ dẫn xuất
ở mục 2.1"*. **Sai, và sai đúng vào chỗ nguy hiểm nhất.** Các tỉ lệ đó (`savings_months`,
`dti`, `savings_rate`) chính là biến mà `g(·)` đặt ngưỡng lên. Đưa chúng vào `X` thì một
cây sâu 3 tầng học thuộc nguyên `g(·)`, mọi thuật toán đạt ~100%, và bảng so sánh 4 thuật
toán mất sạch ý nghĩa — đúng rủi ro "circular labeling" ở §14.

**`X` chỉ gồm biến thô của form** — **17 cột, chốt 11/08/2026** sau khi đối chiếu với
`HouseholdProfile`. Một trường vào được `X` phải thỏa **cả hai** điều kiện:

1. **Luôn thu được** với mọi người dùng ML01 — ML01 chấm sức khỏe tài chính cho *mọi* hồ
   sơ, không riêng người đang tính vay.
2. **Mã hóa được** mà không phải tự bịa ra một tập giá trị chuẩn.

| Nhóm | Trường | Số cột |
|---|---|---|
| Tiền | thu nhập/tháng · chi tiêu/tháng · tiết kiệm · dư nợ · trả nợ/tháng | 5 |
| Nhân khẩu | số người trong nhà · số con · tuổi | 3 |
| Cờ tình trạng | có nợ · có tiết kiệm · phụng dưỡng người già | 3 |
| Tài sản sở hữu | multi-hot 6 loại của `AssetType` (`cash`/`vehicle`/`real_estate`/`insurance`/`gold`/`investment`) | 6 |

Hiện thực: `RAW_FEATURES` trong [labeler.py](src/hfml/ml/ml01_recommendation/labeler.py).

**⚠️ Đính chính bản 11/08/2026 sáng** — bảng cũ có ba chỗ không khớp schema thật:

| Trường trong bảng cũ | Vì sao bỏ |
|---|---|
| `nghề nghiệp`, `số năm đi làm` | Nằm trong **KHỐI VAY** của `HouseholdProfile` — chỉ hiện khi người dùng chọn `home_loan`, nên phần lớn hồ sơ ML01 để trống. Train trên dân số ai cũng có nghề rồi suy luận cho người bỏ trống là **lệch phân phối train/inference**. Hai trường này phục vụ ML02, đúng vai schema đã định |
| `khu vực` | `residence` là `str \| None` tự do tới 255 ký tự, **không có tập giá trị chuẩn**. Muốn mã hóa thì phải tự dựng vocabulary, mà một vocabulary bịa ra không đại diện cho phân bố địa lý nào có thật |
| `tài sản (nhà/xe/đất)` | Giữ lại nhưng **sửa danh sách**: DB thật (`tblassets`, 450 dòng) có 6 loại kể cả tài sản tài chính (`cash`, `gold`, `insurance`, `investment`), không phải 3 loại như bản mô tả ban đầu |

Bảng lý do đầy đủ nằm ở `EXCLUDED_FROM_X` cùng file — có test bắt buộc mọi mục phải trỏ
đúng một trường có thật trong schema và phải kèm lý do.

**Quy ước `None` của trường tiền có điều kiện:** `savings_amount`, `total_current_debt`,
`monthly_debt_payment` để trống nghĩa là **0 đã biết chắc**, không phải "chưa biết" — cờ
`has_savings` / `has_debt` đã mang thông tin có/không. Điền trung vị vào đây là bịa cho hộ
không tiết kiệm một khoản tiết kiệm bằng nửa dân số. Hằng số `ZERO_WHEN_ABSENT` giữ danh
sách này; `pipeline.normalizer` (F05) phải đổi `None → 0.0` trước khi vào Pipeline.

Ba tỉ lệ dẫn xuất **chỉ `g(·)` được dùng**, không bao giờ vào `X`. Chúng vẫn phục vụ tầng
rule và ML02 như mục 2.1 mô tả — ràng buộc này chỉ áp cho ML01.

Đây cũng là điều làm ML01 thành một bài toán thật: cây chẻ nhánh song song trục, nên để
xấp xỉ một ranh giới dạng **tỉ lệ** (`tiết kiệm ÷ chi tiêu < 1`) từ hai cột riêng lẻ, nó
phải dựng nhiều lát cắt. Đó chính là chỗ Boosting được kỳ vọng thắng cây đơn — và là lý do
bảng so sánh 4 thuật toán có nội dung.

### 6.2 ⚠️ Xây dựng label — điểm cần cẩn trọng nhất của F03

xlsx yêu cầu ML01 chạy trên **synthetic dataset** với label do mình thiết kế, kèm ghi chú *"tránh label leakage"*. Rủi ro cụ thể: nếu nhãn sinh ra bằng chính rule ở F02, rồi đưa **kết quả trung gian của rule đó** vào feature set, model chỉ học thuộc lại rule — accuracy ≈ 100% và bài toán trở nên vô nghĩa (circular labeling).

**Ba quy tắc bắt buộc khi sinh dữ liệu (task 2, 3):**

1. **Tách sạch feature khỏi nhãn.** Nhãn sinh từ `g(·)` ở §6.1b; `X` **chỉ là biến thô của
   form** theo danh sách §6.1c. Tuyệt đối không đưa `savings_months`, `dti`,
   `savings_rate` hay bất kỳ output trung gian nào của `g` vào `X`.
2. **Sinh dân số hộ gia đình có phân phối thực tế**, không sinh đều theo nhãn: thu nhập log-normal tham chiếu phân phối thu nhập hộ GSO, số nhân khẩu / số con theo phân bố thực, tương quan thu nhập–chi tiêu dương.
3. **Có vùng biên và nhiễu nhãn.** Thêm ~5–10% hồ sơ nằm sát ngưỡng và một tỉ lệ nhỏ nhãn bị đảo có kiểm soát — nếu không, ranh giới quyết định sạch tuyệt đối và mọi thuật toán đều đạt 100%, bảng so sánh mất hết ý nghĩa.

**Cụ thể hóa quy tắc 3** — tham số sinh dữ liệu, cố định cùng `random_seed = 42`:

| Tham số | Giá trị | Vì sao |
|---|---|---|
| Vùng biên | **8%** hồ sơ được kéo vào dải ±10% quanh một ngưỡng bất kỳ của `g(·)` | Ranh giới có bề dày thì cây phải xấp xỉ, không học thuộc được |
| Nhiễu nhãn | **3%** nhãn bị đảo sang một nhóm liền kề về mức độ | Mô phỏng hộ có hoàn cảnh mà 3 chỉ số không nắm hết |
| Đảo sang đâu | Nhóm **liền kề** trong thang 🔴→🟢, không đảo ngẫu nhiên | Đảo `GROWTH` thành `EMERGENCY` là nhiễu vô nghĩa, không phản ánh thực tế nào |

**Định vị trong báo cáo — phải viết đúng như sau:**

> ML01 là **thí nghiệm có ground truth đã biết**: đánh giá khả năng của họ thuật toán cây trong việc khôi phục một ranh giới quyết định đã biết trước từ biến thô, khi có nhiễu và vùng biên. Chỉ số của ML01 **đo năng lực thuật toán**, **không** chứng minh chất lượng tư vấn tài chính — phần chất lượng tư vấn nằm ở tính đúng đắn của rule F02.

Viết đúng câu này thì ML01 là một thí nghiệm hợp lệ. Viết sai thành *"model học được cách tư vấn tài chính"* thì không đỡ được câu hỏi phản biện.

**Kiểm chứng bắt buộc (task 4)** — ba cổng, không qua thì quay lại task 3:

| Cổng | Điều kiện | Không đạt thì |
|---|---|---|
| Cân bằng lớp | mỗi lớp ≥ **10%** | Chỉnh tham số dân số rồi sinh lại (KHÔNG chỉnh ngưỡng `g(·)` cho vừa) |
| Ranh giới không quá sạch | accuracy model tốt nhất ≤ **0,98** | Tăng vùng biên / nhiễu nhãn |
| Thắng baseline | mọi model > `DummyClassifier(stratified)` rõ rệt | Xem lại feature set |

> Cổng thứ nhất có một bẫy: khi một lớp dưới 10%, phản xạ tự nhiên là nới ngưỡng `g(·)`
> cho lớp đó to ra. **Không được** — ngưỡng đã chốt ở §6.1b và có dẫn nguồn; sửa ngưỡng để
> vừa dữ liệu là đảo ngược quan hệ nhân quả. Chỗ được chỉnh là **tham số sinh dân số**
> (phân phối thu nhập, tỉ lệ hộ có nợ, mức tiết kiệm), không phải định nghĩa nhãn.

### 6.3 Train & đánh giá (task 5–15)

- **Chia dữ liệu (task 5 — chốt 12/08/2026):** mỗi hộ một dòng nên không cần group split.

  | Tập | Tỉ lệ | Vai trò |
  |---|---|---|
  | **train** | 80% | `StratifiedKFold` 5-fold chạy trên đây — vừa là validation, vừa là căn cứ chọn model |
  | **test** | 20% | độc lập hoàn toàn, **chỉ** dùng cho đánh giá cuối |

  **Không** cắt thêm một tập validation cố định: CV bên trong tập train đã đóng vai đó,
  và với lớp nhỏ nhất ~15% thì cắt thêm lần nữa chỉ làm tập validation mỏng đi. Tập test
  không được tham gia CV, chọn model, hay tinh chỉnh siêu tham số — hiện thực ở
  `split_train_test()` ([train.py](src/hfml/ml/ml01_recommendation/train.py)), có test cấu
  trúc canh ràng buộc này.
- **Baseline:** `DummyClassifier(strategy='stratified')` — mọi model phải thắng rõ rệt
- **4 thuật toán** (cùng split, cùng feature, cùng `random_seed = 42`):

| Nhóm | Thuật toán | Vai trò trong báo cáo |
|---|---|---|
| Trees | `DecisionTreeClassifier` | Baseline diễn giải được, cho thấy overfit của cây đơn |
| Bagging | `BaggingClassifier` | Giảm variance bằng lấy mẫu bootstrap |
| Forests | `RandomForestClassifier` | Bagging + lấy mẫu ngẫu nhiên feature |
| Boosting | `XGBoost` | Giảm bias — kỳ vọng tốt nhất |

- **Metric:** Accuracy, **Macro-F1 (chỉ số chọn model — 4 lớp không cân bằng)**, Confusion Matrix, per-class precision/recall
- **Feature importance** (task 13) + so sánh model (task 12) → chọn model tốt nhất (task 14)
- **Export** (task 15): `joblib.dump` **cả pipeline preprocessing + model** vào `src/training/runs/ml01_<algo>_v1.joblib`, kèm `metadata.json` (seed, ngày train, metric, danh sách feature theo đúng thứ tự)

---

## 7. F04 — ML02: Home Credit Risk Classification (15 task · M04 · Tuần 4)

### 7.1 Bài toán

`application_train.csv` — 307.511 hồ sơ, nhãn **`TARGET`** (1 = khó khăn trả nợ). Đây là **nhãn thật, thu thập độc lập** — trụ cột ML mạnh nhất của đồ án.

### 7.2 Train hai phiên bản để so sánh (task 3)

| Phiên bản | Feature | Mục đích |
|---|---|---|
| **Full** | Toàn bộ, kể cả `EXT_SOURCE_1/2/3` | Chứng minh năng lực kỹ thuật, AUC cao |
| **Rút gọn** | Chỉ feature ánh xạ được từ form (mục 2.1) | Model **thực sự deploy** |

`EXT_SOURCE_1/2/3` là điểm tín dụng từ nguồn ngoài — nhóm feature mạnh nhất của Home Credit nhưng **không thể thu được từ form**. Bảng so sánh AUC hai phiên bản chính là mục *"phân tích tính khả thi triển khai"* trong báo cáo.

### 7.3 Class imbalance (task 4)

**8,07% dương** (24.825 / 307.511). Xử lý bằng `scale_pos_weight ≈ 11,4` (XGBoost) hoặc `class_weight='balanced'` (DT / RF / Bagging).

> ⚠️ **Không dùng accuracy làm chỉ số chọn model ở F04.** Model đoán "không ai vỡ nợ" đã đạt **91,93% accuracy**. Vẫn báo cáo accuracy trong bảng cho đủ yêu cầu, nhưng kết luận phải dựa trên **PR-AUC**.

### 7.4 Train & đánh giá (task 5–15)

- **Chia dữ liệu:** `StratifiedKFold` — mỗi khách hàng một dòng
- **Baseline:** `DummyClassifier(strategy='stratified')`
- **4 thuật toán:** giống hệt F03 — Decision Tree / Bagging / Random Forest / XGBoost, cùng seed 42
- **Metric bắt buộc:** **PR-AUC (chỉ số chọn model)**, ROC-AUC, Macro-F1, Confusion Matrix, Accuracy (chỉ để tham chiếu)
- **Calibration:** `CalibratedClassifierCV` + **calibration curve** + **Brier score**. Hệ thống ra quyết định theo ngưỡng xác suất — xác suất chưa calibrate thì ngưỡng vô nghĩa
- **Feature importance** (task 13): built-in importance + **SHAP summary plot (global)**. SHAP local top-5 cho một hồ sơ được đưa vào JSON ở F05 để LLM diễn đạt
- **Export** (task 15): `src/training/runs/ml02_<algo>_<full|reduced>_v1.joblib` + `metadata.json`

---

## 8. F05 — Prediction & LLM Recommendation Pipeline (18 task · M05 + M06 + M08 · Tuần 5–6, 8)

xlsx gộp cả 18 task vào M05; docx tách LLM thành M06 và Demo thành M08. Bản này theo docx:

### 8.1 Inference core — M05 (Tuần 5, task 1–7)

| Task | Nội dung | File |
|---|---|---|
| 1 | Chuẩn hóa input inference | `pipeline/normalizer.py` |
| 2 | Áp dụng preprocessing khi inference | dùng lại Pipeline đã `joblib.dump` ở F01 |
| 3 | Chạy 5 Rule-Based functions | `rules/engine.py` |
| 4 | ML01 inference — Recommendation Group | `pipeline/predictor.py` |
| 5 | ML02 inference — Home Credit Risk | `pipeline/predictor.py` |
| 6 | Tổng hợp kết quả Rule + ML | `pipeline/orchestrator.py` |
| 7 | Kiểm tra confidence/probability | `pipeline/orchestrator.py` |

**Task 7 quan trọng:** nếu `max(predict_proba) < ngưỡng tin cậy` → hạ cấp xuống kết luận của rule và đánh dấu `low_confidence: true` trong JSON. LLM phải diễn đạt điều này ra thành lời, không im lặng bỏ qua.

Đầu ra M05 là một **structured result JSON** cố định schema, chứa: kết quả 5 rule, nhãn + xác suất ML01, P(vỡ nợ) + nhóm rủi ro ML02, SHAP top-5, cờ cảnh báo dữ liệu bất thường.

### 8.2 LLM Assistant — M06 (Tuần 6, task 8–14)

| Task | Nội dung |
|---|---|
| 8 | Thiết kế prompt giải thích kết quả |
| 9 | Xây dựng context từ Rule + ML |
| 10 | Sinh giải thích kết quả bằng LLM |
| 11 | Sinh khuyến nghị tài chính tham khảo |
| 12 | Xử lý hội thoại chatbot (mức demo) |
| 13 | Thiết lập safety guardrails |
| 14 | Validate output LLM |

**Guardrail (task 13, 14):**

1. Prompt **chỉ chứa JSON đã tính sẵn** từ Rule/ML — cấm LLM tự tính
2. **Validate hậu kiểm:** trích mọi số trong câu trả lời bằng regex, đối chiếu với JSON đầu vào; số nào không khớp → chặn và sinh lại
3. **Disclaimer bắt buộc:** chỉ khuyến nghị **phân bổ theo lớp tài sản** (tiền gửi / trái phiếu / quỹ), **không** khuyến nghị mã cụ thể. Ghi rõ *"thông tin tham khảo, không phải tư vấn tài chính chuyên nghiệp"*
4. Kết quả ML02 phải diễn đạt là **ước lượng tham khảo**, không phải cam kết
5. Câu hỏi ngoài phạm vi → từ chối lịch sự + gợi ý bộ câu hỏi mẫu

**Chatbot (task 12) — 2 hướng, không dùng ML router:**

- **Hướng 1 (chính):** bộ câu hỏi mẫu đã gắn sẵn `intent_code` → đi thẳng vào engine, không qua phân loại. Nhanh, chính xác cao, giảm phụ thuộc LLM
- **Hướng 2:** người dùng tự nhập → phân loại bằng **keyword + rule** thành `BUSINESS` / `FINANCE_GENERAL` / `OUT_OF_SCOPE`, `OUT_OF_SCOPE` fallback về Hướng 1

> Ghi chú phạm vi: phiên bản v3 từng dự kiến một intent router bằng ML. Kế hoạch hiện tại chốt **không mở rộng bài toán ML thứ ba**, nên router chuyển sang keyword + rule.

### 8.3 Đóng gói & API — M08 (Tuần 8, task 15–18)

| Task | Nội dung | File |
|---|---|---|
| 15 | Đóng gói Python inference module | `pipeline/analyze.py` — một hàm `analyze(payload) -> Result` |
| 16 | Request/response schema bằng Pydantic | `api/schemas.py` |
| 17 | FastAPI demo endpoints | `api/main.py` — `POST /analyze`, `POST /chat` |
| 18 | Health check và model loading | `GET /health` + load model một lần lúc startup |

---

## 9. F06 — Testing & Validation (6 task · M07 · Tuần 7)

| # | Task | Nội dung cụ thể |
|---|---|---|
| 1 | Unit test preprocessing & feature engineering | Kiểm tra không rò rỉ: `fit` chỉ trên train; encode giá trị lạ không crash |
| 2 | Unit test Rule-Based | Ma trận case biên: mỗi ngưỡng test dưới / đúng bằng / trên |
| 3 | Test ML inference | Load artifact → predict → shape, dải giá trị, thứ tự feature khớp `metadata.json` |
| 4 | Phân tích lỗi model | Xem hồ sơ bị phân loại sai: sai ở lớp nào, ở vùng giá trị nào |
| 5 | Kiểm thử edge cases | Thu nhập = 0, chi > thu, không nợ, không tiết kiệm, hộ 1 người, trường thiếu |
| 6 | Kiểm tra khả năng tái lập | Chạy lại full pipeline với seed 42 → metric trùng khớp đến 4 chữ số thập phân |

Đã có [tests/test_structure.py](tests/test_structure.py) (F01 task 1) — mở rộng thêm `tests/test_{preprocessing,rules,ml01,ml02,pipeline}.py`.

---

## 10. F07 — Experiment Tracking & Documentation (4 task · M08 · Tuần 8)

| # | Task | Thành phẩm |
|---|---|---|
| 1 | Ghi nhận experiment và metrics | `src/training/runs/results.csv` — mỗi dòng: model, feature set, seed, toàn bộ metric, ngày chạy |
| 2 | **Tài liệu hóa dataset** | `docs/dataset.md` — nguồn, license, số dòng, phân bố nhãn, cách sinh synthetic ML01 |
| 3 | Tài liệu hóa model | `docs/model_card.md` — mục đích, feature, metric, **giới hạn sử dụng**, đối tượng không áp dụng |
| 4 | Viết báo cáo pipeline Python/ML | Báo cáo cuối |

---

## 11. Giao thức đánh giá — bảng tổng hợp

| | ML01 | ML02 |
|---|---|---|
| Nguồn nhãn | Synthetic, hàm `g(·)` tự thiết kế | `TARGET` — nhãn thật |
| Số lớp | 4 | 2 |
| Cân bằng | Kiểm tra, mỗi lớp ≥ 10% | 8,07% dương |
| Chia dữ liệu | `StratifiedKFold` | `StratifiedKFold` |
| **Chỉ số chọn model** | **Macro-F1** | **PR-AUC** |
| Báo cáo kèm | Accuracy, Confusion Matrix, per-class P/R | ROC-AUC, Macro-F1, Confusion Matrix, Accuracy |
| Baseline | `DummyClassifier(stratified)` | `DummyClassifier(stratified)` |
| Calibration | — | `CalibratedClassifierCV` + curve + Brier |
| Giải thích | Feature importance | Feature importance + SHAP (global + local) |
| Thuật toán | DT · Bagging · RF · XGBoost | DT · Bagging · RF · XGBoost |

**Hạ tầng cần bổ sung:**
- ✅ [ml/base.py](src/hfml/ml/base.py): contract `BaseClassifier` đã có `predict_proba` (làm ở F01 task 1)
- ✅ [ml/registry.py](src/hfml/ml/registry.py): lưu/tải artifact + `metadata.json` (feature theo đúng thứ tự)
- `ml/evaluation/metrics.py`: `classification_metrics()` — làm trong Tuần 3, trước khi train
- `ml/evaluation/plots.py`: confusion matrix, PR curve, calibration curve, SHAP summary

---

## 12. Timeline & cột mốc

| Tuần | Ngày | Giai đoạn | Nhóm task | Cột mốc | Kết quả cần đạt |
|---|---|---|---|---|---|
| 1 | 10–16/08 | Chuẩn bị môi trường & dữ liệu | F01 (14) | **M01** | Python + Data Pipeline chạy được |
| 2 | 17–23/08 | Rule-Based Engine | F02 (7) | **M02** | 5 rule chạy độc lập, trả kết quả có cấu trúc |
| 3 | 24–30/08 | ML01 | F03 (15) | **M03** | ML01 train + evaluation + export |
| 4 | 31/08–06/09 | ML02 | F04 (15) | **M04** | ML02 train + evaluation + export |
| 5 | 07–13/09 | Inference Pipeline | F05 (1–7) | **M05** | Một input chạy xuyên suốt Rule + ML → structured result |
| 6 | 14–20/09 | LLM Assistant | F05 (8–14) | **M06** | LLM giải thích kết quả + guardrail |
| 7 | 21–27/09 | Testing & Validation | F06 (6) | **M07** | Bộ test đầy đủ + kết quả đánh giá |
| 8 | 28/09–04/10 | Đóng gói & Demo | F05 (15–18) + F07 (4) | **M08** | Pipeline demo được + tài liệu + báo cáo |

**Tổng: 79 task** (RB04 từng bị loại 11/08/2026 rồi khôi phục cùng ngày — xem mục 5). Cột `Ngày bắt đầu` / `Ngày kết thúc` trong xlsx điền theo bảng này.

---

## 13. Thứ tự ưu tiên & phương án cắt giảm

| Mức | Nội dung |
|---|---|
| **P1 — Bắt buộc** | Data → Rule-Based → ML01 → ML02 → Inference → Evaluation |
| **P2 — Quan trọng** | Testing → LLM Explanation → FastAPI Demo |
| **P3 — Có thể cắt** | Chatbot nâng cao → LLM validation nâng cao → Experiment tracking nâng cao |

**Nếu thiếu thời gian, cắt theo thứ tự:** chatbot Hướng 2 → FastAPI (thay bằng script CLI demo) → LLM narrator (thay bằng template tiếng Việt điền chỗ trống) → ML02 phiên bản Full.

**Không được cắt:** F01 task 14 (Pipeline), F02, F03, F04, và toàn bộ mục 11 (giao thức đánh giá) — đó là phần ML và phần "tư vấn", tức trọng tâm chấm điểm.

---

## 14. Rủi ro & cách xử lý

| Rủi ro | Dấu hiệu | Xử lý |
|---|---|---|
| **ML01 circular labeling** | Accuracy > 0,98, mọi thuật toán ngang nhau | Mục 6.2 — tách sạch feature khỏi nhãn, thêm vùng biên + nhiễu, định vị đúng trong báo cáo |
| **ML02 accuracy ảo** | Accuracy ~92% nhưng recall lớp dương ~0 | Mục 7.3 — chọn model bằng PR-AUC, `scale_pos_weight` |
| **Rò rỉ do preprocessing** | Metric CV cao bất thường | Mục 4.4 — mọi bước impute/encode/scale nằm trong Pipeline, fit trên train |
| **Domain gap VNĐ ↔ Home Credit** | Model trả xác suất vô nghĩa với input VN | Mục 2.1 — feature tỉ lệ, bỏ hết giá trị tiền tuyệt đối |
| **LLM bịa số** | Con số trong câu trả lời không có trong JSON | Mục 8.2 — validate hậu kiểm bằng regex |
| **Ngưỡng rule không có căn cứ** | Bị hỏi "sao lấy 36%?" | Mục 5 — `config/rules.yaml` có cột nguồn trích dẫn |

### 14.1 Câu hỏi hội đồng — chuẩn bị sẵn

| Câu hỏi | Trả lời ở mục |
|---|---|
| "Nhãn của bạn lấy từ đâu?" | ML02 dùng `TARGET` — nhãn thật. ML01 là synthetic có ground truth đã biết, mục 6.2 nói rõ định vị và giới hạn kết luận |
| "Accuracy bao nhiêu?" | Mục 11 — ML02 mất cân bằng 8/92 nên chọn model bằng PR-AUC; accuracy vẫn báo cáo nhưng không dùng để kết luận |
| "Sao không dùng ML cho cả 5 chức năng rule?" | Mục 1 + 5 — rule là tính toán xác định, dùng ML sẽ thành circular labeling |
| "Model train bằng dữ liệu nước ngoài sao áp cho VN?" | Mục 2.1 — feature tỉ lệ bất biến đơn vị tiền tệ + bảng so sánh Full vs Rút gọn (7.2) |
| "Vì sao tin được khuyến nghị?" | Mục 7.4 — SHAP global + local; mục 5 — ngưỡng rule có nguồn |
| "LLM có bịa số không?" | Mục 8.2 — LLM chỉ nhận JSON, có validate hậu kiểm đối chiếu số |
| "Boosting sao chọn XGBoost?" | Mục 6.3 — đại diện nhóm Boosting theo yêu cầu môn học, so cùng seed/split với 3 nhóm còn lại |

---

## 15. Việc cần làm ngay (Tuần 1)

| # | Nội dung | Chặn | Trạng thái |
|---|---|---|---|
| 1 | ~~Chốt phương án phân rã chi tiêu cho RB04~~ | ~~F02~~ | ✅ **không còn** — RB04 đã ra ngoài phạm vi (11/08/2026) |
| 2 | Chốt **4 nhóm khuyến nghị ML01** và hàm sinh nhãn `g(·)` | F03 | ✅ chốt 11/08/2026 — xem §6.1, §6.1b, §6.1c |
| 3 | Bổ sung **khối vay 5 ô** vào **form + backend** (mục 4.1) | F04 | ⬜ schema xong, form/backend chưa |
| 4 | Đổi `average_monthly_expense` và `monthly_debt_payment` thành bắt buộc ở backend | F02, F03 | ⬜ chưa |
| 5 | Điền cột `Ngày bắt đầu` / `Ngày kết thúc` trong xlsx theo bảng mục 12 | — | ⬜ chưa |

Dataset Home Credit **đã đủ 5 file** trong `dataset/` — không còn phụ thuộc tải thêm.
