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
| 5 | Chia train/validation/test | ✅ | `split_train_val_test()` — train 70% (14.000) · validation 15% (3.000) · test 15% (3.000), stratified; **không dùng K-Fold CV** (xem §6.3) |
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

1. **Luôn thu được** với mọi người dùng ML01 — ML01 dự đoán nhóm định hướng tài chính của
   hộ gia đình dựa trên các đặc trưng tài chính đầu vào, cho *mọi* hồ sơ, không riêng
   người đang tính vay.
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

  | Tập | Tỉ lệ | Số hộ | Vai trò |
  |---|---|---:|---|
  | **train** | 70% | 14.000 | fit model |
  | **validation** | 15% | 3.000 | **căn cứ chọn model** — so macro-F1 giữa 4 thuật toán; cũng là nơi tinh chỉnh siêu tham số nếu cần |
  | **test** | 15% | 3.000 | độc lập hoàn toàn, **chỉ** dùng cho đánh giá cuối |

  **KHÔNG dùng K-Fold Cross-Validation.** Pipeline:

  ```
  Dataset 20.000 ──stratified──► Train 70% / Validation 15% / Test 15%
                                    │
       Train ──fit──► 4 model (Decision Tree · Bagging · Random Forest · XGBoost)
                                    │
       Validation ──chấm──► so macro-F1 ──► chọn model tốt nhất ──► tuning nếu cần
                                    │
       Test ──chấm MỘT lần──► Final Evaluation
  ```

  Tập test không được tham gia training, chọn model, hay tinh chỉnh siêu tham số — hiện
  thực ở `split_train_val_test()` ([train.py](src/hfml/ml/ml01_recommendation/train.py)),
  có test cấu trúc canh ràng buộc này.

  `stratify` ở cả hai lần cắt để tỉ lệ `EMERGENCY` · `DEBT_FOCUS` · `BUILD_BUFFER` ·
  `GROWTH` giữ nguyên giữa ba tập.

  > **Sửa so với bản chốt 12/08/2026:** bản đầu là 80/20 với CV 5-fold bên trong tập train.
  > Ngày 14/08/2026 phương pháp đổi thành **70/15/15 và bỏ hẳn K-Fold Cross-Validation**
  > theo khuyến nghị của giảng viên cho phạm vi đồ án.
  >
  > **Cái mất khi bỏ CV, ghi ra để đọc số cho đúng:** mỗi chỉ số bây giờ là MỘT điểm đo
  > trên 3.000 hộ, không phải trung bình 5 lần kèm độ lệch. Không còn `macro_f1_std`, nên
  > một khoảng chênh vài phần nghìn giữa hai model không quy chiếu được về độ nhiễu. Kéo
  > theo hai chỗ đã phải viết lại: tiêu chí *"hơn á quân bao nhiêu lần σ"* của task 14
  > (`margin_vs_fold_std`) đã gỡ bỏ, và cổng "Thắng baseline" chỉ còn ngưỡng tuyệt đối
  > `≥ 0,15` thay vì `max(0,15 · 2σ)` — tức cổng này **dễ qua hơn** bản cũ.
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

**Tiến độ: 15/15 task — F04 hoàn thành 16/08/2026.**

**Model cuối: `ml02_xgboost_reduced_vfinal`** · PR-AUC test **0,1714**
(2,12× mức đoán bừa) · ROC-AUC 0,6952 · recall lớp dương 0,4039 · Brier 0,0709 ·
ngưỡng 0,1303. Bộ FULL đạt PR-AUC test 0,2472 nhưng **không deploy được** —
đó là nội dung mục "phân tích tính khả thi triển khai" (§7.2).

| # | Task | Trạng thái | Thành phẩm chính |
|---|---|---|---|
| 1 | Khám phá Home Credit Dataset | ✅ 15/08/2026 | [explore.py](src/hfml/ml/ml02_credit_risk/explore.py) · [report.py](src/hfml/ml/ml02_credit_risk/report.py) · [explore_ml02.py](scripts/explore_ml02.py) → [docs/ml02_eda.md](docs/ml02_eda.md) + 15 bảng ở `src/training/runs/ml02_eda/` · [test](tests/test_explore_ml02.py) (27 test) |
| 2 | Data Cleaning | ✅ 15/08/2026 | [clean.py](src/hfml/ml/ml02_credit_risk/clean.py) · [clean_ml02.py](scripts/clean_ml02.py) → `data/interim/ml02/` + 8 bảng ở `src/training/runs/ml02_cleaning/` · [test](tests/test_clean_ml02.py) (25 test) |
| 3 | Feature Engineering | ✅ 15/08/2026 | [features.py](src/hfml/ml/ml02_credit_risk/features.py) · [build_features_ml02.py](scripts/build_features_ml02.py) → 2 pipeline `.joblib` + 4 bảng ở `src/training/runs/ml02_features/` · [test](tests/test_features_ml02.py) (20 test) |
| 4 | Xử lý class imbalance | ✅ 15/08/2026 | [imbalance.py](src/hfml/ml/ml02_credit_risk/imbalance.py) · [imbalance_ml02.py](scripts/imbalance_ml02.py) → 4 file ở `src/training/runs/ml02_imbalance/` · [test](tests/test_imbalance_ml02.py) (16 test) |
| 5 | Chia train/validation/test | ✅ 15/08/2026 | [split.py](src/hfml/ml/ml02_credit_risk/split.py) · [split_ml02.py](scripts/split_ml02.py) → 4 file ở `src/training/runs/ml02_split/` · [test](tests/test_split_ml02.py) (21 test) |
| 6 | Xây dựng baseline | ✅ 15/08/2026 | [baseline.py](src/hfml/ml/ml02_credit_risk/baseline.py) · [baseline_ml02.py](scripts/baseline_ml02.py) · `binary_metrics()` ở [metrics.py](src/hfml/ml/evaluation/metrics.py) → `runs/ml02_baseline/` + 2 dòng trong `results.csv` · [test](tests/test_baseline_ml02.py) (16 test) |
| 7 | Train Decision Tree | ✅ 15/08/2026 | [train.py](src/hfml/ml/ml02_credit_risk/train.py) · [train_ml02_decision_tree.py](scripts/train_ml02_decision_tree.py) → `runs/ml02_models/` + 2 dòng `results.csv` · [test](tests/test_train_ml02.py) (18 test) |
| 8 | Train Bagging Classifier | ✅ 15/08/2026 | `train_bagging()` ở [train.py](src/hfml/ml/ml02_credit_risk/train.py) · [train_ml02_bagging.py](scripts/train_ml02_bagging.py) → 2 artifact + 2 dòng `results.csv` · [test](tests/test_train_ml02.py) (+8 test) |
| 9 | Train Random Forest | ✅ 15/08/2026 | `train_random_forest()` ở [train.py](src/hfml/ml/ml02_credit_risk/train.py) · [train_ml02_random_forest.py](scripts/train_ml02_random_forest.py) → 2 artifact + 2 dòng `results.csv` · [test](tests/test_train_ml02.py) (+7 test) |
| 10 | Train XGBoost | ✅ 15/08/2026 | `train_xgboost()` ở [train.py](src/hfml/ml/ml02_credit_risk/train.py) · [train_ml02_xgboost.py](scripts/train_ml02_xgboost.py) → 2 artifact + 2 dòng `results.csv` · [test](tests/test_train_ml02.py) (+9 test) |
| 11 | Đánh giá model | ✅ 15/08/2026 | [evaluate.py](src/hfml/ml/ml02_credit_risk/evaluate.py) · [evaluate_ml02.py](scripts/evaluate_ml02.py) → 6 bảng ở `runs/ml02_evaluation/` · [test](tests/test_evaluate_ml02.py) (18 test) |
| 12 | So sánh model | ✅ 15/08/2026 | [compare.py](src/hfml/ml/ml02_credit_risk/compare.py) · [compare_ml02.py](scripts/compare_ml02.py) → 5 bảng ở `runs/ml02_comparison/` · [test](tests/test_compare_ml02.py) (22 test) |
| 13 | Phân tích feature importance | ✅ 15/08/2026 | [importance.py](src/hfml/ml/ml02_credit_risk/importance.py) · [importance_ml02.py](scripts/importance_ml02.py) → 4 bảng ở `runs/ml02_importance/` · [test](tests/test_importance_ml02.py) (18 test) |
| 14 | Chọn model tốt nhất | ✅ 15/08/2026 | [select.py](src/hfml/ml/ml02_credit_risk/select.py) · [select_ml02.py](scripts/select_ml02.py) → `runs/ml02_selection/` · [test](tests/test_select_ml02.py) (18 test) |
| 15 | Export model | ✅ 16/08/2026 | [export.py](src/hfml/ml/ml02_credit_risk/export.py) · [export_ml02.py](scripts/export_ml02.py) → `ml02_xgboost_reduced_vfinal.joblib` + metadata · [test](tests/test_export_ml02.py) (15 test) |

### 7.0o Export (task 15) — **F04 hoàn tất**

**Artifact: `ml02_xgboost_reduced_vfinal`** (25 MB) + `metadata.json`.

Artifact **tự chứa bốn phần**, không phải chỉ có model. Một file chỉ chứa
estimator là artifact chưa dùng được: bên gọi vẫn phải tự dựng feature, tự nhớ
thứ tự cột, tự nhớ ngưỡng — mà nhớ sai thì model **vẫn chạy và vẫn trả xác
suất**, chỉ có điều xác suất đó vô nghĩa.

| # | Phần | Từ task |
|---|---|---|
| 1 | Pipeline feature — nối bureau → dựng tỉ lệ → tiền xử lý | 3 |
| 2 | XGBoost đã train, `scale_pos_weight` từ tập train | 10 |
| 3 | Lớp hiệu chuẩn isotonic, fit trên validation | 14 |
| 4 | Ngưỡng nghiệp vụ **0,1303** | 14 |

> **Bằng chứng mạnh nhất cho việc phải gói ngưỡng vào artifact.** Trên 2.000 hồ
> sơ test, xác suất **đã hiệu chuẩn** cao nhất chỉ **0,3478**. Nếu `predict()`
> dùng quy tắc `argmax` mặc định của sklearn (tương đương ngưỡng 0,5) thì
> **KHÔNG hồ sơ nào** được gắn `HIGH_RISK` — model vẫn chạy, vẫn trả xác suất
> hợp lệ, chỉ là không phân loại gì cả. Với ngưỡng đã chốt: **345/2.000 (17,2%)**.
>
> Ngưỡng nằm ở tài liệu mà không nằm trong file thì sớm muộn có nơi triển khai
> quên áp nó, và không có gì báo lỗi.

**Hợp đồng inference trong `metadata.json`:**

| Trường | Giá trị |
|---|---|
| `feature_names` | 17 cột, **đúng thứ tự** — F06 task 3 đối chiếu danh sách này |
| `label_mapping` | `0 → LOW_RISK` · `1 → HIGH_RISK` · kèm nhãn tiếng Việt |
| `threshold` | 0,1303 + quy tắc + caveat |
| `calibration` | isotonic, fit trên validation |
| `model_config` | toàn bộ siêu tham số XGBoost |
| `metrics_validation` / `metrics_test` | PR-AUC test **0,1714** |
| `data_version` | SHA-256 rút gọn của cả 5 file dataset |
| `limitations` | 5 mục |

Nhãn trả về là **chuỗi nghiệp vụ** `LOW_RISK`/`HIGH_RISK` chứ không phải 0/1:
tầng `api` và `llm` đọc nhãn này ra cho người dùng, để 0/1 thì mỗi nơi tự đặt
tên một kiểu và sớm muộn có nơi đảo ngược ý nghĩa.

**Kiểm nạp lại — 5/5 đạt**, lệch xác suất **0,000e+00**:

| Phép kiểm | Bắt được kiểu hỏng nào |
|---|---|
| nạp lại được | file hỏng / thiếu phụ thuộc |
| xác suất trùng khít | model và pipeline |
| thứ tự feature khớp metadata | hợp đồng inference (F06 task 3) |
| `predict` áp đúng ngưỡng | dùng 0,5 hay dùng ngưỡng chốt |
| ngưỡng ≠ 0,5 | quên gói ngưỡng |

`fit()` của artifact **ném lỗi** — train lại lên bản export thì mọi con số trong
`metadata.json` mô tả một model khác.

**Tích hợp vào hệ thống là bước RIÊNG, chưa làm.** Hai hằng số ở
[api/main.py](src/hfml/api/main.py) đang chờ:

```python
ML02_SLUG           = "ml02_xgboost_reduced_vfinal"   # hiện là "ml02_best_reduced_vfinal"
ML02_RISK_THRESHOLD = 0.1303                          # hiện là None
```

Đổi hai dòng đó thì nhánh `LOAN_RISK_DIAGNOSIS` của Chatbot (đã dựng sẵn đường
đi từ 15/08) tự sống.

### 7.0n Chọn model + kiểm trên test (task 14)

**Model được chọn: `ml02_xgboost_reduced`.** Ba bước, thứ tự không được đảo —
đảo thứ tự thì con số test thôi là ước lượng độc lập mà thành một chỉ số đã
được tối ưu gián tiếp, và đó là dạng rò rỉ không để lại dấu vết nào trong mã.

**Bước 1 — chọn, chỉ bằng bằng chứng validation:**

1. PR-AUC cao nhất ở bộ triển khai (0,1711) — chỉ số chọn model của ML02 (§7.3).
2. Khoảng cách với á quân Bagging (+0,0102) **phân biệt được với nhiễu**: khoảng
   tin cậy 95% [+0,0045 · +0,0151] không chứa 0, thắng **100%** số lần lấy mẫu.
3. Cùng thuật toán cũng dẫn đầu ở bộ full → lựa chọn không phụ thuộc bộ feature.
4. Bộ triển khai là **rút gọn**, không phải full: bộ full có `EXT_SOURCE_1/2/3`
   mà form người dùng **không thu được** (§7.2).

**Bước 2 — chốt cấu hình, vẫn trên validation. Hiệu chuẩn là bước quyết định:**

| | Trước hiệu chuẩn | Sau hiệu chuẩn |
|---|---:|---:|
| Chênh hiệu chuẩn (model nói − thực tế) | **+0,3394** | **−0,0004** |
| Brier | 0,2017 | **0,0710** |

Task 11 đo được cả 8 model đều **nói quá** về rủi ro — hệ quả bắt buộc của
`class_weight` / `scale_pos_weight`. Isotonic kéo chênh lệch về gần 0 và giảm
Brier **65%**. Không có bước này thì con số ngưỡng không mang ý nghĩa xác suất
nào, mà §8.1 lại ra quyết định theo đúng ngưỡng đó và tầng `llm` đọc nó ra cho
người dùng.

Dùng `isotonic` chứ không phải `sigmoid` mặc định: sigmoid giả định lệch hiệu
chuẩn có dạng hàm logistic, mà lệch ở đây do **trọng số lớp** gây ra nên không
có lý do gì mang dạng đó. Isotonic chỉ giả định đơn điệu — đúng thứ ta biết chắc.
`FrozenEstimator` giữ nguyên model đã train, nên model cuối vẫn đúng là model đã
được so sánh ở task 12.

> **Ngưỡng `LOW_RISK` / `HIGH_RISK` = 0,1303** — chốt bằng F1 lớn nhất của lớp
> dương trên validation. **Không phải 0,5**, đúng như đã cảnh báo từ task 4: với
> tỉ lệ nền 8,07% thì 0,5 xếp gần như mọi hồ sơ vào `LOW_RISK`.
>
> ⚠️ **Giới hạn phải ghi vào `model_card.md`:** F1 coi *bỏ lọt một ca vỡ nợ* và
> *gắn nhãn rủi ro cho một hồ sơ tốt* là **đắt như nhau**. Trong tín dụng thật
> thì không. Ngưỡng đúng phải suy từ ma trận chi phí của tổ chức cho vay, mà đồ
> án không có — F1 là lựa chọn **trung tính khi chưa có chi phí**, không phải
> lựa chọn tối ưu.
>
> Điểm vận hành thứ hai để tham chiếu: rà soát 10% hồ sơ rủi ro nhất ứng với
> ngưỡng 0,1483.

**Bước 3 — mở tập test, đúng một lần.** 46.127 hồ sơ chưa từng bị chạm ở task 1–13.

| Chỉ số | Validation | **Test** | Chênh |
|---|---:|---:|---:|
| PR-AUC | 0,1696 | **0,1714** | −0,0018 |
| ROC-AUC | 0,6935 | **0,6952** | −0,0017 |
| F1 lớp 1 | 0,2474 | **0,2494** | −0,0020 |
| Recall lớp 1 | 0,3990 | **0,4039** | −0,0048 |
| Precision lớp 1 | 0,1793 | **0,1804** | −0,0011 |
| Brier | 0,0710 | **0,0709** | +0,0001 |
| Chênh hiệu chuẩn | −0,0004 | **−0,0009** | +0,0005 |

**Chênh lệch tổng quát hoá = −0,0018**, tức test *nhỉnh hơn* validation. Không có
dấu hiệu việc chọn model đã bám vào validation — đúng như mong đợi, vì mọi lựa
chọn đều dựa trên PR-AUC chứ không phải dò siêu tham số.

Confusion matrix trên test tại ngưỡng 0,1303: bắt được **1.504 / 3.724** ca vỡ nợ
(recall 40,4%), đổi bằng 6.831 hồ sơ tốt bị gắn nhãn rủi ro.

**Bộ FULL trên test — chỉ để báo cáo §7.2:** PR-AUC **0,2472** so với 0,1714 của
bộ triển khai (**+0,0758**). Con số cuối cùng cho mục *"phân tích tính khả thi
triển khai"*: model deploy được mất khoảng **44%** năng lực dự báo so với model
dùng đủ dữ liệu Home Credit.

> **Một hiệu ứng nhỏ đáng ghi:** PR-AUC nhích từ 0,1711 xuống 0,1696 sau hiệu
> chuẩn. Isotonic là phép biến đổi đơn điệu nên về nguyên tắc giữ nguyên thứ tự
> — nhưng nó tạo ra các đoạn phẳng, và các hồ sơ rơi vào cùng một đoạn trở thành
> **đồng hạng**. Đó là cái giá rất nhỏ để đổi lấy xác suất đọc được đúng nghĩa.

**Chưa export.** `decision.json` ghi `exported: false`; `select.py` có test chặn
mọi hàm tên chứa `export`/`dump`.

### 7.0m Feature importance (task 13) — ba cách đo, chẩn đoán không phải chọn feature

Một cột "importance" duy nhất rất dễ bị đọc như chân lý, trong khi ba phương pháp
đo ba thứ khác nhau và **thường xếp hạng khác nhau**:

| Cách đo | Đo gì | Điểm yếu |
|---|---|---|
| built-in (impurity) | cây giảm được bao nhiêu tạp chất khi chẻ theo cột | **thiên vị cột nhiều giá trị** — cột liên tục có hàng nghìn điểm cắt khả dĩ, cột nhị phân chỉ có một |
| permutation | xáo trộn cột rồi đo PR-AUC tụt bao nhiêu | đắt; nhưng đo đúng **đóng góp vào năng lực dự báo** |
| SHAP | trung bình \|đóng góp\| ở mức từng hồ sơ | cho cả hướng tác động — thứ tầng `llm` cần (§7.4) |

**Bộ FULL — XGBoost, top 6 theo permutation:**

| Feature | PR-AUC tụt |
|---|---:|
| `EXT_SOURCE_2` | 0,0588 |
| `EXT_SOURCE_3` | 0,0405 |
| **`credit_term_implied`** | **0,0277** |
| `EXT_SOURCE_1` | 0,0197 |
| `DAYS_BIRTH` | 0,0164 |
| `credit_goods_markup` | 0,0101 |

Ba cột `EXT_SOURCE_*` chiếm ba trong bốn vị trí đầu — **xác nhận bằng số** điều
§7.2 giả định từ đầu và §7.0l đã đo qua chênh lệch PR-AUC: đó chính là thứ bộ
RÚT GỌN mất đi.

**Bộ RÚT GỌN — XGBoost, top 6 theo permutation:**

| Feature | PR-AUC tụt |
|---|---:|
| **`credit_term_implied`** | **0,0599** |
| **`bureau_credit_income_ratio`** | 0,0385 |
| **`bureau_debt_income_ratio`** | 0,0302 |
| **`bureau_active_loan_count`** | 0,0130 |
| `dti` | 0,0110 |
| `age_years` | 0,0100 |

> **Hai kết quả đáng chú ý cho phần thiết kế sản phẩm:**
>
> 1. **`credit_term_implied` (`AMT_CREDIT / AMT_ANNUITY`) đứng đầu ở CẢ HAI bộ**
>    theo cả ba cách đo. Đây là feature task 3 mới thêm, và nó vượt cả `dti` lẫn
>    `credit_income_ratio` — hai tỉ lệ vốn được §2.1b coi là trục chính.
> 2. **Ba trong bốn vị trí tiếp theo của bộ rút gọn là nhóm bureau**, tức đúng
>    **mục C của form "Thông tin khoản vay"**. Quyết định gộp `bureau.csv` ở
>    task 1 (trước đó §4.3 xếp nó vào diện "để dành") là quyết định đem lại phần
>    lớn sức mạnh cho model deploy được.

**28/82 feature của bộ full có permutation importance ÂM** — xáo trộn chúng làm
model *tốt lên*, tức chúng chỉ đang thêm nhiễu. Bộ rút gọn chỉ có 3/17. Đây là
thông tin cho vòng sau, **không** dùng để cắt feature ngay (xem dưới).

**Chỗ ba cách đo bất đồng nhất** ở bộ rút gọn là `bureau_overdue_income_ratio`:
built-in xếp **#7**, permutation xếp **#17**. Đúng dạng thiên vị impurity —
cột này liên tục và lệch nặng nên cây hay chẻ theo nó, nhưng lát cắt đó không
giúp dự báo thêm.

> ⚠️ **Đây là CHẨN ĐOÁN, không phải bước chọn feature.** Permutation và SHAP đo
> trên **validation**. Dùng chúng để bỏ bớt feature rồi train lại và chấm lại
> trên chính tập đó là **rò rỉ** — validation khi ấy đã tham gia quyết định
> feature nào tồn tại, và chỉ số thu được sẽ lạc quan mà không có dấu hiệu gì.
> Việc chọn feature có giám sát đã có chỗ của nó: `SupervisedFeatureSelector`
> nằm TRONG Pipeline nên chỉ nhìn thấy tập train (§4.3e). Feature set **giữ
> nguyên** sau task 13; metadata ghi `feature_selection_changed: false`, có test
> canh.

> **Một bất định thật đã tìm ra khi chạy test:** `test_random_forest_is_reproducible`
> trượt đúng một lần, lúc máy đang chạy song song tác vụ phân tích importance.
> Nguyên nhân: với `n_jobs=-1`, `predict_proba` cộng dồn kết quả các cây theo
> thứ tự do bộ lập lịch quyết định, mà cộng số thực không kết hợp. Đo được lệch
> tối đa **1,1e-16** — dưới ngưỡng biểu diễn của `float64`. PR-AUC là chỉ số
> **xếp hạng** nên chênh lệch đó đủ để đảo thứ tự hai hồ sơ sát nhau và làm chỉ
> số nhích. Đã đặt dung sai tường minh `abs=1e-9` kèm giải thích — vẫn chặt hơn
> nhiều so với yêu cầu tái lập của F06 task 6 (trùng tới 4 chữ số thập phân).

### 7.0l So sánh 8 model (task 12) — xếp hạng trên validation

**Bảng xếp hạng, tính RIÊNG trong từng bộ feature.** Xếp chung thì bốn model
của bộ FULL chiếm hết đầu bảng và bộ deploy được — thứ thật sự chạy trong sản
phẩm — không bao giờ hiện ra.

| Hạng | Bộ FULL | PR-AUC | | Bộ RÚT GỌN | PR-AUC |
|---:|---|---:|---|---|---:|
| 1 | **xgboost** | **0,2533** | | **xgboost** | **0,1711** |
| 2 | bagging | 0,2311 | | bagging | 0,1608 |
| 3 | random_forest | 0,2293 | | random_forest | 0,1505 |
| 4 | decision_tree | 0,1914 | | decision_tree | 0,1445 |

#### Chênh lệch có thật hay chỉ là nhiễu — bootstrap thay cho K-Fold

Task 5 bỏ K-Fold nên không có `pr_auc_std` giữa các fold để quy chiếu. Task 9 đã
gặp đúng vấn đề đó và **để ngỏ**: Random Forest thua Bagging 0,0018 ở bộ full —
thật hay nhiễu?

Bỏ trống câu hỏi này thì bảng xếp hạng dẫn tới kết luận sai: xếp theo một con số
mà không biết nó dao động bao nhiêu chẳng khác gì xếp theo nhiễu.

**Bootstrap CẶP ĐÔI trên tập validation trả lời được, không cần K-Fold.** Cặp đôi
là điểm mấu chốt: hai model được chấm trên **cùng 46.127 hồ sơ**, nên phần dao
động do tập validation tác động lên cả hai và **tự triệt tiêu khi lấy hiệu**. So
hai khoảng tin cậy rời nhau là phép so quá bảo thủ — nó kết luận "không phân biệt
được" cho cả những chênh lệch có thật.

**Các cặp đứng liền nhau** (1.000 lần lấy mẫu, tin cậy 95%):

| Bộ | Cặp | Chênh | Khoảng tin cậy | Kết luận |
|---|---|---:|---|---|
| full | xgboost − bagging | +0,0222 | [+0,0144 · +0,0304] | phân biệt được |
| full | **bagging − random_forest** | **+0,0018** | **[−0,0049 · +0,0085]** | ⚠️ **CHƯA phân biệt được** |
| full | random_forest − decision_tree | +0,0379 | [+0,0293 · +0,0455] | phân biệt được |
| rút gọn | xgboost − bagging | +0,0102 | [+0,0045 · +0,0151] | phân biệt được |
| rút gọn | bagging − random_forest | +0,0103 | [+0,0053 · +0,0156] | phân biệt được |
| rút gọn | random_forest − decision_tree | +0,0060 | [+0,0002 · +0,0123] | phân biệt được |

> **Câu trả lời cho task 9: khoảng chênh 0,0018 KHÔNG phân biệt được với nhiễu** —
> khoảng tin cậy chứa 0. Hạng 2 và hạng 3 ở bộ full là **đồng hạng trên thực tế**.
> Báo cáo phải viết đúng như vậy, không được xếp thứ tự hai model theo một con số
> nhỏ hơn sai số của chính phép đo.

**XGBoost dẫn đầu một cách chắc chắn ở CẢ HAI bộ** — mọi khoảng tin cậy so với ba
thuật toán còn lại đều nằm hoàn toàn trên 0, thắng 100% số lần lấy mẫu.

#### Full vs Rút gọn — phân tích tính khả thi triển khai (§7.2)

| Thuật toán | PR-AUC full | PR-AUC rút gọn | Chênh | Tương đối | Chắc chắn? |
|---|---:|---:|---:|---:|---|
| decision_tree | 0,1914 | 0,1445 | +0,0469 | +32,4% | ✅ |
| bagging | 0,2311 | 0,1608 | +0,0703 | +43,7% | ✅ |
| random_forest | 0,2293 | 0,1505 | +0,0788 | +52,4% | ✅ |
| **xgboost** | 0,2533 | 0,1711 | **+0,0822** | **+48,1%** | ✅ |

Đây là **cái giá của việc form không thu được `EXT_SOURCE_1/2/3`**, đo trên cả bốn
thuật toán và cả bốn đều chắc chắn (khoảng tin cậy không chứa 0). Bộ deploy được
mất khoảng **một phần ba tới một nửa** năng lực dự báo so với bộ dùng đủ dữ liệu
Home Credit.

> **Bootstrap KHÔNG tương đương K-Fold.** Nó đo dao động do **mẫu validation**;
> CV đo thêm dao động do **mẫu train**. Nó trả lời đúng câu hỏi của task 12 —
> "hai model chênh ngần này có phân biệt được không" — nhưng không thay thế được
> CV trong việc ước lượng độ ổn định của chính quá trình huấn luyện. Ghi rõ giới
> hạn này trong báo cáo.

**Ranh giới với task 14:** *"dẫn đầu"* ≠ *"được chọn"*. Task 12 chỉ xếp hạng theo
PR-AUC. Task 14 mới chốt, và phải cân nhắc thêm **hiệu chuẩn** (XGBoost gap
+0,2886 — thấp nhất nhưng vẫn lớn), **mức học thuộc** (XGBoost gap train−val
0,1314, gấp bốn lần Random Forest) và **khả năng triển khai**. `compare.py` có test
canh rằng metadata ghi `final_selection_done_here: false`.

### 7.0k Đánh giá 8 model (task 11) — trên validation, chưa xếp hạng

**Bốn nhóm chỉ số, cần cả bốn:**

| Nhóm | Đo gì | Vì sao không bỏ được |
|---|---|---|
| 1 · không phụ thuộc ngưỡng | PR-AUC · ROC-AUC | Khả năng XẾP HẠNG rủi ro, độc lập chỗ cắt |
| 2 · tại một ngưỡng | F1 / precision / recall lớp 1 · confusion | Thứ người dùng thật sự gặp |
| 3 · hiệu chuẩn | Brier · đường tin cậy | §8.1 ra quyết định theo NGƯỠNG xác suất |
| 4 · theo ngân sách rà soát | bắt được bao nhiêu khi soi k% | Gần vận hành nhất, **không cần chọn ngưỡng** |

**Bảng chính** (ngưỡng báo cáo 0,5 — chỉ là quy ước để có một cột so được):

| Thuật toán | Bộ | PR-AUC | ROC-AUC | F1 lớp 1 | Recall lớp 1 | Brier | Accuracy |
|---|---|---:|---:|---:|---:|---:|---:|
| decision_tree | reduced | 0,1445 | 0,6423 | 0,1994 | 0,5913 | 0,2236 | 0,6168 |
| decision_tree | full | 0,1914 | 0,7027 | 0,2299 | 0,6453 | 0,2077 | 0,6510 |
| bagging | reduced | 0,1608 | 0,6765 | 0,2255 | 0,5470 | 0,2032 | 0,6967 |
| bagging | full | 0,2311 | 0,7471 | 0,2687 | 0,6122 | 0,1810 | 0,7310 |
| random_forest | reduced | 0,1505 | 0,6680 | 0,2102 | 0,6034 | 0,2227 | 0,6339 |
| random_forest | full | 0,2293 | 0,7492 | 0,2636 | **0,6716** | 0,2017 | 0,6971 |
| xgboost | reduced | 0,1711 | 0,6916 | 0,2311 | 0,5811 | 0,2017 | 0,6879 |
| xgboost | full | **0,2533** | **0,7639** | **0,2858** | 0,6292 | **0,1717** | 0,7462 |

Bảng giữ nguyên thứ tự nạp, **không sắp xếp** — sắp theo PR-AUC là biến bảng
đánh giá thành bảng xếp hạng, và người đọc sẽ dừng ở dòng đầu. Xếp hạng là task 12.

**Phát hiện quan trọng nhất — mọi model đều NÓI QUÁ về rủi ro:**

| Model (bộ full) | gap trung bình | gap lớn nhất | Brier |
|---|---:|---:|---:|
| decision_tree | +0,3164 | +0,5772 | 0,2077 |
| bagging | +0,3111 | +0,5001 | 0,1810 |
| random_forest | +0,3555 | +0,4528 | 0,2017 |
| xgboost | **+0,2886** | +0,5104 | **0,1717** |

`gap = model nói − thực tế`, dương ở **cả 8 model**. Đây là hệ quả **bắt buộc**
của task 4: `class_weight='balanced'` và `scale_pos_weight` đẩy xác suất lớp
dương lên trên tỉ lệ nền thật. Một hồ sơ được chấm "60% rủi ro" thực tế chỉ vỡ
nợ ở mức thấp hơn nhiều.

> Đây chính là lý do §7.4 yêu cầu **hiệu chuẩn TRƯỚC khi đặt ngưỡng**, và là lý
> do ngưỡng `LOW_RISK/HIGH_RISK` không thể chốt ở task 11 hay 12. Chọn ngưỡng
> trên xác suất chưa hiệu chuẩn thì con số ngưỡng **không mang ý nghĩa xác suất
> nào** — "ngưỡng 0,35" sẽ không có nghĩa "35% khả năng vỡ nợ".

**Bắt được bao nhiêu ca vỡ nợ nếu chỉ soi k% hồ sơ rủi ro nhất** (bộ full):

| Thuật toán | soi 5% | 10% | 20% | 30% | 50% |
|---|---:|---:|---:|---:|---:|
| decision_tree | 0,165 | 0,277 | 0,449 | 0,575 | 0,753 |
| bagging | 0,194 | 0,311 | 0,501 | 0,631 | 0,813 |
| random_forest | 0,191 | 0,314 | 0,499 | 0,636 | 0,814 |
| xgboost | **0,211** | **0,337** | **0,527** | **0,660** | **0,828** |

Cách đọc gần vận hành nhất và **không cần chọn ngưỡng** — chỉ cần một ngân sách
rà soát. Soi 10% hồ sơ rủi ro nhất bắt được 33,7% số ca vỡ nợ, tức **gấp 3,4 lần**
soi ngẫu nhiên.

**Quét ngưỡng** (`threshold_sweep`) cho thấy F1 lớp dương của cả bốn thuật toán
đạt đỉnh quanh **0,55–0,60** chứ không phải 0,5. Đây là **nguyên liệu cho task
14**, không phải quyết định — hàm quét cố ý không trả về ngưỡng nào.

> **Ràng buộc phạm vi cài bằng test:** `evaluate.py` không có hàm nào tên chứa
> `best` / `select` / `rank` / `winner`. Trộn đánh giá với xếp hạng thì phần
> đánh giá bị rút gọn thành "cái nào PR-AUC cao nhất" — mà đó đúng là chỗ bỏ sót
> học thuộc, hiệu chuẩn lệch, và recall cao đổi bằng precision thấp.

Artifact được **nạp lại**, không train lại: train lại tốn ~12 phút và có nguy cơ
ra số khác nếu một tham số lệch, khi đó bảng đánh giá không còn mô tả đúng những
model đã báo cáo ở task 7–10.

### 7.0j XGBoost (task 10) — chấm trên validation

| Bộ | Feature | PR-AUC | lift | ROC-AUC | F1 lớp 1 | Recall lớp 1 | gap train−val |
|---|---:|---:|---:|---:|---:|---:|---:|
| rút gọn | 17 | **0,1711** | 2,12× | 0,6916 | 0,2311 | 0,5811 | +0,1020 |
| full | 82 | **0,2533** | **3,14×** | **0,7639** | **0,2858** | 0,6292 | +0,1314 |

**Bốn thuật toán đã train xong** (PR-AUC validation — bảng so sánh chính thức là
task 12, đây chỉ để đối chiếu):

| Bộ | Decision Tree | Bagging | Random Forest | XGBoost |
|---|---:|---:|---:|---:|
| rút gọn | 0,1445 | 0,1608 | 0,1505 | **0,1711** |
| full | 0,1914 | 0,2311 | 0,2293 | **0,2533** |

XGBoost dẫn đầu ở cả hai bộ, đúng kỳ vọng của §6.3 (Boosting giảm bias). Nhưng
nó cũng **học thuộc nhiều nhất**: gap 0,1314 so với 0,0321 của Random Forest —
gấp bốn lần. Task 11–12 phải đọc hai con số này cùng nhau.

**Vì sao KHÔNG ép `n_estimators = 50` cho "công bằng" với task 8, 9.** Boosting
và bagging dùng cây theo hai cách khác hẳn: bagging trung bình 50 cây **độc lập**,
mỗi cây đã là model đủ mạnh; boosting cộng dồn hàng trăm cây **nông**, mỗi cây chỉ
sửa phần dư của các cây trước. Bắt hai bên cùng số cây là so số lượng của hai thứ
không cùng đơn vị. Cái phải bằng nhau là **điều kiện thí nghiệm** — phép chia,
Pipeline, tỉ số phạt, seed, bộ chỉ số — chứ không phải siêu tham số nội bộ của
từng họ.

**Cấu hình đặt trước, không dò trên validation:** `learning_rate=0.1` +
`n_estimators=200` thay cho mặc định `0.3` + `100` — bước nhỏ hơn và nhiều bước
hơn là cách khắc phục sách vở cho boosting học thuộc. `subsample=0.8`,
`colsample_bytree=0.8` là điều tiết chuẩn của boosting ngẫu nhiên.
`eval_metric='aucpr'` khớp chỉ số chọn model; để mặc định `logloss` thì thứ
XGBoost tối ưu bên trong lệch khỏi thứ đem đi so ở task 12.

> **KHÔNG dùng early stopping.** Nó cần một tập để dừng, mà tập đó ở đây chỉ có
> thể là validation — chính tập dùng để báo cáo và để chọn model ở task 12. Dừng
> theo nó rồi lại chấm trên nó là chọn tham số trên tập đánh giá, và con số báo
> cáo thành lạc quan hơn thực tế. Có test canh.

**Cân bằng lớp qua `scale_pos_weight` = 11,387466** (tính từ riêng tập train), vì
XGBoost không có `class_weight`. Test canh lại rằng con số này trùng khít tỉ số
trọng số mà `class_weight='balanced'` sinh ra cho ba thuật toán kia — đó là điều
kiện để bốn model được so trên cùng một sân.

### 7.0i Random Forest (task 9) — chấm trên validation

| Bộ | Feature | PR-AUC | lift | ROC-AUC | F1 lớp 1 | Recall lớp 1 | gap train−val |
|---|---:|---:|---:|---:|---:|---:|---:|
| rút gọn | 17 | 0,1505 | 1,86× | 0,6680 | 0,2102 | 0,6034 | +0,0373 |
| full | 82 | 0,2293 | 2,84× | **0,7492** | 0,2636 | **0,6716** | **+0,0321** |

**Ba thuật toán trên cùng một phép chia** (PR-AUC validation):

| Bộ | Decision Tree | Bagging | Random Forest |
|---|---:|---:|---:|
| rút gọn | 0,1445 | **0,1608** | 0,1505 |
| full | 0,1914 | **0,2311** | 0,2293 |

> ⚠️ **Random Forest KHÔNG thắng Bagging ở đây** — thấp hơn 0,0103 (−6,4%) ở bộ
> rút gọn và 0,0018 (−0,8%) ở bộ full. Ghi lại đúng như đo được, không làm tròn
> theo kỳ vọng.
>
> Cách đọc: `max_features='sqrt'` ép mỗi lát cắt chỉ xét √p cột — 4/17 ở bộ rút
> gọn, 9/82 ở bộ full. Khi tín hiệu tập trung ở vài cột (`EXT_SOURCE_*`, `dti`),
> việc buộc phần lớn lát cắt bỏ qua chúng làm cây yếu đi nhiều hơn phần lợi từ
> việc các cây bớt giống nhau. Bộ rút gọn thiệt nặng hơn đúng như dự đoán: 17
> cột thì mỗi lát cắt chỉ còn 4.
>
> **Chênh lệch 0,0018 ở bộ full KHÔNG kết luận được.** Vì bỏ K-Fold (§7.0e), mỗi
> chỉ số là một điểm đo trên 46.127 hồ sơ, không có độ lệch giữa các fold để quy
> chiếu. Task 12 phải phát biểu đúng như vậy.

Bù lại, Random Forest **học thuộc ít nhất** trong ba thuật toán (gap 0,0321 ở bộ
full so với 0,0383 của Bagging và 0,0522 của cây đơn) và có **recall lớp dương
cao nhất** (0,6716). Ở ngưỡng 0,5, nó bắt được nhiều ca vỡ nợ hơn Bagging.

**Điều kiện để chênh lệch đọc được đúng:** RF giữ nguyên MỌI tham số của task 8
trừ `max_features` — cùng 50 cây, cùng `min_samples_leaf = 215`,
`max_depth=None`, cùng tỉ số phạt, cùng seed, cùng phép chia. Vì vậy khoảng cách
giữa hai dòng đo **đúng** đóng góp của việc lấy mẫu feature, không lẫn thứ gì khác.

Dùng `class_weight='balanced'` chứ **không** phải `'balanced_subsample'`: bản
subsample tính lại trọng số trên từng mẫu bootstrap nên tỉ số phạt dao động quanh
11,39 thay vì đúng bằng nó — khi đó RF không còn nhận cùng mức phạt với ba thuật
toán kia và task 12 so trên hai sân khác nhau. Có test canh riêng.

### 7.0h Bagging Classifier (task 8) — chấm trên validation

| Bộ | Feature | PR-AUC | lift | ROC-AUC | F1 lớp 1 | Recall lớp 1 | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| rút gọn | 17 | 0,1608 | 1,99× | 0,6765 | 0,2255 | 0,5470 | 0,2032 |
| full | 82 | **0,2311** | **2,86×** | 0,7471 | 0,2687 | 0,6122 | 0,1810 |

**Phần do giảm phương sai đem lại** — so trực tiếp với cây đơn ở task 7:

| Bộ | Decision Tree | Bagging | Chênh |
|---|---:|---:|---:|
| rút gọn | 0,1445 | 0,1608 | **+0,0163 (+11,3%)** |
| full | 0,1914 | 0,2311 | **+0,0397 (+20,7%)** |

Con số này đọc được **chính xác** là đóng góp của bootstrap aggregation, vì cây
con dùng ĐÚNG siêu tham số của task 7 (`min_samples_leaf = 215`, `max_depth=None`)
và mọi thứ khác giữ nguyên: cùng phép chia, cùng Pipeline, cùng tỉ số phạt, cùng
seed. Cho cây con một cấu hình khác thì chênh lệch lẫn cả phần "cây được điều
tiết khác đi".

Học thuộc cũng giảm đúng như lý thuyết: gap train−validation ở bộ full **0,0522 →
0,0383**.

**Ba quyết định giữ cho bảng so sánh đọc được:**

1. **`n_estimators = 50` dùng chung với Random Forest (task 9).** Bagging và RF
   khác nhau đúng ở chỗ RF lấy mẫu thêm feature tại mỗi lát cắt; cho hai bên số
   cây khác nhau thì chênh lệch lẫn cả phần "nhiều cây hơn". Chọn 50 vì phương
   sai của trung bình giảm theo 1/n — phần lợi lớn nhất nằm ở vài chục cây đầu,
   chi phí thì tăng tuyến tính. Đây là lập luận về dạng đường cong, không phải
   con số dò từ validation.
2. **`max_features = 1.0`** — ranh giới với Random Forest. Bagging cũng lấy mẫu
   feature thì hai thuật toán trùng nhau và §6.3 mất chỗ đối chiếu hai cơ chế.
3. **`class_weight` đặt trên CÂY CON**, không phải trên `BaggingClassifier` —
   lớp đó không có tham số này. Đặt nhầm lên ngoài sẽ `TypeError`; tệ hơn là nếu
   bị nuốt trong `**kwargs` thì model train mất cân bằng trong khi bảng cấu hình
   vẫn ghi là đã cân bằng, và task 12 so một model có trọng số với ba model
   không có. Có test canh riêng.

### 7.0g Decision Tree (task 7) — chấm trên validation

| Bộ | Feature | PR-AUC | lift | ROC-AUC | F1 lớp 1 | Recall lớp 1 | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| rút gọn | 17 | 0,1445 | 1,79× | 0,6423 | 0,1994 | 0,5913 | 0,6168 |
| full | 82 | **0,1914** | **2,37×** | 0,7027 | 0,2299 | 0,6453 | 0,6510 |

Cả hai vượt mốc lift ≥ 1,5 của task 6 → cây thật sự học được từ feature.

**Chênh lệch FULL − RÚT GỌN = 0,0469 PR-AUC (+32% tương đối).** Đây là con số
đầu tiên cho mục *"phân tích tính khả thi triển khai"* (§7.2) — cái giá của việc
form không thu được `EXT_SOURCE_1/2/3`. Ba thuật toán còn lại sẽ cho ba cặp số
nữa; kết luận phải dựa trên cả bốn, không phải riêng cây đơn.

**Học thuộc tới đâu** — đây là vai trò của cây đơn trong báo cáo (§6.3):

| Bộ | PR-AUC train | PR-AUC validation | gap |
|---|---:|---:|---:|
| rút gọn | 0,1896 | 0,1445 | +0,0451 |
| full | 0,2436 | 0,1914 | +0,0522 |

Gap ~0,05 là vừa phải — công của `min_samples_leaf`. **Siêu tham số suy từ CỠ
DỮ LIỆU, không phải dò được:** một lá phải chứa ≥ 0,1% dân số train, tức 215 hồ
sơ, tức ~17 ca vỡ nợ ở tỉ lệ 8,07%. Dưới mức đó thì xác suất của lá là ước lượng
từ vài quan sát — con số vô nghĩa mà `predict_proba` vẫn trả về đều đặn.
`max_depth=None` để `min_samples_leaf` một mình điều tiết; thêm một `max_depth`
cụ thể thì phải chọn con số, mà chọn thì phải thử — và thử là việc của bước tinh
chỉnh, không phải task 7.

> **Accuracy 0,62 THẤP HƠN baseline 0,85 — và đó là điều đúng.** Với
> `class_weight='balanced'`, model đánh đổi rất nhiều dương-tính-giả để bắt được
> **59–65% số ca vỡ nợ** (baseline chỉ 8,7%). Đây chính là lý do §7.3 cấm dùng
> accuracy để kết luận: nhìn mỗi accuracy thì model này "tệ hơn đoán bừa".

**Ba ràng buộc chống rò rỉ được cài bằng cấu trúc, không bằng lời dặn:**

1. `TrainingData` **không có** thuộc tính nào chứa tập test — không có gì để lỡ tay.
2. `fit_and_evaluate()` không nhận tham số test — muốn chạm phải sửa chữ ký hàm.
3. Pipeline được **fit lại** trong từng lần train, không nạp `.joblib` của task 3
   (vốn fit trên phép chia 85/15 khác). Pipeline *là một phần của quá trình huấn
   luyện* — nó học trung vị, phân vị, bảng hạng mục từ tập train.

Tỉ số phạt lấy từ `imbalance_params()` của task 4, **không hardcode** ở task
train: một chỗ duy nhất quyết định thì bốn thuật toán không thể lệch nhau.

> Artifact ở `runs/ml02_models/` là **trung gian**, không phải export —
> `artifact_kind: "intermediate"` ghi thẳng trong metadata. Nó tồn tại để task
> 11–14 khỏi train lại. Export là task 15.

### 7.0f Baseline (task 6) — chấm trên validation, test vẫn khoá

| | PR-AUC | lift | ROC-AUC | F1 lớp 1 | Recall lớp 1 | Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| **`baseline_stratified`** ← mốc chính thức | **0,0813** | 1,01 | 0,5036 | 0,0873 | 0,0873 | 0,8527 |
| `reference_most_frequent` *(tham chiếu)* | 0,0807 | 1,00 | 0,5000 | 0,0000 | **0,0000** | **0,9193** |

**Sàn của PR-AUC là TỈ LỆ DƯƠNG (0,0807), KHÔNG phải 0,5.** Đây là chỗ đọc nhầm
nhiều nhất ở bài toán mất cân bằng:

| | Đoán bừa cho ra |
|---|---|
| ROC-AUC | **0,50** — sàn 0,5, ai cũng biết |
| PR-AUC | **0,0807** — bằng tỉ lệ dương, KHÔNG phải 0,5 |

Nên một model đạt PR-AUC 0,20 không hề "tệ hơn ngẫu nhiên" — nó gấp **2,5 lần**
mức ngẫu nhiên. Thiếu hàng baseline trong bảng thì cả người viết lẫn hội đồng đều
dễ kết luận ngược. Cột `pr_auc_lift` tính sẵn tỉ số đó.

Đo được khớp lý thuyết: PR-AUC baseline **0,0813** so với tỉ lệ dương 0,0807
(lệch 0,0006), ROC-AUC **0,5036**.

**Mốc cho task 7–10:** model đạt **PR-AUC ≥ 0,1211** (lift ≥ 1,5) mới coi là học
được gì từ feature.

> **Hàng `most_frequent` không phải baseline** — nó là bằng chứng cho §7.3:
> accuracy **91,93%** mà **recall lớp dương = 0,0000**, tức không bắt được một ca
> vỡ nợ nào. Có nó trong bảng thì câu "accuracy 92%" không còn nghe như một kết
> quả tốt.

`binary_metrics()` báo cáo **riêng lớp dương** thay vì trung bình macro: với 8,07%
dương, macro-recall gộp cả lớp âm nên một model bỏ rơi hoàn toàn lớp dương vẫn có
macro-recall ~0,5 — nghe không tệ, trong khi nó vô dụng. Ngưỡng 0,5 dùng ở đây chỉ
để tính các chỉ số cần nhãn cứng; **ngưỡng nghiệp vụ `LOW_RISK/HIGH_RISK` chốt ở
task 14** sau khi hiệu chuẩn.

Baseline **không đọc feature nào** (`DummyClassifier` bỏ qua `X`), nên nó giống
hệt nhau ở bộ FULL và bộ RÚT GỌN — đo một lần, dùng chung cho cả hai bảng so sánh.

### 7.0e Chia train/validation/test (task 5)

**70 / 15 / 15, phân tầng theo nhãn, seed 42. KHÔNG dùng K-Fold Cross-Validation.**

| Tập | Số hồ sơ | Tỉ lệ | Dương | Tỉ lệ dương | Lệch |
|---|---:|---:|---:|---:|---:|
| train | 215.257 | 70,00% | 17.377 | 8,07% | 0,00σ |
| validation | 46.127 | 15,00% | 3.724 | 8,07% | 0,00σ |
| test | 46.127 | 15,00% | 3.724 | 8,07% | 0,00σ |

Vai trò: **train** fit model · **validation** căn cứ chọn model (so PR-AUC giữa
4 thuật toán, và là nơi tinh chỉnh siêu tham số nếu cần) · **test** khoá lại,
chỉ mở đúng một lần ở task 14.

> **Cái mất khi bỏ CV, ghi ra để đọc số cho đúng:** mỗi chỉ số là MỘT điểm đo
> trên 46.127 hồ sơ validation, không phải trung bình 5 lần kèm độ lệch. Không
> còn `pr_auc_std`, nên chênh vài phần nghìn giữa hai model **không** quy chiếu
> được về độ nhiễu. Task 12 phải phát biểu đúng như vậy — đừng nói "hơn hẳn" cho
> một khoảng chênh không đo được độ tin cậy.

**Cắt hai lần, phải quy đổi tỉ lệ lần hai.** Lần đầu tách `test`, lần sau tách
`validation` khỏi phần còn lại — mà phần còn lại chỉ là 85%, nên tỉ lệ lần hai
là `0,15 / 0,85 = 17,65%`. Lấy thẳng 0,15 thì validation chỉ được **12,75%** và
train phình lên 72,25%. Sai này khó thấy vì 12,75% vẫn "trông như" 15%; có test
canh riêng.

**Lưu danh sách `SK_ID_CURR` chứ không chỉ lưu seed.** Phép chia là tất định,
nhưng chỉ khi đầu vào không đổi — thêm một dòng hay chạy lại task 2 là cả ba tập
đổi hết mà không có gì báo. Khi đó model của task 7 và task 10 được train trên
hai tập khác nhau, còn bảng so sánh ở task 12 vẫn trông bình thường. Task 6 → 15
**bắt buộc** đi qua `load_split()`, không tự chia lại.

**Năm phép kiểm, tất cả đạt:** ba tập rời nhau (0 hồ sơ giao) · phủ kín 307.511
dòng · tỉ lệ đúng 70/15/15 · tỉ lệ dương lệch 0,00σ · train không còn dòng
`INVALID_ROW`.

**`scale_pos_weight` theo tập:** train **11,3875** · validation 11,3864 · test
11,3864. Task 7–10 dùng con số của **train**; số 11,3872 đo trên toàn bộ dataset
ở task 4 chỉ để báo cáo.

> ⚠️ **Phép chia 85/15 trong `build_features_ml02.py` (task 3) KHÔNG phải phép
> chia chính thức.** Nó chỉ để chứng minh Pipeline `fit` được trên train và
> `transform` được trên test. Hai artifact `.joblib` của task 3 vì vậy được fit
> trên một tập train khác — task 7–10 phải **fit lại Pipeline trên tập train của
> task 5**, và đó cũng là cách đúng: Pipeline là một phần của quá trình huấn
> luyện từng model, không phải một bước dùng chung fit sẵn.

### 7.0d Xử lý mất cân bằng lớp (task 4)

| | |
|---|---:|
| Số hồ sơ | 307.511 |
| Dương (khó khăn trả nợ) | 24.825 |
| Tỉ lệ dương | **8,0729%** |
| `scale_pos_weight` | **11,3872** |
| Accuracy nếu đoán toàn `0` | **91,9271%** |

**Phương án: học có trọng số, KHÔNG lấy mẫu lại.** Cả bốn thuật toán nhận cùng
một tỉ số phạt 11,3872 — sai một hồ sơ dương bị phạt gấp 11,39 lần sai một hồ sơ
âm. Không sinh thêm dòng, không bỏ bớt dòng.

| Thuật toán | Cơ chế |
|---|---|
| Decision Tree | `class_weight='balanced'` trên chính model |
| **Bagging** | `class_weight='balanced'` trên **estimator con** — `BaggingClassifier` KHÔNG có tham số này |
| Random Forest | `class_weight='balanced'` trên chính model |
| XGBoost | `scale_pos_weight` tính từ tập train |

> **Đã kiểm bằng số chứ không suy luận:** `class_weight='balanced'` cho trọng số
> lớp 0 = 0,543909 và lớp 1 = 6,193575, **tỉ số 11,387150** — trùng khít
> `scale_pos_weight` tới sáu chữ số. Không có đẳng thức này thì XGBoost và ba
> thuật toán sklearn đang học trên hai mức phạt khác nhau, và bảng so sánh ở
> task 12 so nhầm mà không có gì để lộ ra.

**Bốn lý do không dùng SMOTE / oversample / undersample**, xếp theo mức quan
trọng với chính đồ án này:

1. **Phá hiệu chuẩn xác suất — mà hiệu chuẩn là yêu cầu bắt buộc.** Cân về 50/50
   nghĩa là model học trên quần thể có tỉ lệ vỡ nợ 50%, nên xác suất trả về không
   còn ước lượng P(vỡ nợ) thật. Mà §7.4 yêu cầu `CalibratedClassifierCV` + Brier,
   §8.1 ra quyết định theo **ngưỡng** xác suất. Xác suất chưa hiệu chuẩn thì
   ngưỡng vô nghĩa.
2. **Là một cửa rò rỉ rất dễ mở nhầm.** Phải nằm trong fold, chỉ áp lên train.
   Chạy trước khi chia tập thì SMOTE nội suy giữa các dòng mà sau đó có dòng rơi
   vào validation — tập validation góp phần tạo ra chính dữ liệu huấn luyện.
3. **SMOTE vô nghĩa trên dữ liệu này.** Task 3 mã hoá ordinal, nên
   `ORGANIZATION_TYPE` mã 37 và 38 là hai tổ chức chẳng liên quan gì nhau. Nội
   suy Euclid sinh ra "mã 37,5" — một tổ chức không tồn tại. Bộ FULL có 16 cột
   categorical nên đây không phải chi tiết nhỏ.
4. **Bốn thuật toán phải so công bằng.** Lấy mẫu lại thì mỗi model nhìn thấy một
   tập khác nhau.

**Điểm rò rỉ đã canh:** tỉ lệ dương là một *thống kê của dữ liệu*, nên con số
11,3872 ở trên **chỉ dùng để báo cáo**. Số đem đi train phải tính lại trên riêng
tập train — `scale_pos_weight_from(y_train)` đặt tên tham số là `y_train` để chỗ
gọi phải viết ra chữ đó. Với `class_weight='balanced'` thì sklearn tự tính từ
đúng `y` truyền vào `fit()`, nên không có cửa rò rỉ.

### 7.0c Feature Engineering (task 3)

**Số lượng feature:**

| Bộ | Vào | Ra | Còn cột tiền tuyệt đối? |
|---|---:|---:|---|
| **RÚT GỌN** (deploy) | 127 | **17** | ✅ không còn cột nào |
| **FULL** | 127 | **83** | ⚠️ 10 cột — có chủ ý, xem dưới |

Bộ FULL từ 127 xuống 83 là do các bước CÓ HỌC trong Pipeline: bỏ cột thiếu quá
ngưỡng, khử gần-hằng-số và khử tương quan. Bộ RÚT GỌN không qua bước khử tương
quan (`correlation_threshold=None`) vì 17 cột đã chọn tay theo tiêu chí *"form
có thu được không"* — lọc thêm là bỏ mất feature mà hệ thống đang hỏi người dùng.

**12 feature sinh thêm**, ngoài 7 `SHARED_FEATURES` của §2.1b — trọng tâm là
nhóm lịch sử tín dụng mà form vừa thu được:

| Feature | Công thức | Ở bộ rút gọn |
|---|---|:--:|
| `credit_term_implied` | `AMT_CREDIT / AMT_ANNUITY` | ✅ |
| `bureau_loan_count` | `COUNT(bureau)` | ✅ |
| `bureau_active_loan_count` | `COUNT(CREDIT_ACTIVE = 'Active')` | ✅ |
| `bureau_overdue_loan_count` | `COUNT(CREDIT_DAY_OVERDUE > 0)` | ✅ |
| `bureau_has_overdue` | `MAX(CREDIT_DAY_OVERDUE) > 0` | ✅ |
| `bureau_overdue_loan_share` | `overdue_count / loan_count` | ✅ |
| `bureau_debt_income_ratio` | `SUM(AMT_CREDIT_SUM_DEBT) / AMT_INCOME_TOTAL` | ✅ |
| `bureau_overdue_income_ratio` | `SUM(AMT_CREDIT_SUM_OVERDUE) / AMT_INCOME_TOTAL` | ✅ |
| `bureau_credit_income_ratio` | `SUM(AMT_CREDIT_SUM) / AMT_INCOME_TOTAL` | ✅ |
| `bureau_no_record` | join rỗng | ✅ |
| `bureau_history_years` | `-MIN(DAYS_CREDIT) / 365.25` | ❌ form chưa hỏi |
| `credit_goods_markup` | `AMT_CREDIT / AMT_GOODS_PRICE` | ❌ không phải LTV |

**Nguyên tắc chia đôi nhóm bureau:** số ĐẾM giữ nguyên được vì không mang đơn vị
tiền tệ — "3 khoản vay" ở Việt Nam và ở Home Credit là cùng một thứ. Chỉ ba cột
TIỀN (`AMT_CREDIT_SUM`, `_DEBT`, `_OVERDUE`) mới phải quy về tỉ lệ trên thu nhập.

> **`GoodsPrice/Credit` chính là nghịch đảo của `credit_goods_markup`.** Với bốn
> thuật toán cây, hai đại lượng nghịch đảo nhau mang **thông tin y hệt**: mọi lát
> cắt trên cái này ánh xạ 1-1 sang một lát cắt trên cái kia. Đưa cả hai vào chỉ
> làm feature importance bị xé đôi giữa hai cột cùng nội dung. Giữ một cột, và
> giữ đúng cái đã có caveat: nó **không phải LTV** — Home Credit cộng phí và bảo
> hiểm vào `AMT_CREDIT` nên tỉ lệ luôn ≥ 1,0, tức nó đo *mức đội giá*.

**Encoding:** ordinal (`OrdinalEncoder`), **không one-hot** — theo §4.3d. Bốn
thuật toán đều là cây; one-hot làm số cột nhảy 128 → 249 và xé độ quan trọng của
`ORGANIZATION_TYPE` (57 hạng mục) thành 57 mảnh. Hai mã riêng: `MISSING_CODE = −2`
cho hạng mục thiếu, `UNKNOWN_CODE = −1` cho hạng mục lạ lúc inference — tách
riêng để cây phân biệt được, và để hồ sơ có giá trị lạ **không làm sập service**.

**NaN / inf:** `safe_divide()` cho mẫu số ≤ 0 hoặc thiếu ra `NaN` **chứ không phải
`inf`** — `inf` sẽ làm scaler nổ và `SimpleImputer` không bắt được vì nó chỉ xử lý
`NaN`. Đo trên dữ liệu thật sau Pipeline: **0 NaN, 0 inf** ở cả hai bộ.

**Một bước duy nhất có `fit()` không rỗng:** `income_per_capita_ratio` cần trung
vị thu nhập đầu người làm mẫu số, và trung vị đó **phải học trên riêng tập train**.
Nó được cài thành transformer trong Pipeline chứ không phải một phép tính chạy sẵn
— có test dựng hai quần thể lệch nhau 10 lần để khẳng định `fit` thật sự học.

**Bỏ dòng bất hợp lệ khỏi RIÊNG tập train**, đúng như task 2 đã hẹn: cờ
`INVALID_ROW` được dùng ở đây, tập test giữ nguyên 46.127 dòng. Trên dữ liệu này
là 0 dòng, nhưng đường đi đã đúng.

**Một Pipeline dùng chung train ↔ inference.** `build_feature_pipeline()` trả về
đúng một đối tượng `joblib.dump` được, gồm `BureauJoiner → HomeCreditFeatureBuilder
→ [7 bước tiền xử lý]`. Ba test canh điều này: thứ tự cột train ≡ test, biến đổi
**từng dòng ≡ theo lô** (inference chạy một dòng, train chạy cả lô), và nạp lại từ
đĩa cho kết quả trùng khít.

> **Hai lỗi đã sập lúc chạy thật, cả hai đều ở phần khai báo TÊN cột chứ không
> phải phần tính:**
>
> 1. `get_feature_names_out()` của bộ FULL tự liệt kê lại danh sách tên theo trí
>    nhớ và **sót 6 tỉ lệ dùng chung** → sklearn ném *"Length mismatch: Expected
>    axis has 156 elements, new values have 162"*. Nay cả `transform()` lẫn hàm
>    khai tên cùng đọc `engineered_names_for()`.
> 2. `MissingNormalizer.get_feature_names_out()` khai thừa 6 cờ `_MISSING`.
>    `add_missing_flags` **ghi đè** cột cờ đã có thay vì thêm cột mới, mà task 2 đã
>    sinh sẵn 6 cờ đó — nên số cột không tăng còn số tên thì tăng. Đây là lỗi có
>    sẵn của F01, chỉ lộ ra khi dữ liệu vào Pipeline đã mang sẵn cờ.
>
> Cả hai đều thuộc loại nguy hiểm: nếu độ dài tình cờ khớp thì sklearn **không báo
> gì**, model vẫn chạy, chỉ có tên cột lệch khỏi nội dung cột.

**Một chỗ gom lại:** phép gộp bureau trước đây có hai bản cài đặt riêng ở
`explore.py` và `features.py`. Nay `features.py` sở hữu, `explore.py` import lại —
hai bản cài đặt cho cùng một câu hỏi thì sớm muộn cho hai câu trả lời khác nhau về
cùng một khách hàng.

### 7.0b Làm sạch dữ liệu (task 2) — ranh giới quan trọng nhất

Task này **không** cho ra "dữ liệu sẵn sàng cho model", mà cho ra "dữ liệu đã hết
bẩn ở mức từng dòng". Phân biệt đó là toàn bộ nội dung của task, vì làm sai chỗ
này thì mọi chỉ số về sau đều lạc quan giả mà không có dấu hiệu gì:

| | Việc | Ghi ra đĩa? | Vì sao |
|---|---|---|---|
| **KHÔNG học** | sentinel → NaN + 6 cờ · chuỗi giả → NaN · chuẩn kiểu · bỏ dòng trùng · gắn cờ dòng bất hợp lệ | ✅ | Biến đổi theo từng dòng, `fit()` rỗng → chạy trước split cũng không rò rỉ |
| **CÓ học** | `HighMissingDropper` · `OutlierClipper` · `SimpleImputer` · encoder · khử NZV/tương quan · chọn feature có giám sát | ❌ | Cần thống kê cả tập. Chạy trước rồi lưu = trung vị/phân vị được tính trên cả phần sau này là test |

Bảy bước loại hai nằm trong `pipeline_steps_remaining` của metadata, để ai đọc
file dữ liệu cũng đọc luôn được là còn thiếu gì.

**Bỏ dòng trùng và gắn cờ dòng bất hợp lệ là HAI việc khác nhau** — đây là chỗ
dễ làm gộp nhất:

- **Dòng trùng `SK_ID_CURR` → BỎ, và phải bỏ trước khi chia tập.** Cùng một
  khách nằm ở cả train lẫn test là rò rỉ theo nghĩa đen. *(Kết quả: 0 dòng.)*
- **Dòng bất hợp lệ → chỉ GẮN CỜ `INVALID_ROW`.** Bỏ trước khi chia thì tập test
  cũng sạch theo, và chỉ số báo cáo sẽ cao hơn năng lực thật — lúc chạy thật hồ
  sơ bất hợp lệ vẫn cứ đến. Task 3 bỏ chúng khỏi **riêng** tập train.
  *(Kết quả: 0 dòng vi phạm 6 quy tắc.)*

**Kết quả trước/sau:**

| Bảng | Trước | Sau | Thay đổi |
|---|---|---|---|
| `application_train.csv` | 307.511 × 122 | 307.511 × **129** | +6 cờ `_MISSING`, +1 `INVALID_ROW`; **0 dòng bị bỏ** |
| `bureau.csv` | 1.716.428 × 17 | **1.716.411** × 17 | −17 dòng có thông tin tương lai |

126 cột được phép làm feature; ba cột `TARGET` · `SK_ID_CURR` · `INVALID_ROW`
loại vĩnh viễn qua `feature_columns()` — một nơi duy nhất trả lời "cột nào vào X".

**Kiểm toán rò rỉ — sáu phép kiểm, tất cả đều ĐO được:**

| Phép kiểm | Đo được | Đạt |
|---|---|:--:|
| Nhãn không trong feature set | `TARGET` không nằm trong 126 feature | ✅ |
| Khoá hồ sơ không trong feature set | `SK_ID_CURR` đã loại | ✅ |
| Khoá hồ sơ không mang tín hiệu thời gian | vỡ nợ theo thập phân vị ID **7,91%–8,29%** (chênh 0,0038, ngưỡng 3σ = 0,0047) | ✅ |
| Không cột nào tương quan bất thường với nhãn | \|r\| cao nhất **0,1789** (`EXT_SOURCE_3`), ngưỡng 0,50 | ✅ |
| Không cột nào trùng khít nhãn | 0 cột | ✅ |
| Không còn khách hàng trùng | 0 dòng | ✅ |

Ngưỡng của phép kiểm thứ ba **theo cỡ mẫu, không phải hằng số**: 10 nhóm chia từ
200 dòng thì mỗi nhóm 20 dòng, chênh 15 điểm phần trăm là nhiễu lấy mẫu thuần
tuý. So với 3 lần sai số chuẩn của một tỉ lệ trong nhóm mới là so đúng. Trên dữ
liệu thật, chênh lệch **nằm trong** nhiễu — tức ID không phải biến thời gian trá
hình, nhưng vẫn loại nó khỏi feature vì lý do loại là nguyên tắc chứ không phải
kết quả đo.

**Dữ liệu tương lai ở `bureau.csv`** — 17 dòng `DAYS_CREDIT_UPDATE > 0` (10→372
ngày SAU khi nộp đơn, 17 khách hàng). Đã bỏ.

> ⚠️ **`DAYS_CREDIT_ENDDATE` CỐ Ý không bị coi là dữ liệu tương lai**, dù **35,11%**
> giá trị của nó là số dương. Đó là *ngày kết thúc dự kiến* của khoản vay còn hiệu
> lực — con số đã biết ngay lúc ký hợp đồng, nên biết nó tại thời điểm nộp đơn là
> hoàn toàn hợp lệ. Hằng số `BUREAU_FUTURE_LOOKING_OK` tồn tại để lần sau có người
> thấy "35% dương" rồi tưởng là lỗi và đi "sửa" — sửa là mất một feature hợp lệ.

**Hai quyết định giữ nguyên dữ liệu, cả hai đều ngược với phản xạ thông thường:**

1. **Không bỏ 2.059 dòng trùng nội dung ở `bureau.csv`** (khác mỗi `SK_ID_BUREAU`,
   thuộc 1.865 khách hàng). `SK_ID_BUREAU` không trùng dòng nào nên đó là những
   khoản vay riêng biệt tình cờ cùng số tiền cùng ngày — chuyện bình thường với
   hai khoản tiêu dùng nhỏ mở cùng lúc. Bỏ đi sẽ làm `previous_loan_count`, đúng ô
   *"Số khoản vay trước đây"* của form, **đếm thiếu**. Đây là cái bẫy §4.3c đã ghi.
2. **Không kẹp ngoại lai ở task này.** `AMT_INCOME_TOTAL` cao nhất 117.000.000
   (**247× phân vị 99**) vẫn giữ nguyên — kẹp biên học phân vị từ tập train nên
   thuộc Pipeline.

**Kiểu dữ liệu:** 73 numeric · 40 binary · 16 categorical. **0 cột bị đọc sai
kiểu** (ép `to_numeric` trên cả 16 cột chuỗi ra 100% `NaN`). 40 cột "binary" là
biến nhị phân đội lốt số (`FLAG_DOCUMENT_*`…) — với cây thì vô hại, ghi ra để
bảng feature importance đọc dễ hơn.

> **Một bẫy đã sập lúc chạy thật:** bảng "thông tin tương lai" của bureau bị nhét
> vào cùng ô với bảng kiểm toán của application, mà hai bảng khác cột hoàn toàn.
> `passed_leakage_audit` đọc cột `passed` không tồn tại và ném `KeyError` — **sau
> khi mọi phép đo đã chạy xong 2,5 phút**, đúng lúc ghi file. Đã tách thành hai
> trường riêng, có test canh.

### 7.0 Kết quả khám phá (task 1) — xem [docs/ml02_eda.md](docs/ml02_eda.md)

Thước đo là **Information Value (WoE binning)** chứ không phải tương quan
Pearson: §4.3e đã đo được |r| tuyến tính cao nhất của mọi cột số chỉ **0,179**,
tin vào Pearson thì kết luận nhầm là không cột nào có ích. IV còn so được cột số
với cột hạng mục trên cùng một thang, và coi `NaN` là một khoảng riêng nên giữ
được tín hiệu của việc thiếu dữ liệu (§4.3b).

| Nhóm | Số cột | Cột đầu bảng |
|---|---:|---|
| mạnh (IV ≥ 0,30) | 2 | `EXT_SOURCE_3` 0,3293 · `EXT_SOURCE_2` 0,3063 |
| trung bình (≥ 0,10) | 2 | `EXT_SOURCE_1` 0,1508 · `DAYS_EMPLOYED` 0,1111 |
| yếu (≥ 0,02) | 51 | `DAYS_BIRTH` · `AMT_GOODS_PRICE` · `OCCUPATION_TYPE` |
| gần như vô dụng | 65 | |

**Không cột nào có IV > 0,5** — tức không có cột nào chứa sẵn câu trả lời. Bài
toán là bài toán thật, không có rò rỉ nhãn.

**Phủ sóng của form (căn cứ cho §7.2):** 23 trường form, **17 ánh xạ được** sang
một cột Home Credit, tổng IV **0,6449 / 3,2147 ≈ 20%**. Con số đó là *chỉ dấu,
không phải kết luận* — IV cộng dồn giữa các cột tương quan sẽ đếm trùng cùng một
lượng thông tin. Con số dùng để kết luận là **PR-AUC của hai model train thật**.

Ba cột form không lấy được đứng đầu bảng đều là `EXT_SOURCE_1/2/3`, đúng như
§7.2 dự đoán. Tiếp theo mới tới `NAME_INCOME_TYPE`, `ORGANIZATION_TYPE`,
`REGION_RATING_CLIENT*` — đều ở mức "yếu".

**`bureau.csv` đã vào phạm vi** (chốt 15/08/2026, trước đây xếp "để dành").
1.716.428 khoản vay của 263.491 khách hàng, gộp về một dòng mỗi khách thành đúng
bốn ô mục C của form:

| Cột tổng hợp | ← ô form | IV | lift xấu nhất |
|---|---|---:|---:|
| `BUREAU_LOAN_COUNT` | Số khoản vay trước đây | 0,0170 | 1,25 |
| `BUREAU_OVERDUE_LOAN_COUNT` | Số lần trả chậm | 0,0112 | 1,79 |
| `BUREAU_HAS_OVERDUE` | Có khoản vay quá hạn | 0,0091 | **1,97** |
| `BUREAU_TOTAL_OVERDUE` | Tổng nợ quá hạn | 0,0095 | **2,01** |
| `BUREAU_HISTORY_YEARS` | *(form chưa hỏi)* | **0,0761** | 1,54 |
| `BUREAU_ACTIVE_LOAN_COUNT` | *(form chưa hỏi)* | 0,0299 | 1,68 |

⚠️ **IV thấp ở đây KHÔNG có nghĩa là vô dụng, và đây là chỗ dễ kết luận sai
nhất.** IV cân theo tỉ trọng dân số; `BUREAU_HAS_OVERDUE` chỉ bật ở **1,10%** hồ
sơ nên IV nhỏ, nhưng trong nhóm đó tỉ lệ vỡ nợ là **15,90% so với 7,99%** —
**gần gấp đôi**. Loại các cột này theo một ngưỡng IV là loại đúng tín hiệu mạnh
nhất trên nhóm mà hệ thống cần cảnh báo nhất. Việc giữ/bỏ để `SupervisedFeature
Selector` trong Pipeline làm (§4.3e), không cắt tay.

**Hai điều phải ghi vào `docs/model_card.md`:**

1. **Lệch định nghĩa ở ô "số lần trả chậm".** `bureau.csv` chỉ ghi trạng thái
   quá hạn **hiện tại** của từng khoản (`CREDIT_DAY_OVERDUE`), không ghi lịch sử
   từng kỳ. `BUREAU_OVERDUE_LOAN_COUNT` vì vậy là số **khoản** đang quá hạn,
   không phải số **lần** trả chậm. Gần nhau nhưng không bằng nhau.
2. **14,31% hồ sơ không có bản ghi bureau nào**, và nhóm đó vỡ nợ **10,12% so
   với 7,73%** của nhóm có bản ghi. Điền **0 chứ không phải NaN**: không tìm thấy
   gì ở trung tâm tín dụng nghĩa là *chưa từng vay*, đúng bằng câu trả lời
   `previous_loan_count = 0` mà form cho phép chọn. Riêng `BUREAU_HISTORY_YEARS`
   giữ `NaN` — số năm có lịch sử tín dụng của người chưa từng vay không phải 0
   năm, nó **không tồn tại**.

**Hai bẫy đo lường đã sập trong lúc làm, cả hai đều cho ra con số trông bình
thường** — ghi lại vì chúng là loại lỗi không có test thì không bao giờ phát hiện:

| | Hiện tượng | Nguyên nhân | Cách sửa |
|---|---|---|---|
| 1 | `BUREAU_TOTAL_OVERDUE` IV = **0,0000** | 98,92% giá trị bằng 0 → mọi mốc phân vị của `qcut` trùng nhau → `duplicates="drop"` gộp còn **một** khoảng. Không phải "cột không có tín hiệu" mà là **phép đo không chạy** | Tách giá trị chiếm ≥ 5% dân số thành khoảng riêng trước, phần còn lại mới chia phân vị (`MASS_POINT_SHARE`) |
| 2 | Cùng cột đó rồi báo lift **0,99** — "an toàn hơn trung bình" | Sau khi tách khối 0 chỉ còn 1,08% dân số, chia tiếp thành 10 thì mỗi khoảng ~0,1%, đều dưới `min_share` nên bị loại khỏi phép tính lift | Co số khoảng theo phần dân số còn lại |

Bẫy 1 cũng suýt nuốt mất sentinel `DAYS_EMPLOYED = 365243` (18,01%): sau khi sửa,
IV của cột này tăng từ 0,1012 lên **0,1111**.

Phép tính lift còn phải loại nhóm `__RARE__` — nó là túi gom nhiều hạng mục
chẳng liên quan gì nhau nên lift của nó không mô tả nhóm người nào cả. Không lọc
thì `FLAG_DOCUMENT_2` (bật ở **13/307.511** hồ sơ) leo lên đầu bảng với lift
3,81, thuần tuý do ngẫu nhiên.

**Xác nhận lại hai đính chính của §2.1b bằng số đo trên toàn bộ dữ liệu:**

| Feature tỉ lệ | p1 | trung vị | p99 |
|---|---:|---:|---:|
| `dti` | 0,0395 | **0,1628** | 0,4835 |
| `credit_income_ratio` | 0,5954 | **3,2651** | 13,0273 |
| `credit_goods_markup` | **1,0000** | 1,1188 | 1,4752 |
| `employment_ratio` | 0,0075 | 0,1187 | 0,5906 |

`credit_goods_markup` có p1 = 1,0000 — **luôn ≥ 1**, xác nhận nó đo *mức đội
giá* chứ không phải tỉ lệ vay trên tài sản, nên không được gộp với `ltv` của
form. `dti` trung vị 0,1628 xác nhận `AMT_INCOME_TOTAL` và `AMT_ANNUITY` cùng kỳ.

### 7.1 Bài toán

`application_train.csv` — 307.511 hồ sơ, nhãn **`TARGET`** (1 = khó khăn trả nợ). Đây là **nhãn thật, thu thập độc lập** — trụ cột ML mạnh nhất của đồ án.

### 7.2 Train hai phiên bản để so sánh (task 3)

| Phiên bản | Feature | Mục đích |
|---|---|---|
| **Full** | Toàn bộ, kể cả `EXT_SOURCE_1/2/3` | Chứng minh năng lực kỹ thuật, AUC cao |
| **Rút gọn** | Chỉ feature ánh xạ được từ form (mục 2.1) | Model **thực sự deploy** |

`EXT_SOURCE_1/2/3` là điểm tín dụng từ nguồn ngoài — nhóm feature mạnh nhất của Home Credit nhưng **không thể thu được từ form**. Task 1 đã xác nhận bằng số: đó là **hai trong hai** cột duy nhất đạt mức "mạnh", và cả ba chiếm đầu bảng những cột form không lấy được. Bảng so sánh AUC hai phiên bản chính là mục *"phân tích tính khả thi triển khai"* trong báo cáo.

> **Bộ Rút gọn đã rộng hơn hẳn so với bản 11/08/2026.** Lúc đó nó chỉ có **6/7
> feature** vì form chưa hỏi nghề nghiệp, học vấn, hôn nhân hay lịch sử tín dụng.
> Sau khi màn **"Thông tin khoản vay"** lên (15/08/2026), form thu đủ 23 trường và
> bộ Rút gọn có thêm ba nhóm: nhân thân (`CODE_GENDER`, `NAME_FAMILY_STATUS`,
> `NAME_EDUCATION_TYPE`), nghề nghiệp (`OCCUPATION_TYPE`, `DAYS_EMPLOYED`), và
> lịch sử tín dụng tổng hợp từ `bureau.csv`. Bảng phủ sóng đầy đủ ở
> `src/training/runs/ml02_eda/form_coverage.csv`.

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
