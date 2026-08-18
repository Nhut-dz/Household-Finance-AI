# ML02 — Khám phá Home Credit Dataset (F04 task 1)

> File này sinh tự động bởi `scripts/explore_ml02.py`. Đừng sửa tay.

Bước này KHÁC với kiểm tra chất lượng dữ liệu ở F01 task 6. `docs/dataset.md` trả lời *"dữ liệu sạch tới đâu"*; file này trả lời *"cột nào dự báo được vỡ nợ, và cột nào trong số đó form của mình lấy được"* — câu hỏi quyết định kiến trúc hai phiên bản model của §7.2.

## 1. Nhãn `TARGET` — điểm xuất phát của mọi quyết định sau đó

| | |
|---|---:|
| Số hồ sơ | 307,511 |
| `TARGET = 1` (khó khăn trả nợ) | 24,825 |
| Tỉ lệ dương | 8.0729% |
| `scale_pos_weight` | 11.39 |
| Accuracy của model đoán toàn `0` | 91.9271% |

Dòng cuối là lý do **chọn model bằng PR-AUC chứ không phải accuracy**: một model không học gì đã đạt hơn 91%. Accuracy vẫn báo cáo cho đủ bộ chỉ số, nhưng không được dùng để kết luận (PLAN.md §7.3).

## 2. Sức mạnh dự báo của từng cột — Information Value

Thước đo là **IV (Information Value)** chứ không phải tương quan Pearson. Lý do đã đo được: trên chính dataset này, |r| tuyến tính cao nhất của mọi cột số chỉ **0,179** — tin vào Pearson thì kết luận nhầm là không cột nào có ích. IV còn so được cột số với cột hạng mục trên cùng một thang, và coi `NaN` là một khoảng riêng nên không vứt mất tín hiệu của việc thiếu dữ liệu.

Thang diễn giải (quy ước ngành chấm điểm tín dụng):

| IV | Mức | Số cột |
|---|---|---:|
| ≥ 0.50 | đáng ngờ (nghi rò rỉ nhãn) | 0 |
| ≥ 0.30 | mạnh | 2 |
| ≥ 0.10 | trung bình | 2 |
| ≥ 0.02 | yếu | 52 |
| ≥ 0.00 | gần như vô dụng | 64 |

### 20 cột mạnh nhất

| Cột | Kiểu | Thiếu | IV | Mức | lift xấu nhất |
|---|---|---|---|---|---|
| EXT_SOURCE_3 | numeric | 0.1983 | 0.3293 | mạnh | 2.4778 |
| EXT_SOURCE_2 | numeric | 0.0021 | 0.3063 | mạnh | 2.2731 |
| EXT_SOURCE_1 | numeric | 0.5638 | 0.1508 | trung bình | 2.1756 |
| DAYS_EMPLOYED | numeric | 0.0000 | 0.1111 | trung bình | 1.4204 |
| DAYS_BIRTH | numeric | 0.0000 | 0.0842 | yếu | 1.4345 |
| AMT_GOODS_PRICE | numeric | 0.0009 | 0.0841 | yếu | 1.7023 |
| OCCUPATION_TYPE | categorical | 0.3135 | 0.0759 | yếu | 1.4030 |
| NAME_INCOME_TYPE | categorical | 0.0000 | 0.0579 | yếu | 1.1877 |
| ORGANIZATION_TYPE | categorical | 0.0000 | 0.0578 | yếu | 1.4468 |
| REGION_RATING_CLIENT_W_CITY | numeric | 0.0000 | 0.0512 | yếu | 1.4124 |
| NAME_EDUCATION_TYPE | categorical | 0.0000 | 0.0507 | yếu | 1.3536 |
| REGION_RATING_CLIENT | numeric | 0.0000 | 0.0483 | yếu | 1.3753 |
| DAYS_LAST_PHONE_CHANGE | numeric | 0.0000 | 0.0468 | yếu | 1.2302 |
| AMT_CREDIT | numeric | 0.0000 | 0.0451 | yếu | 1.3034 |
| CODE_GENDER | categorical | 0.0000 | 0.0386 | yếu | 1.2563 |
| DAYS_ID_PUBLISH | numeric | 0.0000 | 0.0384 | yếu | 1.2595 |
| FLOORSMAX_MODE | numeric | 0.4976 | 0.0383 | yếu | 1.1385 |
| REGION_POPULATION_RELATIVE | numeric | 0.0000 | 0.0377 | yếu | 1.2365 |
| FLOORSMAX_AVG | numeric | 0.4976 | 0.0377 | yếu | 1.1385 |
| FLOORSMAX_MEDI | numeric | 0.4976 | 0.0372 | yếu | 1.1385 |

Bảng đầy đủ 120 cột: `src/training/runs/ml02_eda/iv_ranking.csv`.

### ⚠️ IV thấp **không** đồng nghĩa vô dụng

IV cân theo tỉ trọng dân số, nên một cột chỉ bật ở nhóm nhỏ sẽ có IV thấp dù nhóm đó rủi ro rất cao. Hai con số đo hai thứ khác nhau:

| | Trả lời câu hỏi |
|---|---|
| **IV** | cột này giúp phân biệt được bao nhiêu hồ sơ trong TOÀN BỘ danh mục |
| **lift xấu nhất** | khi cột này bật, hồ sơ đó nguy hiểm tới đâu |

Các cột bị xếp mức thấp theo IV nhưng có nhóm rủi ro gấp rưỡi trở lên:

| Cột | Khoảng xấu nhất | IV | Mức theo IV | lift |
|---|---|---|---|---|
| NAME_HOUSING_TYPE | Rented apartment | 0.0157 | gần như vô dụng | 1.5252 |
| DEF_60_CNT_SOCIAL_CIRCLE | 2.0 | 0.0137 | gần như vô dụng | 1.5044 |

Loại các cột này chỉ vì IV thấp là loại đúng tín hiệu mạnh nhất trên nhóm mà hệ thống cần cảnh báo nhất. Quyết định giữ/bỏ để cho bước feature selection có giám sát trong Pipeline làm (F01 task 13), không cắt tay theo một ngưỡng IV.

## 3. Form lấy được bao nhiêu — căn cứ của bảng Full vs Rút gọn (§7.2)

Sau khi màn **Thông tin khoản vay** lên (15/08/2026), form thu được **23 trường**, trong đó **17** ánh xạ được sang một cột Home Credit và **6** thì không.

| Trường form | Màn | Cột Home Credit | IV | Mức | Ghi chú |
|---|---|---|---|---|---|
| household_size | household | CNT_FAM_MEMBERS | 0.0053 | gần như vô dụng |  |
| children_count | household | CNT_CHILDREN | 0.0057 | gần như vô dụng |  |
| birth_year | household | DAYS_BIRTH | 0.0842 | yếu | đổi sang tuổi |
| average_monthly_income | household | AMT_INCOME_TOTAL | 0.0117 | gần như vô dụng | CHỈ dùng làm mẫu số của feature tỉ lệ, không vào X (§2.1) |
| average_monthly_expense | household | — | — | — | Home Credit không có cột chi tiêu — chỉ phục vụ tầng rule |
| savings_amount | household | — | — | — | Home Credit không có cột tiết kiệm |
| total_current_debt | household | — | — | — | gần nhất là bureau.AMT_CREDIT_SUM_DEBT, không phải cùng một thứ |
| monthly_debt_payment | household | — | — | — | AMT_ANNUITY là kỳ trả của khoản ĐANG XIN, không phải nợ đang có |
| borrower_age | loan | DAYS_BIRTH | 0.0842 | yếu |  |
| gender | loan | CODE_GENDER | 0.0386 | yếu |  |
| marital_status | loan | NAME_FAMILY_STATUS | 0.0217 | yếu |  |
| education_level | loan | NAME_EDUCATION_TYPE | 0.0507 | yếu |  |
| occupation | loan | OCCUPATION_TYPE | 0.0759 | yếu |  |
| employment_years | loan | DAYS_EMPLOYED | 0.1111 | trung bình |  |
| loan_amount | loan | AMT_CREDIT | 0.0451 | yếu |  |
| monthly_payment | loan | AMT_ANNUITY | 0.0266 | yếu |  |
| asset_price | loan | AMT_GOODS_PRICE | 0.0841 | yếu |  |
| loan_term_months | loan | — | — | — | suy ra từ AMT_CREDIT/AMT_ANNUITY |
| loan_purpose | loan | — | — | — | chỉ có ở previous_application, 95,8% là XAP/XNA |
| previous_loan_count | loan | BUREAU_LOAN_COUNT | — | — |  |
| late_payment_count | loan | BUREAU_OVERDUE_LOAN_COUNT | — | — |  |
| has_overdue_loan | loan | BUREAU_HAS_OVERDUE | — | — |  |
| total_overdue_amount | loan | BUREAU_TOTAL_OVERDUE | — | — |  |

Tổng IV của các cột form lấy được: **0.6448** trên tổng **3.2171** của toàn bộ dataset (**20.0%**).

> Con số đó là **chỉ dấu, không phải kết luận**. IV cộng dồn giữa các cột > tương quan với nhau sẽ đếm trùng cùng một lượng thông tin, nên > "giữ được 40% IV" không có nghĩa là "giữ được 40% năng lực dự báo". > Con số dùng để kết luận là **PR-AUC của hai model train thật**, và đó > chính là việc của task tiếp theo.

### Những cột mạnh nhất mà form KHÔNG lấy được

Đây là cái giá của bộ Rút gọn, phải nêu đích danh trong báo cáo — nói "bộ rút gọn kém hơn" mà không chỉ ra kém vì mất gì thì hội đồng hỏi ngay.

| Cột | Kiểu | Thiếu | IV | Mức |
|---|---|---|---|---|
| EXT_SOURCE_3 | numeric | 0.1983 | 0.3293 | mạnh |
| EXT_SOURCE_2 | numeric | 0.0021 | 0.3063 | mạnh |
| EXT_SOURCE_1 | numeric | 0.5638 | 0.1508 | trung bình |
| NAME_INCOME_TYPE | categorical | 0.0000 | 0.0579 | yếu |
| ORGANIZATION_TYPE | categorical | 0.0000 | 0.0578 | yếu |
| REGION_RATING_CLIENT_W_CITY | numeric | 0.0000 | 0.0512 | yếu |
| REGION_RATING_CLIENT | numeric | 0.0000 | 0.0483 | yếu |
| DAYS_LAST_PHONE_CHANGE | numeric | 0.0000 | 0.0468 | yếu |
| DAYS_ID_PUBLISH | numeric | 0.0000 | 0.0384 | yếu |
| FLOORSMAX_MODE | numeric | 0.4976 | 0.0383 | yếu |

## 4. `bureau.csv` — nguồn của mục C trên form

`bureau.csv` có **1,716,428 khoản vay** của **263,491 khách hàng**. Gộp về một dòng mỗi khách thì ra đúng bốn ô mục C mà form đang hỏi.

| | |
|---|---:|
| Hồ sơ KHÔNG có bản ghi bureau | 44,020 (14.31%) |
| Vỡ nợ khi KHÔNG có bản ghi | 10.1249% |
| Vỡ nợ khi CÓ bản ghi | 7.7301% |

Nhóm không có bản ghi được điền **0 chứ không phải NaN**: không tìm thấy gì ở trung tâm tín dụng nghĩa là *chưa từng vay*, đúng bằng câu trả lời `previous_loan_count = 0` mà form cho phép chọn. Impute trung vị vào đây là gán cho người chưa từng vay một lịch sử tín dụng trung bình mà họ không có. Riêng `BUREAU_HISTORY_YEARS` giữ `NaN` — số năm có lịch sử tín dụng của người chưa từng vay không phải 0 năm, nó **không tồn tại**.

### IV của phần tổng hợp bureau

| Cột tổng hợp | Thiếu | IV | Mức | lift xấu nhất |
|---|---|---|---|---|
| BUREAU_HISTORY_YEARS | 0.1431 | 0.0761 | yếu | 1.5371 |
| BUREAU_ACTIVE_LOAN_COUNT | 0.0000 | 0.0299 | yếu | 1.6838 |
| BUREAU_LOAN_COUNT | 0.0000 | 0.0170 | gần như vô dụng | 1.2542 |
| BUREAU_NO_RECORD | 0.0000 | 0.0117 | gần như vô dụng | 1.2542 |
| BUREAU_OVERDUE_LOAN_COUNT | 0.0000 | 0.0112 | gần như vô dụng | 1.7935 |
| BUREAU_TOTAL_OVERDUE | 0.0000 | 0.0095 | gần như vô dụng | 2.0063 |
| BUREAU_HAS_OVERDUE | 0.0000 | 0.0091 | gần như vô dụng | 1.9691 |

Đọc bảng này phải kèm mục 2.1: `BUREAU_HAS_OVERDUE` có IV rất thấp, nhưng đó là vì chỉ **1.10%** hồ sơ đang có khoản quá hạn — còn trong nhóm đó thì tỉ lệ vỡ nợ **15.90% so với 7.99%**, tức gấp **1.97 lần** trung bình. Đây đúng là loại tín hiệu mà mục C của form sinh ra để bắt.

⚠️ **Một chỗ lệch định nghĩa phải ghi vào `model_card.md`:** form hỏi *"số lần trả chậm"*, nhưng `bureau.csv` chỉ ghi trạng thái quá hạn **hiện tại** của từng khoản (`CREDIT_DAY_OVERDUE`), không ghi lịch sử từng kỳ. `BUREAU_OVERDUE_LOAN_COUNT` vì vậy là số **khoản** đang quá hạn, không phải số **lần** trả chậm. Hai đại lượng gần nhau nhưng không bằng nhau — đừng để nó thành một phép đồng nhất ngầm.

## 5. Khoảng cách miền VNĐ ↔ Home Credit (§2.1)

Home Credit không dùng VNĐ: `AMT_INCOME_TOTAL` trung vị ≈ 147.150, người dùng Việt Nam nhập 50.000.000 — lệch ~340 lần. Model gặp giá trị ngoài phân phối huấn luyện sẽ **trả về số vô nghĩa mà không báo lỗi**. Cách xử lý là bỏ hết giá trị tiền tuyệt đối, chỉ giữ feature **tỉ lệ** — và tỉ lệ thì bất biến với đơn vị tiền tệ.

Phân phối các tỉ lệ đó trên chính `application_train.csv`:

| Feature | Công thức | Thiếu | p1 | p25 | trung vị | p75 | p99 |
|---|---|---|---|---|---|---|---|
| dti | AMT_ANNUITY / AMT_INCOME_TOTAL | 0.0000 | 0.0395 | 0.1148 | 0.1628 | 0.2291 | 0.4835 |
| credit_income_ratio | AMT_CREDIT / AMT_INCOME_TOTAL | 0.0000 | 0.5954 | 2.0187 | 3.2651 | 5.1599 | 13.0273 |
| credit_goods_markup | AMT_CREDIT / AMT_GOODS_PRICE | 0.0009 | 1.0000 | 1.0000 | 1.1188 | 1.1980 | 1.4752 |
| children_ratio | CNT_CHILDREN / CNT_FAM_MEMBERS | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.3333 | 0.6667 |
| employment_ratio | DAYS_EMPLOYED / DAYS_BIRTH | 0.1801 | 0.0075 | 0.0561 | 0.1187 | 0.2192 | 0.5906 |

Hai điều bảng này xác nhận, cả hai đều đã ghi ở PLAN.md §2.1b:

1. **`credit_goods_markup` luôn ≥ 1,0** — Home Credit cộng phí và bảo hiểm vào `AMT_CREDIT`, nên tỉ lệ này đo **mức đội giá**, KHÔNG phải tỉ lệ vay trên tài sản. Nó không cùng đại lượng với `loan_amount / asset_price` của form (vay 70%, tự có 30%), nên hai thứ phải tách riêng.
2. **`dti` trung vị ≈ 0,16** xác nhận `AMT_INCOME_TOTAL` và `AMT_ANNUITY` cùng kỳ. Nếu thu nhập theo NĂM mà kỳ trả theo THÁNG thì DTI sẽ là 0,16 × 12 ≈ 196% — bất khả.

## 6. Kết luận rút ra cho các task sau

| # | Kết luận | Ảnh hưởng tới |
|---|---|---|
| 1 | Chọn model bằng PR-AUC, `scale_pos_weight` ≈ 11,39 | task 4, 11 |
| 2 | `EXT_SOURCE_*` mạnh nhất nhưng form không lấy được | task 3 — bộ Full vs Rút gọn |
| 3 | Phần tổng hợp bureau bổ sung được nhóm tín hiệu lịch sử tín dụng | task 2 — feature engineering |
| 4 | Thiếu dữ liệu tự nó là tín hiệu → giữ cờ `_MISSING` | task 2 |
| 5 | Chỉ dùng feature tỉ lệ, bỏ mọi giá trị tiền tuyệt đối | task 1, 2 |
