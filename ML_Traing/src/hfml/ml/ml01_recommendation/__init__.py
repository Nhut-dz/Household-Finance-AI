"""ML01 — Financial Recommendation Group Classification (F03 · M03 · Tuần 3).

Phân loại hồ sơ hộ gia đình vào 4 nhóm, xếp theo mức độ nghiêm trọng giảm
dần (chốt 11/08/2026):

    EMERGENCY      Tài chính nguy cấp                          🔴 Rất rủi ro
    DEBT_FOCUS     Cần tập trung xử lý nợ                      🟠 Rủi ro
    BUILD_BUFFER   Cần xây dựng quỹ dự phòng                   🟡 Trung bình
    GROWTH         Tài chính tương đối tốt, có thể tăng trưởng 🟢 Tốt

    labeler.py   Hàm sinh nhãn g(·) trên dân số hộ synthetic
    train.py     Train 4 thuật toán, đánh giá, export artifact

Hàm sinh nhãn — bản chốt (PLAN.md §6.1b). Thứ tự kiểm tra CHÍNH LÀ thang mức
độ ở trên: hộ thỏa nhiều điều kiện thì nhận nhãn nặng nhất, nhờ vậy g(·) đơn
trị mà không cần luật phá hòa riêng.

    savings_months = tiết kiệm ÷ chi tiêu/tháng
    dti            = trả nợ/tháng ÷ thu nhập/tháng
    savings_rate   = (thu nhập − chi tiêu) ÷ thu nhập

    savings_rate < 0  hoặc  savings_months < 1     → EMERGENCY
    dti ≥ 0.40                                     → DEBT_FOCUS
    savings_months < 3  hoặc  savings_rate < 0.10  → BUILD_BUFFER
    còn lại                                        → GROWTH

Ngưỡng lấy từ: quy tắc 28/36 (DTI back-end ≤ 36%, nới 40% cho mức "cần xử
lý") và khuyến nghị quỹ dự phòng 3–6 tháng chi tiêu.

⚠️ BA CHỈ SỐ TRÊN KHÔNG BAO GIỜ ĐƯỢC VÀO `X`. Chúng là biến mà g(·) đặt
ngưỡng lên; đưa vào feature set thì một cây sâu 3 tầng học thuộc nguyên g(·),
mọi thuật toán đạt ~100% và bảng so sánh mất hết ý nghĩa. `X` chỉ gồm biến
thô của form — danh sách đầy đủ ở PLAN.md §6.1c.

Đây cũng là điều làm ML01 thành bài toán thật: cây chẻ nhánh song song trục,
nên để xấp xỉ một ranh giới dạng tỉ lệ (`tiết kiệm ÷ chi tiêu < 1`) từ hai
cột riêng lẻ nó phải dựng nhiều lát cắt. Đó là chỗ Boosting được kỳ vọng
thắng cây đơn.

RỦI RO LỚN NHẤT CỦA F03 — circular labeling (PLAN.md §6.2). Nếu nhãn sinh
bằng rule ở F02 rồi lại đưa kết quả trung gian của chính rule đó vào feature
set, model chỉ học thuộc lại rule: accuracy ≈ 100% và bài toán vô nghĩa.

Ba quy tắc bắt buộc khi sinh dữ liệu:

1.  Tách sạch feature khỏi nhãn. Nhãn sinh từ `g(·)`; X chỉ gồm biến THÔ của
    form. Tuyệt đối không đưa output trung gian của `g` (điểm sức khỏe, cờ
    cảnh báo, nhóm DTI…) vào X.
2.  Sinh dân số có phân phối thực tế, không sinh đều theo nhãn: thu nhập
    log-normal tham chiếu GSO, nhân khẩu/số con theo phân bố thực, tương
    quan thu nhập–chi tiêu dương.
3.  Có vùng biên và nhiễu nhãn: **8%** hồ sơ kéo vào dải ±10% quanh một
    ngưỡng, **3%** nhãn đảo sang nhóm LIỀN KỀ về mức độ (không đảo ngẫu
    nhiên — `GROWTH` thành `EMERGENCY` là nhiễu vô nghĩa). Không có bước này
    thì ranh giới sạch tuyệt đối và mọi thuật toán đạt 100%.

Ba cổng kiểm chứng, không qua thì quay lại bước sinh dữ liệu:

    mỗi lớp ≥ 10%                          nếu không → chỉnh THAM SỐ DÂN SỐ
    accuracy model tốt nhất ≤ 0,98         nếu không → tăng vùng biên/nhiễu
    mọi model thắng rõ DummyClassifier     nếu không → xem lại feature set

⚠️ Khi một lớp dưới 10%, phản xạ tự nhiên là nới ngưỡng g(·) cho lớp đó to
ra. KHÔNG ĐƯỢC — ngưỡng đã chốt và có dẫn nguồn; sửa ngưỡng để vừa dữ liệu là
đảo ngược quan hệ nhân quả. Chỗ được chỉnh là tham số sinh dân số (phân phối
thu nhập, tỉ lệ hộ có nợ, mức tiết kiệm), không phải định nghĩa nhãn.

Định vị trong báo cáo — phải viết đúng như sau: ML01 là thí nghiệm có ground
truth đã biết, đo NĂNG LỰC THUẬT TOÁN trong việc khôi phục một ranh giới
quyết định đã biết trước từ biến thô khi có nhiễu và vùng biên. Chỉ số của
ML01 KHÔNG chứng minh chất lượng tư vấn tài chính — phần đó nằm ở tính đúng
đắn của rule F02.
"""
