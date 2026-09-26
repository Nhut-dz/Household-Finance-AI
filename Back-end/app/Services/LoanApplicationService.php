<?php

namespace App\Services;

use App\Models\Household;
use App\Models\LoanApplication;
use Illuminate\Database\Eloquent\ModelNotFoundException;

/**
 * Nghiệp vụ màn "Thông tin khoản vay".
 *
 * Quyền sở hữu KHÔNG kiểm tra ở đây: mọi thao tác đều đi qua một `Household`
 * đã được `HouseholdService::findOwned()` xác thực. Kiểm hai lần ở hai nơi thì
 * sớm muộn hai nơi lệch nhau, và chỗ lỏng hơn mới là chỗ bị lợi dụng.
 *
 * Cùng tinh thần đó, form này KHÔNG hỏi số con: `tblhouseholds.children_count`
 * đã giữ con số ấy rồi. Hỏi lại là tạo nguồn sự thật thứ hai, và hai nguồn thì
 * sớm muộn lệch nhau.
 */
class LoanApplicationService
{
    /**
     * Lưu hoặc ghi đè phương án vay của hộ.
     *
     * Một hộ giữ đúng một phương án đang xét (unique household_id), nên nộp lại
     * là ghi đè chứ không tạo bản ghi mới. Dùng `updateOrCreate` để cùng một
     * request PUT xử lý được cả lần đầu lẫn các lần sửa sau — FE không phải
     * biết trước hộ đã có phương án hay chưa.
     *
     * Nạp bản ghi cũ TRƯỚC khi ghi đè, vì `mapToColumns()` cần biết ba trường
     * quyết định EMI có thay đổi hay không — xem `PAYMENT_DRIVERS`.
     *
     * @param  array<string, mixed>  $data  Dữ liệu đã qua StoreLoanApplicationRequest.
     */
    public function upsert(Household $household, array $data): LoanApplication
    {
        $existing = LoanApplication::query()
            ->where('household_id', $household->getKey())
            ->first();

        return LoanApplication::updateOrCreate(
            ['household_id' => $household->getKey()],
            $this->mapToColumns($data, $existing),
        );
    }

    /**
     * Phương án vay của hộ.
     *
     * @throws ModelNotFoundException khi hộ chưa từng khai khoản vay (404).
     */
    public function findFor(Household $household): LoanApplication
    {
        $application = LoanApplication::query()
            ->where('household_id', $household->getKey())
            ->first();

        if ($application === null) {
            throw (new ModelNotFoundException)->setModel(LoanApplication::class);
        }

        return $application;
    }

    public function delete(LoanApplication $application): void
    {
        $application->delete();
    }

    /**
     * Khoản trả hàng tháng (EMI) suy ra từ số tiền vay, kỳ hạn và lãi suất.
     *
     * Đây là NGUỒN SỰ THẬT của con số đó. Form không nhận nó từ người dùng nữa
     * (`StoreLoanApplicationRequest` không còn rule cho `monthly_payment`), nên
     * mọi bản ghi trong bảng đều phải đi qua hàm này.
     *
     * Chưa biết lãi suất (`null`)
     * ---------------------------
     * Chỉ chia đều tiền gốc: `EMI = P / n`. KHÔNG coi là vay 0% — đây là câu
     * trả lời tạm cho người chưa được ngân hàng nào báo lãi suất, và giao diện
     * nói rõ con số chưa gồm lãi.
     *
     * Có lãi suất
     * -----------
     * Công thức trả góp đều (annuity), gồm cả gốc lẫn lãi:
     *
     *     r = lãi suất năm / 100 / 12
     *     EMI = P · r · (1+r)^n / ((1+r)^n − 1)
     *
     * Làm tròn LÊN đồng, không làm tròn xuống. Hai lý do, cả hai đều thực tế:
     * đồng Việt Nam không tiêu được phần lẻ, và làm tròn xuống có thể đẩy EMI
     * xuống dưới `loan_amount / loan_term_months` — vi phạm đúng bất đẳng thức
     * mà `hfml.data.schema.LoanApplication` vẫn kiểm ở phía Python, khiến cả hồ
     * sơ bị từ chối vì lệch một đồng.
     *
     * @param  float|null  $annualRatePercent  Lãi suất %/năm; `null` = chưa biết.
     */
    public static function monthlyPayment(
        float $amount,
        int $termMonths,
        ?float $annualRatePercent,
    ): float {
        if ($amount <= 0 || $termMonths <= 0) {
            return 0.0;
        }

        if ($annualRatePercent === null || $annualRatePercent <= 0) {
            return ceil($amount / $termMonths);
        }

        $monthlyRate = $annualRatePercent / 100 / 12;
        $growth = (1 + $monthlyRate) ** $termMonths;

        return ceil($amount * $monthlyRate * $growth / ($growth - 1));
    }

    /**
     * Ba trường quyết định EMI. Chỉ khi một trong ba đổi thì mới tính lại.
     *
     * @var array<int, string>
     */
    private const PAYMENT_DRIVERS = ['loan_amount', 'loan_term_months', 'interest_rate'];

    /**
     * Ba trường quyết định EMI có giữ nguyên so với bản ghi đã lưu không?
     *
     * Vì sao cần câu hỏi này
     * ----------------------
     * Bảng có sẵn những bản ghi lập TRƯỚC khi có trường lãi suất, mang một
     * `monthly_payment` do người dùng tự gõ — thường cao hơn gốc chia đều vì họ
     * đã tự nhẩm phần lãi. Với `interest_rate = NULL`, công thức mới cho ra
     * đúng gốc chia đều, tức THẤP HƠN con số họ đã khai.
     *
     * Nếu cứ tính lại vô điều kiện thì chỉ cần mở form sửa một ô không liên
     * quan — chẳng hạn số con — rồi bấm Lưu là khoản trả hàng tháng bị hạ
     * xuống. Con số đó đi thẳng vào `AMT_ANNUITY` của ML02, nên điểm rủi ro của
     * hộ đổi theo mà người dùng không hề chạm vào thông tin khoản vay. Một thao
     * tác vô hại làm đổi kết luận rủi ro là điều không được phép xảy ra.
     *
     * Bản ghi mới (`$existing === null`) luôn tính — không có gì để bảo toàn.
     *
     * @param  array<string, mixed>  $data
     */
    private static function paymentDriversUnchanged(?LoanApplication $existing, array $data): bool
    {
        if ($existing === null) {
            return false;
        }

        foreach (self::PAYMENT_DRIVERS as $field) {
            if (! self::sameNumber($existing->{$field}, $data[$field] ?? null)) {
                return false;
            }
        }

        return true;
    }

    /**
     * Hai giá trị số có bằng nhau không, `null`-safe.
     *
     * So sau khi quy về chuỗi 2 chữ số thập phân chứ không so float trực tiếp:
     * cột `numeric` của Postgres về PHP là chuỗi ("8.50"), còn request gửi lên
     * là float (8.5). `===` trên hai thứ đó luôn sai, và hệ quả là KHÔNG bản
     * ghi nào được bảo toàn — đúng cái bug mà hàm này sinh ra để tránh.
     */
    private static function sameNumber(mixed $a, mixed $b): bool
    {
        if ($a === null || $b === null) {
            return $a === null && $b === null;
        }

        return number_format((float) $a, 2, '.', '')
            === number_format((float) $b, 2, '.', '');
    }

    /**
     * Ánh xạ field của form sang cột thật của bảng tblloan_applications.
     *
     * @param  array<string, mixed>  $data
     * @param  LoanApplication|null  $existing  Bản ghi đang có, `null` khi tạo mới.
     * @return array<string, mixed>
     */
    private function mapToColumns(array $data, ?LoanApplication $existing = null): array
    {
        $hasOverdue = (bool) $data['has_overdue_loan'];
        $interestRate = $data['interest_rate'] ?? null;

        // Giữ nguyên khoản trả hàng tháng đã lưu khi cả ba trường quyết định
        // EMI đều không đổi — xem paymentDriversUnchanged() để biết vì sao.
        $keepPayment = self::paymentDriversUnchanged($existing, $data);

        return [
            'borrower_age' => $data['borrower_age'],
            'gender' => $data['gender'],
            'marital_status' => $data['marital_status'],
            'education_level' => $data['education_level'],
            'occupation' => $data['occupation'],
            'employment_years' => $data['employment_years'],

            'loan_amount' => $data['loan_amount'],
            'loan_term_months' => $data['loan_term_months'],
            'interest_rate' => $interestRate,
            // Tính ở đây chứ không nhận từ client: con số này đi thẳng vào
            // `AMT_ANNUITY` của ML02, nên nó phải là hệ quả của ba trường trên
            // chứ không phải một giá trị người gọi API tự đặt.
            'monthly_payment' => $keepPayment
                ? (float) $existing->monthly_payment
                : self::monthlyPayment(
                    (float) $data['loan_amount'],
                    (int) $data['loan_term_months'],
                    $interestRate === null ? null : (float) $interestRate,
                ),
            'asset_price' => $data['asset_price'],
            'loan_purpose' => $data['loan_purpose'],

            'previous_loan_count' => $data['previous_loan_count'],
            'late_payment_count' => $data['late_payment_count'],
            'has_overdue_loan' => $hasOverdue,
            // Bỏ chọn "có khoản vay quá hạn" thì số nợ quá hạn luôn về 0. Giữ
            // lại số cũ sẽ vi phạm chk_loan_overdue_consistency ở DB, và tệ hơn
            // là để tầng ML đọc được một con số mà người dùng đã rút lại.
            'total_overdue_amount' => $hasOverdue ? ($data['total_overdue_amount'] ?? 0) : 0,
        ];
    }
}
