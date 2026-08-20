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
     * @param  array<string, mixed>  $data  Dữ liệu đã qua StoreLoanApplicationRequest.
     */
    public function upsert(Household $household, array $data): LoanApplication
    {
        return LoanApplication::updateOrCreate(
            ['household_id' => $household->getKey()],
            $this->mapToColumns($data),
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
     * Ánh xạ field của form sang cột thật của bảng tblloan_applications.
     *
     * @param  array<string, mixed>  $data
     * @return array<string, mixed>
     */
    private function mapToColumns(array $data): array
    {
        $hasOverdue = (bool) $data['has_overdue_loan'];

        return [
            'borrower_age' => $data['borrower_age'],
            'gender' => $data['gender'],
            'marital_status' => $data['marital_status'],
            'children_count' => $data['children_count'],
            'education_level' => $data['education_level'],
            'occupation' => $data['occupation'],
            'employment_years' => $data['employment_years'],

            'loan_amount' => $data['loan_amount'],
            'loan_term_months' => $data['loan_term_months'],
            'monthly_payment' => $data['monthly_payment'],
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
