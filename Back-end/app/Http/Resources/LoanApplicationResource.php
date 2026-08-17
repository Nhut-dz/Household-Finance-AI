<?php

namespace App\Http\Resources;

use App\Models\LoanApplication;
use Illuminate\Http\Request;
use Illuminate\Http\Resources\Json\JsonResource;
use OpenApi\Attributes as OA;

/**
 * @mixin LoanApplication
 */
#[OA\Schema(
    schema: 'LoanApplication',
    title: 'Thông tin khoản vay',
    properties: [
        new OA\Property(property: 'id', type: 'integer', example: 7),
        new OA\Property(property: 'household_id', type: 'integer', example: 12),

        new OA\Property(property: 'borrower_age', type: 'integer', example: 35),
        new OA\Property(property: 'gender', type: 'string', example: 'male'),
        new OA\Property(property: 'gender_label', type: 'string', example: 'Nam'),
        new OA\Property(property: 'marital_status', type: 'string', example: 'married'),
        new OA\Property(property: 'marital_status_label', type: 'string', example: 'Đã kết hôn'),
        new OA\Property(property: 'children_count', type: 'integer', example: 2),
        new OA\Property(property: 'education_level', type: 'string', example: 'higher'),
        new OA\Property(property: 'education_level_label', type: 'string', example: 'Đại học'),
        new OA\Property(property: 'occupation', type: 'string', example: 'office_staff'),
        new OA\Property(property: 'occupation_label', type: 'string', example: 'Nhân viên văn phòng'),
        new OA\Property(property: 'employment_years', type: 'number', example: 8.5),

        new OA\Property(property: 'loan_amount', type: 'number', example: 1400000000),
        new OA\Property(property: 'loan_term_months', type: 'integer', example: 240),
        new OA\Property(property: 'monthly_payment', type: 'number', example: 12000000),
        new OA\Property(property: 'asset_price', type: 'number', example: 2000000000),
        new OA\Property(property: 'loan_purpose', type: 'string', example: 'buy_house'),
        new OA\Property(property: 'loan_purpose_label', type: 'string', example: 'Mua nhà, căn hộ'),

        new OA\Property(property: 'previous_loan_count', type: 'integer', example: 3),
        new OA\Property(property: 'late_payment_count', type: 'integer', example: 1),
        new OA\Property(property: 'has_overdue_loan', type: 'boolean', example: false),
        new OA\Property(property: 'total_overdue_amount', type: 'number', example: 0),

        new OA\Property(property: 'loan_to_value', type: 'number', nullable: true, description: 'loan_amount / asset_price. Tính sẵn để FE và tầng LLM khỏi mỗi nơi tính một kiểu.', example: 0.7),
        new OA\Property(property: 'created_at', type: 'string', format: 'date-time'),
        new OA\Property(property: 'updated_at', type: 'string', format: 'date-time'),
    ],
    type: 'object'
)]
class LoanApplicationResource extends JsonResource
{
    /**
     * Trả về đúng bộ field mà form gửi lên, kèm nhãn tiếng Việt của từng giá
     * trị enum.
     *
     * Nhãn đi kèm chứ không để FE tự dựng bảng tra: form, thông báo lỗi và câu
     * chữ của tầng LLM phải gọi tên một hạng mục giống hệt nhau. Chép bảng nhãn
     * sang FE là tạo ra nguồn sự thật thứ hai.
     *
     * @return array<string, mixed>
     */
    public function toArray(Request $request): array
    {
        return [
            'id' => $this->id,
            'household_id' => $this->household_id,

            // A. Thông tin người vay
            'borrower_age' => $this->borrower_age,
            'gender' => $this->gender->value,
            'gender_label' => $this->gender->label(),
            'marital_status' => $this->marital_status->value,
            'marital_status_label' => $this->marital_status->label(),
            'children_count' => $this->children_count,
            'education_level' => $this->education_level->value,
            'education_level_label' => $this->education_level->label(),
            'occupation' => $this->occupation->value,
            'occupation_label' => $this->occupation->label(),
            'employment_years' => $this->asNumber($this->employment_years),

            // B. Thông tin khoản vay
            'loan_amount' => $this->asNumber($this->loan_amount),
            'loan_term_months' => $this->loan_term_months,
            'monthly_payment' => $this->asNumber($this->monthly_payment),
            'asset_price' => $this->asNumber($this->asset_price),
            'loan_purpose' => $this->loan_purpose->value,
            'loan_purpose_label' => $this->loan_purpose->label(),

            // C. Lịch sử tín dụng
            'previous_loan_count' => $this->previous_loan_count,
            'late_payment_count' => $this->late_payment_count,
            'has_overdue_loan' => $this->has_overdue_loan,
            'total_overdue_amount' => $this->asNumber($this->total_overdue_amount),

            'loan_to_value' => $this->loanToValue(),

            'created_at' => $this->created_at,
            'updated_at' => $this->updated_at,
        ];
    }

    /**
     * Tỉ lệ vay trên giá trị tài sản, làm tròn 4 chữ số.
     *
     * `asset_price > 0` đã được CHECK constraint bảo đảm, nhưng vẫn phòng chia
     * cho 0 ở đây: bản ghi có thể đến từ script SQL chạy trước khi constraint
     * được thêm.
     */
    private function loanToValue(): ?float
    {
        $price = (float) $this->asset_price;

        return $price > 0 ? round((float) $this->loan_amount / $price, 4) : null;
    }

    /**
     * Cột numeric của PostgreSQL trả về chuỗi "1400000000.00"; FE cần số thuần.
     */
    private function asNumber(mixed $value): int|float|null
    {
        return $value === null ? null : (float) $value;
    }
}
