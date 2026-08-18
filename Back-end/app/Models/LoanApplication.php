<?php

namespace App\Models;

use App\Enums\EducationLevelEnum;
use App\Enums\GenderEnum;
use App\Enums\LoanPurposeEnum;
use App\Enums\MaritalStatusEnum;
use App\Enums\OccupationEnum;
use Illuminate\Database\Eloquent\Attributes\Fillable;
use Illuminate\Database\Eloquent\Attributes\Table;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;

/**
 * Dữ liệu form "Thông tin khoản vay" — đầu vào của ML02 (Home Credit Risk).
 *
 * Một hộ giữ đúng một phương án vay đang xét (unique household_id). Xem
 * migration `create_tblloan_applications_table` để biết vì sao tách bảng riêng
 * và từng cột ánh xạ sang cột nào của Home Credit.
 */
#[Table('tblloan_applications')]
#[Fillable([
    'household_id',
    // A. Thông tin người vay
    'borrower_age',
    'gender',
    'marital_status',
    'children_count',
    'education_level',
    'occupation',
    'employment_years',
    // B. Thông tin khoản vay
    'loan_amount',
    'loan_term_months',
    'monthly_payment',
    'asset_price',
    'loan_purpose',
    // C. Lịch sử tín dụng
    'previous_loan_count',
    'late_payment_count',
    'has_overdue_loan',
    'total_overdue_amount',
])]
class LoanApplication extends Model
{
    /**
     * Get the attributes that should be cast.
     *
     * @return array<string, string>
     */
    protected function casts(): array
    {
        return [
            'borrower_age' => 'integer',
            'gender' => GenderEnum::class,
            'marital_status' => MaritalStatusEnum::class,
            'children_count' => 'integer',
            'education_level' => EducationLevelEnum::class,
            'occupation' => OccupationEnum::class,
            'employment_years' => 'decimal:1',

            'loan_amount' => 'decimal:2',
            'loan_term_months' => 'integer',
            'monthly_payment' => 'decimal:2',
            'asset_price' => 'decimal:2',
            'loan_purpose' => LoanPurposeEnum::class,

            'previous_loan_count' => 'integer',
            'late_payment_count' => 'integer',
            'has_overdue_loan' => 'boolean',
            'total_overdue_amount' => 'decimal:2',
        ];
    }

    /**
     * Hộ gia đình nộp phương án vay này.
     */
    public function household(): BelongsTo
    {
        return $this->belongsTo(Household::class);
    }
}
