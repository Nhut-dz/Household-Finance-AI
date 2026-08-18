<?php

namespace App\Http\Requests\Api;

use App\Enums\EducationLevelEnum;
use App\Enums\GenderEnum;
use App\Enums\LoanPurposeEnum;
use App\Enums\MaritalStatusEnum;
use App\Enums\OccupationEnum;
use App\Http\Requests\BaseRequest;
use Illuminate\Validation\Rule;
use Illuminate\Validation\Validator;
use OpenApi\Attributes as OA;

/**
 * Validate dữ liệu màn "Thông tin khoản vay".
 *
 * Đây là RANH GIỚI của dữ liệu ML02: sai kiểu, số vô lý, mâu thuẫn nội tại đều
 * phải chặn tại đây. Để chúng chảy xuống model thì hệ thống vẫn trả về một xác
 * suất vỡ nợ trông rất bình thường — chỉ có điều nó vô nghĩa, và không ai biết.
 */
#[OA\Schema(
    schema: 'LoanApplicationInput',
    title: 'Dữ liệu form "Thông tin khoản vay"',
    required: [
        'borrower_age', 'gender', 'marital_status', 'children_count',
        'education_level', 'occupation', 'employment_years',
        'loan_amount', 'loan_term_months', 'monthly_payment', 'asset_price',
        'loan_purpose', 'has_overdue_loan',
    ],
    properties: [
        new OA\Property(property: 'borrower_age', type: 'integer', minimum: 18, maximum: 100, example: 35),
        new OA\Property(property: 'gender', type: 'string', enum: ['male', 'female'], example: 'male'),
        new OA\Property(property: 'marital_status', type: 'string', enum: ['single', 'married', 'civil_marriage', 'separated', 'widow'], example: 'married'),
        new OA\Property(property: 'children_count', type: 'integer', minimum: 0, maximum: 20, example: 2),
        new OA\Property(property: 'education_level', type: 'string', enum: ['lower_secondary', 'secondary', 'incomplete_higher', 'higher', 'academic_degree'], example: 'higher'),
        new OA\Property(property: 'occupation', type: 'string', enum: ['office_staff', 'manager', 'accountant', 'it_staff', 'teacher', 'medical_staff', 'sales_staff', 'driver', 'security_staff', 'service_staff', 'laborer', 'farmer', 'self_employed', 'retired', 'unemployed', 'other'], example: 'office_staff'),
        new OA\Property(property: 'employment_years', type: 'number', minimum: 0, maximum: 60, description: 'Không được lớn hơn borrower_age - 15.', example: 8.5),

        new OA\Property(property: 'loan_amount', type: 'number', minimum: 1, example: 1400000000),
        new OA\Property(property: 'loan_term_months', type: 'integer', enum: [12, 24, 36, 60, 120, 180, 240, 300], example: 240),
        new OA\Property(property: 'monthly_payment', type: 'number', minimum: 1, description: 'Phải ≥ loan_amount / loan_term_months (không âm lãi).', example: 12000000),
        new OA\Property(property: 'asset_price', type: 'number', minimum: 1, example: 2000000000),
        new OA\Property(property: 'loan_purpose', type: 'string', enum: ['buy_house', 'buy_land', 'buy_car', 'home_repair', 'business', 'education', 'medical', 'consumer', 'debt_consolidation', 'other'], example: 'buy_house'),

        new OA\Property(property: 'previous_loan_count', type: 'integer', minimum: 0, maximum: 100, example: 3),
        new OA\Property(property: 'late_payment_count', type: 'integer', minimum: 0, description: 'Không được lớn hơn previous_loan_count.', example: 1),
        new OA\Property(property: 'has_overdue_loan', type: 'boolean', example: false),
        new OA\Property(property: 'total_overdue_amount', type: 'number', nullable: true, description: 'Bắt buộc khi has_overdue_loan = true; bỏ qua khi false.', example: 0),

        new OA\Property(property: 'guest_session_id', type: 'string', maxLength: 64, nullable: true, description: 'Bắt buộc khi chưa đăng nhập.'),
    ],
    type: 'object'
)]
class StoreLoanApplicationRequest extends BaseRequest
{
    /**
     * Giá trị tiền tệ tối đa, nằm trong giới hạn numeric(18,2) của PostgreSQL.
     */
    private const MAX_MONEY = 999999999999999;

    /**
     * Kỳ hạn cho chọn (tháng). 12→60 cho vay tiêu dùng/mua xe, 120→300 cho vay
     * mua nhà/đất. Giữ khớp `LOAN_TERM_CHOICES` của
     * `ML_Training/src/hfml/data/schema.py`.
     *
     * @var array<int, int>
     */
    public const TERM_CHOICES = [12, 24, 36, 60, 120, 180, 240, 300];

    /**
     * Tuổi tối thiểu được tính là đã đi làm. Dùng cho ràng buộc
     * employment_years ≤ borrower_age - 15.
     */
    private const MIN_WORKING_AGE = 15;

    /**
     * @return array<string, mixed>
     */
    public function rules(): array
    {
        return [
            // -- A. Thông tin người vay ----------------------------------
            'borrower_age' => ['required', 'integer', 'min:18', 'max:100'],
            'gender' => ['required', Rule::enum(GenderEnum::class)],
            'marital_status' => ['required', Rule::enum(MaritalStatusEnum::class)],
            'children_count' => ['required', 'integer', 'min:0', 'max:20'],
            'education_level' => ['required', Rule::enum(EducationLevelEnum::class)],
            'occupation' => ['required', Rule::enum(OccupationEnum::class)],
            'employment_years' => ['required', 'numeric', 'min:0', 'max:60'],

            // -- B. Thông tin khoản vay ----------------------------------
            'loan_amount' => ['required', 'numeric', 'min:1', 'max:'.self::MAX_MONEY],
            'loan_term_months' => ['required', 'integer', Rule::in(self::TERM_CHOICES)],
            'monthly_payment' => ['required', 'numeric', 'min:1', 'max:'.self::MAX_MONEY],
            'asset_price' => ['required', 'numeric', 'min:1', 'max:'.self::MAX_MONEY],
            'loan_purpose' => ['required', Rule::enum(LoanPurposeEnum::class)],

            // -- C. Lịch sử tín dụng -------------------------------------
            'previous_loan_count' => ['required', 'integer', 'min:0', 'max:100'],
            'late_payment_count' => ['required', 'integer', 'min:0', 'max:1000'],
            'has_overdue_loan' => ['required', 'boolean'],
            'total_overdue_amount' => [
                Rule::requiredIf(fn () => $this->boolean('has_overdue_loan')),
                'nullable', 'numeric', 'min:0', 'max:'.self::MAX_MONEY,
            ],

            'guest_session_id' => [
                Rule::requiredIf(fn () => $this->resolvedUser() === null),
                'nullable', 'string', 'max:64',
            ],
        ];
    }

    /**
     * Bốn luật LIÊN TRƯỜNG — không luật nào diễn đạt được bằng rule đơn lẻ.
     *
     * Cả bốn đều là mâu thuẫn nội tại chứ không phải "giá trị đáng ngờ". Hồ sơ
     * rủi ro cao (vay 95% giá trị tài sản, nợ quá hạn lớn) vẫn phải đi lọt —
     * đó chính là nhóm ML02 sinh ra để đánh giá, chặn nó là chặn đúng đối tượng
     * cần đánh giá nhất.
     */
    public function withValidator(Validator $validator): void
    {
        $validator->after(function (Validator $validator) {
            $age = $this->integer('borrower_age');
            $years = (float) $this->input('employment_years');
            $maxYears = $age - self::MIN_WORKING_AGE;

            if ($age > 0 && $years > $maxYears) {
                $validator->errors()->add(
                    'employment_years',
                    "Thời gian làm việc không thể vượt quá {$maxYears} năm với người {$age} tuổi."
                );
            }

            // Trả góp tối thiểu là gốc chia đều; nhỏ hơn thế thì đến hạn vẫn
            // chưa trả xong gốc, chưa nói tới lãi.
            $amount = (float) $this->input('loan_amount');
            $term = $this->integer('loan_term_months');
            $payment = (float) $this->input('monthly_payment');

            if ($amount > 0 && $term > 0 && $payment > 0 && $payment < $amount / $term) {
                $minimum = number_format(ceil($amount / $term), 0, ',', '.');
                $validator->errors()->add(
                    'monthly_payment',
                    "Khoản trả hàng tháng phải từ {$minimum} VNĐ trở lên mới trả hết gốc trong {$term} tháng."
                );
            }

            $previous = $this->integer('previous_loan_count');
            $late = $this->integer('late_payment_count');

            if ($late > 0 && $previous === 0) {
                $validator->errors()->add(
                    'late_payment_count',
                    'Chưa có khoản vay nào trước đây thì không thể có lần trả chậm.'
                );
            }

            if ($this->boolean('has_overdue_loan') && $previous === 0) {
                $validator->errors()->add(
                    'has_overdue_loan',
                    'Chưa có khoản vay nào trước đây thì không thể có khoản vay quá hạn.'
                );
            }
        });
    }

    /**
     * @return array<string, string>
     */
    public function attributes(): array
    {
        return [
            'borrower_age' => 'tuổi',
            'gender' => 'giới tính',
            'marital_status' => 'tình trạng hôn nhân',
            'children_count' => 'số con',
            'education_level' => 'trình độ học vấn',
            'occupation' => 'nghề nghiệp',
            'employment_years' => 'thời gian làm việc',
            'loan_amount' => 'số tiền vay',
            'loan_term_months' => 'thời hạn vay',
            'monthly_payment' => 'khoản trả hàng tháng',
            'asset_price' => 'giá trị tài sản',
            'loan_purpose' => 'mục đích vay',
            'previous_loan_count' => 'số khoản vay trước đây',
            'late_payment_count' => 'số lần trả chậm',
            'has_overdue_loan' => 'tình trạng khoản vay quá hạn',
            'total_overdue_amount' => 'tổng nợ quá hạn',
            'guest_session_id' => 'mã phiên khách',
        ];
    }

    /**
     * @return array<string, string>
     */
    public function messages(): array
    {
        return [
            'loan_term_months.in' => 'Thời hạn vay phải là một trong các mốc: '
                .implode(', ', self::TERM_CHOICES).' tháng.',
            'total_overdue_amount.required' => 'Vui lòng nhập tổng nợ quá hạn.',
            'guest_session_id.required' => 'Cần guest_session_id khi chưa đăng nhập.',
        ];
    }
}
