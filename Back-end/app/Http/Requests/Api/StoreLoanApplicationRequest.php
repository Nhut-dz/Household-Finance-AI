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
 *
 * `monthly_payment` CỐ Ý không có trong danh sách rule: từ khi có trường lãi
 * suất, nó là giá trị SUY RA (`LoanApplicationService::monthlyPayment()`) chứ
 * không phải người dùng nhập. Client gửi kèm cũng bị `validated()` bỏ qua —
 * chính là hành vi mong muốn, vì một khoản trả hàng tháng do client tự đặt sẽ
 * đi thẳng vào `AMT_ANNUITY` của ML02 mà không ai kiểm chứng được.
 */
#[OA\Schema(
    schema: 'LoanApplicationInput',
    title: 'Dữ liệu form "Thông tin khoản vay"',
    required: [
        'borrower_age', 'gender', 'marital_status',
        'education_level', 'occupation', 'employment_years',
        'loan_amount', 'loan_term_months', 'asset_price',
        'loan_purpose', 'has_overdue_loan',
    ],
    properties: [
        new OA\Property(property: 'borrower_age', type: 'integer', minimum: 18, maximum: 100, example: 35),
        new OA\Property(property: 'gender', type: 'string', enum: ['male', 'female'], example: 'male'),
        new OA\Property(property: 'marital_status', type: 'string', enum: ['single', 'married', 'civil_marriage', 'separated', 'widow'], example: 'married'),
        new OA\Property(property: 'education_level', type: 'string', enum: ['lower_secondary', 'secondary', 'incomplete_higher', 'higher', 'academic_degree'], example: 'higher'),
        new OA\Property(property: 'occupation', type: 'string', enum: ['office_staff', 'manager', 'accountant', 'it_staff', 'teacher', 'medical_staff', 'sales_staff', 'driver', 'security_staff', 'service_staff', 'laborer', 'farmer', 'self_employed', 'retired', 'unemployed', 'other'], example: 'office_staff'),
        new OA\Property(property: 'employment_years', type: 'number', minimum: 0, maximum: 60, description: 'Không được lớn hơn borrower_age - 15.', example: 8.5),

        new OA\Property(property: 'loan_amount', type: 'number', minimum: 1, example: 1400000000),
        new OA\Property(property: 'loan_term_months', type: 'integer', enum: [12, 24, 36, 60, 120, 180, 240, 300], example: 240),
        new OA\Property(property: 'interest_rate', type: 'number', minimum: 6, maximum: 10, nullable: true, description: 'Lãi suất %/năm, TÙY CHỌN, tối đa 2 chữ số thập phân. Bỏ trống = chưa biết lãi suất, khi đó khoản trả hàng tháng chỉ gồm tiền gốc.', example: 8.5),
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
     * Khoảng lãi suất %/năm được chấp nhận. Giữ khớp CHECK
     * `chk_loan_interest_rate` ở DB và `INTEREST_RATE_MIN`/`INTEREST_RATE_MAX`
     * của `Front-end/src/data/loan.ts`.
     */
    public const MIN_INTEREST_RATE = 6;

    public const MAX_INTEREST_RATE = 10;

    /**
     * Số chữ số thập phân tối đa của lãi suất. Cột DB là numeric(5,2), nên
     * nhận 9.999 ở đây là để Postgres làm tròn thành 10.00 — một giá trị người
     * dùng không hề nhập.
     */
    public const INTEREST_RATE_DECIMALS = 2;

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
            // KHÔNG có `children_count`: số con thuộc hồ sơ hộ gia đình
            // (`StoreHouseholdRequest`), form này không hỏi lại.
            'education_level' => ['required', Rule::enum(EducationLevelEnum::class)],
            'occupation' => ['required', Rule::enum(OccupationEnum::class)],
            'employment_years' => ['required', 'numeric', 'min:0', 'max:60'],

            // -- B. Thông tin khoản vay ----------------------------------
            'loan_amount' => ['required', 'numeric', 'min:1', 'max:'.self::MAX_MONEY],
            'loan_term_months' => ['required', 'integer', Rule::in(self::TERM_CHOICES)],
            // Tùy chọn. `decimal:0,2` chặn 9.999 — xem INTEREST_RATE_DECIMALS.
            'interest_rate' => [
                'nullable', 'numeric',
                'min:'.self::MIN_INTEREST_RATE,
                'max:'.self::MAX_INTEREST_RATE,
                'decimal:0,'.self::INTEREST_RATE_DECIMALS,
            ],
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
     * Ba luật LIÊN TRƯỜNG — không luật nào diễn đạt được bằng rule đơn lẻ.
     *
     * Cả ba đều là mâu thuẫn nội tại chứ không phải "giá trị đáng ngờ". Hồ sơ
     * rủi ro cao (vay 95% giá trị tài sản, nợ quá hạn lớn) vẫn phải đi lọt —
     * đó chính là nhóm ML02 sinh ra để đánh giá, chặn nó là chặn đúng đối tượng
     * cần đánh giá nhất.
     *
     * Luật thứ tư trước đây — "khoản trả hàng tháng ≥ gốc chia đều" — đã bỏ.
     * Nó chỉ có nghĩa khi con số đó do người dùng nhập; nay hệ thống tự tính
     * nên bất đẳng thức luôn đúng theo cách dựng, và giữ lại chỉ là một nhánh
     * chết không test nào chạm tới được.
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
            'education_level' => 'trình độ học vấn',
            'occupation' => 'nghề nghiệp',
            'employment_years' => 'thời gian làm việc',
            'loan_amount' => 'số tiền vay',
            'loan_term_months' => 'thời hạn vay',
            'interest_rate' => 'lãi suất',
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
            // Một câu chung cho cả min lẫn max: người dùng cần biết KHOẢNG hợp
            // lệ, chứ "lãi suất phải ít nhất 6" không nói được cận trên là 10.
            'interest_rate.min' => 'Lãi suất phải nằm trong khoảng '
                .self::MIN_INTEREST_RATE.'% đến '.self::MAX_INTEREST_RATE.'%/năm.',
            'interest_rate.max' => 'Lãi suất phải nằm trong khoảng '
                .self::MIN_INTEREST_RATE.'% đến '.self::MAX_INTEREST_RATE.'%/năm.',
            'interest_rate.decimal' => 'Lãi suất chỉ được có tối đa '
                .self::INTEREST_RATE_DECIMALS.' chữ số sau dấu thập phân.',
            'total_overdue_amount.required' => 'Vui lòng nhập tổng nợ quá hạn.',
            'guest_session_id.required' => 'Cần guest_session_id khi chưa đăng nhập.',
        ];
    }
}
