<?php

namespace App\Http\Requests\Api;

use App\Enums\AssetTypeEnum;
use App\Enums\GoalTypeEnum;
use App\Http\Requests\BaseRequest;
use Illuminate\Validation\Rule;
use OpenApi\Attributes as OA;

/**
 * Validate dữ liệu màn "Nhập thông tin", dùng chung cho cả tạo mới và cập nhật.
 *
 * Hỗ trợ cả Guest lẫn User đăng nhập: nếu request không kèm access token thì
 * bắt buộc phải có guest_session_id để định danh phiên.
 */
#[OA\Schema(
    schema: 'HouseholdInput',
    title: 'Dữ liệu form "Nhập thông tin"',
    required: [
        'representative_name', 'household_size', 'children_count',
        'average_monthly_income', 'has_debt', 'has_savings', 'has_dependents',
    ],
    properties: [
        new OA\Property(property: 'representative_name', type: 'string', maxLength: 150, example: 'Nguyễn Văn A'),
        new OA\Property(property: 'birth_year', type: 'integer', nullable: true, example: 1991),
        new OA\Property(property: 'household_size', type: 'integer', minimum: 1, example: 5),
        new OA\Property(property: 'children_count', type: 'integer', minimum: 0, description: 'Phải nhỏ hơn household_size.', example: 2),
        new OA\Property(property: 'residence', type: 'string', nullable: true, example: 'TP. Hồ Chí Minh'),
        new OA\Property(property: 'average_monthly_income', type: 'number', example: 35000000),
        new OA\Property(property: 'average_monthly_expense', type: 'number', nullable: true, example: 17000000),
        new OA\Property(property: 'has_debt', type: 'boolean', example: true),
        new OA\Property(property: 'total_current_debt', type: 'number', nullable: true, description: 'Bắt buộc khi has_debt = true.', example: 500000000),
        new OA\Property(property: 'monthly_debt_payment', type: 'number', nullable: true, example: 5000000),
        new OA\Property(property: 'has_savings', type: 'boolean', example: true),
        new OA\Property(property: 'savings_amount', type: 'number', nullable: true, description: 'Bắt buộc khi has_savings = true.', example: 150000000),
        new OA\Property(property: 'has_dependents', type: 'boolean', example: true),
        new OA\Property(property: 'assets', type: 'array', items: new OA\Items(type: 'string', enum: ['house', 'car', 'land', 'other']), example: ['house', 'land']),
        new OA\Property(property: 'financial_needs', type: 'array', items: new OA\Items(type: 'string', enum: ['buy_house', 'buy_car', 'buy_land', 'loan', 'other']), example: ['buy_house']),
        new OA\Property(property: 'guest_session_id', type: 'string', maxLength: 64, nullable: true, description: 'Bắt buộc khi chưa đăng nhập.', example: '9f1c2b7a-3d4e-4a55-8f0b-2c6d1e7a9b30'),
    ],
    type: 'object'
)]
class StoreHouseholdRequest extends BaseRequest
{
    /**
     * Giá trị tiền tệ tối đa, nằm trong giới hạn numeric(18,2) của PostgreSQL.
     */
    private const MAX_MONEY = 999999999999999;

    /**
     * @return array<string, mixed>
     */
    public function rules(): array
    {
        return [
            'representative_name' => ['required', 'string', 'max:150'],
            'birth_year' => ['nullable', 'integer', 'min:1900', 'max:'.date('Y')],
            'household_size' => ['required', 'integer', 'min:1', 'max:50'],
            'children_count' => ['required', 'integer', 'min:0', 'lt:household_size'],
            'residence' => ['nullable', 'string', 'max:255'],

            'average_monthly_income' => ['required', 'numeric', 'min:0', 'max:'.self::MAX_MONEY],
            'average_monthly_expense' => ['nullable', 'numeric', 'min:0', 'max:'.self::MAX_MONEY],

            'has_debt' => ['required', 'boolean'],
            'total_current_debt' => [
                Rule::requiredIf(fn () => $this->boolean('has_debt')),
                'nullable', 'numeric', 'min:0', 'max:'.self::MAX_MONEY,
            ],
            'monthly_debt_payment' => ['nullable', 'numeric', 'min:0', 'max:'.self::MAX_MONEY],

            'has_savings' => ['required', 'boolean'],
            'savings_amount' => [
                Rule::requiredIf(fn () => $this->boolean('has_savings')),
                'nullable', 'numeric', 'min:0', 'max:'.self::MAX_MONEY,
            ],

            'has_dependents' => ['required', 'boolean'],

            'assets' => ['nullable', 'array', 'max:'.count(AssetTypeEnum::cases())],
            'assets.*' => ['distinct', Rule::enum(AssetTypeEnum::class)],

            'financial_needs' => ['nullable', 'array', 'max:'.count(GoalTypeEnum::cases())],
            'financial_needs.*' => ['distinct', Rule::enum(GoalTypeEnum::class)],

            'guest_session_id' => [
                Rule::requiredIf(fn () => $this->resolvedUser() === null),
                'nullable', 'string', 'max:64',
            ],
        ];
    }

    /**
     * @return array<string, string>
     */
    public function attributes(): array
    {
        return [
            'representative_name' => 'họ và tên người đại diện',
            'birth_year' => 'năm sinh',
            'household_size' => 'số người trong nhà',
            'children_count' => 'số con',
            'residence' => 'nơi ở',
            'average_monthly_income' => 'thu nhập trung bình tháng',
            'average_monthly_expense' => 'chi tiêu trung bình tháng',
            'has_debt' => 'tình trạng nợ',
            'total_current_debt' => 'tổng dư nợ hiện tại',
            'monthly_debt_payment' => 'số tiền trả nợ hàng tháng',
            'has_savings' => 'tình trạng tiết kiệm',
            'savings_amount' => 'số tiền tiết kiệm',
            'has_dependents' => 'tình trạng phụng dưỡng người già',
            'assets' => 'tài sản đang sở hữu',
            'financial_needs' => 'nhu cầu tài chính',
            'guest_session_id' => 'mã phiên khách',
        ];
    }

    /**
     * @return array<string, string>
     */
    public function messages(): array
    {
        return [
            'children_count.lt' => 'Số con phải nhỏ hơn số người trong nhà.',
            'guest_session_id.required' => 'Cần guest_session_id khi chưa đăng nhập.',
            'assets.*.distinct' => 'Tài sản đang sở hữu bị trùng.',
            'financial_needs.*.distinct' => 'Nhu cầu tài chính bị trùng.',
        ];
    }
}
