<?php

namespace App\Http\Resources;

use App\Models\Asset;
use App\Models\FinancialGoal;
use App\Models\Household;
use Illuminate\Http\Request;
use Illuminate\Http\Resources\Json\JsonResource;
use OpenApi\Attributes as OA;

/**
 * @mixin Household
 */
#[OA\Schema(
    schema: 'Household',
    title: 'Hồ sơ tài chính hộ gia đình',
    properties: [
        new OA\Property(property: 'id', type: 'integer', example: 12),
        new OA\Property(property: 'user_id', type: 'integer', nullable: true, example: null),
        new OA\Property(property: 'guest_session_id', type: 'string', nullable: true, example: '9f1c2b7a-3d4e-4a55-8f0b-2c6d1e7a9b30'),
        new OA\Property(property: 'representative_name', type: 'string', example: 'Nguyễn Văn A'),
        new OA\Property(property: 'birth_year', type: 'integer', nullable: true, example: 1991),
        new OA\Property(property: 'household_size', type: 'integer', example: 5),
        new OA\Property(property: 'children_count', type: 'integer', example: 2),
        new OA\Property(property: 'residence', type: 'string', nullable: true, example: 'TP. Hồ Chí Minh'),
        new OA\Property(property: 'average_monthly_income', type: 'number', example: 35000000),
        new OA\Property(property: 'average_monthly_expense', type: 'number', nullable: true, example: 17000000),
        new OA\Property(property: 'has_debt', type: 'boolean', example: true),
        new OA\Property(property: 'total_current_debt', type: 'number', example: 500000000),
        new OA\Property(property: 'monthly_debt_payment', type: 'number', example: 5000000),
        new OA\Property(property: 'has_savings', type: 'boolean', example: true),
        new OA\Property(property: 'savings_amount', type: 'number', example: 150000000),
        new OA\Property(property: 'has_dependents', type: 'boolean', example: true),
        new OA\Property(property: 'assets', type: 'array', items: new OA\Items(type: 'string'), example: ['house', 'land']),
        new OA\Property(property: 'financial_needs', type: 'array', items: new OA\Items(type: 'string'), example: ['buy_house']),
        new OA\Property(property: 'created_at', type: 'string', format: 'date-time'),
        new OA\Property(property: 'updated_at', type: 'string', format: 'date-time'),
    ],
    type: 'object'
)]
class HouseholdResource extends JsonResource
{
    /**
     * Trả về đúng bộ field mà form "Nhập thông tin" gửi lên, kèm field hệ thống.
     * Tiền tệ trả về dạng số để FE tự định dạng theo vi-VN.
     *
     * @return array<string, mixed>
     */
    public function toArray(Request $request): array
    {
        return [
            'id' => $this->id,
            'user_id' => $this->user_id,
            'guest_session_id' => $this->session_token,

            'representative_name' => $this->representative_name,
            'birth_year' => $this->birth_year,
            'household_size' => $this->household_size,
            'children_count' => $this->children_count,
            'residence' => $this->location,

            'average_monthly_income' => $this->asNumber($this->monthly_income),
            'average_monthly_expense' => $this->asNumber($this->monthly_living_cost),

            'has_debt' => $this->has_debt,
            'total_current_debt' => $this->asNumber($this->total_debt),
            'monthly_debt_payment' => $this->asNumber($this->monthly_debt_payment),

            'has_savings' => $this->has_savings,
            'savings_amount' => $this->asNumber($this->current_savings),

            'has_dependents' => $this->supports_elderly,

            'assets' => $this->whenLoaded(
                'assets',
                fn () => $this->assets->map(fn (Asset $asset) => $asset->asset_type->value)->all()
            ),
            'financial_needs' => $this->whenLoaded(
                'financialGoals',
                fn () => $this->financialGoals->map(fn (FinancialGoal $goal) => $goal->goal_type->value)->all()
            ),

            'created_at' => $this->created_at,
            'updated_at' => $this->updated_at,
        ];
    }

    /**
     * Cột numeric của PostgreSQL trả về chuỗi "35000000.00"; FE cần số thuần.
     */
    private function asNumber(mixed $value): int|float|null
    {
        return $value === null ? null : (float) $value;
    }
}
