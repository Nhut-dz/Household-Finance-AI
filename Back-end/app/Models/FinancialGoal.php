<?php

namespace App\Models;

use App\Enums\GoalTypeEnum;
use Illuminate\Database\Eloquent\Attributes\Fillable;
use Illuminate\Database\Eloquent\Attributes\Table;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;

#[Table('tblfinancial_goals')]
#[Fillable(['household_id', 'goal_type', 'description', 'target_amount', 'priority'])]
class FinancialGoal extends Model
{
    /**
     * Bảng chỉ có created_at, không có updated_at.
     */
    public const UPDATED_AT = null;

    /**
     * Get the attributes that should be cast.
     *
     * @return array<string, string>
     */
    protected function casts(): array
    {
        return [
            'goal_type' => GoalTypeEnum::class,
            'target_amount' => 'decimal:2',
            'priority' => 'integer',
        ];
    }

    /**
     * Hộ gia đình có nhu cầu tài chính này.
     */
    public function household(): BelongsTo
    {
        return $this->belongsTo(Household::class);
    }
}
