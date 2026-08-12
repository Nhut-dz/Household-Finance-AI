<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Attributes\Fillable;
use Illuminate\Database\Eloquent\Attributes\Table;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;

#[Table('tblcalculation_results')]
#[Fillable([
    'consultation_id',
    'dti_ratio',
    'dti_status',
    'safe_loan_limit',
    'recommended_monthly_saving',
    'budget_needs',
    'budget_wants',
    'budget_savings',
    'allocation_rule',
    'raw_json',
])]
class CalculationResult extends Model
{
    public const UPDATED_AT = null;

    /**
     * @return array<string, string>
     */
    protected function casts(): array
    {
        return [
            'dti_ratio' => 'decimal:2',
            'safe_loan_limit' => 'decimal:2',
            'recommended_monthly_saving' => 'decimal:2',
            'budget_needs' => 'decimal:2',
            'budget_wants' => 'decimal:2',
            'budget_savings' => 'decimal:2',
            'raw_json' => 'array',
        ];
    }

    public function consultation(): BelongsTo
    {
        return $this->belongsTo(Consultation::class);
    }
}
