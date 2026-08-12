<?php

namespace App\Models;

use App\Enums\AssetTypeEnum;
use Illuminate\Database\Eloquent\Attributes\Fillable;
use Illuminate\Database\Eloquent\Attributes\Table;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;

#[Table('tblassets')]
#[Fillable(['household_id', 'asset_type', 'description', 'estimated_value'])]
class Asset extends Model
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
            'asset_type' => AssetTypeEnum::class,
            'estimated_value' => 'decimal:2',
        ];
    }

    /**
     * Hộ gia đình sở hữu tài sản này.
     */
    public function household(): BelongsTo
    {
        return $this->belongsTo(Household::class);
    }
}
