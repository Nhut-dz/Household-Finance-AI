<?php

namespace App\Models;

use App\Enums\PackageTypeEnum;
use Illuminate\Database\Eloquent\Attributes\Fillable;
use Illuminate\Database\Eloquent\Attributes\Table;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;

#[Table('tblrecommendation_packages')]
#[Fillable(['consultation_id', 'package_type', 'title', 'content'])]
class RecommendationPackage extends Model
{
    public const UPDATED_AT = null;

    /**
     * @return array<string, string>
     */
    protected function casts(): array
    {
        return [
            'package_type' => PackageTypeEnum::class,
            'content' => 'array',
        ];
    }

    public function consultation(): BelongsTo
    {
        return $this->belongsTo(Consultation::class);
    }
}
