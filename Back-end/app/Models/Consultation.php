<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Attributes\Fillable;
use Illuminate\Database\Eloquent\Attributes\Table;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;
use Illuminate\Database\Eloquent\Relations\HasMany;
use Illuminate\Database\Eloquent\Relations\HasOne;

/**
 * Một lượt hỏi của người dùng trong Chatbot.
 */
#[Table('tblconsultations')]
#[Fillable(['household_id', 'user_question'])]
class Consultation extends Model
{
    public const UPDATED_AT = null;

    public function household(): BelongsTo
    {
        return $this->belongsTo(Household::class);
    }

    /**
     * Câu trả lời của AI cho lượt hỏi này (quan hệ 1-1).
     */
    public function aiResponse(): HasOne
    {
        return $this->hasOne(AiResponse::class);
    }

    /**
     * Kết quả tính toán tài chính kèm theo lượt tư vấn.
     */
    public function calculationResult(): HasOne
    {
        return $this->hasOne(CalculationResult::class);
    }

    /**
     * Các gói khuyến nghị (tiết kiệm / đầu tư / vay).
     */
    public function recommendationPackages(): HasMany
    {
        return $this->hasMany(RecommendationPackage::class);
    }
}
