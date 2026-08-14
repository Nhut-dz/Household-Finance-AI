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
#[Fillable(['household_id', 'conversation_id', 'user_question'])]
class Consultation extends Model
{
    public const UPDATED_AT = null;

    public function household(): BelongsTo
    {
        return $this->belongsTo(Household::class);
    }

    /**
     * Phiên trò chuyện chứa lượt hỏi này.
     *
     * Null chỉ xảy ra với dữ liệu chưa qua backfill của migration
     * `create_tblconversations_table`; mọi lượt hỏi mới đều có phiên.
     */
    public function conversation(): BelongsTo
    {
        return $this->belongsTo(Conversation::class);
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
