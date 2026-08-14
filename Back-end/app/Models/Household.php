<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Attributes\Fillable;
use Illuminate\Database\Eloquent\Attributes\Table;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;
use Illuminate\Database\Eloquent\Relations\HasMany;
use Illuminate\Database\Eloquent\Relations\HasOne;

#[Table('tblhouseholds')]
#[Fillable([
    'user_id',
    'session_token',
    'representative_name',
    'birth_year',
    'location',
    'monthly_income',
    'monthly_living_cost',
    'household_size',
    'children_count',
    'supports_elderly',
    'has_debt',
    'total_debt',
    'monthly_debt_payment',
    'has_savings',
    'current_savings',
    'occupation',
    'employment_years',
    'asset_price',
    'loan_amount',
    'loan_term_months',
])]
class Household extends Model
{
    /**
     * Get the attributes that should be cast.
     *
     * @return array<string, string>
     */
    protected function casts(): array
    {
        return [
            'birth_year' => 'integer',
            'household_size' => 'integer',
            'children_count' => 'integer',
            'supports_elderly' => 'boolean',
            'has_debt' => 'boolean',
            'has_savings' => 'boolean',
            'monthly_income' => 'decimal:2',
            'monthly_living_cost' => 'decimal:2',
            'total_debt' => 'decimal:2',
            'monthly_debt_payment' => 'decimal:2',
            'current_savings' => 'decimal:2',
            'employment_years' => 'decimal:1',
            'asset_price' => 'decimal:2',
            'loan_amount' => 'decimal:2',
            'loan_term_months' => 'integer',
        ];
    }

    /**
     * Chủ hộ đã đăng nhập. Null khi hồ sơ được tạo bởi Guest.
     */
    public function user(): BelongsTo
    {
        return $this->belongsTo(User::class);
    }

    /**
     * Tài sản đang sở hữu.
     */
    public function assets(): HasMany
    {
        return $this->hasMany(Asset::class);
    }

    /**
     * Nhu cầu tài chính.
     */
    public function financialGoals(): HasMany
    {
        return $this->hasMany(FinancialGoal::class);
    }

    /**
     * Các lượt hỏi đáp với Chatbot, cũ nhất trước.
     *
     * Đây là TẤT CẢ lượt hỏi của hộ, xuyên mọi phiên. Muốn hội thoại đang diễn
     * ra thì đi qua `activeConversation()` — dùng trực tiếp quan hệ này để dựng
     * lịch sử chat sẽ lôi cả hội thoại của những hồ sơ cũ đã bị thay thế.
     */
    public function consultations(): HasMany
    {
        return $this->hasMany(Consultation::class);
    }

    /**
     * Các phiên trò chuyện, mới nhất trước.
     */
    public function conversations(): HasMany
    {
        return $this->hasMany(Conversation::class)->latest('id');
    }

    /**
     * Phiên đang mở. Tối đa một phiên nhờ partial unique index ở DB.
     */
    public function activeConversation(): HasOne
    {
        return $this->hasOne(Conversation::class)
            ->where('status', Conversation::STATUS_ACTIVE);
    }
}
