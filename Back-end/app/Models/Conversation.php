<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Attributes\Fillable;
use Illuminate\Database\Eloquent\Attributes\Table;
use Illuminate\Database\Eloquent\Builder;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;
use Illuminate\Database\Eloquent\Relations\HasMany;

/**
 * Một phiên trò chuyện của Chatbot, gắn với một hồ sơ hộ gia đình.
 *
 * Mỗi hộ có tối đa MỘT phiên `active`; các phiên trước đó ở trạng thái `closed`
 * và vẫn nằm trong DB để người dùng xem lại. Chỉ phiên đang mở được dùng làm
 * ngữ cảnh cho lượt hỏi mới.
 */
#[Table('tblconversations')]
#[Fillable(['household_id', 'status', 'profile_fingerprint', 'closed_reason', 'closed_at'])]
class Conversation extends Model
{
    public const UPDATED_AT = null;

    public const STATUS_ACTIVE = 'active';

    public const STATUS_CLOSED = 'closed';

    /** Phiên bị đóng vì dữ liệu tài chính của hồ sơ đã thay đổi. */
    public const REASON_PROFILE_UPDATED = 'profile_updated';

    protected function casts(): array
    {
        return ['closed_at' => 'datetime'];
    }

    public function household(): BelongsTo
    {
        return $this->belongsTo(Household::class);
    }

    public function consultations(): HasMany
    {
        return $this->hasMany(Consultation::class);
    }

    public function scopeActive(Builder $query): Builder
    {
        return $query->where('status', self::STATUS_ACTIVE);
    }

    public function isActive(): bool
    {
        return $this->status === self::STATUS_ACTIVE;
    }
}
