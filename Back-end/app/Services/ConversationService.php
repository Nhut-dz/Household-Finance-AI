<?php

namespace App\Services;

use App\Models\Conversation;
use App\Models\Household;
use Illuminate\Support\Facades\DB;

/**
 * Quản lý phiên trò chuyện của Chatbot theo vòng đời của hồ sơ tài chính.
 *
 * Quy tắc nghiệp vụ: một phiên chỉ có nghĩa với ĐÚNG bộ số liệu tài chính mà
 * nó được mở ra cùng. Người dùng sửa thu nhập, chi tiêu, nợ, tiết kiệm, thành
 * viên hay nhu cầu tài chính thì mọi câu trả lời trước đó đã được sinh trên số
 * liệu không còn đúng — giữ chúng lại trong cùng một phiên là để người dùng đọc
 * lời khuyên đã hết hiệu lực như thể vẫn còn hiệu lực.
 *
 * Cách phát hiện: `profile_fingerprint` — sha256 của riêng các trường ảnh hưởng
 * tới phân tích. Sửa tên hoặc nơi ở không đổi vân tay nên phiên được giữ; sửa
 * bất kỳ số liệu tài chính nào thì vân tay đổi và phiên bị thay.
 */
class ConversationService
{
    /**
     * Các cột của `tblhouseholds` đi vào vân tay.
     *
     * Đây đúng là tập trường mà ML01 và tầng rule đọc. `representative_name` và
     * `location` CỐ Ý không có mặt: chúng không vào 17 feature của ML01 cũng
     * không vào rule nào, nên sửa chúng không làm lời khuyên cũ sai đi.
     *
     * `birth_year` CÓ mặt vì ML01 dùng `age` làm feature — đây là chỗ dễ tưởng
     * là thông tin hành chính rồi bỏ sót.
     *
     * @var array<int, string>
     */
    private const FINGERPRINT_COLUMNS = [
        'birth_year',
        'household_size',
        'children_count',
        'monthly_income',
        'monthly_living_cost',
        'supports_elderly',
        'has_debt',
        'total_debt',
        'monthly_debt_payment',
        'has_savings',
        'current_savings',
    ];

    /**
     * Vân tay dữ liệu tài chính của hồ sơ, gồm cả tài sản và nhu cầu.
     *
     * Tài sản và nhu cầu nằm ở bảng khác nhưng vẫn vào vân tay: cả hai đều là
     * đầu vào của phân tích (tài sản là 6 cột multi-hot của ML01, nhu cầu quyết
     * định gói khuyến nghị nào được đề xuất).
     */
    public function fingerprint(Household $household): string
    {
        $household->loadMissing(['assets', 'financialGoals']);

        $payload = [];
        foreach (self::FINGERPRINT_COLUMNS as $column) {
            // Chuẩn hoá về chuỗi: cột số của PostgreSQL về PHP có thể là '5000000'
            // hoặc '5000000.00' tuỳ driver, và bool có thể là true hoặc 1. Không
            // chuẩn hoá thì vân tay đổi dù dữ liệu không đổi, và người dùng bị
            // mất hội thoại sau mỗi lần bấm lưu.
            $payload[$column] = $this->normalise($household->getAttribute($column));
        }

        // Sắp xếp để thứ tự bản ghi trong DB không ảnh hưởng vân tay.
        // `asset_type` / `goal_type` được model cast sang enum, nên phải lấy
        // `->value` chứ không ép chuỗi thẳng.
        $payload['assets'] = $household->assets
            ->pluck('asset_type')->map($this->enumValue(...))->sort()->values()->all();
        $payload['financial_needs'] = $household->financialGoals
            ->pluck('goal_type')->map($this->enumValue(...))->sort()->values()->all();

        return hash('sha256', json_encode($payload, JSON_THROW_ON_ERROR));
    }

    /**
     * Phiên đang mở của hộ, tạo mới nếu chưa có.
     *
     * Dùng ở đường gửi tin nhắn: hộ vừa tạo hồ sơ chưa có phiên nào, và lượt
     * hỏi đầu tiên là thời điểm hợp lý để mở phiên.
     */
    public function currentOrNew(Household $household): Conversation
    {
        $active = $household->activeConversation()->first();

        return $active ?? $this->open($household);
    }

    /**
     * Đóng phiên đang mở (nếu có) rồi mở phiên mới, trong một transaction.
     *
     * Trả về phiên mới. Phiên cũ KHÔNG bị xoá — nó chuyển sang `closed` và ở
     * lại DB để người dùng xem lại lịch sử.
     */
    public function rotate(Household $household, string $reason): Conversation
    {
        return DB::transaction(function () use ($household, $reason) {
            $this->closeActive($household, $reason);

            return $this->open($household);
        });
    }

    /**
     * Xoay phiên khi và chỉ khi dữ liệu tài chính đã thay đổi.
     *
     * @param  string  $fingerprintBefore  Vân tay chụp TRƯỚC khi cập nhật hồ sơ.
     * @return array{rotated: bool, conversation: Conversation|null}
     */
    public function rotateIfProfileChanged(
        Household $household,
        string $fingerprintBefore,
    ): array {
        $fingerprintAfter = $this->fingerprint($household);

        if ($fingerprintAfter === $fingerprintBefore) {
            // Sửa thông tin không ảnh hưởng phân tích (tên, nơi ở) → giữ phiên.
            //
            // Trả về phiên ĐANG CÓ, không gọi `currentOrNew()`: hộ chưa từng hỏi
            // câu nào thì chưa có phiên, và một lần bấm lưu không đổi gì về tài
            // chính không phải lý do để sinh ra một phiên rỗng.
            return [
                'rotated' => false,
                'conversation' => $household->activeConversation()->first(),
            ];
        }

        return [
            'rotated' => true,
            'conversation' => $this->rotate(
                $household,
                Conversation::REASON_PROFILE_UPDATED,
            ),
        ];
    }

    private function open(Household $household): Conversation
    {
        return $household->conversations()->create([
            'status' => Conversation::STATUS_ACTIVE,
            'profile_fingerprint' => $this->fingerprint($household),
        ]);
    }

    private function closeActive(Household $household, string $reason): void
    {
        $household->activeConversation()->first()?->update([
            'status' => Conversation::STATUS_CLOSED,
            'closed_reason' => $reason,
            'closed_at' => now(),
        ]);
    }

    /**
     * Giá trị chuỗi của một cột có thể đã được cast sang enum.
     *
     * Cột `asset_type` là `string(50)` không ràng buộc ở DB, nên một hàng lạ
     * vẫn lọt vào được dưới dạng chuỗi thường — xử lý cả hai trường hợp thay vì
     * giả định luôn là enum.
     */
    private function enumValue(mixed $value): string
    {
        return $value instanceof \BackedEnum ? (string) $value->value : (string) $value;
    }

    /**
     * Đưa giá trị về dạng so sánh được, không phụ thuộc kiểu mà driver trả về.
     */
    private function normalise(mixed $value): string
    {
        if ($value === null) {
            return '';
        }

        if (is_bool($value)) {
            return $value ? '1' : '0';
        }

        if (is_numeric($value)) {
            // '5000000.00' và 5000000 phải cho cùng một chuỗi.
            return rtrim(rtrim(number_format((float) $value, 4, '.', ''), '0'), '.');
        }

        return (string) $value;
    }
}
