<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\Schema;

/**
 * Phiên trò chuyện (Conversation) của Chatbot.
 *
 * Trước migration này không có thực thể phiên nào: `tblconsultations` chỉ trỏ
 * tới `household_id`, nên lịch sử chat là *toàn bộ* lượt hỏi của hộ từ đầu đến
 * cuối. Hậu quả nghiệp vụ: người dùng sửa hồ sơ xong vẫn thấy nguyên hội thoại
 * cũ — những câu trả lời được sinh ra trên số liệu tài chính đã không còn đúng.
 *
 * `profile_fingerprint` là vân tay của các trường tài chính tại thời điểm mở
 * phiên. So vân tay trước/sau khi cập nhật hồ sơ là cách biết một lần sửa có
 * ảnh hưởng tới phân tích tài chính hay không, mà không phải so từng cột ở mọi
 * nơi gọi.
 */
return new class extends Migration
{
    public function up(): void
    {
        if (! Schema::hasTable('tblconversations')) {
            Schema::create('tblconversations', function (Blueprint $table) {
                $table->id();
                $table->foreignId('household_id');

                // 'active' | 'closed'. Mỗi hộ có TỐI ĐA một phiên 'active' —
                // ràng buộc này được canh bằng partial unique index bên dưới.
                $table->string('status', 16)->default('active');

                //: Vân tay dữ liệu tài chính lúc mở phiên (sha256).
                $table->string('profile_fingerprint', 64)->nullable();

                //: Vì sao phiên bị đóng, ví dụ 'profile_updated'. Null khi đang mở.
                $table->string('closed_reason', 32)->nullable();

                $table->timestampTz('created_at')->useCurrent();
                $table->timestampTz('closed_at')->nullable();

                $table->index('household_id', 'idx_conversations_household_id');

                $table->foreign('household_id', 'fk_conversations_household')
                    ->references('id')
                    ->on('tblhouseholds')
                    ->cascadeOnDelete();
            });

            // Một hộ không được có hai phiên đang mở cùng lúc. Đặt ràng buộc ở
            // DB chứ không chỉ ở tầng service: hai request cập nhật hồ sơ gần
            // như đồng thời sẽ cùng thấy "chưa có phiên active" và cùng tạo.
            DB::statement(
                'CREATE UNIQUE INDEX uniq_conversations_one_active_per_household '
                .'ON tblconversations (household_id) '
                ."WHERE status = 'active'"
            );
        }

        if (! Schema::hasColumn('tblconsultations', 'conversation_id')) {
            Schema::table('tblconsultations', function (Blueprint $table) {
                // Nullable để migration chạy được trên dữ liệu đang có, rồi
                // backfill ngay bên dưới.
                $table->foreignId('conversation_id')->nullable()->after('household_id');

                $table->index('conversation_id', 'idx_consultations_conversation_id');

                // Xoá phiên thì xoá luôn lượt hỏi thuộc phiên đó — cùng quy ước
                // cascade như quan hệ household → consultations.
                $table->foreign('conversation_id', 'fk_consultations_conversation')
                    ->references('id')
                    ->on('tblconversations')
                    ->cascadeOnDelete();
            });
        }

        $this->backfillLegacyConversations();
    }

    /**
     * Gom các lượt hỏi đã có vào một phiên ĐÃ ĐÓNG cho mỗi hộ.
     *
     * Để `conversation_id` rỗng thì mọi truy vấn sau này phải mang thêm một
     * nhánh "null nghĩa là lịch sử cũ", và cái nhánh đó sẽ bị quên ở đúng chỗ
     * quan trọng. Đóng sẵn (chứ không mở) vì các lượt hỏi đó được trả lời trên
     * số liệu của một thời điểm không xác định được nữa.
     */
    private function backfillLegacyConversations(): void
    {
        $householdIds = DB::table('tblconsultations')
            ->whereNull('conversation_id')
            ->distinct()
            ->pluck('household_id');

        foreach ($householdIds as $householdId) {
            $conversationId = DB::table('tblconversations')->insertGetId([
                'household_id' => $householdId,
                'status' => 'closed',
                'profile_fingerprint' => null,
                'closed_reason' => 'legacy_backfill',
                'created_at' => now(),
                'closed_at' => now(),
            ]);

            DB::table('tblconsultations')
                ->where('household_id', $householdId)
                ->whereNull('conversation_id')
                ->update(['conversation_id' => $conversationId]);
        }
    }

    public function down(): void
    {
        if (Schema::hasColumn('tblconsultations', 'conversation_id')) {
            Schema::table('tblconsultations', function (Blueprint $table) {
                $table->dropForeign('fk_consultations_conversation');
                $table->dropIndex('idx_consultations_conversation_id');
                $table->dropColumn('conversation_id');
            });
        }

        Schema::dropIfExists('tblconversations');
    }
};
