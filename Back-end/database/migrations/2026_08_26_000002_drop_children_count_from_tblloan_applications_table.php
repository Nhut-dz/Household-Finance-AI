<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\Schema;

/**
 * Bỏ cột `children_count` khỏi bảng "Thông tin khoản vay".
 *
 * Vì sao bỏ
 * ---------
 * Số con được hỏi ở CẢ HAI màn, tạo ra hai nguồn cho cùng một sự thật. Tệ hơn,
 * ML02 lại đọc hai cột liên quan chặt với nhau từ hai nguồn khác nhau
 * (`hfml.pipeline.adapters.to_ml02_frame`):
 *
 *     CNT_CHILDREN    ← form Thông tin khoản vay   ← cột này
 *     CNT_FAM_MEMBERS ← tblhouseholds.household_size
 *
 * Mỗi form tự nó hợp lệ nên không tầng validate nào bắt được mâu thuẫn: khai
 * hộ 2 người ở màn này rồi 4 con ở màn kia là lọt xuống tới model. Nay
 * `CNT_CHILDREN` lấy thẳng từ `tblhouseholds.children_count`, cùng nguồn với
 * `CNT_FAM_MEMBERS`, và `HouseholdProfile` đã bảo đảm sẵn
 * `children_count < household_size`.
 *
 * Về dữ liệu bị mất
 * -----------------
 * Cột này KHÔNG mang thông tin nào mà `tblhouseholds.children_count` chưa có —
 * đó là định nghĩa của việc nó thừa. Vẫn là thao tác không hoàn tác được, nên
 * `down()` dựng lại cột với mặc định 0 chứ không hứa khôi phục giá trị cũ.
 *
 * Feature schema của ML02 KHÔNG đổi: vẫn đúng `CNT_CHILDREN`, chỉ đổi nguồn.
 */
return new class extends Migration
{
    public function up(): void
    {
        if (! Schema::hasColumn('tblloan_applications', 'children_count')) {
            return;
        }

        // CHECK phải bỏ TRƯỚC: Postgres không cho drop cột còn ràng buộc trỏ vào.
        DB::statement(
            'ALTER TABLE tblloan_applications DROP CONSTRAINT IF EXISTS chk_loan_children_count'
        );

        Schema::table('tblloan_applications', function (Blueprint $table) {
            $table->dropColumn('children_count');
        });
    }

    public function down(): void
    {
        if (Schema::hasColumn('tblloan_applications', 'children_count')) {
            return;
        }

        Schema::table('tblloan_applications', function (Blueprint $table) {
            $table->smallInteger('children_count')->default(0)->after('marital_status');
        });

        DB::statement(
            'ALTER TABLE tblloan_applications ADD CONSTRAINT chk_loan_children_count '
            .'CHECK (children_count >= 0)'
        );
    }
};
