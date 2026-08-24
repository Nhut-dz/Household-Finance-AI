<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\Schema;

/**
 * Bổ sung ON DELETE CASCADE cho 6 khoá ngoại đang thiếu.
 *
 * Vì sao cần
 * ----------
 * Xoá một hộ gia đình đang NỔ LỖI chứ không phải xoá được:
 *
 *     SQLSTATE[23503]: update or delete on table "tblhouseholds" violates
 *     foreign key constraint "tblassets_household_id_fkey"
 *
 * Nghĩa là nút "Xóa dữ liệu" ở màn Chẩn đoán hồ sơ chưa từng chạy được, và
 * `HouseholdService::delete()` cũng vậy — dù docblock của nó ghi rằng "tài sản,
 * nhu cầu tài chính và lịch sử hội thoại tự động bị xoá theo nhờ ràng buộc
 * ON DELETE CASCADE". Câu đó đúng với ý ĐỊNH nhưng sai với DB thật.
 *
 * Nguyên nhân: các bảng đời đầu (`2026_08_06_*`) khai `foreignId(...)` rồi
 * `constrained()`, sinh ra ràng buộc tên mặc định kiểu `{bảng}_{cột}_fkey` với
 * hành vi NO ACTION. Các bảng làm sau (`tblconversations`, `tblloan_applications`)
 * khai `foreign(..., 'tên_rõ_ràng')->cascadeOnDelete()` nên có cascade. Hai lối
 * viết cùng tồn tại, và không ai phát hiện vì chưa có test nào xoá thật.
 *
 * Trạng thái trước migration này (đo trên DB dev bằng pg_constraint):
 *
 *     tblconversations   <- tblhouseholds     CASCADE     ✅
 *     tblloan_applications <- tblhouseholds   CASCADE     ✅
 *     tblconsultations   <- tblconversations  CASCADE     ✅
 *     tblassets          <- tblhouseholds     NO ACTION   ❌
 *     tblfinancial_goals <- tblhouseholds     NO ACTION   ❌
 *     tblconsultations   <- tblhouseholds     NO ACTION   ❌
 *     tblai_responses    <- tblconsultations  NO ACTION   ❌
 *     tblcalculation_results     <- tblconsultations  NO ACTION   ❌
 *     tblrecommendation_packages <- tblconsultations  NO ACTION   ❌
 *
 * `tblhouseholds <- tblusers` CỐ Ý không đổi: xoá một tài khoản có nên kéo theo
 * toàn bộ hồ sơ tài chính của họ không là quyết định về lưu trữ dữ liệu, không
 * thuộc phạm vi nút "Xóa dữ liệu".
 */
return new class extends Migration
{
    /**
     * Ràng buộc cần sửa: bảng con → [cột, bảng cha, tên ràng buộc hiện tại].
     *
     * Tên ràng buộc lấy đúng như DB đang có, không đoán theo quy ước: chúng do
     * Laravel sinh tự động nên phải khớp từng ký tự thì DROP mới trúng.
     */
    private const FOREIGN_KEYS = [
        ['tblassets', 'household_id', 'tblhouseholds', 'tblassets_household_id_fkey'],
        ['tblfinancial_goals', 'household_id', 'tblhouseholds', 'tblfinancial_goals_household_id_fkey'],
        ['tblconsultations', 'household_id', 'tblhouseholds', 'tblconsultations_household_id_fkey'],
        ['tblai_responses', 'consultation_id', 'tblconsultations', 'tblai_responses_consultation_id_fkey'],
        ['tblcalculation_results', 'consultation_id', 'tblconsultations', 'tblcalculation_results_consultation_id_fkey'],
        ['tblrecommendation_packages', 'consultation_id', 'tblconsultations', 'tblrecommendation_packages_consultation_id_fkey'],
    ];

    public function up(): void
    {
        $this->rebuild('CASCADE');
    }

    /**
     * Trả về NO ACTION — đúng trạng thái trước migration, kể cả khi đó là
     * trạng thái hỏng. `down()` là để quay lui, không phải để sửa thêm.
     */
    public function down(): void
    {
        $this->rebuild('NO ACTION');
    }

    private function rebuild(string $onDelete): void
    {
        // SQLite (dùng khi chạy phpunit) không có ALTER TABLE DROP CONSTRAINT.
        // Ở đó ràng buộc được dựng lại từ đầu mỗi lần migrate nên không có gì
        // để vá; bỏ qua thay vì để migration ném lỗi.
        if (DB::connection()->getDriverName() !== 'pgsql') {
            return;
        }

        foreach (self::FOREIGN_KEYS as [$table, $column, $parent, $name]) {
            if (! Schema::hasTable($table)) {
                continue;
            }

            // IF EXISTS: DB dựng mới sau migration này sẽ không có ràng buộc
            // tên cũ, và migration vẫn phải chạy qua được.
            DB::statement("ALTER TABLE {$table} DROP CONSTRAINT IF EXISTS {$name}");
            DB::statement(
                "ALTER TABLE {$table} ADD CONSTRAINT {$name} "
                ."FOREIGN KEY ({$column}) REFERENCES {$parent} (id) "
                ."ON DELETE {$onDelete}"
            );
        }
    }
};
