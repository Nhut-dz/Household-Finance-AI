<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\Schema;

/**
 * Bổ sung cột lãi suất cho bảng "Thông tin khoản vay".
 *
 * Vì sao nullable
 * ---------------
 * Lãi suất là trường TÙY CHỌN. Phần lớn người dùng mở màn này lúc mới cân nhắc
 * vay, chưa ngân hàng nào báo lãi suất cho họ. `NULL` mang nghĩa "chưa biết" —
 * khác hẳn `0` là "vay không lãi", và hai trạng thái đó dẫn tới hai câu chữ
 * khác nhau trên form.
 *
 * Vì sao phải LƯU chứ không chỉ dùng để tính
 * ------------------------------------------
 * `monthly_payment` từ nay là giá trị SUY RA từ (số tiền vay, kỳ hạn, lãi suất).
 * Không lưu lãi suất thì mất mất số hạng thứ ba: form nạp lại bản ghi cũ sẽ
 * hiện một khoản trả hàng tháng không giải thích được, và người dùng chỉ cần
 * sửa số tiền vay là EMI bị tính lại như thể chưa từng có lãi suất — âm thầm
 * biến một khoản vay có lãi thành khoản vay không lãi.
 *
 * Vì sao CHECK đặt ở DB
 * ---------------------
 * Cùng lý do với 11 CHECK sẵn có của bảng (xem migration tạo bảng): dữ liệu còn
 * vào qua seeder và script SQL, những đường không chạy qua tầng validate của
 * Laravel. Khoảng 6–10 %/năm là ràng buộc nghiệp vụ, giữ khớp
 * `StoreLoanApplicationRequest::MIN_INTEREST_RATE`/`MAX_INTEREST_RATE`.
 *
 * KHÔNG ảnh hưởng ML02: lãi suất không phải feature của model. Nó chỉ đi vào
 * `monthly_payment`, và cột đó vẫn ánh xạ sang `AMT_ANNUITY` y như trước.
 */
return new class extends Migration
{
    public function up(): void
    {
        // Guard cả cột lẫn CHECK trong một điều kiện: Postgres không có
        // ADD CONSTRAINT IF NOT EXISTS, nên chạy lại migration trên DB đã có
        // cột sẽ nổ lỗi trùng tên ràng buộc.
        if (Schema::hasColumn('tblloan_applications', 'interest_rate')) {
            return;
        }

        Schema::table('tblloan_applications', function (Blueprint $table) {
            $table->decimal('interest_rate', 5, 2)
                ->nullable()
                ->after('loan_term_months');
        });

        DB::statement(
            'ALTER TABLE tblloan_applications ADD CONSTRAINT chk_loan_interest_rate '
            .'CHECK (interest_rate IS NULL OR interest_rate BETWEEN 6 AND 10)'
        );
    }

    public function down(): void
    {
        if (! Schema::hasColumn('tblloan_applications', 'interest_rate')) {
            return;
        }

        DB::statement(
            'ALTER TABLE tblloan_applications DROP CONSTRAINT IF EXISTS chk_loan_interest_rate'
        );

        Schema::table('tblloan_applications', function (Blueprint $table) {
            $table->dropColumn('interest_rate');
        });
    }
};
