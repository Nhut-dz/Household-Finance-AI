<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\Schema;

/**
 * Bảng "Thông tin khoản vay" — dữ liệu đầu vào của ML02 (Home Credit Risk).
 *
 * Vì sao tách bảng riêng thay vì thêm cột vào tblhouseholds:
 *
 *   1. Đây là dữ liệu TÙY CHỌN. Chỉ hộ nào muốn đánh giá khoản vay mới nhập;
 *      nhồi 16 cột nullable vào tblhouseholds thì mọi hồ sơ đều mang chúng.
 *   2. Vòng đời khác nhau. Hồ sơ tài chính sửa lại thì phiên chat bị xoay
 *      (ConversationService::rotateIfProfileChanged); khai lại khoản vay thì
 *      không nhất thiết — đây là hai luồng nghiệp vụ tách bạch.
 *   3. Nhóm C (lịch sử tín dụng) không phải thông tin hộ gia đình, mà là
 *      thông tin quan hệ tín dụng của người vay.
 *
 * Ràng buộc 1-1: unique(household_id). Một hộ giữ ĐÚNG MỘT phương án vay đang
 * xét, nộp lại thì ghi đè (LoanApplicationService::upsert). Muốn so nhiều kịch
 * bản vay thì bỏ unique và thêm cột nhãn kịch bản — hiện chưa nằm trong phạm vi.
 */
return new class extends Migration
{
    public function up(): void
    {
        if (Schema::hasTable('tblloan_applications')) {
            return;
        }

        Schema::create('tblloan_applications', function (Blueprint $table) {
            $table->id();
            $table->foreignId('household_id');

            // -- A. Thông tin người vay ------------------------------------
            // Tuổi nhập trực tiếp chứ không suy từ tblhouseholds.birth_year:
            // trang này phải điền được cả khi hồ sơ chưa khai năm sinh, và
            // người vay không nhất thiết là người đại diện hộ.
            $table->smallInteger('borrower_age');            // → DAYS_BIRTH
            $table->string('gender', 8);                     // → CODE_GENDER
            $table->string('marital_status', 20);            // → NAME_FAMILY_STATUS
            $table->smallInteger('children_count')->default(0);   // → CNT_CHILDREN
            $table->string('education_level', 24);           // → NAME_EDUCATION_TYPE
            $table->string('occupation', 24);                // → OCCUPATION_TYPE
            $table->decimal('employment_years', 4, 1);       // → DAYS_EMPLOYED

            // -- B. Thông tin khoản vay ------------------------------------
            $table->decimal('loan_amount', 18, 2);           // → AMT_CREDIT
            $table->smallInteger('loan_term_months');
            $table->decimal('monthly_payment', 18, 2);       // → AMT_ANNUITY
            $table->decimal('asset_price', 18, 2);           // → AMT_GOODS_PRICE
            $table->string('loan_purpose', 24);              // → NAME_CASH_LOAN_PURPOSE

            // -- C. Lịch sử tín dụng ---------------------------------------
            // Bốn cột này KHÔNG có trong application_train.csv — chúng là bản
            // tự khai tương ứng với phần tổng hợp từ bureau.csv:
            //   previous_loan_count  ← COUNT(SK_ID_BUREAU)
            //   late_payment_count   ← SUM(CREDIT_DAY_OVERDUE > 0)
            //   has_overdue_loan     ← MAX(CREDIT_DAY_OVERDUE) > 0
            //   total_overdue_amount ← SUM(AMT_CREDIT_SUM_OVERDUE)
            $table->smallInteger('previous_loan_count')->default(0);
            $table->smallInteger('late_payment_count')->default(0);
            $table->boolean('has_overdue_loan')->default(false);
            $table->decimal('total_overdue_amount', 18, 2)->default(0);

            $table->timestampTz('created_at')->useCurrent();
            $table->timestampTz('updated_at')->useCurrent();

            $table->unique('household_id', 'uq_loan_applications_household');

            $table->foreign('household_id', 'fk_loan_applications_household')
                ->references('id')
                ->on('tblhouseholds')
                ->cascadeOnDelete();
        });

        // Ràng buộc miền giá trị đặt ở DB chứ không chỉ ở FormRequest: dữ liệu
        // còn vào bảng qua seeder và script SQL, mà những đường đó không chạy
        // qua tầng validate của Laravel.
        foreach ([
            'chk_loan_borrower_age' => 'borrower_age BETWEEN 18 AND 100',
            'chk_loan_children_count' => 'children_count >= 0',
            'chk_loan_employment_years' => 'employment_years >= 0 AND employment_years <= 60',
            'chk_loan_amount' => 'loan_amount > 0',
            'chk_loan_term_months' => 'loan_term_months BETWEEN 6 AND 360',
            'chk_loan_monthly_payment' => 'monthly_payment > 0',
            'chk_loan_asset_price' => 'asset_price > 0',
            'chk_loan_previous_count' => 'previous_loan_count >= 0',
            'chk_loan_late_count' => 'late_payment_count >= 0',
            'chk_loan_total_overdue' => 'total_overdue_amount >= 0',
            // Khai "không có khoản quá hạn" mà vẫn có số nợ quá hạn là mâu
            // thuẫn nội tại — chặn ở đây thì tầng ML khỏi phải đoán bên nào đúng.
            'chk_loan_overdue_consistency' => 'has_overdue_loan OR total_overdue_amount = 0',
        ] as $name => $expression) {
            DB::statement("ALTER TABLE tblloan_applications ADD CONSTRAINT {$name} CHECK ({$expression})");
        }
    }

    public function down(): void
    {
        Schema::dropIfExists('tblloan_applications');
    }
};
