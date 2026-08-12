<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    /**
     * Run the migrations.
     */
    public function up(): void
    {
        // Schema nghiệp vụ đã được dựng sẵn bằng script SQL trên DB dev.
        // Migration này để dựng lại cho môi trường mới, nên bỏ qua nếu đã có bảng.
        if (Schema::hasTable('tblhouseholds')) {
            return;
        }

        Schema::create('tblhouseholds', function (Blueprint $table) {
            $table->id();
            $table->foreignId('user_id')->nullable();
            $table->string('session_token', 64)->nullable();
            $table->string('representative_name', 150);
            $table->smallInteger('birth_year')->nullable();
            $table->string('location', 255)->nullable();
            $table->decimal('monthly_income', 18, 2)->default(0);
            $table->integer('household_size')->default(1);
            $table->integer('children_count')->default(0);
            $table->boolean('supports_elderly')->default(false);
            $table->boolean('has_debt')->default(false);
            $table->decimal('total_debt', 18, 2)->default(0);
            $table->decimal('monthly_debt_payment', 18, 2)->default(0);
            $table->boolean('has_savings')->default(false);
            $table->decimal('current_savings', 18, 2)->default(0);
            $table->decimal('monthly_living_cost', 18, 2)->nullable();
            $table->timestampTz('created_at')->useCurrent();
            $table->timestampTz('updated_at')->useCurrent();

            $table->index('user_id', 'idx_households_user_id');

            $table->foreign('user_id', 'fk_households_user')
                ->references('id')
                ->on('tblusers')
                ->nullOnDelete();
        });

        foreach ([
            'chk_household_size' => 'household_size >= 1',
            'chk_children_count' => 'children_count >= 0',
            'chk_household_income' => 'monthly_income >= 0',
            'chk_total_debt' => 'total_debt >= 0',
            'chk_monthly_debt_payment' => 'monthly_debt_payment >= 0',
            'chk_current_savings' => 'current_savings >= 0',
        ] as $name => $expression) {
            DB::statement("ALTER TABLE tblhouseholds ADD CONSTRAINT {$name} CHECK ({$expression})");
        }
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('tblhouseholds');
    }
};
