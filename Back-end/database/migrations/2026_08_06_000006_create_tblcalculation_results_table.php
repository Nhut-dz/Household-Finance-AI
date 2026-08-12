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
        if (Schema::hasTable('tblcalculation_results')) {
            return;
        }

        Schema::create('tblcalculation_results', function (Blueprint $table) {
            $table->id();
            $table->foreignId('consultation_id')->unique();
            $table->decimal('dti_ratio', 5, 2)->nullable();
            $table->string('dti_status', 20)->nullable();
            $table->decimal('safe_loan_limit', 18, 2)->nullable();
            $table->decimal('recommended_monthly_saving', 18, 2)->nullable();
            $table->decimal('budget_needs', 18, 2)->nullable();
            $table->decimal('budget_wants', 18, 2)->nullable();
            $table->decimal('budget_savings', 18, 2)->nullable();
            $table->string('allocation_rule', 20)->nullable();
            $table->jsonb('raw_json')->nullable();
            $table->timestampTz('created_at')->useCurrent();

            $table->foreign('consultation_id', 'fk_calculation_results_consultation')
                ->references('id')
                ->on('tblconsultations')
                ->cascadeOnDelete();
        });

        DB::statement('ALTER TABLE tblcalculation_results ADD CONSTRAINT chk_dti_ratio CHECK (dti_ratio IS NULL OR dti_ratio >= 0)');
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('tblcalculation_results');
    }
};
