<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    /**
     * Run the migrations.
     */
    public function up(): void
    {
        // Các cột này đã được thêm sẵn trên DB dev bằng ALTER TABLE.
        // Guard hasColumn để migration chạy được cả trên môi trường mới.
        Schema::table('tblhouseholds', function (Blueprint $table) {
            if (! Schema::hasColumn('tblhouseholds', 'occupation')) {
                $table->string('occupation', 32)->nullable();
            }
            if (! Schema::hasColumn('tblhouseholds', 'employment_years')) {
                $table->decimal('employment_years', 4, 1)->nullable();
            }
            if (! Schema::hasColumn('tblhouseholds', 'asset_price')) {
                $table->decimal('asset_price', 15, 2)->nullable();
            }
            if (! Schema::hasColumn('tblhouseholds', 'loan_amount')) {
                $table->decimal('loan_amount', 15, 2)->nullable();
            }
            if (! Schema::hasColumn('tblhouseholds', 'loan_term_months')) {
                $table->smallInteger('loan_term_months')->nullable();
            }
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::table('tblhouseholds', function (Blueprint $table) {
            $table->dropColumn([
                'occupation',
                'employment_years',
                'asset_price',
                'loan_amount',
                'loan_term_months',
            ]);
        });
    }
};
