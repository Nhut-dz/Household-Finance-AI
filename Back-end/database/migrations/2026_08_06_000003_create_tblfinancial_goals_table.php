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
        if (Schema::hasTable('tblfinancial_goals')) {
            return;
        }

        Schema::create('tblfinancial_goals', function (Blueprint $table) {
            $table->id();
            $table->foreignId('household_id');
            $table->string('goal_type', 50)->default('other');
            $table->string('description', 255)->nullable();
            $table->decimal('target_amount', 18, 2)->nullable();
            $table->integer('priority')->nullable();
            $table->timestampTz('created_at')->useCurrent();

            $table->index('household_id', 'idx_financial_goals_household_id');

            $table->foreign('household_id', 'fk_financial_goals_household')
                ->references('id')
                ->on('tblhouseholds')
                ->cascadeOnDelete();
        });

        DB::statement('ALTER TABLE tblfinancial_goals ADD CONSTRAINT chk_goal_amount CHECK (target_amount IS NULL OR target_amount >= 0)');
        DB::statement('ALTER TABLE tblfinancial_goals ADD CONSTRAINT chk_goal_priority CHECK (priority IS NULL OR priority >= 1)');
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('tblfinancial_goals');
    }
};
