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
        if (Schema::hasTable('tblconsultations')) {
            return;
        }

        Schema::create('tblconsultations', function (Blueprint $table) {
            $table->id();
            $table->foreignId('household_id');
            $table->text('user_question')->nullable();
            $table->timestampTz('created_at')->useCurrent();

            $table->index('household_id', 'idx_consultations_household_id');

            $table->foreign('household_id', 'fk_consultations_household')
                ->references('id')
                ->on('tblhouseholds')
                ->cascadeOnDelete();
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('tblconsultations');
    }
};
