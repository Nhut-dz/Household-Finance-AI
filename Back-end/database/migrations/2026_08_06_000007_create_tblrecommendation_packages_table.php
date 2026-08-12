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
        if (Schema::hasTable('tblrecommendation_packages')) {
            return;
        }

        Schema::create('tblrecommendation_packages', function (Blueprint $table) {
            $table->id();
            $table->foreignId('consultation_id');
            $table->string('package_type', 50);
            $table->string('title', 150);
            $table->jsonb('content')->nullable();
            $table->timestampTz('created_at')->useCurrent();

            $table->index('consultation_id', 'idx_recommendation_packages_consultation_id');

            $table->foreign('consultation_id', 'fk_recommendation_packages_consultation')
                ->references('id')
                ->on('tblconsultations')
                ->cascadeOnDelete();
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('tblrecommendation_packages');
    }
};
