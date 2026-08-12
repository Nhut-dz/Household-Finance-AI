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
        if (Schema::hasTable('tblassets')) {
            return;
        }

        Schema::create('tblassets', function (Blueprint $table) {
            $table->id();
            $table->foreignId('household_id');
            $table->string('asset_type', 50)->default('other');
            $table->string('description', 255)->nullable();
            $table->decimal('estimated_value', 18, 2)->nullable();
            $table->timestampTz('created_at')->useCurrent();

            $table->index('household_id', 'idx_assets_household_id');

            $table->foreign('household_id', 'fk_assets_household')
                ->references('id')
                ->on('tblhouseholds')
                ->cascadeOnDelete();
        });

        DB::statement('ALTER TABLE tblassets ADD CONSTRAINT chk_asset_value CHECK (estimated_value IS NULL OR estimated_value >= 0)');
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('tblassets');
    }
};
