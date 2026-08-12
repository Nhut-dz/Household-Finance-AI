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
        if (Schema::hasTable('tblai_responses')) {
            return;
        }

        Schema::create('tblai_responses', function (Blueprint $table) {
            $table->id();
            // Mỗi lượt hỏi chỉ có đúng một câu trả lời.
            $table->foreignId('consultation_id')->unique();
            $table->text('response_text');
            $table->string('model_used', 100);
            $table->jsonb('suggested_questions')->nullable();
            $table->integer('tokens_used')->nullable();
            $table->timestampTz('created_at')->useCurrent();

            $table->foreign('consultation_id', 'fk_ai_responses_consultation')
                ->references('id')
                ->on('tblconsultations')
                ->cascadeOnDelete();
        });

        DB::statement('ALTER TABLE tblai_responses ADD CONSTRAINT chk_tokens_used CHECK (tokens_used IS NULL OR tokens_used >= 0)');
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('tblai_responses');
    }
};
