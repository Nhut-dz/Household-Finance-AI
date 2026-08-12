<?php

namespace App\Services;

use App\Models\AiResponse;
use App\Models\Consultation;
use App\Models\Household;
use Illuminate\Support\Facades\DB;

/**
 * Lịch sử hội thoại Chatbot.
 *
 * DB lưu mỗi lượt hỏi đáp là một tblconsultations (câu hỏi) kèm tối đa một
 * tblai_responses (câu trả lời). FE lại cần một danh sách tin nhắn phẳng, nên
 * service này chịu trách nhiệm trải hai bảng đó thành các message.
 */
class ChatService
{
    public function __construct(private readonly AdvisorClient $advisor) {}

    /**
     * Toàn bộ hội thoại của hồ sơ, sắp xếp theo created_at tăng dần.
     *
     * @return array<int, array<string, mixed>>
     */
    public function history(Household $household): array
    {
        return $household->consultations()
            ->with('aiResponse')
            ->orderBy('id')
            ->get()
            ->flatMap(fn (Consultation $consultation) => $this->toMessages($consultation))
            ->all();
    }

    /**
     * Gửi câu hỏi sang service Python rồi lưu cả câu hỏi lẫn câu trả lời.
     *
     * Gọi Python trước khi ghi DB để nếu service lỗi thì không để lại lượt hỏi
     * mồ côi không có câu trả lời.
     *
     * @return array{user_message: array<string, mixed>, ai_message: array<string, mixed>}
     */
    public function send(Household $household, string $content): array
    {
        $answer = $this->advisor->ask($household, $content);

        $consultation = DB::transaction(function () use ($household, $content, $answer) {
            $consultation = $household->consultations()->create([
                'user_question' => $content,
            ]);

            $consultation->aiResponse()->create([
                'response_text' => $answer['response_text'],
                'model_used' => $answer['model_used'],
                'suggested_questions' => $answer['suggested_questions'],
                'tokens_used' => $answer['tokens_used'],
            ]);

            return $consultation->load('aiResponse');
        });

        return [
            'user_message' => $this->userMessage($consultation),
            'ai_message' => $this->aiMessage($consultation, $consultation->aiResponse),
        ];
    }

    /**
     * @return array<int, array<string, mixed>>
     */
    private function toMessages(Consultation $consultation): array
    {
        $messages = [];

        if ($consultation->user_question !== null) {
            $messages[] = $this->userMessage($consultation);
        }

        if ($consultation->aiResponse !== null) {
            $messages[] = $this->aiMessage($consultation, $consultation->aiResponse);
        }

        return $messages;
    }

    /**
     * @return array<string, mixed>
     */
    private function userMessage(Consultation $consultation): array
    {
        return [
            'id' => $this->messageId($consultation, isUser: true),
            'role' => 'user',
            'content' => $consultation->user_question,
            'created_at' => $consultation->created_at,
        ];
    }

    /**
     * @return array<string, mixed>
     */
    private function aiMessage(Consultation $consultation, AiResponse $response): array
    {
        return [
            'id' => $this->messageId($consultation, isUser: false),
            'role' => 'ai',
            'content' => $response->response_text,
            'created_at' => $response->created_at,
            'suggested_questions' => $response->suggested_questions,
        ];
    }

    /**
     * Id của message không có sẵn trong DB vì một lượt tư vấn nằm ở hai bảng
     * khác nhau và id của hai bảng đó có thể trùng nhau. Suy ra từ id lượt tư
     * vấn để mỗi message có một id số nguyên duy nhất và ổn định giữa các lần
     * gọi: câu hỏi là số lẻ, câu trả lời là số chẵn liền sau.
     */
    private function messageId(Consultation $consultation, bool $isUser): int
    {
        return $isUser
            ? $consultation->id * 2 - 1
            : $consultation->id * 2;
    }
}
