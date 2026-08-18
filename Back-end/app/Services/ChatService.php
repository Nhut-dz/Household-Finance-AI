<?php

namespace App\Services;

use App\Enums\IntentCodeEnum;
use App\Models\AiResponse;
use App\Models\Consultation;
use App\Models\Conversation;
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
    public function __construct(
        private readonly AdvisorClient $advisor,
        private readonly ConversationService $conversations,
    ) {}

    /**
     * Hội thoại của PHIÊN ĐANG MỞ, sắp xếp theo thời gian tăng dần.
     *
     * Chỉ phiên đang mở, không phải toàn bộ lịch sử của hộ. Sau khi người dùng
     * sửa số liệu tài chính, phiên cũ đã bị đóng nên các lượt hỏi của nó không
     * còn xuất hiện ở đây — chúng vẫn nằm trong DB và đọc lại được qua
     * `conversationHistory()`.
     *
     * @return array{conversation_id: int|null, messages: array<int, array<string, mixed>>}
     */
    public function history(Household $household): array
    {
        $conversation = $household->activeConversation()->first();

        if ($conversation === null) {
            // Hộ chưa từng hỏi câu nào. Không mở phiên ở đường ĐỌC — mở phiên
            // là việc của đường ghi, để một lần vào xem màn chat không tạo ra
            // phiên rỗng.
            return ['conversation_id' => null, 'messages' => []];
        }

        return [
            'conversation_id' => $conversation->id,
            'messages' => $conversation->consultations()
                ->with('aiResponse')
                ->orderBy('id')
                ->get()
                ->flatMap(fn (Consultation $consultation) => $this->toMessages($consultation))
                ->all(),
        ];
    }

    /**
     * Danh sách các phiên của hộ kèm số lượt hỏi — cho màn xem lại lịch sử.
     *
     * @return array<int, array<string, mixed>>
     */
    public function conversations(Household $household): array
    {
        return $household->conversations()
            ->withCount('consultations')
            ->get()
            ->map(fn (Conversation $conversation) => [
                'id' => $conversation->id,
                'status' => $conversation->status,
                'closed_reason' => $conversation->closed_reason,
                // Số LƯỢT hỏi đáp, không phải số message: một lượt trải ra
                // thành hai message (câu hỏi + câu trả lời) ở `history()`.
                'turn_count' => $conversation->consultations_count,
                'created_at' => $conversation->created_at,
                'closed_at' => $conversation->closed_at,
            ])
            ->all();
    }

    /**
     * Hội thoại của MỘT phiên cụ thể, kể cả phiên đã đóng.
     *
     * @return array<int, array<string, mixed>>
     */
    public function conversationHistory(Conversation $conversation): array
    {
        return $conversation->consultations()
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
     * Lượt hỏi luôn được gắn vào phiên ĐANG MỞ của hộ, lấy ở server chứ không
     * nhận `conversation_id` từ client: client giữ id cũ sau khi hồ sơ đổi là
     * đúng tình huống nghiệp vụ này cấm, và nhận id từ ngoài vào thì cái cấm đó
     * chỉ còn là lời hứa.
     *
     * `$intent` chỉ có khi người dùng bấm một chip gợi ý. Nó KHÔNG được lưu
     * vào tblconsultations: bảng đó ghi lại cuộc hội thoại như người dùng
     * thấy, mà thứ họ thấy là câu hỏi. Mã ý định là chi tiết định tuyến của
     * một lần gọi, lưu lại sẽ thành một trường luôn null với mọi câu tự gõ.
     *
     * @return array{conversation_id: int, user_message: array<string, mixed>, ai_message: array<string, mixed>}
     */
    public function send(Household $household, string $content, ?IntentCodeEnum $intent = null): array
    {
        $conversation = $this->conversations->currentOrNew($household);
        $answer = $this->advisor->ask($household, $content, $intent);

        $consultation = DB::transaction(function () use ($household, $conversation, $content, $answer) {
            $consultation = $household->consultations()->create([
                'conversation_id' => $conversation->id,
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
            'conversation_id' => $conversation->id,
            'user_message' => $this->userMessage($consultation),
            'ai_message' => [
                ...$this->aiMessage($consultation, $consultation->aiResponse),
                // Hai trường của LƯỢT GỌI này, không thuộc bản ghi trong DB nên
                // chỉ có ở response của lần gửi, không có khi tải lại lịch sử.
                'intent_code' => $answer['intent_code'],
                'requires_loan_application' => $answer['requires_loan_application'],
            ],
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
