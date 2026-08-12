<?php

namespace App\Http\Controllers\Api;

use App\Http\Controllers\Controller;
use App\Http\Requests\Api\HouseholdAccessRequest;
use App\Http\Requests\Api\StoreChatMessageRequest;
use App\Services\ChatService;
use App\Services\HouseholdService;
use Illuminate\Http\JsonResponse;
use Illuminate\Http\Response;
use OpenApi\Attributes as OA;

class ChatMessageController extends Controller
{
    public function __construct(
        private readonly HouseholdService $householdService,
        private readonly ChatService $chatService,
    ) {}

    #[OA\Get(
        path: '/households/{id}/messages',
        operationId: 'listChatMessages',
        description: 'Trả về danh sách phẳng gồm câu hỏi của người dùng và câu '
            .'trả lời của AI, sắp xếp theo created_at tăng dần.',
        summary: 'Lấy lịch sử hội thoại Chatbot',
        security: [[], ['bearerAuth' => []]],
        tags: ['Chat'],
        parameters: [
            new OA\Parameter(name: 'id', in: 'path', required: true, schema: new OA\Schema(type: 'integer')),
            new OA\Parameter(name: 'guest_session_id', in: 'query', required: false, schema: new OA\Schema(type: 'string', maxLength: 64)),
        ],
        responses: [
            new OA\Response(
                response: 200,
                description: 'Thành công',
                content: new OA\JsonContent(
                    properties: [
                        new OA\Property(property: 'status', type: 'boolean', example: true),
                        new OA\Property(property: 'message', type: 'string'),
                        new OA\Property(
                            property: 'result',
                            properties: [
                                new OA\Property(
                                    property: 'data',
                                    type: 'array',
                                    items: new OA\Items(ref: '#/components/schemas/ChatMessage')
                                ),
                            ],
                            type: 'object'
                        ),
                    ],
                    type: 'object'
                )
            ),
            new OA\Response(response: 403, description: 'Hồ sơ không thuộc về người gọi'),
            new OA\Response(response: 404, description: 'Không tìm thấy hồ sơ'),
        ]
    )]
    public function index(HouseholdAccessRequest $request, int $id): JsonResponse
    {
        $household = $this->householdService->findOwned(
            $id,
            $request->resolvedUser(),
            $request->guestSessionId()
        );

        return $this->successResponse(
            $this->chatService->history($household),
            __('lang.Messages_fetched')
        );
    }

    #[OA\Post(
        path: '/households/{id}/messages',
        operationId: 'storeChatMessage',
        description: 'Chuyển câu hỏi sang service tư vấn của nhóm Python rồi lưu '
            .'cả câu hỏi lẫn câu trả lời. Trả về cả hai để FE render một lần. '
            .'Chưa cấu hình PYTHON_ADVISOR_URL thì trả 503 và không ghi gì vào DB.',
        summary: 'Gửi câu hỏi tự do cho Chatbot',
        security: [[], ['bearerAuth' => []]],
        requestBody: new OA\RequestBody(
            required: true,
            content: new OA\JsonContent(
                required: ['content'],
                properties: [
                    new OA\Property(property: 'content', type: 'string', maxLength: 2000, example: 'Gia đình tôi nên trả nợ trước hay tiết kiệm trước?'),
                    new OA\Property(property: 'guest_session_id', type: 'string', maxLength: 64, nullable: true, description: 'Bắt buộc khi chưa đăng nhập.'),
                ],
                type: 'object'
            )
        ),
        tags: ['Chat'],
        parameters: [
            new OA\Parameter(name: 'id', in: 'path', required: true, schema: new OA\Schema(type: 'integer')),
        ],
        responses: [
            new OA\Response(
                response: 201,
                description: 'Gửi thành công',
                content: new OA\JsonContent(
                    properties: [
                        new OA\Property(property: 'status', type: 'boolean', example: true),
                        new OA\Property(property: 'message', type: 'string'),
                        new OA\Property(
                            property: 'result',
                            properties: [
                                new OA\Property(
                                    property: 'data',
                                    properties: [
                                        new OA\Property(property: 'user_message', ref: '#/components/schemas/ChatMessage'),
                                        new OA\Property(property: 'ai_message', ref: '#/components/schemas/ChatMessage'),
                                    ],
                                    type: 'object'
                                ),
                            ],
                            type: 'object'
                        ),
                    ],
                    type: 'object'
                )
            ),
            new OA\Response(response: 403, description: 'Hồ sơ không thuộc về người gọi'),
            new OA\Response(response: 422, description: 'Dữ liệu không hợp lệ'),
            new OA\Response(response: 503, description: 'Service tư vấn AI chưa cấu hình hoặc không phản hồi'),
        ]
    )]
    public function store(StoreChatMessageRequest $request, int $id): JsonResponse
    {
        $household = $this->householdService->findOwned(
            $id,
            $request->resolvedUser(),
            $request->guestSessionId()
        );

        $messages = $this->chatService->send($household, $request->validated('content'));

        return $this->successResponse($messages, __('lang.Message_sent'), Response::HTTP_CREATED);
    }
}
