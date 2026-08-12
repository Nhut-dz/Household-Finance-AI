<?php

namespace App\OpenApi;

use OpenApi\Attributes as OA;

/**
 * Lớp chỉ dùng để khai báo schema cho tài liệu Swagger của tin nhắn Chatbot.
 * Dữ liệu thật do App\Services\ChatService dựng ra.
 */
#[OA\Schema(
    schema: 'ChatMessage',
    title: 'Tin nhắn trong hội thoại Chatbot',
    properties: [
        new OA\Property(
            property: 'id',
            description: 'Suy ra từ id lượt tư vấn: câu hỏi là số lẻ, câu trả lời là số chẵn liền sau.',
            type: 'integer',
            example: 3
        ),
        new OA\Property(property: 'role', type: 'string', enum: ['user', 'ai'], example: 'user'),
        new OA\Property(property: 'content', type: 'string'),
        new OA\Property(property: 'created_at', type: 'string', format: 'date-time'),
        new OA\Property(
            property: 'suggested_questions',
            description: 'Chỉ có ở message role = ai.',
            type: 'array',
            items: new OA\Items(type: 'string'),
            nullable: true
        ),
    ],
    type: 'object'
)]
final class ChatMessageSchema {}
