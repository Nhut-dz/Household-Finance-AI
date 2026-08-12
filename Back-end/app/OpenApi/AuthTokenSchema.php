<?php

namespace App\OpenApi;

use OpenApi\Attributes as OA;

/**
 * Lớp chỉ dùng để khai báo schema phản hồi của đăng ký / đăng nhập.
 */
#[OA\Schema(
    schema: 'AuthToken',
    title: 'Phản hồi đăng nhập',
    properties: [
        new OA\Property(property: 'status', type: 'boolean', example: true),
        new OA\Property(property: 'message', type: 'string', example: 'Đăng nhập thành công.'),
        new OA\Property(
            property: 'result',
            properties: [
                new OA\Property(
                    property: 'data',
                    properties: [
                        new OA\Property(property: 'token', type: 'string', example: '1|xxxxxxxxxxxxxxxxxxxxx'),
                        new OA\Property(property: 'token_type', type: 'string', example: 'Bearer'),
                        new OA\Property(property: 'user', ref: '#/components/schemas/User'),
                    ],
                    type: 'object'
                ),
            ],
            type: 'object'
        ),
    ],
    type: 'object'
)]
final class AuthTokenSchema {}
