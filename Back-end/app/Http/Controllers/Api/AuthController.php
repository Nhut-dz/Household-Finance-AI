<?php

namespace App\Http\Controllers\Api;

use App\Http\Controllers\Controller;
use App\Http\Requests\Api\Auth\LoginRequest;
use App\Http\Requests\Api\Auth\RegisterRequest;
use App\Http\Resources\UserResource;
use App\Services\AuthService;
use Illuminate\Http\JsonResponse;
use Illuminate\Http\Request;
use Illuminate\Http\Response;
use OpenApi\Attributes as OA;

class AuthController extends Controller
{
    public function __construct(private readonly AuthService $authService) {}

    #[OA\Post(
        path: '/auth/register',
        operationId: 'register',
        description: 'Gửi kèm guest_session_id để backend gán các hồ sơ đã nhập '
            .'lúc chưa đăng nhập về tài khoản vừa tạo.',
        summary: 'Đăng ký tài khoản',
        requestBody: new OA\RequestBody(
            required: true,
            content: new OA\JsonContent(
                required: ['name', 'email', 'password', 'password_confirmation'],
                properties: [
                    new OA\Property(property: 'name', type: 'string', maxLength: 150, example: 'Nguyễn Văn A'),
                    new OA\Property(property: 'email', type: 'string', format: 'email', example: 'a@example.com'),
                    new OA\Property(property: 'password', type: 'string', minLength: 8, example: 'secret123'),
                    new OA\Property(property: 'password_confirmation', type: 'string', example: 'secret123'),
                    new OA\Property(property: 'guest_session_id', type: 'string', maxLength: 64, nullable: true),
                ],
                type: 'object'
            )
        ),
        tags: ['Auth'],
        responses: [
            new OA\Response(response: 201, description: 'Đăng ký thành công', content: new OA\JsonContent(ref: '#/components/schemas/AuthToken')),
            new OA\Response(response: 422, description: 'Dữ liệu không hợp lệ'),
        ]
    )]
    public function register(RegisterRequest $request): JsonResponse
    {
        return $this->successResponse(
            $this->presentToken($this->authService->register($request->validated())),
            __('lang.Registered'),
            Response::HTTP_CREATED
        );
    }

    #[OA\Post(
        path: '/auth/login',
        operationId: 'login',
        summary: 'Đăng nhập',
        requestBody: new OA\RequestBody(
            required: true,
            content: new OA\JsonContent(
                required: ['email', 'password'],
                properties: [
                    new OA\Property(property: 'email', type: 'string', format: 'email', example: 'a@example.com'),
                    new OA\Property(property: 'password', type: 'string', example: 'secret123'),
                    new OA\Property(property: 'guest_session_id', type: 'string', maxLength: 64, nullable: true),
                ],
                type: 'object'
            )
        ),
        tags: ['Auth'],
        responses: [
            new OA\Response(response: 200, description: 'Đăng nhập thành công', content: new OA\JsonContent(ref: '#/components/schemas/AuthToken')),
            new OA\Response(response: 422, description: 'Sai thông tin đăng nhập'),
        ]
    )]
    public function login(LoginRequest $request): JsonResponse
    {
        return $this->successResponse(
            $this->presentToken($this->authService->login($request->validated())),
            __('lang.Logged_in')
        );
    }

    #[OA\Post(
        path: '/auth/logout',
        operationId: 'logout',
        description: 'Thu hồi đúng token đang dùng cho request hiện tại.',
        summary: 'Đăng xuất',
        security: [['bearerAuth' => []]],
        tags: ['Auth'],
        responses: [
            new OA\Response(response: 200, description: 'Đăng xuất thành công'),
            new OA\Response(response: 401, description: 'Chưa xác thực'),
        ]
    )]
    public function logout(Request $request): JsonResponse
    {
        $this->authService->logout($request->user());

        return $this->successResponse(null, __('lang.Logged_out'));
    }

    /**
     * @param  array<string, mixed>  $payload
     * @return array<string, mixed>
     */
    private function presentToken(array $payload): array
    {
        return [
            'token' => $payload['token'],
            'token_type' => $payload['token_type'],
            'user' => new UserResource($payload['user']),
        ];
    }
}
