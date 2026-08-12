<?php

use Illuminate\Auth\AuthenticationException;
use Illuminate\Foundation\Application;
use Illuminate\Foundation\Configuration\Exceptions;
use Illuminate\Foundation\Configuration\Middleware;
use Illuminate\Http\Request;
use Illuminate\Http\Response;
use Illuminate\Validation\ValidationException;
use Symfony\Component\HttpKernel\Exception\AccessDeniedHttpException;
use Symfony\Component\HttpKernel\Exception\NotFoundHttpException;

/**
 * Dựng phản hồi lỗi theo đúng khung của ApiResponseTrait.
 */
$apiError = static fn (string $message, int $status) => response()->json([
    'status' => false,
    'message' => $message,
], $status);

return Application::configure(basePath: dirname(__DIR__))
    ->withRouting(
        web: __DIR__.'/../routes/web.php',
        api: __DIR__.'/../routes/api.php',
        commands: __DIR__.'/../routes/console.php',
        health: '/up',
    )
    ->withMiddleware(function (Middleware $middleware): void {
        //
    })
    ->withExceptions(function (Exceptions $exceptions) use ($apiError): void {
        $exceptions->shouldRenderJsonWhen(
            fn (Request $request) => $request->is('api/*'),
        );

        $exceptions->renderable(
            fn (AuthenticationException $e, Request $request) => $request->is('api/*')
                ? $apiError('Bạn cần đăng nhập để thực hiện thao tác này.', Response::HTTP_UNAUTHORIZED)
                : null
        );

        // Laravel đổi AuthorizationException thành AccessDeniedHttpException và
        // ModelNotFoundException thành NotFoundHttpException ở prepareException,
        // tức là trước khi các callback dưới đây chạy, nên phải bắt kiểu đã đổi.
        $exceptions->renderable(
            fn (AccessDeniedHttpException $e, Request $request) => $request->is('api/*')
                ? $apiError($e->getMessage() ?: 'Bạn không có quyền truy cập tài nguyên này.', Response::HTTP_FORBIDDEN)
                : null
        );

        $exceptions->renderable(
            fn (NotFoundHttpException $e, Request $request) => $request->is('api/*')
                ? $apiError('Không tìm thấy dữ liệu.', Response::HTTP_NOT_FOUND)
                : null
        );

        // Lỗi validate ném từ tầng Service (ví dụ sai thông tin đăng nhập) cũng
        // phải theo cùng format với lỗi validate của FormRequest.
        $exceptions->renderable(
            fn (ValidationException $e, Request $request) => $request->is('api/*')
                ? response()->json([
                    'status' => false,
                    'message' => __('lang.Validation_failed'),
                    'result' => ['errors' => $e->errors()],
                ], Response::HTTP_UNPROCESSABLE_ENTITY)
                : null
        );
    })->create();
