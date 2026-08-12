<?php

namespace App\Exceptions;

use Illuminate\Http\JsonResponse;
use Illuminate\Http\Response;
use RuntimeException;

/**
 * Ném ra khi service tư vấn của nhóm Python chưa cấu hình hoặc không phản hồi.
 */
class AdvisorUnavailableException extends RuntimeException
{
    /**
     * Trả 503 theo đúng khung của ApiResponseTrait.
     */
    public function render(): JsonResponse
    {
        return response()->json([
            'status' => false,
            'message' => $this->getMessage(),
        ], Response::HTTP_SERVICE_UNAVAILABLE);
    }
}
