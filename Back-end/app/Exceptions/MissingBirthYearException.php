<?php

namespace App\Exceptions;

use Illuminate\Http\JsonResponse;
use Illuminate\Http\Response;
use RuntimeException;

/**
 * Ném ra khi hồ sơ thiếu năm sinh nên chưa dự đoán được bằng ML01.
 *
 * Model ML01 có `age` trong 17 feature bắt buộc, còn `birth_year` trong DB thì
 * cho phép bỏ trống. Đây là lỗi DỮ LIỆU của người dùng, không phải service
 * hỏng — nên trả 422 kèm field cụ thể để form biết ô nào cần điền, chứ không
 * dùng chung 503 với `AdvisorUnavailableException`.
 */
class MissingBirthYearException extends RuntimeException
{
    /**
     * Trả 422 theo đúng khung của ApiResponseTrait, kèm lỗi theo field.
     */
    public function render(): JsonResponse
    {
        return response()->json([
            'status' => false,
            'message' => $this->getMessage(),
            'result' => ['errors' => ['birth_year' => [$this->getMessage()]]],
        ], Response::HTTP_UNPROCESSABLE_ENTITY);
    }
}
