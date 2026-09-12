<?php

namespace App\Exceptions;

use Illuminate\Http\JsonResponse;
use Illuminate\Http\Response;
use RuntimeException;

/**
 * Ném ra khi chưa đánh giá được rủi ro khoản vay (ML02) vì dữ liệu khoản vay
 * thiếu hoặc không hợp lệ.
 *
 * Cùng lý do với MissingBirthYearException: đây là chuyện DỮ LIỆU của người
 * dùng — hộ chưa khai màn "Thông tin khoản vay" — không phải service hỏng.
 * Trả 422 kèm field để FE hướng thẳng sang màn nhập, thay vì 503 chung với
 * AdvisorUnavailableException khiến người dùng tưởng hệ thống đang sập.
 */
class MissingLoanApplicationException extends RuntimeException
{
    public function render(): JsonResponse
    {
        return response()->json([
            'status' => false,
            'message' => $this->getMessage(),
            'result' => ['errors' => ['loan_application' => [$this->getMessage()]]],
        ], Response::HTTP_UNPROCESSABLE_ENTITY);
    }
}
