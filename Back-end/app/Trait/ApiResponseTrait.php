<?php

namespace App\Trait;

use Illuminate\Http\JsonResponse;
use Illuminate\Pagination\LengthAwarePaginator;
use Illuminate\Http\Response;

use Illuminate\Contracts\Support\Arrayable;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Support\Collection;
use Illuminate\Pagination\AbstractPaginator;
use Illuminate\Http\Resources\Json\JsonResource;

/**
 * Trait ApiResponseTrait
 * @package App\Traits
 */
trait ApiResponseTrait
{
    /**
     * Tạo response thành công.
     *
     * @param mixed       $data Dữ liệu trả về, có thể là Paginator, mảng, đối tượng, etc.
     * @param string      $message Thông điệp.
     * @param int         $statusCode Mã HTTP status.
     * @return JsonResponse
     */
    protected function successResponse($data = null, ?string $message = null, $statusCode = 200): JsonResponse
    {
        $message = $message ?? __('lang.Send_successfully');
        $data = self::convertToArray($data);
        $response = [];
        $response['status'] = $data['status'] ?? true;
        unset($data['status']);
        $response['message'] = $data['message'] ?? $message;
        unset($data['message']);
        if (isset($data['result']['data'])) {
            $response['result'] = ['data' => $data['result']['data']];
        } else if (isset($data['result'])) {
            $response['result'] = $data['result'];
        } else if (isset($data['data'])) {
            $response['result'] = $data;
        } else {
            $response['result'] = ['data' => $data];
        }
        return response()->json($response, $statusCode);
    }
    /**
     * Tạo response lỗi.
     *
     * @param string      $message Thông điệp lỗi.
     * @param int         $statusCode Mã HTTP status.
     * @param array|null  $errors Mảng chứa các chi tiết lỗi (ví dụ: validation errors).
     * @return JsonResponse
     */
    protected function errorResponse(?string $message = null, int $statusCode = Response::HTTP_BAD_REQUEST, ?array $errors = null): JsonResponse
    {
        $message = $message ?? __('lang.Send_failed');
        $response = [
            'status' => false,
            'message' => $message,
        ];

        if (!is_null($errors) && !empty($errors)) {
            $response['result']['errors'] = $errors;
        }

        return response()->json($response, $statusCode);
    }

    static function convertToArray($data): array
    {
        // Null => []
        if (is_null($data)) {
            return [];
        }

        // Laravel Resource (JsonResource)
        if ($data instanceof JsonResource) {
            return $data->resolve();
        }

        // Laravel Collection (gồm cả Eloquent\Collection, với() -> get())
        if ($data instanceof Collection) {
            return $data->toArray();
        }

        if ($data instanceof LengthAwarePaginator) {
            $data = [
                'data' => $data->items(),
                'pagination' => [
                    'total' => $data->total(),
                    'per_page' => $data->perPage(),
                    'current_page' => $data->currentPage(),
                    'last_page' => $data->lastPage(),
                    'from' => $data->firstItem(),
                    'to' => $data->lastItem(),
                    'prev_page_url' => $data->previousPageUrl(),
                    'next_page_url' => $data->nextPageUrl(),
                ]
            ];
        }

        // Laravel Model (kể cả có with() hay không)
        if ($data instanceof Model) {
            return $data->toArray();
        }

        // Object implement Arrayable (custom object)
        if ($data instanceof Arrayable) {
            return $data->toArray();
        }

        // Object thường (stdClass, anonymous class, json decode)
        if (is_object($data)) {
            return (array) $data;
        }

        // Mảng sẵn
        if (is_array($data)) {
            return $data;
        }
        // Kiểu còn lại (string, int, bool, float, v.v.)
        return [$data];
    }
}
