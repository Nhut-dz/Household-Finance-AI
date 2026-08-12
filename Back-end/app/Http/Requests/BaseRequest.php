<?php

namespace App\Http\Requests;

use App\Models\User;
use App\Trait\ApiResponseTrait;
use Illuminate\Contracts\Validation\Validator;
use Illuminate\Foundation\Http\FormRequest;
use Illuminate\Http\Exceptions\HttpResponseException;
use Illuminate\Http\Response;

/**
 * Request nền cho API, trả lỗi validate theo đúng format của ApiResponseTrait.
 */
abstract class BaseRequest extends FormRequest
{
    use ApiResponseTrait;

    public function authorize(): bool
    {
        return true;
    }

    /**
     * User đang đăng nhập qua Sanctum, null nếu là Guest.
     */
    public function resolvedUser(): ?User
    {
        return $this->user('sanctum');
    }

    /**
     * Mã phiên khách, đọc được từ cả query string lẫn body.
     * Guest đã đăng nhập thì giá trị này bị bỏ qua ở tầng Service.
     */
    public function guestSessionId(): ?string
    {
        $value = $this->input('guest_session_id');

        return is_string($value) && $value !== '' ? $value : null;
    }

    /**
     * Trả 422 theo format { status, message, result.errors } thay vì format mặc định.
     */
    protected function failedValidation(Validator $validator): void
    {
        throw new HttpResponseException(
            $this->errorResponse(
                __('lang.Validation_failed'),
                Response::HTTP_UNPROCESSABLE_ENTITY,
                $validator->errors()->toArray()
            )
        );
    }
}
