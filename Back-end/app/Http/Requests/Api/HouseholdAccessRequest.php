<?php

namespace App\Http\Requests\Api;

use App\Http\Requests\BaseRequest;
use Illuminate\Validation\Rule;

/**
 * Dùng cho các endpoint chỉ cần định danh người gọi (GET / DELETE),
 * không có body nghiệp vụ.
 */
class HouseholdAccessRequest extends BaseRequest
{
    /**
     * @return array<string, mixed>
     */
    public function rules(): array
    {
        return [
            'guest_session_id' => [
                Rule::requiredIf(fn () => $this->resolvedUser() === null),
                'nullable', 'string', 'max:64',
            ],
        ];
    }

    /**
     * @return array<string, string>
     */
    public function messages(): array
    {
        return [
            'guest_session_id.required' => 'Cần guest_session_id khi chưa đăng nhập.',
        ];
    }
}
