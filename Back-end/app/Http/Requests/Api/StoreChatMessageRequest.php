<?php

namespace App\Http\Requests\Api;

use App\Http\Requests\BaseRequest;
use Illuminate\Validation\Rule;

class StoreChatMessageRequest extends BaseRequest
{
    /**
     * @return array<string, mixed>
     */
    public function rules(): array
    {
        return [
            'content' => ['required', 'string', 'max:2000'],
            'guest_session_id' => [
                Rule::requiredIf(fn () => $this->resolvedUser() === null),
                'nullable', 'string', 'max:64',
            ],
        ];
    }

    /**
     * @return array<string, string>
     */
    public function attributes(): array
    {
        return [
            'content' => 'nội dung câu hỏi',
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
