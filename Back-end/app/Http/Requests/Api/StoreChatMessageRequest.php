<?php

namespace App\Http\Requests\Api;

use App\Enums\IntentCodeEnum;
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

            // Chỉ có khi người dùng bấm một chip gợi ý. Câu tự gõ để trống, và
            // khi đó service Python mới đoán ý định bằng từ khoá.
            'intent_code' => ['nullable', Rule::enum(IntentCodeEnum::class)],

            'guest_session_id' => [
                Rule::requiredIf(fn () => $this->resolvedUser() === null),
                'nullable', 'string', 'max:64',
            ],
        ];
    }

    /**
     * Mã ý định đã kiểm, `null` khi là câu hỏi tự gõ.
     */
    public function intentCode(): ?IntentCodeEnum
    {
        $value = $this->input('intent_code');

        return is_string($value) && $value !== ''
            ? IntentCodeEnum::tryFrom($value)
            : null;
    }

    /**
     * @return array<string, string>
     */
    public function attributes(): array
    {
        return [
            'content' => 'nội dung câu hỏi',
            'intent_code' => 'mã chức năng',
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
