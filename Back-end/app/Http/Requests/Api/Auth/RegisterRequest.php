<?php

namespace App\Http\Requests\Api\Auth;

use App\Http\Requests\BaseRequest;
use Illuminate\Validation\Rules\Password;

class RegisterRequest extends BaseRequest
{
    /**
     * @return array<string, mixed>
     */
    public function rules(): array
    {
        return [
            'name' => ['required', 'string', 'max:150'],
            'email' => ['required', 'string', 'email', 'max:191', 'unique:tblusers,email'],
            'password' => ['required', 'confirmed', Password::min(8)],
            // Cho phép gán các hồ sơ đã nhập lúc chưa đăng nhập về tài khoản mới.
            'guest_session_id' => ['nullable', 'string', 'max:64'],
        ];
    }

    /**
     * @return array<string, string>
     */
    public function attributes(): array
    {
        return [
            'name' => 'họ và tên',
            'email' => 'email',
            'password' => 'mật khẩu',
        ];
    }
}
