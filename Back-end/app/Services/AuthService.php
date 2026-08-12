<?php

namespace App\Services;

use App\Models\Household;
use App\Models\User;
use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\Hash;
use Illuminate\Validation\ValidationException;

/**
 * Đăng ký, đăng nhập và đăng xuất bằng Sanctum personal access token.
 */
class AuthService
{
    /**
     * Tên gán cho token phát ra, để dễ thu hồi theo thiết bị sau này.
     */
    private const TOKEN_NAME = 'api';

    /**
     * @param  array<string, mixed>  $data
     * @return array<string, mixed>
     */
    public function register(array $data): array
    {
        $user = DB::transaction(function () use ($data) {
            $user = User::create([
                'full_name' => $data['name'],
                'email' => $data['email'],
                'password_hash' => $data['password'],
            ]);

            $this->claimGuestHouseholds($user, $data['guest_session_id'] ?? null);

            return $user;
        });

        return $this->tokenPayload($user);
    }

    /**
     * @param  array<string, mixed>  $data
     * @return array<string, mixed>
     *
     * @throws ValidationException khi email hoặc mật khẩu không đúng.
     */
    public function login(array $data): array
    {
        $user = User::where('email', $data['email'])->first();

        if ($user === null || ! Hash::check($data['password'], $user->getAuthPassword())) {
            throw ValidationException::withMessages([
                'email' => ['Email hoặc mật khẩu không đúng.'],
            ]);
        }

        $this->claimGuestHouseholds($user, $data['guest_session_id'] ?? null);

        return $this->tokenPayload($user);
    }

    /**
     * Thu hồi đúng token đang dùng cho request hiện tại.
     */
    public function logout(User $user): void
    {
        $user->currentAccessToken()?->delete();
    }

    /**
     * Gán các hồ sơ đã tạo lúc chưa đăng nhập về tài khoản vừa xác thực, để
     * người dùng không mất dữ liệu đã nhập ở chế độ Guest.
     */
    private function claimGuestHouseholds(User $user, ?string $guestSessionId): void
    {
        if (blank($guestSessionId)) {
            return;
        }

        Household::whereNull('user_id')
            ->where('session_token', $guestSessionId)
            ->update(['user_id' => $user->getKey()]);
    }

    /**
     * @return array<string, mixed>
     */
    private function tokenPayload(User $user): array
    {
        return [
            'token' => $user->createToken(self::TOKEN_NAME)->plainTextToken,
            'token_type' => 'Bearer',
            'user' => $user,
        ];
    }
}
