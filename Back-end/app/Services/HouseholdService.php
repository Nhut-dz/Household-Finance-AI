<?php

namespace App\Services;

use App\Enums\AssetTypeEnum;
use App\Enums\GoalTypeEnum;
use App\Models\Household;
use App\Models\User;
use Illuminate\Auth\Access\AuthorizationException;
use Illuminate\Database\Eloquent\ModelNotFoundException;
use Illuminate\Support\Facades\DB;

/**
 * Nghiệp vụ hồ sơ tài chính hộ gia đình cho màn "Nhập thông tin".
 */
class HouseholdService
{
    /**
     * Quan hệ luôn nạp kèm khi trả hồ sơ về cho FE.
     *
     * @var array<int, string>
     */
    private const WITH_RELATIONS = ['assets', 'financialGoals'];

    /**
     * Lưu hồ sơ mới cùng tài sản và nhu cầu tài chính trong một transaction.
     *
     * @param  array<string, mixed>  $data  Dữ liệu đã qua StoreHouseholdRequest.
     * @param  User|null  $user  User đăng nhập, null nếu là Guest.
     */
    public function store(array $data, ?User $user = null): Household
    {
        return DB::transaction(function () use ($data, $user) {
            $household = Household::create($this->mapToColumns($data, $user));

            $this->replaceAssets($household, $data['assets'] ?? []);
            $this->replaceFinancialGoals($household, $data['financial_needs'] ?? []);

            return $household->load(self::WITH_RELATIONS);
        });
    }

    /**
     * Cập nhật hồ sơ. Tài sản và nhu cầu tài chính được ghi đè theo danh sách
     * mới, không cộng dồn với lần lưu trước.
     *
     * @param  array<string, mixed>  $data
     */
    public function update(Household $household, array $data, ?User $user = null): Household
    {
        return DB::transaction(function () use ($household, $data, $user) {
            $columns = $this->mapToColumns($data, $user);

            // Giữ nguyên chủ sở hữu của bản ghi, tránh việc sửa hồ sơ làm mất
            // liên kết user_id / session_token ban đầu.
            unset($columns['user_id'], $columns['session_token']);

            $household->update($columns);

            $this->replaceAssets($household, $data['assets'] ?? []);
            $this->replaceFinancialGoals($household, $data['financial_needs'] ?? []);

            return $household->fresh(self::WITH_RELATIONS);
        });
    }

    /**
     * Xoá hồ sơ. Tài sản, nhu cầu tài chính và lịch sử hội thoại tự động bị xoá
     * theo nhờ ràng buộc ON DELETE CASCADE của PostgreSQL.
     */
    public function delete(Household $household): void
    {
        $household->delete();
    }

    /**
     * Hồ sơ gần nhất của người gọi, dùng để khôi phục màn Chatbot sau khi FE
     * tải lại trang.
     *
     * @throws ModelNotFoundException khi người gọi chưa từng gửi hồ sơ nào.
     */
    public function findLatestFor(?User $user, ?string $guestSessionId): Household
    {
        $query = Household::query()->with(self::WITH_RELATIONS)->latest('id');

        if ($user !== null) {
            $query->where('user_id', $user->getKey());
        } else {
            $query->whereNull('user_id')->where('session_token', $guestSessionId);
        }

        $household = $query->first();

        if ($household === null) {
            throw (new ModelNotFoundException)->setModel(Household::class);
        }

        return $household;
    }

    /**
     * Lấy hồ sơ theo id và đảm bảo người gọi đúng là chủ sở hữu.
     *
     * @throws ModelNotFoundException khi id không tồn tại (404).
     * @throws AuthorizationException khi hồ sơ thuộc về người khác (403).
     */
    public function findOwned(int $id, ?User $user, ?string $guestSessionId): Household
    {
        $household = Household::with(self::WITH_RELATIONS)->findOrFail($id);

        if (! $this->isOwnedBy($household, $user, $guestSessionId)) {
            throw new AuthorizationException('Hồ sơ này không thuộc về bạn.');
        }

        return $household;
    }

    /**
     * User đăng nhập sở hữu hồ sơ qua user_id; Guest sở hữu qua session_token.
     */
    private function isOwnedBy(Household $household, ?User $user, ?string $guestSessionId): bool
    {
        if ($user !== null) {
            return $household->user_id === $user->getKey();
        }

        return $guestSessionId !== null
            && $household->user_id === null
            && $household->session_token === $guestSessionId;
    }

    /**
     * Ánh xạ field của form sang cột thật của bảng tblhouseholds.
     *
     * @param  array<string, mixed>  $data
     * @return array<string, mixed>
     */
    private function mapToColumns(array $data, ?User $user): array
    {
        $hasDebt = (bool) $data['has_debt'];
        $hasSavings = (bool) $data['has_savings'];

        return [
            'user_id' => $user?->getKey(),
            'session_token' => $data['guest_session_id'] ?? null,
            'representative_name' => $data['representative_name'],
            'birth_year' => $data['birth_year'] ?? null,
            'location' => $data['residence'] ?? null,
            'household_size' => $data['household_size'],
            'children_count' => $data['children_count'],
            'monthly_income' => $data['average_monthly_income'],
            'monthly_living_cost' => $data['average_monthly_expense'] ?? null,
            'supports_elderly' => (bool) $data['has_dependents'],
            'has_debt' => $hasDebt,
            // Bỏ chọn "Có nợ" thì các số liệu nợ luôn về 0, tránh dữ liệu mồ côi.
            'total_debt' => $hasDebt ? ($data['total_current_debt'] ?? 0) : 0,
            'monthly_debt_payment' => $hasDebt ? ($data['monthly_debt_payment'] ?? 0) : 0,
            'has_savings' => $hasSavings,
            'current_savings' => $hasSavings ? ($data['savings_amount'] ?? 0) : 0,
        ];
    }

    /**
     * @param  array<int, string>  $assetTypes
     */
    private function replaceAssets(Household $household, array $assetTypes): void
    {
        $household->assets()->delete();

        if ($assetTypes === []) {
            return;
        }

        $household->assets()->createMany(
            collect($assetTypes)
                ->unique()
                ->map(fn (string $type) => [
                    'asset_type' => $type,
                    'description' => AssetTypeEnum::from($type)->label(),
                ])
                ->all()
        );
    }

    /**
     * Thứ tự người dùng chọn trên form được dùng làm độ ưu tiên của nhu cầu.
     *
     * @param  array<int, string>  $goalTypes
     */
    private function replaceFinancialGoals(Household $household, array $goalTypes): void
    {
        $household->financialGoals()->delete();

        if ($goalTypes === []) {
            return;
        }

        $household->financialGoals()->createMany(
            collect($goalTypes)
                ->unique()
                ->values()
                ->map(fn (string $type, int $index) => [
                    'goal_type' => $type,
                    'description' => GoalTypeEnum::from($type)->label(),
                    'priority' => $index + 1,
                ])
                ->all()
        );
    }
}
