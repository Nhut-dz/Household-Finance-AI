<?php

namespace App\Services;

use App\Enums\PackageTypeEnum;
use App\Models\Consultation;
use App\Models\Household;
use App\Models\RecommendationPackage;
use Illuminate\Database\Eloquent\ModelNotFoundException;
use Illuminate\Support\Collection;

/**
 * Dựng dữ liệu cho trang "Phương án đề xuất".
 *
 * Số liệu do nhóm Python ghi vào tblcalculation_results và
 * tblrecommendation_packages; service này chỉ đọc lượt tư vấn gần nhất có kết
 * quả tính toán rồi chuyển sang đúng cấu trúc FE cần.
 */
class ProposalService
{
    /**
     * Số tháng chi phí sinh hoạt tối thiểu mà quỹ dự phòng nên đạt.
     */
    private const EMERGENCY_FUND_MONTHS = 3;

    /**
     * @return array<string, mixed>
     *
     * @throws ModelNotFoundException khi hồ sơ chưa được tính phương án.
     */
    public function forHousehold(Household $household): array
    {
        $consultation = $this->latestCalculatedConsultation($household);
        $calculation = $consultation->calculationResult;
        $packages = $consultation->recommendationPackages->keyBy(
            fn (RecommendationPackage $package) => $package->package_type->value
        );

        return [
            'household_id' => $household->id,
            'summary' => $consultation->aiResponse?->response_text,
            'overview' => [
                'net_income' => $this->netIncome($household),
                'current_debt' => (float) $household->total_debt,
                'current_savings' => (float) $household->current_savings,
                'target_accumulation' => $this->targetAccumulation($household),
            ],
            'savings_plan' => $this->savingsPlan($packages, $calculation?->recommended_monthly_saving),
            'investment_plan' => $this->investmentPlan($packages),
            'loan_plan' => $this->loanPlan($packages, $calculation?->safe_loan_limit),
            'roadmap' => $this->roadmap($household, $calculation?->raw_json ?? []),
            'generated_at' => $calculation?->created_at,
        ];
    }

    /**
     * Lượt tư vấn gần nhất đã có kết quả tính toán.
     */
    private function latestCalculatedConsultation(Household $household): Consultation
    {
        $consultation = $household->consultations()
            ->has('calculationResult')
            ->with(['calculationResult', 'recommendationPackages', 'aiResponse'])
            ->latest('id')
            ->first();

        if ($consultation === null) {
            throw (new ModelNotFoundException)->setModel(Consultation::class);
        }

        return $consultation;
    }

    /**
     * Thu nhập ròng hàng tháng = thu nhập - chi tiêu - trả nợ.
     */
    private function netIncome(Household $household): float
    {
        return (float) $household->monthly_income
            - (float) ($household->monthly_living_cost ?? 0)
            - (float) $household->monthly_debt_payment;
    }

    /**
     * Tổng nhu cầu tích lũy, cộng từ target_amount của các nhu cầu tài chính.
     */
    private function targetAccumulation(Household $household): ?float
    {
        $total = $household->financialGoals->sum(fn ($goal) => (float) ($goal->target_amount ?? 0));

        return $total > 0 ? (float) $total : null;
    }

    /**
     * @param  Collection<string, RecommendationPackage>  $packages
     * @return array<string, mixed>|null
     */
    private function savingsPlan(Collection $packages, mixed $recommendedMonthlySaving): ?array
    {
        $package = $packages->get(PackageTypeEnum::SAVING->value);

        if ($package === null) {
            return null;
        }

        $content = $package->content ?? [];

        return [
            'monthly_contribution' => $this->toFloat(
                $content['monthly_contribution'] ?? $recommendedMonthlySaving
            ),
            'term_months' => isset($content['term_months']) ? (int) $content['term_months'] : null,
            // Lãi suất chưa có trong dữ liệu do Python ghi ra; trả null để FE ẩn.
            'interest_rate' => $this->toFloat($content['interest_rate'] ?? null),
            'note' => $content['explanation'] ?? null,
        ];
    }

    /**
     * @param  Collection<string, RecommendationPackage>  $packages
     * @return array<string, mixed>|null
     */
    private function investmentPlan(Collection $packages): ?array
    {
        $package = $packages->get(PackageTypeEnum::INVESTMENT->value);

        if ($package === null) {
            return null;
        }

        $content = $package->content ?? [];

        return [
            'allocation' => $this->formatAllocation($content['allocation'] ?? null),
            'risk_level' => $this->riskLabel($content['risk_level'] ?? null),
            'note' => $content['social_safeguard'] ?? null,
        ];
    }

    /**
     * @param  Collection<string, RecommendationPackage>  $packages
     * @return array<string, mixed>|null
     */
    private function loanPlan(Collection $packages, mixed $safeLoanLimit): ?array
    {
        $package = $packages->get(PackageTypeEnum::LOAN->value);

        if ($package === null) {
            return null;
        }

        $content = $package->content ?? [];

        return [
            'product' => $package->title,
            'credit_limit' => $this->toFloat($safeLoanLimit),
            'monthly_payment' => $this->toFloat($content['safe_new_monthly_payment'] ?? null),
            'note' => $content['recommended_action'] ?? null,
        ];
    }

    /**
     * Tỉ trọng danh mục đầu tư dạng "cổ phiếu / trái phiếu / tiền gửi".
     *
     * @param  array<string, mixed>|null  $allocation
     */
    private function formatAllocation(?array $allocation): ?string
    {
        if ($allocation === null) {
            return null;
        }

        $parts = collect(['equity_etf_pct', 'bond_fund_pct', 'cash_deposit_pct'])
            ->filter(fn (string $key) => isset($allocation[$key]))
            ->map(fn (string $key) => ((float) $allocation[$key]).'%');

        return $parts->isEmpty() ? null : $parts->implode(' / ');
    }

    /**
     * Nhãn tiếng Việt cho mức rủi ro do Python trả về.
     */
    private function riskLabel(?string $level): ?string
    {
        return match (strtoupper((string) $level)) {
            'LOW' => 'Rủi ro thấp',
            'MEDIUM' => 'Rủi ro vừa phải',
            'HIGH' => 'Rủi ro cao',
            default => null,
        };
    }

    /**
     * Lộ trình hành động.
     *
     * Ưu tiên dùng roadmap do nhóm Python ghi vào raw_json; nếu chưa có thì sinh
     * bằng luật đơn giản dựa trên chính số liệu của hộ, để FE luôn có nội dung
     * hiển thị thay cho phần đang hard-code.
     *
     * @param  array<string, mixed>  $rawJson
     * @return array<int, array{step: int, text: string}>
     */
    private function roadmap(Household $household, array $rawJson): array
    {
        if (isset($rawJson['roadmap']) && is_array($rawJson['roadmap'])) {
            return collect($rawJson['roadmap'])
                ->values()
                ->map(fn (mixed $item, int $index) => [
                    'step' => $index + 1,
                    'text' => is_array($item) ? ($item['text'] ?? '') : (string) $item,
                ])
                ->all();
        }

        $livingCost = (float) ($household->monthly_living_cost ?? 0);
        $steps = [];

        if ((float) $household->current_savings < $livingCost * self::EMERGENCY_FUND_MONTHS) {
            $steps[] = 'Tăng quỹ dự phòng rủi ro lên tối thiểu 3-6 tháng chi phí sinh hoạt.';
        }

        if ($household->has_debt) {
            $steps[] = 'Giảm nợ ngắn hạn để cải thiện chỉ số DTI và tối ưu lãi suất vay.';
        }

        $surplus = $this->netIncome($household);

        if ($surplus > 0) {
            $steps[] = sprintf(
                'Duy trì tiết kiệm %sđ/tháng để đạt mục tiêu tích lũy đúng kế hoạch.',
                number_format($surplus, 0, ',', '.')
            );
        }

        return collect($steps)
            ->map(fn (string $text, int $index) => ['step' => $index + 1, 'text' => $text])
            ->all();
    }

    private function toFloat(mixed $value): ?float
    {
        return $value === null ? null : (float) $value;
    }
}
