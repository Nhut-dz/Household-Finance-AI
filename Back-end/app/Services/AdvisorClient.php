<?php

namespace App\Services;

use App\Exceptions\AdvisorUnavailableException;
use App\Exceptions\MissingBirthYearException;
use App\Models\Household;
use Illuminate\Http\Client\ConnectionException;
use Illuminate\Support\Facades\Http;
use Illuminate\Support\Facades\Log;

/**
 * Cầu nối sang service tư vấn tài chính của nhóm Python.
 *
 * Cấu hình qua .env: PYTHON_ADVISOR_URL, PYTHON_ADVISOR_TIMEOUT,
 * PYTHON_ADVISOR_TOKEN. Không cấu hình thì mọi lời gọi trả 503.
 *
 * @phpstan-type AdvisorAnswer array{response_text: string, model_used: string, suggested_questions: array<int, string>|null, tokens_used: int|null}
 */
class AdvisorClient
{
    /**
     * Sáu cột tài sản multi-hot mà ML01 được train. Thứ tự không quan trọng —
     * ML service tự sắp theo `RAW_FEATURES` — nhưng phải đủ cả sáu, vì thiếu
     * một cột là request bị từ chối ở tầng validate.
     */
    private const ML_ASSET_FEATURES = [
        'has_asset_cash',
        'has_asset_vehicle',
        'has_asset_real_estate',
        'has_asset_insurance',
        'has_asset_gold',
        'has_asset_investment',
    ];

    /**
     * `tblassets.asset_type` ↔ cột feature của ML01.
     *
     * Map CẢ HAI bộ từ vựng đang cùng tồn tại: bộ 4 giá trị của
     * `AssetTypeEnum` (house/car/land/other) và bộ 6 giá trị mà ML01 được
     * train. Cột DB là `string(50)` không ràng buộc nên cả hai đều lọt vào
     * được, và một hàng lạ không được phép làm hỏng cả lời gọi.
     *
     * `other` cố ý không map: nó không nói được là loại tài sản gì, mà đoán
     * bừa sang một trong sáu cột thì model nhận thông tin sai.
     */
    private const ASSET_TO_FEATURE = [
        // Bộ của AssetTypeEnum
        'house' => 'has_asset_real_estate',
        'land' => 'has_asset_real_estate',
        'car' => 'has_asset_vehicle',
        // Bộ mà ML01 được train
        'real_estate' => 'has_asset_real_estate',
        'vehicle' => 'has_asset_vehicle',
        'cash' => 'has_asset_cash',
        'insurance' => 'has_asset_insurance',
        'gold' => 'has_asset_gold',
        'investment' => 'has_asset_investment',
    ];

    /**
     * Gửi câu hỏi kèm hồ sơ hộ gia đình và nhận lại câu trả lời của AI.
     *
     * @return AdvisorAnswer
     *
     * @throws AdvisorUnavailableException
     */
    public function ask(Household $household, string $question): array
    {
        $baseUrl = config('services.python_advisor.url');

        if (blank($baseUrl)) {
            throw new AdvisorUnavailableException(
                'Chưa cấu hình service tư vấn AI (PYTHON_ADVISOR_URL).'
            );
        }

        try {
            $response = Http::baseUrl(rtrim((string) $baseUrl, '/'))
                ->timeout((int) config('services.python_advisor.timeout'))
                ->acceptJson()
                ->withToken((string) config('services.python_advisor.token'))
                ->post('/advise', [
                    'question' => $question,
                    'household' => $this->householdPayload($household),
                ]);
        } catch (ConnectionException $e) {
            Log::warning('Không kết nối được service tư vấn AI.', ['error' => $e->getMessage()]);

            throw new AdvisorUnavailableException('Không kết nối được service tư vấn AI.');
        }

        if ($response->failed()) {
            Log::warning('Service tư vấn AI trả về lỗi.', [
                'status' => $response->status(),
                'body' => $response->body(),
            ]);

            throw new AdvisorUnavailableException('Service tư vấn AI đang không phản hồi.');
        }

        return $this->normalize($response->json() ?? []);
    }

    /**
     * Phân loại hộ vào 4 nhóm khuyến nghị bằng model ML01 (F03).
     *
     * Khác `ask()` ở chỗ endpoint này nhận ĐÚNG 17 feature đã chuẩn hoá, không
     * nhận hồ sơ dạng cột DB. Việc quy đổi nằm ở `predictionPayload()`.
     *
     * @return array<string, mixed>
     *
     * @throws AdvisorUnavailableException khi chưa cấu hình hoặc ML không phản hồi
     * @throws MissingBirthYearException khi hồ sơ thiếu năm sinh
     */
    public function predict(Household $household): array
    {
        $baseUrl = config('services.python_advisor.url');

        if (blank($baseUrl)) {
            throw new AdvisorUnavailableException(
                'Chưa cấu hình service ML (PYTHON_ADVISOR_URL).'
            );
        }

        try {
            $response = Http::baseUrl(rtrim((string) $baseUrl, '/'))
                ->timeout((int) config('services.python_advisor.timeout'))
                ->acceptJson()
                ->withToken((string) config('services.python_advisor.token'))
                ->post('/predict', $this->predictionPayload($household));
        } catch (ConnectionException $e) {
            Log::warning('Không kết nối được service ML.', ['error' => $e->getMessage()]);

            throw new AdvisorUnavailableException('Không kết nối được service ML.');
        }

        if ($response->failed()) {
            Log::warning('Service ML trả về lỗi khi dự đoán.', [
                'household_id' => $household->id,
                'status' => $response->status(),
                'body' => $response->body(),
            ]);

            throw new AdvisorUnavailableException(
                'Service ML đang không phản hồi (mã '.$response->status().').'
            );
        }

        return $response->json() ?? [];
    }

    /**
     * Quy đổi hồ sơ DB sang đúng 17 feature mà ML01 được train.
     *
     * Hai chỗ hai bên không khớp nhau, xử lý ở đây:
     *
     * 1. `age` — DB chỉ lưu `birth_year`, và trường đó cho phép bỏ trống. Model
     *    thì bắt buộc có tuổi. Thiếu thì ném lỗi để người dùng bổ sung, KHÔNG
     *    điền tuổi mặc định: điền bừa thì model vẫn trả về một nhãn trông hợp
     *    lý và không ai biết nó dựa trên tuổi bịa.
     *
     * 2. Tài sản — `AssetTypeEnum` của backend có 4 giá trị (house/car/land/
     *    other) còn ML01 được train trên 6 cột (cash/vehicle/real_estate/
     *    insurance/gold/investment). `tblassets.asset_type` là `string(50)`
     *    không ràng buộc nên có thể chứa cả hai bộ; `ASSET_TO_FEATURE` map cả
     *    hai. Giá trị lạ bị bỏ qua, mọi cột không map được để `false`.
     *
     * @return array<string, mixed>
     */
    private function predictionPayload(Household $household): array
    {
        $birthYear = $household->birth_year;

        if ($birthYear === null) {
            throw new MissingBirthYearException(
                'Hồ sơ chưa có năm sinh nên chưa dự đoán được. Vui lòng bổ sung năm sinh.'
            );
        }

        $features = [
            'average_monthly_income' => (float) $household->monthly_income,
            'average_monthly_expense' => (float) ($household->monthly_living_cost ?? 0),
            'savings_amount' => (float) ($household->current_savings ?? 0),
            'total_current_debt' => (float) ($household->total_debt ?? 0),
            'monthly_debt_payment' => (float) ($household->monthly_debt_payment ?? 0),
            'household_size' => (int) $household->household_size,
            'children_count' => (int) $household->children_count,
            'age' => (int) date('Y') - (int) $birthYear,
            'has_debt' => (bool) $household->has_debt,
            'has_savings' => (bool) $household->has_savings,
            'has_dependents' => (bool) $household->supports_elderly,
        ];

        foreach (self::ML_ASSET_FEATURES as $feature) {
            $features[$feature] = false;
        }

        foreach ($household->assets as $asset) {
            // Đọc giá trị THÔ, không qua cast. `Asset::$casts` ép asset_type
            // sang AssetTypeEnum (4 giá trị), trong khi cột DB là string(50)
            // không ràng buộc và thực tế đang chứa 10 giá trị khác nhau —
            // truy cập `$asset->asset_type` với hàng 'cash' hay
            // 'savings_certificate' sẽ ném ValueError và làm sập cả lời gọi.
            $type = (string) $asset->getRawOriginal('asset_type');

            if (isset(self::ASSET_TO_FEATURE[$type])) {
                $features[self::ASSET_TO_FEATURE[$type]] = true;
            }
        }

        return $features;
    }

    /**
     * Hồ sơ gửi sang Python, dùng đúng tên cột trong DB để hai bên khỏi lệch.
     *
     * @return array<string, mixed>
     */
    private function householdPayload(Household $household): array
    {
        return [
            'id' => $household->id,
            'representative_name' => $household->representative_name,
            'birth_year' => $household->birth_year,
            'location' => $household->location,
            'household_size' => $household->household_size,
            'children_count' => $household->children_count,
            'supports_elderly' => $household->supports_elderly,
            'monthly_income' => (float) $household->monthly_income,
            'monthly_living_cost' => (float) ($household->monthly_living_cost ?? 0),
            'has_debt' => $household->has_debt,
            'total_debt' => (float) $household->total_debt,
            'monthly_debt_payment' => (float) $household->monthly_debt_payment,
            'has_savings' => $household->has_savings,
            'current_savings' => (float) $household->current_savings,
        ];
    }

    /**
     * @param  array<string, mixed>  $payload
     * @return AdvisorAnswer
     */
    private function normalize(array $payload): array
    {
        $suggested = $payload['suggested_questions'] ?? null;

        return [
            'response_text' => (string) ($payload['response_text'] ?? $payload['answer'] ?? ''),
            'model_used' => (string) ($payload['model_used'] ?? 'PYTHON-ADVISOR'),
            'suggested_questions' => is_array($suggested) ? array_values($suggested) : null,
            'tokens_used' => isset($payload['tokens_used']) ? (int) $payload['tokens_used'] : null,
        ];
    }
}
