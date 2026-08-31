<?php

namespace App\Services;

use App\Enums\IntentCodeEnum;
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
 * @phpstan-type AdvisorAnswer array{response_text: string, model_used: string, suggested_questions: array<int, string>|null, tokens_used: int|null, intent_code: string|null, requires_loan_application: bool}
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
     * `tblassets.asset_type` ↔ giá trị enum `AssetType` mà `HouseholdProfile`
     * của Python chấp nhận cho trường `assets` (khác đích với
     * `ASSET_TO_FEATURE` ở trên — đó là 6 cột boolean rời cho `/predict`, còn
     * đây là ĐÚNG MỘT giá trị enum cho từng dòng tài sản gửi trong `/advise`).
     *
     * Bắt buộc phải map, không được gửi giá trị thô: `HouseholdProfile` của
     * Python validate nghiêm ngặt (`extra`/enum không khớp → lỗi), và lỗi đó
     * làm hỏng VALIDATION CỦA CẢ HỒ SƠ chứ không riêng gì trường `assets` —
     * hộ có một tài sản kiểu `house` (giá trị hợp lệ của `AssetTypeEnum`) gửi
     * thẳng xuống thì mất luôn cả ML01 lẫn ML02, và người dùng nhận về "Chưa
     * đủ dữ liệu để đánh giá" dù đã điền đủ mọi trường (phát hiện 24/08/2026).
     *
     * `other` cố ý không map, cùng lý do với `ASSET_TO_FEATURE`: không nói
     * được là loại tài sản gì, đoán bừa sang một enum khác là thông tin sai.
     * Giá trị không map được bị LOẠI ở `householdPayload()`, không gửi xuống.
     */
    private const ASSET_TYPE_TO_PYTHON = [
        // Bộ của AssetTypeEnum
        'house' => 'real_estate',
        'land' => 'real_estate',
        'car' => 'vehicle',
        // Bộ mà HouseholdProfile.assets chấp nhận — đi thẳng qua
        'real_estate' => 'real_estate',
        'vehicle' => 'vehicle',
        'cash' => 'cash',
        'insurance' => 'insurance',
        'gold' => 'gold',
        'investment' => 'investment',
    ];

    /**
     * Gửi câu hỏi kèm hồ sơ hộ gia đình và nhận lại câu trả lời của AI.
     *
     * `$intent` chỉ có khi người dùng bấm một chip gợi ý; câu tự gõ để `null`
     * và service Python sẽ tự đoán ý định bằng từ khoá. Hai intent chạy model
     * CHỈ kích hoạt được qua tham số này — chúng cố ý không đoán được từ nhãn
     * tiếng Việt, xem `IntentCodeEnum`.
     *
     * Dữ liệu đính kèm được chọn theo intent chứ không gửi hết mọi lúc: dựng
     * 17 feature ML01 cho một câu hỏi về quy tắc 50/30/20 là công vô ích, và
     * tệ hơn là nó ném lỗi thiếu năm sinh cho một luồng vốn không cần tuổi.
     *
     * @return AdvisorAnswer
     *
     * @throws AdvisorUnavailableException
     */
    public function ask(Household $household, string $question, ?IntentCodeEnum $intent = null): array
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
                    'intent_code' => $intent?->value,
                    'household' => $this->householdPayload($household),
                    'ml_features' => $this->mlFeaturesFor($household, $intent),
                    'loan_application' => $this->loanApplicationFor($household, $intent),
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
     * 17 feature của ML01, chỉ dựng khi intent thực sự cần.
     *
     * Nuốt `MissingBirthYearException` ở đây là CÓ CHỦ Ý: trong luồng chat,
     * hồ sơ thiếu năm sinh không được phép làm hỏng cả câu hỏi. Gửi `null`
     * xuống thì service Python nói rõ thiếu gì và chỉ chỗ bổ sung — vẫn là
     * một câu trả lời dùng được, thay vì một lỗi 500.
     *
     * Ở luồng `/predict` thì ngược lại: ngoại lệ được ném ra tới FE, vì ở đó
     * người dùng đang yêu cầu ĐÚNG kết quả phân loại chứ không phải hội thoại.
     *
     * @return array<string, mixed>|null
     */
    private function mlFeaturesFor(Household $household, ?IntentCodeEnum $intent): ?array
    {
        if ($intent === null || ! $intent->needsMlFeatures()) {
            return null;
        }

        try {
            return $this->predictionPayload($household);
        } catch (MissingBirthYearException $e) {
            Log::info('Bỏ qua ML01 trong luồng chat vì hồ sơ thiếu năm sinh.', [
                'household_id' => $household->id,
            ]);

            return null;
        }
    }

    /**
     * Thông tin khoản vay của hộ, chỉ nạp khi intent thực sự cần.
     *
     * `null` = hộ chưa khai. Đó là trạng thái bình thường chứ không phải lỗi:
     * service Python sẽ hướng người dùng sang màn "Thông tin khoản vay" thay
     * vì chạy ML02 trên dữ liệu rỗng — chạy thì vẫn ra một xác suất, và đó là
     * con số vô nghĩa mà không có gì báo hiệu.
     *
     * @return array<string, mixed>|null
     */
    private function loanApplicationFor(Household $household, ?IntentCodeEnum $intent): ?array
    {
        if ($intent === null || ! $intent->needsLoanApplication()) {
            return null;
        }

        $application = $household->loanApplication()->first();

        if ($application === null) {
            return null;
        }

        return [
            'borrower_age' => $application->borrower_age,
            'gender' => $application->gender->value,
            'marital_status' => $application->marital_status->value,
            'children_count' => $application->children_count,
            'education_level' => $application->education_level->value,
            'occupation' => $application->occupation->value,
            'employment_years' => (float) $application->employment_years,

            'loan_amount' => (float) $application->loan_amount,
            'loan_term_months' => $application->loan_term_months,
            'monthly_payment' => (float) $application->monthly_payment,
            'asset_price' => (float) $application->asset_price,
            'loan_purpose' => $application->loan_purpose->value,

            'previous_loan_count' => $application->previous_loan_count,
            'late_payment_count' => $application->late_payment_count,
            'has_overdue_loan' => $application->has_overdue_loan,
            'total_overdue_amount' => (float) $application->total_overdue_amount,
        ];
    }

    /**
     * Hồ sơ gửi sang Python, dùng đúng tên cột trong DB để hai bên khỏi lệch.
     *
     * @return array<string, mixed>
     */
    private function householdPayload(Household $household): array
    {
        $household->loadMissing('assets');

        return [
            'id' => $household->id,
            'representative_name' => $household->representative_name,
            'birth_year' => $household->birth_year,
            'location' => $household->location,
            'household_size' => $household->household_size,
            'children_count' => $household->children_count,
            'supports_elderly' => $household->supports_elderly,
            'has_dependents' => (bool) $household->supports_elderly,
            'monthly_income' => (float) $household->monthly_income,
            'monthly_living_cost' => (float) ($household->monthly_living_cost ?? 0),
            'has_debt' => $household->has_debt,
            'total_debt' => (float) $household->total_debt,
            'monthly_debt_payment' => (float) $household->monthly_debt_payment,
            'has_savings' => $household->has_savings,
            'current_savings' => (float) $household->current_savings,
<<<<<<< HEAD
            'assets' => $household->assets->map(fn ($a) => (string) $a->getRawOriginal('asset_type'))->all(),
=======
            'assets' => $household->assets
                ->map(fn ($a) => self::ASSET_TYPE_TO_PYTHON[(string) $a->getRawOriginal('asset_type')] ?? null)
                ->filter()
                ->unique()
                ->values()
                ->all(),
>>>>>>> main
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

            // Ý định mà engine THỰC SỰ đã chạy. Trả ngược lên FE để kiểm chứng
            // được rằng chip đã vào đúng nhánh — không có trường này thì câu
            // "chip có chạy đúng model không" phải đọc log server mới biết.
            'intent_code' => isset($payload['intent_code'])
                ? (string) $payload['intent_code']
                : null,

            // ML02 báo hộ chưa khai thông tin khoản vay. FE dựa vào đây để hiện
            // nút điều hướng thay vì chỉ in ra một đoạn chữ bảo người dùng tự
            // đi tìm màn nhập.
            'requires_loan_application' => (bool) ($payload['requires_loan_application'] ?? false),
        ];
    }
}
