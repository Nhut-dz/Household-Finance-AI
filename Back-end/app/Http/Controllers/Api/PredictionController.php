<?php

namespace App\Http\Controllers\Api;

use App\Http\Controllers\Controller;
use App\Http\Requests\Api\HouseholdAccessRequest;
use App\Services\AdvisorClient;
use App\Services\HouseholdService;
use Illuminate\Http\JsonResponse;
use OpenApi\Attributes as OA;

/**
 * Phân loại hộ vào 4 nhóm khuyến nghị bằng model ML01 (F03).
 *
 * Đi theo đúng khuôn của ProposalController: quyền sở hữu do HouseholdService
 * xét (user_id khi có Bearer token, guest_session_id khi không), rồi mới gọi
 * sang service Python.
 *
 * Khác ProposalController ở chỗ dữ liệu KHÔNG đọc từ DB mà tính trực tiếp:
 * ML01 chạy trên hồ sơ hiện tại, không phụ thuộc lượt tư vấn nào đã lưu.
 */
class PredictionController extends Controller
{
    public function __construct(
        private readonly HouseholdService $householdService,
        private readonly AdvisorClient $advisorClient,
    ) {}

    #[OA\Get(
        path: '/households/{id}/prediction',
        operationId: 'getHouseholdPrediction',
        description: 'Gọi model ML01 (Financial Recommendation Group Classification) dự đoán '
            .'nhóm định hướng tài chính của hộ. Output nghiệp vụ là MỘT nhãn ở `prediction`: '
            .'EMERGENCY, DEBT_FOCUS, BUILD_BUFFER hoặc GROWTH. Khối `model_confidence` là số '
            .'liệu kỹ thuật (xác suất 4 lớp, cờ low_confidence) — không phải 4 kết quả dự đoán.',
        summary: 'Dự đoán nhóm định hướng tài chính (ML01)',
        security: [[], ['bearerAuth' => []]],
        tags: ['Prediction'],
        parameters: [
            new OA\Parameter(name: 'id', in: 'path', required: true, schema: new OA\Schema(type: 'integer')),
            new OA\Parameter(name: 'guest_session_id', in: 'query', required: false, schema: new OA\Schema(type: 'string', maxLength: 64)),
        ],
        responses: [
            new OA\Response(
                response: 200,
                description: 'Thành công',
                content: new OA\JsonContent(
                    properties: [
                        new OA\Property(property: 'status', type: 'boolean', example: true),
                        new OA\Property(property: 'message', type: 'string'),
                        new OA\Property(
                            property: 'result',
                            properties: [
                                new OA\Property(
                                    property: 'data',
                                    properties: [
                                        new OA\Property(property: 'prediction', type: 'string', example: 'BUILD_BUFFER', description: 'Output nghiệp vụ — MỘT nhóm định hướng'),
                                        new OA\Property(property: 'prediction_vi', type: 'string', example: 'Cần xây dựng quỹ dự phòng'),
                                        new OA\Property(
                                            property: 'model_confidence',
                                            description: 'Số liệu kỹ thuật, không phải kết quả dự đoán',
                                            properties: [
                                                new OA\Property(property: 'confidence', type: 'number', example: 0.87),
                                                new OA\Property(property: 'low_confidence', type: 'boolean', example: false),
                                                new OA\Property(property: 'probabilities', type: 'array', items: new OA\Items(
                                                    properties: [
                                                        new OA\Property(property: 'label', type: 'string'),
                                                        new OA\Property(property: 'label_vi', type: 'string'),
                                                        new OA\Property(property: 'probability', type: 'number'),
                                                    ],
                                                    type: 'object'
                                                )),
                                            ],
                                            type: 'object'
                                        ),
                                        new OA\Property(property: 'model_version', type: 'string', example: 'ml01_xgboost_vfinal'),
                                    ],
                                    type: 'object'
                                ),
                            ],
                            type: 'object'
                        ),
                    ],
                    type: 'object'
                )
            ),
            new OA\Response(response: 403, description: 'Hồ sơ không thuộc về người gọi'),
            new OA\Response(response: 404, description: 'Không tìm thấy hồ sơ'),
            new OA\Response(response: 422, description: 'Hồ sơ thiếu năm sinh nên chưa dự đoán được'),
            new OA\Response(response: 503, description: 'Service ML chưa cấu hình hoặc không phản hồi'),
        ]
    )]
    public function show(HouseholdAccessRequest $request, int $id): JsonResponse
    {
        $household = $this->householdService->findOwned(
            $id,
            $request->resolvedUser(),
            $request->guestSessionId()
        );

        // `assets` cần cho 6 cột multi-hot; nạp sẵn để tránh N+1 khi duyệt.
        $household->loadMissing('assets');

        return $this->successResponse(
            $this->advisorClient->predict($household),
            __('lang.Prediction_fetched')
        );
    }
}
