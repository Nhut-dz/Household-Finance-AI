<?php

namespace App\Http\Controllers\Api;

use App\Http\Controllers\Controller;
use App\Http\Requests\Api\HouseholdAccessRequest;
use App\Services\HouseholdService;
use App\Services\ProposalService;
use Illuminate\Http\JsonResponse;
use OpenApi\Attributes as OA;

class ProposalController extends Controller
{
    public function __construct(
        private readonly HouseholdService $householdService,
        private readonly ProposalService $proposalService,
    ) {}

    #[OA\Get(
        path: '/households/{id}/proposal',
        operationId: 'getHouseholdProposal',
        description: 'Trả về toàn bộ số liệu cho trang "Phương án đề xuất". '
            .'Ba khối savings_plan / investment_plan / loan_plan có thể là null '
            .'nếu không áp dụng cho hộ đó, FE sẽ ẩn thẻ tương ứng. '
            .'Mọi số tiền đều là số thuần để FE tự định dạng theo vi-VN.',
        summary: 'Lấy phương án đề xuất của hồ sơ',
        security: [[], ['bearerAuth' => []]],
        tags: ['Proposal'],
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
                                new OA\Property(property: 'data', ref: '#/components/schemas/Proposal'),
                            ],
                            type: 'object'
                        ),
                    ],
                    type: 'object'
                )
            ),
            new OA\Response(response: 403, description: 'Hồ sơ không thuộc về người gọi'),
            new OA\Response(response: 404, description: 'Hồ sơ chưa được tính phương án'),
        ]
    )]
    public function show(HouseholdAccessRequest $request, int $id): JsonResponse
    {
        $household = $this->householdService->findOwned(
            $id,
            $request->resolvedUser(),
            $request->guestSessionId()
        );

        return $this->successResponse(
            $this->proposalService->forHousehold($household),
            __('lang.Proposal_fetched')
        );
    }
}
