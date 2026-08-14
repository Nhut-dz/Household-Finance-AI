<?php

namespace App\Http\Controllers\Api;

use App\Http\Controllers\Controller;
use App\Http\Requests\Api\HouseholdAccessRequest;
use App\Http\Requests\Api\StoreHouseholdRequest;
use App\Http\Resources\HouseholdResource;
use App\Services\HouseholdService;
use Illuminate\Http\JsonResponse;
use Illuminate\Http\Response;
use OpenApi\Attributes as OA;

class HouseholdController extends Controller
{
    public function __construct(private readonly HouseholdService $householdService) {}

    #[OA\Post(
        path: '/households',
        operationId: 'storeHousehold',
        description: 'Nhận dữ liệu màn "Nhập thông tin" và lưu vào PostgreSQL. '
            .'Gửi kèm Bearer token thì bản ghi gắn với user_id; không có token thì '
            .'user_id = null và bắt buộc truyền guest_session_id.',
        summary: 'Tạo hồ sơ tài chính hộ gia đình (hỗ trợ Guest và User)',
        security: [[], ['bearerAuth' => []]],
        requestBody: new OA\RequestBody(required: true, content: new OA\JsonContent(ref: '#/components/schemas/HouseholdInput')),
        tags: ['Household'],
        responses: [
            new OA\Response(response: 201, description: 'Lưu thành công'),
            new OA\Response(response: 422, description: 'Dữ liệu không hợp lệ'),
        ]
    )]
    public function store(StoreHouseholdRequest $request): JsonResponse
    {
        $household = $this->householdService->store(
            $request->validated(),
            $request->resolvedUser()
        );

        return $this->successResponse(
            new HouseholdResource($household),
            __('lang.Household_stored'),
            Response::HTTP_CREATED
        );
    }

    #[OA\Get(
        path: '/households/latest',
        operationId: 'getLatestHousehold',
        description: 'Dùng để khôi phục màn Chatbot và Phương án sau khi FE tải lại trang.',
        summary: 'Lấy hồ sơ gần nhất của người gọi',
        security: [[], ['bearerAuth' => []]],
        tags: ['Household'],
        parameters: [
            new OA\Parameter(
                name: 'guest_session_id',
                description: 'Bắt buộc khi chưa đăng nhập. Có token thì bỏ qua và lấy theo user_id.',
                in: 'query',
                required: false,
                schema: new OA\Schema(type: 'string', maxLength: 64)
            ),
        ],
        responses: [
            new OA\Response(response: 200, description: 'Thành công'),
            new OA\Response(response: 404, description: 'Chưa từng gửi hồ sơ'),
            new OA\Response(response: 422, description: 'Thiếu guest_session_id'),
        ]
    )]
    public function latest(HouseholdAccessRequest $request): JsonResponse
    {
        $household = $this->householdService->findLatestFor(
            $request->resolvedUser(),
            $request->guestSessionId()
        );

        return $this->successResponse(
            new HouseholdResource($household),
            __('lang.Household_fetched')
        );
    }

    #[OA\Get(
        path: '/households/{id}',
        operationId: 'showHousehold',
        summary: 'Lấy hồ sơ theo id',
        security: [[], ['bearerAuth' => []]],
        tags: ['Household'],
        parameters: [
            new OA\Parameter(name: 'id', in: 'path', required: true, schema: new OA\Schema(type: 'integer')),
            new OA\Parameter(name: 'guest_session_id', in: 'query', required: false, schema: new OA\Schema(type: 'string', maxLength: 64)),
        ],
        responses: [
            new OA\Response(response: 200, description: 'Thành công'),
            new OA\Response(response: 403, description: 'Hồ sơ không thuộc về người gọi'),
            new OA\Response(response: 404, description: 'Không tìm thấy hồ sơ'),
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
            new HouseholdResource($household),
            __('lang.Household_fetched')
        );
    }

    #[OA\Put(
        path: '/households/{id}',
        operationId: 'updateHousehold',
        description: 'Body giống hệt POST /households. Danh sách assets và '
            .'financial_needs được ghi đè theo lựa chọn mới, không cộng dồn.',
        summary: 'Cập nhật hồ sơ (nút "Sửa hồ sơ")',
        security: [[], ['bearerAuth' => []]],
        requestBody: new OA\RequestBody(required: true, content: new OA\JsonContent(ref: '#/components/schemas/HouseholdInput')),
        tags: ['Household'],
        parameters: [
            new OA\Parameter(name: 'id', in: 'path', required: true, schema: new OA\Schema(type: 'integer')),
        ],
        responses: [
            new OA\Response(response: 200, description: 'Cập nhật thành công'),
            new OA\Response(response: 403, description: 'Hồ sơ không thuộc về người gọi'),
            new OA\Response(response: 404, description: 'Không tìm thấy hồ sơ'),
            new OA\Response(response: 422, description: 'Dữ liệu không hợp lệ'),
        ]
    )]
    public function update(StoreHouseholdRequest $request, int $id): JsonResponse
    {
        $user = $request->resolvedUser();

        $household = $this->householdService->findOwned($id, $user, $request->guestSessionId());
        $result = $this->householdService->update($household, $request->validated(), $user);

        // FE cần biết phiên có bị xoay hay không để quyết định xoá hội thoại
        // đang hiển thị. Suy đoán ở FE bằng cách tự so số liệu là chép lại luật
        // nghiệp vụ ở hai nơi, và hai nơi đó sẽ lệch nhau.
        //
        // Hai khoá phiên nằm TRONG `data`, không phải cạnh nó: lớp gọi API của
        // FE trả về đúng `result.data`, nên thứ gì đặt ngoài đó sẽ không bao giờ
        // tới nơi. Cũng không dùng `JsonResource::additional()` —
        // `ApiResponseTrait::convertToArray()` gọi `resolve()`, mà `resolve()`
        // bỏ qua phần `additional`.
        return $this->successResponse(
            [
                ...(new HouseholdResource($result['household']))->resolve(),
                'conversation_id' => $result['conversation_id'],
                'conversation_rotated' => $result['conversation_rotated'],
            ],
            __('lang.Household_updated')
        );
    }

    #[OA\Delete(
        path: '/households/{id}',
        operationId: 'deleteHousehold',
        description: 'Xoá kèm tài sản, nhu cầu tài chính và lịch sử hội thoại của hồ sơ.',
        summary: 'Xoá hồ sơ (nút "Xóa dữ liệu")',
        security: [[], ['bearerAuth' => []]],
        tags: ['Household'],
        parameters: [
            new OA\Parameter(name: 'id', in: 'path', required: true, schema: new OA\Schema(type: 'integer')),
            new OA\Parameter(name: 'guest_session_id', in: 'query', required: false, schema: new OA\Schema(type: 'string', maxLength: 64)),
        ],
        responses: [
            new OA\Response(response: 200, description: 'Xoá thành công'),
            new OA\Response(response: 403, description: 'Hồ sơ không thuộc về người gọi'),
            new OA\Response(response: 404, description: 'Không tìm thấy hồ sơ'),
        ]
    )]
    public function destroy(HouseholdAccessRequest $request, int $id): JsonResponse
    {
        $household = $this->householdService->findOwned(
            $id,
            $request->resolvedUser(),
            $request->guestSessionId()
        );

        $this->householdService->delete($household);

        return $this->successResponse(null, __('lang.Household_deleted'));
    }
}
