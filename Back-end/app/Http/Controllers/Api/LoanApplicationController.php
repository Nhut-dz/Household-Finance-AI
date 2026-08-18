<?php

namespace App\Http\Controllers\Api;

use App\Http\Controllers\Controller;
use App\Http\Requests\Api\HouseholdAccessRequest;
use App\Http\Requests\Api\StoreLoanApplicationRequest;
use App\Http\Resources\LoanApplicationResource;
use App\Services\HouseholdService;
use App\Services\LoanApplicationService;
use Illuminate\Http\JsonResponse;

/**
 * Màn "Thông tin khoản vay" — dữ liệu đầu vào của ML02.
 *
 * Endpoint nằm dưới `/households/{id}` vì phương án vay không tồn tại độc lập:
 * ML02 cần cả thu nhập và chi tiêu của hộ để dựng feature tỉ lệ (dti,
 * credit_income_ratio), mà hai số đó nằm ở tblhouseholds.
 */
class LoanApplicationController extends Controller
{
    public function __construct(
        private readonly HouseholdService $householdService,
        private readonly LoanApplicationService $loanApplications,
    ) {}

    /**
     * Lưu hoặc ghi đè phương án vay của hộ.
     *
     * Dùng PUT chứ không POST: một hộ giữ đúng một phương án đang xét, gửi
     * cùng một body nhiều lần cho ra cùng một trạng thái. FE nhờ vậy không phải
     * biết trước hộ đã khai khoản vay hay chưa để chọn method.
     */
    public function store(StoreLoanApplicationRequest $request, int $id): JsonResponse
    {
        $household = $this->householdService->findOwned(
            $id,
            $request->resolvedUser(),
            $request->guestSessionId()
        );

        $application = $this->loanApplications->upsert($household, $request->validated());

        return $this->successResponse(
            new LoanApplicationResource($application),
            __('lang.Loan_application_saved')
        );
    }

    /**
     * Phương án vay đang xét của hộ. 404 khi hộ chưa từng khai — đây là trạng
     * thái bình thường, FE hiển thị form trống chứ không báo lỗi.
     */
    public function show(HouseholdAccessRequest $request, int $id): JsonResponse
    {
        $household = $this->householdService->findOwned(
            $id,
            $request->resolvedUser(),
            $request->guestSessionId()
        );

        return $this->successResponse(
            new LoanApplicationResource($this->loanApplications->findFor($household)),
            __('lang.Loan_application_fetched')
        );
    }

    /**
     * Xoá phương án vay. Hồ sơ hộ gia đình giữ nguyên — người dùng chỉ thôi
     * không muốn đánh giá khoản vay nữa, không phải xoá cả hồ sơ.
     */
    public function destroy(HouseholdAccessRequest $request, int $id): JsonResponse
    {
        $household = $this->householdService->findOwned(
            $id,
            $request->resolvedUser(),
            $request->guestSessionId()
        );

        $this->loanApplications->delete($this->loanApplications->findFor($household));

        return $this->successResponse(null, __('lang.Loan_application_deleted'));
    }
}
