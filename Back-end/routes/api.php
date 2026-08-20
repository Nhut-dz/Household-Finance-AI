<?php

use App\Enums\Routes\ApiEnum;
use App\Http\Controllers\Api\AuthController;
use App\Http\Controllers\Api\ChatMessageController;
use App\Http\Controllers\Api\HouseholdController;
use App\Http\Controllers\Api\LoanApplicationController;
use App\Http\Controllers\Api\PredictionController;
use App\Http\Controllers\Api\ProposalController;
use App\Http\Controllers\Api\UserController;
use Illuminate\Support\Facades\Route;

Route::prefix('auth')->group(function () {
    Route::post('/register', [AuthController::class, 'register'])
        ->name(ApiEnum::AUTH_REGISTER->routeName());

    Route::post('/login', [AuthController::class, 'login'])
        ->name(ApiEnum::AUTH_LOGIN->routeName());

    Route::post('/logout', [AuthController::class, 'logout'])
        ->middleware('auth:sanctum')
        ->name(ApiEnum::AUTH_LOGOUT->routeName());
});

/*
 * Nhóm hồ sơ hộ gia đình là route công khai: Guest lẫn User đăng nhập đều gọi
 * được. Có Bearer token thì quyền sở hữu xét theo user_id, không có thì xét
 * theo guest_session_id. Việc kiểm tra do HouseholdService đảm nhiệm.
 */
Route::prefix('households')->group(function () {
    Route::post('/', [HouseholdController::class, 'store'])
        ->name(ApiEnum::HOUSEHOLD_STORE->routeName());

    // Phải khai báo trước '/{id}' để "latest" không bị hiểu thành id.
    Route::get('/latest', [HouseholdController::class, 'latest'])
        ->name(ApiEnum::HOUSEHOLD_LATEST->routeName());

    Route::get('/{id}', [HouseholdController::class, 'show'])
        ->whereNumber('id')
        ->name(ApiEnum::HOUSEHOLD_SHOW->routeName());

    Route::put('/{id}', [HouseholdController::class, 'update'])
        ->whereNumber('id')
        ->name(ApiEnum::HOUSEHOLD_UPDATE->routeName());

    Route::delete('/{id}', [HouseholdController::class, 'destroy'])
        ->whereNumber('id')
        ->name(ApiEnum::HOUSEHOLD_DESTROY->routeName());

    Route::get('/{id}/proposal', [ProposalController::class, 'show'])
        ->whereNumber('id')
        ->name(ApiEnum::HOUSEHOLD_PROPOSAL->routeName());

    Route::get('/{id}/prediction', [PredictionController::class, 'show'])
        ->whereNumber('id')
        ->name(ApiEnum::HOUSEHOLD_PREDICTION->routeName());

    /*
     * Màn "Thông tin khoản vay" — đầu vào của ML02. PUT thay vì POST vì một hộ
     * giữ đúng một phương án vay đang xét: gửi lại cùng body cho ra cùng trạng
     * thái, nên FE không phải biết trước đã có bản ghi hay chưa.
     */
    Route::get('/{id}/loan-application', [LoanApplicationController::class, 'show'])
        ->whereNumber('id')
        ->name(ApiEnum::HOUSEHOLD_LOAN_APPLICATION_SHOW->routeName());

    Route::put('/{id}/loan-application', [LoanApplicationController::class, 'store'])
        ->whereNumber('id')
        ->name(ApiEnum::HOUSEHOLD_LOAN_APPLICATION_STORE->routeName());

    Route::delete('/{id}/loan-application', [LoanApplicationController::class, 'destroy'])
        ->whereNumber('id')
        ->name(ApiEnum::HOUSEHOLD_LOAN_APPLICATION_DESTROY->routeName());

    Route::get('/{id}/messages', [ChatMessageController::class, 'index'])
        ->whereNumber('id')
        ->name(ApiEnum::HOUSEHOLD_MESSAGE_INDEX->routeName());

    Route::post('/{id}/messages', [ChatMessageController::class, 'store'])
        ->whereNumber('id')
        ->name(ApiEnum::HOUSEHOLD_MESSAGE_STORE->routeName());

    /*
     * Xem lại các phiên trò chuyện đã đóng. Phiên bị đóng khi hồ sơ tài chính
     * thay đổi; nội dung vẫn ở lại DB, chỉ không còn được dùng làm ngữ cảnh.
     */
    Route::get('/{id}/conversations', [ChatMessageController::class, 'conversations'])
        ->whereNumber('id')
        ->name(ApiEnum::HOUSEHOLD_CONVERSATION_INDEX->routeName());

    Route::get('/{id}/conversations/{conversationId}/messages',
        [ChatMessageController::class, 'conversationMessages'])
        ->whereNumber('id')
        ->whereNumber('conversationId')
        ->name(ApiEnum::HOUSEHOLD_CONVERSATION_MESSAGES->routeName());
});

Route::middleware('auth:sanctum')->group(function () {
    Route::get('/user', [UserController::class, 'show'])
        ->name(ApiEnum::USER_SHOW->routeName());
});
