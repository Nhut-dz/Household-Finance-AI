<?php

namespace App\Enums;

/**
 * Mã ý định của một lượt hỏi Chatbot.
 *
 * Bốn giá trị đầu ứng với đúng bốn chip gợi ý trên màn Chatbot. FE gửi mã
 * xuống thay vì để engine đoán từ nhãn tiếng Việt — đoán thì hỏng ngay ở cả
 * hai chức năng mới:
 *
 *     "Chẩn đoán rủi ro vay vốn"      chứa chữ "vay" → rơi vào nhánh hạn mức
 *                                     vay (RB05), KHÔNG bao giờ tới ML02
 *     "Chẩn đoán sức khỏe tài chính"  không chứa từ khoá nào → rơi xuống
 *                                     nhánh trả lời chung, KHÔNG tới ML01
 *
 * Cả hai vẫn trả lời trôi chảy, chỉ là bằng nhánh sai — loại lỗi không nhìn
 * màn hình mà thấy được.
 *
 * Giữ khớp `IntentCode` của `ML_Training/src/hfml/api/intents.py`. Hai bên
 * lệch nhau thì mã gửi xuống không khớp giá trị nào, service Python coi như
 * không có mã và quay về đoán từ khoá — tức im lặng mất đúng tính năng này.
 */
enum IntentCodeEnum: string
{
    // -- Bốn chip gợi ý --------------------------------------------------
    case SAVINGS_PACKAGE = 'SAVINGS_PACKAGE';
    case FINANCIAL_HEALTH_DIAGNOSIS = 'FINANCIAL_HEALTH_DIAGNOSIS';   // → ML01
    case LOAN_RISK_DIAGNOSIS = 'LOAN_RISK_DIAGNOSIS';                 // → ML02
    case BUDGET_50_30_20 = 'BUDGET_50_30_20';

    // -- Chỉ đến từ câu người dùng tự gõ ---------------------------------
    case LOAN_CAPACITY = 'LOAN_CAPACITY';
    case DEBT = 'DEBT';
    case INVESTMENT = 'INVESTMENT';
    case GENERAL = 'GENERAL';

    /**
     * Nhãn tiếng Việt hiển thị trên chip.
     */
    public function label(): string
    {
        return match ($this) {
            self::SAVINGS_PACKAGE => 'Gói tiết kiệm',
            self::FINANCIAL_HEALTH_DIAGNOSIS => 'Chẩn đoán sức khỏe tài chính',
            self::LOAN_RISK_DIAGNOSIS => 'Chẩn đoán rủi ro vay vốn',
            self::BUDGET_50_30_20 => 'Quy tắc 50/30/20',
            self::LOAN_CAPACITY => 'Khả năng vay vốn',
            self::DEBT => 'Xử lý nợ',
            self::INVESTMENT => 'Phân bổ đầu tư',
            self::GENERAL => 'Tư vấn chung',
        };
    }

    /**
     * Intent này có chạy model hay không.
     *
     * Quyết định backend phải đính kèm dữ liệu gì cho service Python: ML01
     * cần 17 feature đã chuẩn hoá, ML02 cần bản ghi thông tin khoản vay.
     */
    public function usesModel(): bool
    {
        return $this === self::FINANCIAL_HEALTH_DIAGNOSIS
            || $this === self::LOAN_RISK_DIAGNOSIS;
    }

    /**
     * Intent cần 17 feature của ML01.
     */
    public function needsMlFeatures(): bool
    {
        return $this === self::FINANCIAL_HEALTH_DIAGNOSIS;
    }

    /**
     * Intent cần bản ghi tblloan_applications.
     */
    public function needsLoanApplication(): bool
    {
        return $this === self::LOAN_RISK_DIAGNOSIS;
    }
}
