<?php

namespace App\Enums;

/**
 * Mục đích vay — cột tblloan_applications.loan_purpose.
 *
 * ⚠️ Giới hạn đã biết, phải ghi vào báo cáo chứ đừng phát hiện lúc bảo vệ:
 * `previous_application.NAME_CASH_LOAN_PURPOSE` có 1.600.579 / 1.670.214 dòng
 * (95,8%) mang giá trị 'XAP' hoặc 'XNA' — tức KHÔNG khai mục đích. Phần khai
 * thật chỉ còn ~70.000 dòng, trải trên 23 hạng mục.
 *
 * Hệ quả: trường này phục vụ RB05 và phần diễn giải của tầng LLM (vay mua nhà
 * khác vay tiêu dùng về kỳ hạn và LTV hợp lý), chứ KHÔNG kỳ vọng là feature
 * mạnh của ML02. Đừng trình bày nó như một biến dự báo.
 */
enum LoanPurposeEnum: string
{
    case BUY_HOUSE = 'buy_house';
    case BUY_LAND = 'buy_land';
    case BUY_CAR = 'buy_car';
    case HOME_REPAIR = 'home_repair';
    case BUSINESS = 'business';
    case EDUCATION = 'education';
    case MEDICAL = 'medical';
    case CONSUMER = 'consumer';
    case DEBT_CONSOLIDATION = 'debt_consolidation';
    case OTHER = 'other';

    public function label(): string
    {
        return match ($this) {
            self::BUY_HOUSE => 'Mua nhà, căn hộ',
            self::BUY_LAND => 'Mua đất',
            self::BUY_CAR => 'Mua xe',
            self::HOME_REPAIR => 'Sửa chữa, xây dựng nhà',
            self::BUSINESS => 'Kinh doanh, sản xuất',
            self::EDUCATION => 'Học tập',
            self::MEDICAL => 'Chữa bệnh',
            self::CONSUMER => 'Tiêu dùng, mua sắm',
            self::DEBT_CONSOLIDATION => 'Trả nợ khoản vay khác',
            self::OTHER => 'Khác',
        };
    }

    /**
     * Giá trị gốc của `previous_application.NAME_CASH_LOAN_PURPOSE`.
     *
     * Trả về nhiều giá trị vì Home Credit chia nhỏ hơn form: "Mua xe" ứng với
     * cả xe mới lẫn xe cũ, "Sửa chữa, xây dựng nhà" ứng với cả 'Repairs' lẫn
     * 'Building a house or an annex'.
     *
     * @return array<int, string>
     */
    public function homeCredit(): array
    {
        return match ($this) {
            self::BUY_HOUSE => ['Buying a home'],
            self::BUY_LAND => ['Buying a holiday home / land'],
            self::BUY_CAR => ['Buying a new car', 'Buying a used car'],
            self::HOME_REPAIR => ['Repairs', 'Building a house or an annex'],
            self::BUSINESS => ['Business development'],
            self::EDUCATION => ['Education'],
            self::MEDICAL => ['Medicine'],
            self::CONSUMER => ['Purchase of electronic equipment', 'Furniture', 'Everyday expenses'],
            self::DEBT_CONSOLIDATION => ['Payments on other loans'],
            self::OTHER => ['Other', 'Urgent needs'],
        };
    }

    /**
     * Mục đích có tài sản bảo đảm hình thành từ chính khoản vay. Với nhóm này,
     * "giá trị tài sản" là giá tài sản định mua nên LTV = vay / giá có ý nghĩa;
     * với nhóm còn lại đó là tài sản bảo đảm sẵn có.
     */
    public function isAssetBacked(): bool
    {
        return match ($this) {
            self::BUY_HOUSE, self::BUY_LAND, self::BUY_CAR => true,
            default => false,
        };
    }
}
