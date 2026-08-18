<?php

namespace App\Enums;

/**
 * Tình trạng hôn nhân — cột tblloan_applications.marital_status.
 *
 * Năm giá trị lấy ĐÚNG theo `application_train.NAME_FAMILY_STATUS` (2 dòng
 * 'Unknown' coi là missing, không đưa vào form):
 *
 *     Married               196.432
 *     Single / not married   45.444
 *     Civil marriage         29.775
 *     Separated              19.770
 *     Widow                  16.088
 *
 * `CIVIL_MARRIAGE` giữ riêng chứ không gộp vào `MARRIED` vì Home Credit tách
 * chúng, và gộp ở phía form thì tầng ML mất một hạng mục có sẵn mẫu.
 */
enum MaritalStatusEnum: string
{
    case SINGLE = 'single';
    case MARRIED = 'married';
    case CIVIL_MARRIAGE = 'civil_marriage';
    case SEPARATED = 'separated';
    case WIDOW = 'widow';

    public function label(): string
    {
        return match ($this) {
            self::SINGLE => 'Độc thân',
            self::MARRIED => 'Đã kết hôn',
            self::CIVIL_MARRIAGE => 'Sống chung, chưa đăng ký kết hôn',
            self::SEPARATED => 'Ly thân, ly hôn',
            self::WIDOW => 'Góa',
        };
    }

    /**
     * Giá trị gốc của `application_train.NAME_FAMILY_STATUS`.
     */
    public function homeCredit(): string
    {
        return match ($this) {
            self::SINGLE => 'Single / not married',
            self::MARRIED => 'Married',
            self::CIVIL_MARRIAGE => 'Civil marriage',
            self::SEPARATED => 'Separated',
            self::WIDOW => 'Widow',
        };
    }
}
