<?php

namespace App\Enums;

/**
 * Giới tính người vay — cột tblloan_applications.gender.
 *
 * Chỉ hai giá trị vì đích đến là `CODE_GENDER` của Home Credit, mà cột đó chỉ
 * có F (202.448) / M (105.059) và 4 dòng 'XNA' được coi là missing. Thêm lựa
 * chọn thứ ba vào form sẽ tạo ra hạng mục không có mẫu huấn luyện nào.
 */
enum GenderEnum: string
{
    case MALE = 'male';
    case FEMALE = 'female';

    public function label(): string
    {
        return match ($this) {
            self::MALE => 'Nam',
            self::FEMALE => 'Nữ',
        };
    }

    /**
     * Giá trị gốc của `application_train.CODE_GENDER`.
     */
    public function homeCredit(): string
    {
        return match ($this) {
            self::MALE => 'M',
            self::FEMALE => 'F',
        };
    }
}
