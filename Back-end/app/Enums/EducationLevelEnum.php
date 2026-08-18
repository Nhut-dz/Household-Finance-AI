<?php

namespace App\Enums;

/**
 * Trình độ học vấn — cột tblloan_applications.education_level.
 *
 * Năm bậc theo `application_train.NAME_EDUCATION_TYPE`, xếp từ thấp lên cao.
 * Thứ tự khai báo có ý nghĩa: đây là biến THỨ BẬC, nên tầng ML mã hoá ordinal
 * theo đúng thứ tự này chứ không one-hot.
 *
 *     Secondary / secondary special  218.391
 *     Higher education                74.863
 *     Incomplete higher               10.277
 *     Lower secondary                  3.816
 *     Academic degree                    164
 *
 * `ACADEMIC_DEGREE` chỉ có 164 mẫu (0,05%). Vẫn giữ trên form vì bỏ đi thì
 * người có học vị phải tự xếp mình vào bậc khác; tầng ML sẽ tự gộp nó vào
 * `HIGHER` nếu bước gộp hạng mục hiếm thấy cần.
 */
enum EducationLevelEnum: string
{
    case LOWER_SECONDARY = 'lower_secondary';
    case SECONDARY = 'secondary';
    case INCOMPLETE_HIGHER = 'incomplete_higher';
    case HIGHER = 'higher';
    case ACADEMIC_DEGREE = 'academic_degree';

    public function label(): string
    {
        return match ($this) {
            self::LOWER_SECONDARY => 'Trung học cơ sở',
            self::SECONDARY => 'Trung học phổ thông, trung cấp',
            self::INCOMPLETE_HIGHER => 'Cao đẳng, đại học dở dang',
            self::HIGHER => 'Đại học',
            self::ACADEMIC_DEGREE => 'Sau đại học',
        };
    }

    /**
     * Giá trị gốc của `application_train.NAME_EDUCATION_TYPE`.
     */
    public function homeCredit(): string
    {
        return match ($this) {
            self::LOWER_SECONDARY => 'Lower secondary',
            self::SECONDARY => 'Secondary / secondary special',
            self::INCOMPLETE_HIGHER => 'Incomplete higher',
            self::HIGHER => 'Higher education',
            self::ACADEMIC_DEGREE => 'Academic degree',
        };
    }

    /**
     * Hạng thứ bậc, 1 = thấp nhất. Dùng khi mã hoá ordinal cho ML02.
     */
    public function rank(): int
    {
        return match ($this) {
            self::LOWER_SECONDARY => 1,
            self::SECONDARY => 2,
            self::INCOMPLETE_HIGHER => 3,
            self::HIGHER => 4,
            self::ACADEMIC_DEGREE => 5,
        };
    }
}
