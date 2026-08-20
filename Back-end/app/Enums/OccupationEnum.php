<?php

namespace App\Enums;

/**
 * Nghề nghiệp người vay — cột tblloan_applications.occupation.
 *
 * Danh sách rút gọn cho người dùng Việt Nam. Home Credit có 18 giá trị
 * `OCCUPATION_TYPE` nhưng nhiều giá trị quá hẹp (Waiters/barmen staff,
 * Realty agents…), gộp lại còn 16 để dropdown chọn được nhanh.
 *
 * Giữ khớp 1-1 với `OccupationType` của
 * `ML_Training/src/hfml/data/schema.py` — hai nơi lệch nhau là lỗi im lặng:
 * form vẫn gửi được, model vẫn trả xác suất, chỉ có điều hạng mục rơi vào
 * nhánh "giá trị lạ" và mất sạch tín hiệu.
 */
enum OccupationEnum: string
{
    case OFFICE_STAFF = 'office_staff';
    case MANAGER = 'manager';
    case ACCOUNTANT = 'accountant';
    case IT_STAFF = 'it_staff';
    case TEACHER = 'teacher';
    case MEDICAL_STAFF = 'medical_staff';
    case SALES_STAFF = 'sales_staff';
    case DRIVER = 'driver';
    case SECURITY_STAFF = 'security_staff';
    case SERVICE_STAFF = 'service_staff';
    case LABORER = 'laborer';
    case FARMER = 'farmer';
    case SELF_EMPLOYED = 'self_employed';
    case RETIRED = 'retired';
    case UNEMPLOYED = 'unemployed';
    case OTHER = 'other';

    public function label(): string
    {
        return match ($this) {
            self::OFFICE_STAFF => 'Nhân viên văn phòng',
            self::MANAGER => 'Quản lý, lãnh đạo',
            self::ACCOUNTANT => 'Kế toán, tài chính',
            self::IT_STAFF => 'Công nghệ thông tin',
            self::TEACHER => 'Giáo viên, giảng viên',
            self::MEDICAL_STAFF => 'Y tế',
            self::SALES_STAFF => 'Kinh doanh, bán hàng',
            self::DRIVER => 'Lái xe',
            self::SECURITY_STAFF => 'Bảo vệ',
            self::SERVICE_STAFF => 'Dịch vụ, giúp việc',
            self::LABORER => 'Công nhân, lao động phổ thông',
            self::FARMER => 'Nông, lâm, ngư nghiệp',
            self::SELF_EMPLOYED => 'Tự kinh doanh, tự do',
            self::RETIRED => 'Nghỉ hưu',
            self::UNEMPLOYED => 'Chưa có việc làm',
            self::OTHER => 'Khác',
        };
    }

    /**
     * Giá trị gốc của `application_train.OCCUPATION_TYPE`.
     *
     * `null` nghĩa là Home Credit ĐỂ TRỐNG cột này cho nhóm đó (nghỉ hưu, thất
     * nghiệp, tự kinh doanh) — đó là NaN hợp lệ, không được điền bừa một nghề
     * vào. Bản thân việc thiếu `OCCUPATION_TYPE` đã là tín hiệu: nhóm thiếu vỡ
     * nợ 6,51% so với 8,79% của nhóm có khai.
     */
    public function homeCredit(): ?string
    {
        return match ($this) {
            self::OFFICE_STAFF => 'Core staff',
            self::MANAGER => 'Managers',
            self::ACCOUNTANT => 'Accountants',
            self::IT_STAFF => 'IT staff',
            self::TEACHER => 'High skill tech staff',
            self::MEDICAL_STAFF => 'Medicine staff',
            self::SALES_STAFF => 'Sales staff',
            self::DRIVER => 'Drivers',
            self::SECURITY_STAFF => 'Security staff',
            self::SERVICE_STAFF => 'Private service staff',
            self::LABORER => 'Laborers',
            self::FARMER => 'Low-skill Laborers',
            self::SELF_EMPLOYED, self::RETIRED, self::UNEMPLOYED, self::OTHER => null,
        };
    }

    /**
     * Nhóm không đi làm hưởng lương. Với họ, "số năm đi làm" = 0 là giá trị
     * hợp lệ chứ không phải bỏ trống — tương ứng sentinel `DAYS_EMPLOYED
     * = 365243` của Home Credit (18,01% dữ liệu, gần như trùng khít nhóm nghỉ
     * hưu và thất nghiệp).
     */
    public function isOutOfWorkforce(): bool
    {
        return $this === self::RETIRED || $this === self::UNEMPLOYED;
    }
}
