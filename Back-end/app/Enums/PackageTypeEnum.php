<?php

namespace App\Enums;

/**
 * Loại gói khuyến nghị, tương ứng cột tblrecommendation_packages.package_type.
 * Giá trị viết hoa theo đúng dữ liệu do nhóm Python ghi vào DB.
 */
enum PackageTypeEnum: string
{
    case SAVING = 'SAVING';
    case INVESTMENT = 'INVESTMENT';
    case LOAN = 'LOAN';
}
