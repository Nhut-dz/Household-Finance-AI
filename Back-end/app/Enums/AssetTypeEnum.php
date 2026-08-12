<?php

namespace App\Enums;

/**
 * Loại tài sản đang sở hữu, tương ứng cột tblassets.asset_type.
 */
enum AssetTypeEnum: string
{
    case HOUSE = 'house';
    case CAR = 'car';
    case LAND = 'land';
    case OTHER = 'other';

    /**
     * Nhãn tiếng Việt hiển thị trên form.
     */
    public function label(): string
    {
        return match ($this) {
            self::HOUSE => 'Nhà',
            self::CAR => 'Xe',
            self::LAND => 'Đất',
            self::OTHER => 'Khác',
        };
    }
}
