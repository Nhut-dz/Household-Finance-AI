<?php

namespace App\Enums;

/**
 * Nhu cầu tài chính, tương ứng cột tblfinancial_goals.goal_type.
 */
enum GoalTypeEnum: string
{
    case BUY_HOUSE = 'buy_house';
    case BUY_CAR = 'buy_car';
    case BUY_LAND = 'buy_land';
    case LOAN = 'loan';
    case OTHER = 'other';

    /**
     * Nhãn tiếng Việt hiển thị trên form.
     */
    public function label(): string
    {
        return match ($this) {
            self::BUY_HOUSE => 'Mua nhà',
            self::BUY_CAR => 'Mua xe',
            self::BUY_LAND => 'Mua đất',
            self::LOAN => 'Vay vốn',
            self::OTHER => 'Khác',
        };
    }
}
