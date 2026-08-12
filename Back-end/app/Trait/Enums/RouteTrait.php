<?php

namespace App\Trait\Enums;

trait RouteTrait
{
    /**
     * Tiền tố của route, lấy từ hằng MODULE nếu enum có khai báo.
     * Enum không khai báo MODULE thì không có tiền tố.
     */
    private function prefix(): string
    {
        if (!defined(static::class . '::MODULE')) {
            return '';
        }

        $module = constant(static::class . '::MODULE');

        return !empty($module) ? $module . '.' : '';
    }

    /**
     * Tên route đầy đủ, ví dụ: 'user.show' hoặc 'admin.user.show'.
     */
    public function routeName(): string
    {
        return $this->prefix() . $this->value;
    }

    /**
     * URL đã resolve của route.
     */
    public function route(mixed $parameters = [], bool $absolute = true): string
    {
        return route($this->routeName(), $parameters, $absolute);
    }
}
