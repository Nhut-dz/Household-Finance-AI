<?php

namespace App\Enums\Routes;

use App\Trait\Enums\RouteTrait;

enum WebEnum: string
{
    use RouteTrait;

    case HOME = 'home';
}
