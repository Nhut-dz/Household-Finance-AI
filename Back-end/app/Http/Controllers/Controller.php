<?php

namespace App\Http\Controllers;

use App\Trait\ApiResponseTrait;
use Illuminate\Routing\Controller as BaseController;
use OpenApi\Attributes as OA;

#[OA\Info(
    version: '1.0.0',
    description: 'Tài liệu API của Household Finance, sinh bằng L5 Swagger.',
    title: 'Household Finance API',
    contact: new OA\Contact(email: 'nhut3475@gmail.com'),
    license: new OA\License(
        name: 'Apache 2.0',
        url: 'http://www.apache.org/licenses/LICENSE-2.0.html'
    )
)]
#[OA\Server(
    url: L5_SWAGGER_CONST_HOST,
    description: 'API Server'
)]
#[OA\SecurityScheme(
    securityScheme: 'bearerAuth',
    type: 'http',
    description: 'Nhập Bearer token của bạn vào đây',
    bearerFormat: 'JWT',
    scheme: 'bearer'
)]
abstract class Controller extends BaseController
{
    use ApiResponseTrait;
}
