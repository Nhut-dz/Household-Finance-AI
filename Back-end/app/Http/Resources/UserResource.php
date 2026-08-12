<?php

namespace App\Http\Resources;

use App\Models\User;
use Illuminate\Http\Request;
use Illuminate\Http\Resources\Json\JsonResource;
use OpenApi\Attributes as OA;

/**
 * @mixin User
 */
#[OA\Schema(
    schema: 'User',
    title: 'Người dùng',
    properties: [
        new OA\Property(property: 'id', type: 'integer', example: 1),
        new OA\Property(property: 'name', type: 'string', example: 'Nguyễn Văn A'),
        new OA\Property(property: 'email', type: 'string', format: 'email', example: 'a@example.com'),
    ],
    type: 'object'
)]
class UserResource extends JsonResource
{
    /**
     * FE dùng khoá "name"; DB lưu ở cột full_name.
     *
     * @return array<string, mixed>
     */
    public function toArray(Request $request): array
    {
        return [
            'id' => $this->id,
            'name' => $this->full_name,
            'email' => $this->email,
        ];
    }
}
