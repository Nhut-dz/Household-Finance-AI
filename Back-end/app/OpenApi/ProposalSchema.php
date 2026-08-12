<?php

namespace App\OpenApi;

use OpenApi\Attributes as OA;

/**
 * Lớp chỉ dùng để khai báo schema cho tài liệu Swagger của phương án đề xuất.
 * Dữ liệu thật do App\Services\ProposalService dựng ra.
 */
#[OA\Schema(
    schema: 'Proposal',
    title: 'Phương án đề xuất',
    properties: [
        new OA\Property(property: 'household_id', type: 'integer', example: 12),
        new OA\Property(property: 'summary', type: 'string', nullable: true),
        new OA\Property(
            property: 'overview',
            properties: [
                new OA\Property(property: 'net_income', type: 'number', example: 18000000),
                new OA\Property(property: 'current_debt', type: 'number', example: 500000000),
                new OA\Property(property: 'current_savings', type: 'number', example: 150000000),
                new OA\Property(property: 'target_accumulation', type: 'number', nullable: true, example: 1800000000),
            ],
            type: 'object'
        ),
        new OA\Property(
            property: 'savings_plan',
            properties: [
                new OA\Property(property: 'monthly_contribution', type: 'number', nullable: true, example: 9000000),
                new OA\Property(property: 'term_months', type: 'integer', nullable: true, example: 24),
                new OA\Property(property: 'interest_rate', type: 'number', nullable: true, description: '%/năm', example: 12),
                new OA\Property(property: 'note', type: 'string', nullable: true),
            ],
            type: 'object',
            nullable: true
        ),
        new OA\Property(
            property: 'investment_plan',
            properties: [
                new OA\Property(property: 'allocation', type: 'string', nullable: true, example: '40% / 40% / 20%'),
                new OA\Property(property: 'risk_level', type: 'string', nullable: true, example: 'Rủi ro vừa phải'),
                new OA\Property(property: 'note', type: 'string', nullable: true),
            ],
            type: 'object',
            nullable: true
        ),
        new OA\Property(
            property: 'loan_plan',
            properties: [
                new OA\Property(property: 'product', type: 'string', nullable: true, example: 'Vay mua nhà trả góp'),
                new OA\Property(property: 'credit_limit', type: 'number', nullable: true, example: 1200000000),
                new OA\Property(property: 'monthly_payment', type: 'number', nullable: true, example: 32000000),
                new OA\Property(property: 'note', type: 'string', nullable: true),
            ],
            type: 'object',
            nullable: true
        ),
        new OA\Property(
            property: 'roadmap',
            type: 'array',
            items: new OA\Items(
                properties: [
                    new OA\Property(property: 'step', type: 'integer', example: 1),
                    new OA\Property(property: 'text', type: 'string'),
                ],
                type: 'object'
            )
        ),
        new OA\Property(property: 'generated_at', type: 'string', format: 'date-time', nullable: true),
    ],
    type: 'object'
)]
final class ProposalSchema {}
