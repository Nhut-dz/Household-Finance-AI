<?php

namespace App\Enums\Routes;

use App\Trait\Enums\RouteTrait;

enum ApiEnum: string
{
    use RouteTrait;

    case USER_SHOW = 'user.show';

    case AUTH_REGISTER = 'auth.register';
    case AUTH_LOGIN = 'auth.login';
    case AUTH_LOGOUT = 'auth.logout';

    case HOUSEHOLD_STORE = 'household.store';
    case HOUSEHOLD_LATEST = 'household.latest';
    case HOUSEHOLD_SHOW = 'household.show';
    case HOUSEHOLD_UPDATE = 'household.update';
    case HOUSEHOLD_DESTROY = 'household.destroy';

    case HOUSEHOLD_PROPOSAL = 'household.proposal';
    case HOUSEHOLD_PREDICTION = 'household.prediction';

    case HOUSEHOLD_LOAN_APPLICATION_SHOW = 'household.loan-application.show';
    case HOUSEHOLD_LOAN_APPLICATION_STORE = 'household.loan-application.store';
    case HOUSEHOLD_LOAN_APPLICATION_DESTROY = 'household.loan-application.destroy';

    case HOUSEHOLD_MESSAGE_INDEX = 'household.message.index';
    case HOUSEHOLD_MESSAGE_STORE = 'household.message.store';

    case HOUSEHOLD_CONVERSATION_INDEX = 'household.conversation.index';
    case HOUSEHOLD_CONVERSATION_MESSAGES = 'household.conversation.messages';
    case HOUSEHOLD_CONVERSATION_DESTROY = 'household.conversation.destroy';
}
