<?php

use App\Enums\Routes\WebEnum;
use Illuminate\Support\Facades\Route;

Route::get('/', function () {
    return view('welcome');
})->name(WebEnum::HOME->routeName());
