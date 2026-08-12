<?php

namespace App\Repositories;

interface BaseRepositoryInterface
{
    public function get();
    public function all($perPage = 10);
    public function find($id, $columns = ['*']);
    public function findOrFail($id, $columns = ['*']);
    public function insert(array $data);
    public function create(array $data);
    public function update(array $data, $id);
    public function delete($id);
    public function updateOrCreate(array $attributes, array $values = []);
    public function updateOrInsert(array $attributes, array $values = []);
    public function getTable();
    public function exists($id);
    public function findWith($id, array $relations = [], $columns = ['*']);
    public function chunk(int $size, callable $callback);
}
