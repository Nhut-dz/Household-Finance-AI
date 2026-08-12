<?php

namespace App\Repositories;

use Illuminate\Database\Eloquent\Model;

class BaseRepository implements BaseRepositoryInterface
{
    protected $model;

    public function __construct(Model $model)
    {
        $this->model = $model;
    }

    public function get()
    {
        return $this->model::when(in_array('order', $this->model->getFillable()), fn($q) =>  $q->orderBy('order', 'ASC'))
            ->orderBy($this->model->getKeyName(), 'DESC')
            ->get();
    }

    public function all($perPage = 10)
    {
        return $this->model::when(in_array('order', $this->model->getFillable()), fn($q) =>  $q->orderBy('order', 'ASC'))
            ->orderBy($this->model->getKeyName(), 'DESC')
            ->paginate($perPage);
    }

    public function find($id, $columns = ['*'])
    {
        return $this->model->find($id, $columns);
    }

    public function findOrFail($id, $columns = ['*'])
    {
        return $this->model->findOrFail($id, $columns);
    }

    public function create(array $data)
    {
        $fillables =  $this->model->getFillable();
        $fillables = array_filter($fillables, fn($fillable) => in_array($fillable, ['created_by', 'updated_by']));
        foreach ($fillables as $fillable) {
            if ($auth_id = auth()->id()) $data[$fillable] = $auth_id;
        }
        return $this->model->create($data);
    }

    public function insert(array $data)
    {
        $fillables =  $this->model->getFillable();
        $fillables = array_filter($fillables, fn($fillable) => in_array($fillable, ['created_by', 'updated_by', 'created_at', 'updated_at']));
        $data = array_map(function ($item) use ($fillables) {
            foreach ($fillables as $fillable) {
                if (in_array($fillable, ['created_by', 'updated_by'])) {
                    if ($auth_id = auth()->id()) $item[$fillable] = $auth_id;
                }
                if (in_array($fillable, ['created_at', 'updated_at'])) $item[$fillable] = !empty($item[$fillable]) ? $item[$fillable] : now();
            }
            return $item;
        }, $data);
        return $this->model->insert($data);
    }


    public function update(array $data, $id)
    {
        $item = $this->find($id);
        if (!$item) return NULL;
        $fillables =  $this->model->getFillable();
        $fillables = array_filter($fillables, fn($fillable) => in_array($fillable, ['updated_by']));
        foreach ($fillables as $fillable) {
            if ($auth_id = auth()->id()) $data[$fillable] = $auth_id;
        }
        $item->update($data);
        return $item->refresh();
    }

    public function delete($id)
    {
        $item = $this->find($id);
        if (!$item) {
            return false;
        }
        return $item->delete();
    }

    public function updateOrCreate(array $attributes, array $values = [])
    {
        $fillables =  $this->model->getFillable();
        $fillables = array_filter($fillables, fn($fillable) => in_array($fillable, ['created_by', 'updated_by', 'created_at', 'updated_at']));
        foreach ($fillables as $fillable) {
            if (in_array($fillable, ['created_by', 'updated_by'])) {
                if ($auth_id = auth()->id()) $values[$fillable] = $auth_id;
            }
            if (in_array($fillable, ['created_at', 'updated_at'])) $values[$fillable] = !empty($values[$fillable]) ? $values[$fillable] : now();
        }

        return $this->model->updateOrCreate($attributes, $values);
    }

    public function updateOrInsert(array $attributes, array $values = [])
    {
        $fillables =  $this->model->getFillable();
        $fillables = array_filter($fillables, fn($fillable) => in_array($fillable, ['created_by', 'updated_by', 'created_at', 'updated_at']));
        foreach ($fillables as $fillable) {
            if (in_array($fillable, ['created_by', 'updated_by'])) {
                if ($auth_id = auth()->id()) $values[$fillable] = $auth_id;
            }
            if (in_array($fillable, ['created_at', 'updated_at'])) $values[$fillable] = !empty($values[$fillable]) ? $values[$fillable] : now();
        }

        return $this->model->updateOrInsert($attributes, $values);
    }

    public function getTable()
    {
        return $this->model->getTable();
    }

    public function exists($id)
    {
        return $this->model->where($this->model->getKeyName(), $id)->exists();
    }

    public function findWith($id, array $relations = [], $columns = ['*'])
    {
        return $this->model->query()->with($relations)->find($id, $columns);
    }

    public function chunk(int $size, callable $callback){
        return $this->model->chunk($size, $callback);
    }
}
