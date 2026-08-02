import importlib
import sys
import types
from types import SimpleNamespace


def _import_sharded_eval_with_stubs(monkeypatch):
    libero_pkg = types.ModuleType("libero")
    libero_libero = types.ModuleType("libero.libero")
    libero_envs = types.ModuleType("libero.libero.envs")
    model_interface = types.ModuleType("examples.LIBERO.model2libero_interface")
    imageio = types.ModuleType("imageio")
    tqdm = types.ModuleType("tqdm")
    tyro = types.ModuleType("tyro")
    requests = types.ModuleType("requests")

    libero_libero.benchmark = SimpleNamespace(get_benchmark_dict=lambda: {})
    libero_libero.get_libero_path = lambda key: "/unused"
    libero_envs.OffScreenRenderEnv = object
    model_interface.M1Inference = object
    imageio.mimwrite = lambda *args, **kwargs: None
    tqdm.tqdm = lambda iterable, *args, **kwargs: iterable
    tyro.cli = lambda *args, **kwargs: None
    libero_pkg.libero = libero_libero

    monkeypatch.setitem(sys.modules, "imageio", imageio)
    monkeypatch.setitem(sys.modules, "tqdm", tqdm)
    monkeypatch.setitem(sys.modules, "tyro", tyro)
    monkeypatch.setitem(sys.modules, "requests", requests)
    monkeypatch.setitem(sys.modules, "libero", libero_pkg)
    monkeypatch.setitem(sys.modules, "libero.libero", libero_libero)
    monkeypatch.setitem(sys.modules, "libero.libero.envs", libero_envs)
    monkeypatch.setitem(sys.modules, "examples.LIBERO.model2libero_interface", model_interface)
    sys.modules.pop("examples.LIBERO.eval_libero_sharded", None)
    return importlib.import_module("examples.LIBERO.eval_libero_sharded")


def test_resolve_task_range_uses_explicit_start_and_end(monkeypatch):
    eval_libero = _import_sharded_eval_with_stubs(monkeypatch)
    args = SimpleNamespace(task_start=10, task_end=20, shard_index=0, num_shards=1)

    assert eval_libero.resolve_task_range(num_tasks=100, args=args) == range(10, 20)


def test_resolve_task_range_splits_tasks_by_shard(monkeypatch):
    eval_libero = _import_sharded_eval_with_stubs(monkeypatch)
    args = SimpleNamespace(task_start=None, task_end=None, shard_index=2, num_shards=4)

    assert eval_libero.resolve_task_range(num_tasks=10, args=args) == range(2, 10, 4)


def test_resolve_task_range_clamps_explicit_end(monkeypatch):
    eval_libero = _import_sharded_eval_with_stubs(monkeypatch)
    args = SimpleNamespace(task_start=8, task_end=99, shard_index=0, num_shards=1)

    assert eval_libero.resolve_task_range(num_tasks=10, args=args) == range(8, 10)


def test_resolve_task_range_rejects_invalid_shard_index(monkeypatch):
    eval_libero = _import_sharded_eval_with_stubs(monkeypatch)
    args = SimpleNamespace(task_start=None, task_end=None, shard_index=4, num_shards=4)

    try:
        eval_libero.resolve_task_range(num_tasks=10, args=args)
    except ValueError as exc:
        assert "shard_index" in str(exc)
    else:
        raise AssertionError("Expected invalid shard_index to fail")
