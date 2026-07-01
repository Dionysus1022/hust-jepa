import importlib
import sys
import types
from types import SimpleNamespace


def _import_eval_libero_with_stubs(monkeypatch):
    libero_pkg = types.ModuleType("libero")
    libero_libero = types.ModuleType("libero.libero")
    libero_envs = types.ModuleType("libero.libero.envs")
    model_interface = types.ModuleType("examples.LIBERO.model2libero_interface")

    libero_libero.benchmark = SimpleNamespace(get_benchmark_dict=lambda: {})
    libero_libero.get_libero_path = lambda key: "/unused"
    libero_envs.OffScreenRenderEnv = object
    model_interface.M1Inference = object
    libero_pkg.libero = libero_libero

    monkeypatch.setitem(sys.modules, "libero", libero_pkg)
    monkeypatch.setitem(sys.modules, "libero.libero", libero_libero)
    monkeypatch.setitem(sys.modules, "libero.libero.envs", libero_envs)
    monkeypatch.setitem(sys.modules, "examples.LIBERO.model2libero_interface", model_interface)
    sys.modules.pop("examples.LIBERO.eval_libero", None)
    return importlib.import_module("examples.LIBERO.eval_libero")


def test_get_libero_env_passes_bddl_file_name_as_string(monkeypatch, tmp_path):
    eval_libero = _import_eval_libero_with_stubs(monkeypatch)
    captured = {}

    class FakeOffScreenRenderEnv:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def seed(self, seed):
            captured["seed"] = seed

    monkeypatch.setattr(eval_libero, "get_libero_path", lambda key: tmp_path)
    monkeypatch.setattr(eval_libero, "OffScreenRenderEnv", FakeOffScreenRenderEnv)

    task = SimpleNamespace(
        language="put the bowl on the plate",
        problem_folder="libero_goal",
        bddl_file="put_bowl_on_plate.bddl",
    )

    env, task_description = eval_libero._get_libero_env(task, resolution=256, seed=7)

    assert isinstance(env, FakeOffScreenRenderEnv)
    assert task_description == "put the bowl on the plate"
    assert isinstance(captured["bddl_file_name"], str)
    assert captured["bddl_file_name"].endswith("libero_goal/put_bowl_on_plate.bddl")
    assert captured["camera_heights"] == 256
    assert captured["camera_widths"] == 256
    assert captured["seed"] == 7


class _FakeTaskSuite:
    def __init__(self, n_tasks):
        self.n_tasks = n_tasks


def test_vanilla_libero_benchmark_mode_rejects_libero_plus_task_counts(monkeypatch):
    eval_libero = _import_eval_libero_with_stubs(monkeypatch)
    task_suite = _FakeTaskSuite(n_tasks=408)
    args = SimpleNamespace(task_suite_name="libero_goal", benchmark_mode="libero")

    try:
        eval_libero._validate_benchmark_mode(task_suite, args)
    except ValueError as exc:
        assert "Expected vanilla LIBERO suite `libero_goal` to have 10 tasks" in str(exc)
        assert "LIBERO-plus" in str(exc)
    else:
        raise AssertionError("Expected vanilla LIBERO mode to reject LIBERO-plus task counts")


def test_libero_plus_benchmark_mode_allows_expanded_task_counts(monkeypatch):
    eval_libero = _import_eval_libero_with_stubs(monkeypatch)
    task_suite = _FakeTaskSuite(n_tasks=408)
    args = SimpleNamespace(task_suite_name="libero_goal", benchmark_mode="libero_plus")

    eval_libero._validate_benchmark_mode(task_suite, args)


def test_vanilla_libero_benchmark_mode_rejects_plus_only_suite(monkeypatch):
    eval_libero = _import_eval_libero_with_stubs(monkeypatch)
    task_suite = _FakeTaskSuite(n_tasks=1000)
    args = SimpleNamespace(task_suite_name="libero_mix", benchmark_mode="libero")

    try:
        eval_libero._validate_benchmark_mode(task_suite, args)
    except ValueError as exc:
        assert "`libero_mix` is not a vanilla LIBERO suite" in str(exc)
    else:
        raise AssertionError("Expected vanilla LIBERO mode to reject plus-only suites")
