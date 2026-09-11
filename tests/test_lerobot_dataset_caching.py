from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from starVLA.dataloader.gr00t_lerobot import datasets


def _bare_dataset(
    trajectory_ids: tuple[int, ...] = (11, 12),
    trajectory_lengths: tuple[int, ...] = (3, 3),
) -> datasets.LeRobotSingleDataset:
    dataset = datasets.LeRobotSingleDataset.__new__(datasets.LeRobotSingleDataset)
    dataset._trajectory_ids = np.asarray(trajectory_ids)
    dataset._trajectory_lengths = np.asarray(trajectory_lengths)
    dataset._trajectory_id_to_index = {
        trajectory_id: index for index, trajectory_id in enumerate(trajectory_ids)
    }
    dataset.curr_traj_id = None
    dataset.curr_traj_data = None
    dataset._curr_traj_array_cache = {}
    return dataset


def test_trajectory_index_uses_precomputed_lookup_instead_of_np_where(monkeypatch):
    dataset = _bare_dataset(trajectory_ids=(4, 92, 301))

    def fail_if_called(*args, **kwargs):
        raise AssertionError("get_trajectory_index must not scan all trajectory IDs with np.where")

    monkeypatch.setattr(datasets.np, "where", fail_if_called)

    assert dataset.get_trajectory_index(92) == 1
    with pytest.raises(ValueError, match="999"):
        dataset.get_trajectory_index(999)


def test_trajectory_data_cache_updates_and_invalidates_stacked_columns(tmp_path, monkeypatch):
    dataset = _bare_dataset()
    dataset._dataset_path = tmp_path
    dataset._data_path_pattern = "data/chunk-{episode_chunk}/episode-{episode_index}.parquet"
    dataset._chunk_size = 100

    paths = {}
    frames = {}
    for trajectory_id in (11, 12):
        path = tmp_path / f"data/chunk-0/episode-{trajectory_id}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        paths[trajectory_id] = path
        frames[trajectory_id] = pd.DataFrame({"trajectory": [trajectory_id]})

    read_paths = []

    def fake_read_parquet(path):
        path = type(paths[11])(path)
        read_paths.append(path)
        trajectory_id = int(path.stem.removeprefix("episode-"))
        return frames[trajectory_id]

    monkeypatch.setattr(datasets.pd, "read_parquet", fake_read_parquet)

    first = dataset.get_trajectory_data(11)
    assert first is frames[11]
    assert dataset.curr_traj_id == 11
    assert dataset.curr_traj_data is first
    assert dataset._curr_traj_array_cache == {}

    cached_column = np.asarray([[1.0, 2.0]])
    dataset._curr_traj_array_cache["observation.state"] = cached_column
    assert dataset.get_trajectory_data(11) is first
    assert read_paths == [paths[11]]
    assert dataset._curr_traj_array_cache["observation.state"] is cached_column

    second = dataset.get_trajectory_data(12)
    assert second is frames[12]
    assert dataset.curr_traj_id == 12
    assert dataset.curr_traj_data is second
    assert read_paths == [paths[11], paths[12]]
    assert dataset._curr_traj_array_cache == {}


def test_state_and_action_columns_are_each_stacked_once_per_trajectory(monkeypatch):
    dataset = _bare_dataset(trajectory_ids=(11,), trajectory_lengths=(3,))
    dataset.curr_traj_id = 11
    dataset.curr_traj_data = pd.DataFrame(
        {
            "observation.state": [
                np.asarray([1.0, 2.0, 3.0]),
                np.asarray([4.0, 5.0, 6.0]),
                np.asarray([7.0, 8.0, 9.0]),
            ],
            "action": [
                np.asarray([10.0, 11.0, 12.0]),
                np.asarray([13.0, 14.0, 15.0]),
                np.asarray([16.0, 17.0, 18.0]),
            ],
        }
    )
    dataset._delta_indices = {
        "state.arm": np.asarray([0, 1]),
        "state.gripper": np.asarray([0, 1]),
        "action.arm": np.asarray([0, 1]),
        "action.gripper": np.asarray([0, 1]),
    }
    dataset._lerobot_modality_meta = SimpleNamespace(
        state={
            "arm": SimpleNamespace(original_key="observation.state", start=0, end=2),
            "gripper": SimpleNamespace(original_key="observation.state", start=2, end=3),
        },
        action={
            "arm": SimpleNamespace(original_key="action", start=0, end=2),
            "gripper": SimpleNamespace(original_key="action", start=2, end=3),
        },
    )

    stack_calls = Counter()
    original_stack = datasets.np.stack

    def counting_stack(arrays, *args, **kwargs):
        stack_calls[getattr(arrays, "name", None)] += 1
        return original_stack(arrays, *args, **kwargs)

    monkeypatch.setattr(datasets.np, "stack", counting_stack)

    state_arm = dataset.get_state_or_action(11, "state", "state.arm", base_index=0)
    state_gripper = dataset.get_state_or_action(11, "state", "state.gripper", base_index=0)
    action_arm = dataset.get_state_or_action(11, "action", "action.arm", base_index=0)
    action_gripper = dataset.get_state_or_action(11, "action", "action.gripper", base_index=0)
    dataset.get_state_or_action(11, "state", "state.arm", base_index=1)
    dataset.get_state_or_action(11, "action", "action.arm", base_index=1)

    assert stack_calls == Counter({"observation.state": 1, "action": 1})
    np.testing.assert_array_equal(state_arm, [[1.0, 2.0], [4.0, 5.0]])
    np.testing.assert_array_equal(state_gripper, [[3.0], [6.0]])
    np.testing.assert_array_equal(action_arm, [[10.0, 11.0], [13.0, 14.0]])
    np.testing.assert_array_equal(action_gripper, [[12.0], [15.0]])

