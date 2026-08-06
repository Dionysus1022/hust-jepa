from types import SimpleNamespace

import pytest
import torch
from torch import nn

from starVLA.model.framework.VLA_JEPA import VLA_JEPA
from starVLA.model.modules.world_model.vj2_tensors import (
    merge_multiview_tubelet_embeddings,
    split_video_into_tubelets,
)


class _DummyBidirectionalEncoder(nn.Module):
    config = SimpleNamespace(tubelet_size=2)

    def get_vision_features(self, pixel_values_videos):
        return pixel_values_videos.float().mean(dim=(1, 2, 3, 4)).reshape(-1, 1, 1)


def _make_framework():
    framework = VLA_JEPA.__new__(VLA_JEPA)
    nn.Module.__init__(framework)
    framework.config = SimpleNamespace(
        framework=SimpleNamespace(vj2_model=SimpleNamespace(num_frames=8))
    )
    framework.vj_encoder = _DummyBidirectionalEncoder()
    return framework


def test_framework_encoding_path_is_leakage_free():
    videos = torch.arange(16, dtype=torch.float32).reshape(2, 1, 8, 1, 1, 1)
    videos[1, :, :4] = videos[0, :, :4]
    input_videos = videos.reshape(2, 8, 1, 1, 1)

    states, tokens_per_state = _make_framework()._encode_video_states(
        input_videos,
        batch_size=2,
        num_views=1,
        num_frames=8,
    )

    assert tokens_per_state == 1
    assert states.shape == (2, 4, 1)
    assert torch.equal(states[0, :2], states[1, :2])
    assert not torch.equal(states[0, 2:], states[1, 2:])
    assert not states.requires_grad


def test_multiview_embeddings_stay_with_their_batch_element():
    videos = torch.empty(2, 2, 4, 1, 1, 1)
    for batch_index in range(2):
        for view_index in range(2):
            for step_index in range(2):
                value = 100 * batch_index + 10 * view_index + step_index
                start = step_index * 2
                videos[batch_index, view_index, start : start + 2] = value

    tubelets, dimensions = split_video_into_tubelets(videos, tubelet_size=2)
    embeddings = tubelets.mean(dim=(1, 2, 3, 4)).reshape(-1, 1, 1)
    states, _ = merge_multiview_tubelet_embeddings(embeddings, *dimensions)

    expected = torch.tensor(
        [
            [[0.0, 10.0], [1.0, 11.0]],
            [[100.0, 110.0], [101.0, 111.0]],
        ]
    )
    assert torch.equal(states, expected)


def test_temporal_states_align_as_next_step_targets():
    input_videos = torch.arange(8, dtype=torch.float32).reshape(1, 8, 1, 1, 1)
    states, tokens_per_state = _make_framework()._encode_video_states(
        input_videos,
        batch_size=1,
        num_views=1,
        num_frames=8,
    )

    input_states = states[:, :-tokens_per_state]
    target_states = states[:, tokens_per_state:]

    assert input_states.shape == target_states.shape == (1, 3, 1)
    assert torch.equal(input_states[0, 1:], target_states[0, :-1])


def test_frame_count_must_match_prompt_and_tubelet_size():
    framework = _make_framework()

    with pytest.raises(ValueError, match="must be divisible"):
        split_video_into_tubelets(torch.zeros(1, 1, 7, 1, 1, 1), tubelet_size=2)

    with pytest.raises(ValueError, match="Expected 8 video frames"):
        framework._encode_video_states(
            torch.zeros(1, 6, 1, 1, 1),
            batch_size=1,
            num_views=1,
            num_frames=6,
        )
