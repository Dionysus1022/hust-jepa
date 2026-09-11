from __future__ import annotations

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from starVLA.dataloader.lerobot_datasets import (
    _augment_video_frames,
    _select_augmented_image_indices,
    causal_state_delta_indices,
    collate_fn,
)
from starVLA.model.modules.history_position_encoding import HistorySinusoidalEncoding
from starVLA.model.framework.VLA_JEPA import SoftQueryConnector
from starVLA.model.modules.action_model import GR00T_ActionHeader as gr00t_action_header


def _fake_sample():
    image = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
    return {
        "video": np.random.randint(0, 255, (2, 4, 32, 32, 3), dtype=np.uint8),
        "image": [image, image.copy()],
        "lang": "put the object in the bowl",
        "action": np.random.randn(7, 7).astype(np.float16),
        "state": np.random.randn(1, 8).astype(np.float16),
    }


def test_causal_state_history_uses_past_through_current_indices():
    assert causal_state_delta_indices(8) == [-7, -6, -5, -4, -3, -2, -1, 0]


def test_state_history_position_encoding_distinguishes_timesteps():
    encoding = HistorySinusoidalEncoding(hidden_size=16, max_history_len=8)
    tokens = torch.zeros(2, 8, 16)

    positioned = encoding(tokens)

    assert positioned.shape == tokens.shape
    assert not torch.equal(positioned[:, 0], positioned[:, -1])


def test_short_state_history_is_right_aligned_to_current_position():
    encoding = HistorySinusoidalEncoding(hidden_size=16, max_history_len=8)

    full = encoding(torch.zeros(1, 8, 16))
    current_only = encoding(torch.zeros(1, 1, 16))

    torch.testing.assert_close(current_only[:, 0], full[:, -1])


def test_lerobot_collate_keeps_fast_path_when_augmentation_disabled():
    sample = _fake_sample()
    batch = collate_fn([sample, sample], augmentation={"enabled": False})

    assert batch["video"].shape == (2, 2, 4, 32, 32, 3)
    assert batch["action"].shape == (2, 7, 7)
    assert batch["state"].shape == (2, 1, 8)
    assert len(batch["image"]) == 2
    assert len(batch["image"][0]) == 2


def test_lerobot_collate_augments_vla_images_but_keeps_vjepa_video_clean():
    sample = _fake_sample()
    original_video = sample["video"].copy()
    original_image = np.asarray(sample["image"][0]).copy()
    augmentation = {
        "enabled": True,
        "random_brightness": [0.5, 0.5],
        "augment_order": ["random_brightness"],
    }

    batch = collate_fn([sample], augmentation=augmentation)

    assert batch["video"].shape == (1, 2, 4, 32, 32, 3)
    assert batch["video"].dtype == torch.uint8
    assert torch.equal(batch["video"][0], torch.from_numpy(original_video))
    assert not np.array_equal(np.asarray(batch["image"][0][0]), original_image)
    assert batch["action"].shape == (1, 7, 7)


def test_lerobot_global_augmentation_probability_can_keep_sample_clean(monkeypatch):
    sample = _fake_sample()
    original_video = sample["video"].copy()
    original_images = [np.asarray(image).copy() for image in sample["image"]]
    augmentation = {
        "enabled": True,
        "probability": 0.5,
        "random_brightness": [0.5, 0.5],
        "augment_order": ["random_brightness"],
    }
    monkeypatch.setattr(np.random, "random", lambda: 0.75)

    batch = collate_fn([sample], augmentation=augmentation)

    assert torch.equal(batch["video"][0], torch.from_numpy(original_video))
    for actual, expected in zip(batch["image"][0], original_images):
        assert np.array_equal(np.asarray(actual), expected)


def test_lerobot_camera_aug_policy_uses_quarter_buckets(monkeypatch):
    augmentation = {
        "enabled": True,
        "camera_aug_policy": "third_wrist_quarters",
        "third_camera_index": 0,
        "wrist_camera_index": 1,
    }

    for bucket, expected_indices in enumerate([set(), {0}, {1}, {0, 1}]):
        monkeypatch.setattr(np.random, "randint", lambda high, bucket=bucket: bucket)

        assert _select_augmented_image_indices(augmentation, num_images=2) == expected_indices


def test_lerobot_camera_aug_policy_augments_only_selected_camera(monkeypatch):
    sample = _fake_sample()
    original_video = sample["video"].copy()
    original_images = [np.asarray(image).copy() for image in sample["image"]]
    monkeypatch.setattr(np.random, "randint", lambda high: 2)
    augmentation = {
        "enabled": True,
        "camera_aug_policy": "third_wrist_quarters",
        "third_camera_index": 0,
        "wrist_camera_index": 1,
        "random_brightness": [0.5, 0.5],
        "augment_order": ["random_brightness"],
    }

    batch = collate_fn([sample], augmentation=augmentation)

    assert torch.equal(batch["video"][0], torch.from_numpy(original_video))
    assert np.array_equal(np.asarray(batch["image"][0][0]), original_images[0])
    assert not np.array_equal(np.asarray(batch["image"][0][1]), original_images[1])


def test_lerobot_image_augmentation_supports_camera_robustness_ops():
    image = np.full((32, 32, 3), 128, dtype=np.uint8)
    augmentation = {
        "enabled": True,
        "random_gamma": [0.8],
        "random_exposure_ev": [0.25, 0.25],
        "gaussian_noise_std": 0.015,
        "random_rotation_degrees": [8, 8],
        "augment_order": [
            "random_gamma",
            "random_exposure_ev",
            "gaussian_noise",
            "random_rotation",
        ],
    }

    augmented = _augment_video_frames(image, augmentation)

    assert augmented.shape == image.shape
    assert augmented.dtype == np.uint8
    assert not np.array_equal(augmented, image)


def test_lerobot_random_affine_supports_fractional_translation(monkeypatch):
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[12:20, 12:20] = 255
    augmentation = {
        "enabled": True,
        "random_affine": {
            "translate": [0.25, 0.0],
            "scale": [1.0, 1.0],
            "degrees": [0.0, 0.0],
            "probability": 1.0,
        },
        "augment_order": ["random_affine"],
    }
    monkeypatch.setattr(np.random, "random", lambda: 0.0)
    monkeypatch.setattr(np.random, "uniform", lambda low, high: high)

    augmented = _augment_video_frames(image, augmentation)

    original_x = np.argwhere(image[..., 0] > 127)[:, 1].mean()
    augmented_x = np.argwhere(augmented[..., 0] > 127)[:, 1].mean()
    assert augmented_x > original_x


def test_lerobot_random_affine_keeps_vjepa_video_clean(monkeypatch):
    sample = _fake_sample()
    original_video = sample["video"].copy()
    augmentation = {
        "enabled": True,
        "random_affine": {
            "translate": [0.25, 0.0],
            "scale": [1.0, 1.0],
            "degrees": [0.0, 0.0],
            "probability": 1.0,
        },
        "augment_order": ["random_affine"],
    }
    monkeypatch.setattr(np.random, "random", lambda: 0.0)
    monkeypatch.setattr(np.random, "uniform", lambda low, high: high)

    batch = collate_fn([sample], augmentation=augmentation)

    assert torch.equal(batch["video"][0], torch.from_numpy(original_video))


def test_action_head_can_disable_independent_state_encoder(monkeypatch):
    class DummyDiT(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()

    monkeypatch.setattr(gr00t_action_header, "DiT", DummyDiT)
    config = OmegaConf.create(
        {
            "framework": {
                "action_model": {
                    "action_model_type": "DiT-B",
                    "hidden_size": 32,
                    "action_dim": 7,
                    "state_dim": 7,
                    "use_state_encoder": False,
                    "future_action_window_size": 6,
                    "num_inference_timesteps": 4,
                    "num_target_vision_tokens": 4,
                    "add_pos_embed": False,
                    "max_seq_len": 64,
                    "noise_beta_alpha": 1.5,
                    "noise_beta_beta": 1.0,
                    "noise_s": 0.999,
                    "num_timestep_buckets": 1000,
                    "diffusion_model_cfg": {},
                }
            }
        }
    )

    action_head = gr00t_action_header.FlowmatchingActionHead(config)

    assert action_head.use_state_encoder is False
    assert action_head.state_encoder is None


def test_soft_query_connector_uses_embodied_action_tokens_as_query():
    connector = SoftQueryConnector(hidden_size=32, depth=1, num_heads=4)
    query = torch.randn(2, 6, 32)
    context = torch.randn(2, 20, 32)

    out = connector(query=query, context=context)

    assert out.shape == query.shape
    assert not hasattr(connector, "query_embed")
