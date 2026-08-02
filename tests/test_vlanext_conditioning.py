from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from starVLA.dataloader.lerobot_datasets import _augment_video_frames, collate_fn
from starVLA.model.framework.VLA_JEPA import SoftQueryConnector


def _fake_sample():
    image = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
    return {
        "video": np.random.randint(0, 255, (2, 4, 32, 32, 3), dtype=np.uint8),
        "image": [image, image.copy()],
        "lang": "put the object in the bowl",
        "action": np.random.randn(7, 7).astype(np.float16),
        "state": np.random.randn(1, 8).astype(np.float16),
    }


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


def test_soft_query_connector_uses_embodied_action_tokens_as_query():
    connector = SoftQueryConnector(hidden_size=32, depth=1, num_heads=4)
    query = torch.randn(2, 6, 32)
    context = torch.randn(2, 20, 32)

    out = connector(query=query, context=context)

    assert out.shape == query.shape
    assert not hasattr(connector, "query_embed")
