from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from starVLA.dataloader.lerobot_datasets import collate_fn
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


def test_lerobot_collate_applies_vlanext_style_augmentation():
    sample = _fake_sample()
    augmentation = {
        "enabled": True,
        "random_resized_crop": {"scale": [0.8, 1.0], "ratio": [0.9, 1.1]},
        "random_brightness": [0.2],
        "random_contrast": [0.8, 1.2],
        "random_saturation": [0.8, 1.2],
        "random_hue": [0.05],
        "augment_order": [
            "random_resized_crop",
            "random_brightness",
            "random_contrast",
            "random_saturation",
            "random_hue",
        ],
    }

    batch = collate_fn([sample, sample], augmentation=augmentation)

    assert batch["video"].shape == (2, 2, 4, 32, 32, 3)
    assert batch["video"].dtype == torch.uint8
    assert batch["action"].shape == (2, 7, 7)


def test_soft_query_connector_uses_embodied_action_tokens_as_query():
    connector = SoftQueryConnector(hidden_size=32, depth=1, num_heads=4)
    query = torch.randn(2, 6, 32)
    context = torch.randn(2, 20, 32)

    out = connector(query=query, context=context)

    assert out.shape == query.shape
    assert not hasattr(connector, "query_embed")
