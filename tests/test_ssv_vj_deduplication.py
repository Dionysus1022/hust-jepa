from __future__ import annotations

import numpy as np
import torch

from starVLA.dataloader import video_datasets
from starVLA.model.framework.VLA_JEPA import _expand_vj_video_embeddings
from starVLA.training.train_vlajepa_cotrain import VLAMTrainer


def test_ssv_collate_preprocesses_each_source_video_only_once(monkeypatch):
    batch_size = 3
    n_views = 2
    n_frames = 4
    height = 6
    width = 8
    batch = [
        (
            np.full((n_frames, height, width, 3), fill_value=i, dtype=np.uint8),
            f"instruction {i}",
        )
        for i in range(batch_size)
    ]
    processor_sentinel = object()
    processed = torch.arange(batch_size * 5, dtype=torch.float32).reshape(batch_size, 5)
    process_calls = []

    monkeypatch.setattr(video_datasets, "_get_vj_processor", lambda _: processor_sentinel)

    def fake_process_videos(processor, videos):
        process_calls.append((processor, videos.copy()))
        return processed

    monkeypatch.setattr(video_datasets, "_process_vj_videos", fake_process_videos)

    collated = video_datasets.collate_fn(
        batch,
        n_views=n_views,
        resolution_size=4,
        vj_processor_path="unused-in-test",
        preprocess_vj_inputs=True,
    )

    assert len(process_calls) == 1
    assert process_calls[0][0] is processor_sentinel
    assert process_calls[0][1].shape == (batch_size, n_frames, 3, height, width)
    assert collated["video"] is None
    assert collated["video_shape"] == (batch_size, n_views, n_frames, 3, height, width)
    torch.testing.assert_close(collated["vj_pixel_values_videos"], processed)
    torch.testing.assert_close(
        collated["vj_feature_expand_indices"],
        torch.arange(batch_size, dtype=torch.long).repeat_interleave(n_views),
    )


def test_feature_expansion_is_equivalent_to_encoding_repeated_inputs():
    batch_size = 3
    n_views = 2
    tokens = 4
    embed_dim = 5
    physical_inputs = torch.arange(batch_size * 7, dtype=torch.float32).reshape(batch_size, 7)
    expand_indices = torch.arange(batch_size, dtype=torch.long).repeat_interleave(n_views)

    class ToyEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            weight = torch.arange(7 * tokens * embed_dim, dtype=torch.float32)
            self.register_buffer("weight", weight.reshape(7, tokens * embed_dim) / 100.0)

        def forward(self, inputs):
            return (inputs @ self.weight).reshape(inputs.shape[0], tokens, embed_dim)

    encoder = ToyEncoder().eval()
    old_embeddings = encoder(physical_inputs.repeat_interleave(n_views, dim=0))
    deduplicated_embeddings = _expand_vj_video_embeddings(
        encoder(physical_inputs),
        expand_indices,
        expected_videos=batch_size * n_views,
    )

    torch.testing.assert_close(deduplicated_embeddings, old_embeddings, rtol=0, atol=0)

    # Match VLA_JEPA's sample-major view fusion after V-JEPA encoding.
    old_fused = (
        old_embeddings.reshape(batch_size, n_views, tokens, embed_dim)
        .transpose(1, 2)
        .reshape(batch_size, tokens, n_views * embed_dim)
    )
    deduplicated_fused = (
        deduplicated_embeddings.reshape(batch_size, n_views, tokens, embed_dim)
        .transpose(1, 2)
        .reshape(batch_size, tokens, n_views * embed_dim)
    )
    torch.testing.assert_close(deduplicated_fused, old_fused, rtol=0, atol=0)


def test_cotrain_merge_offsets_ssv_expansion_after_vla_identity_rows():
    trainer = VLAMTrainer.__new__(VLAMTrainer)
    n_views = 2
    video_tail = (n_views, 4, 3, 6, 8)

    vla_batch_size = 2
    vla_physical_rows = vla_batch_size * n_views
    batch_vla = {
        "image": [[object()] for _ in range(vla_batch_size)],
        "lang": [f"vla {i}" for i in range(vla_batch_size)],
        "action": torch.zeros(vla_batch_size, 1, 7),
        "video": None,
        "video_shape": (vla_batch_size, *video_tail),
        "vj_pixel_values_videos": torch.zeros(vla_physical_rows, 4, 3, 6, 8),
    }

    ssv_batch_size = 3
    batch_ssv = {
        "image": [[object()] for _ in range(ssv_batch_size)],
        "lang": [f"ssv {i}" for i in range(ssv_batch_size)],
        "video": None,
        "video_shape": (ssv_batch_size, *video_tail),
        "vj_pixel_values_videos": torch.ones(ssv_batch_size, 4, 3, 6, 8),
        "vj_feature_expand_indices": torch.arange(ssv_batch_size, dtype=torch.long).repeat_interleave(n_views),
    }

    mixed = trainer._merge_cotrain_batches(batch_vla, batch_ssv)

    assert mixed["video"] is None
    assert mixed["video_shape"] == (vla_batch_size + ssv_batch_size, *video_tail)
    assert [tensor.shape[0] for tensor in mixed["vj_pixel_values_videos"]] == [vla_physical_rows, ssv_batch_size]
    expected_indices = torch.cat(
        [
            torch.arange(vla_physical_rows, dtype=torch.long),
            vla_physical_rows
            + torch.arange(ssv_batch_size, dtype=torch.long).repeat_interleave(n_views),
        ]
    )
    torch.testing.assert_close(mixed["vj_feature_expand_indices"], expected_indices)
