from __future__ import annotations

import json

import pytest
import torch

from starVLA.training.trainer_utils.trainer_tools import (
    compose_vlajepa_loss,
    read_accelerate_checkpoint_metadata,
    read_accelerate_checkpoint_step,
    resolve_accelerate_checkpoint,
)


def test_compose_vlajepa_loss_scales_disjoint_components_once():
    losses = {
        "action_loss": torch.tensor(2.0),
        "action_dct_loss": torch.tensor(3.0),
        "wm_loss": torch.tensor(5.0),
        "future_token_loss": torch.tensor(7.0),
    }

    total = compose_vlajepa_loss(
        losses,
        {"vla": 10.0, "vlm": 0.1, "future": 0.01},
    )

    # (2 + 3) * 10 + 5 * .1 + 7 * .01
    torch.testing.assert_close(total, torch.tensor(50.57))


def test_compose_vlajepa_loss_rejects_accidentally_summed_new_components():
    with pytest.raises(KeyError, match="Unrecognized"):
        compose_vlajepa_loss({"action_loss": torch.tensor(1.0), "debug_metric": torch.tensor(2.0)})


def test_resolve_latest_complete_accelerate_checkpoint(tmp_path):
    incomplete = tmp_path / "steps_30"
    incomplete.mkdir()
    for step in (10, 20):
        checkpoint = tmp_path / f"steps_{step}"
        checkpoint.mkdir()
        (checkpoint / "trainer_state.json").write_text(
            json.dumps({"completed_steps": step}),
            encoding="utf-8",
        )

    latest = resolve_accelerate_checkpoint(tmp_path, "latest")

    assert latest == tmp_path / "steps_20"
    assert read_accelerate_checkpoint_step(latest) == 20
    assert read_accelerate_checkpoint_metadata(latest) == {"completed_steps": 20}


def test_model_only_checkpoint_is_not_accepted_as_resume_state(tmp_path):
    model_only = tmp_path / "steps_20_pytorch_model.pt"
    model_only.touch()

    with pytest.raises(ValueError, match="model-only checkpoint"):
        resolve_accelerate_checkpoint(tmp_path, model_only)
