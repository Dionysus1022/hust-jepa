from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import torch

import starVLA.training.train_vlajepa_cotrain as cotrain


class _AttrDict(dict):
    """Small OmegaConf-like mapping for constructing a trainer without Hydra."""

    __getattr__ = dict.__getitem__


class _DeepSpeedLikeModel:
    """Exercise the short DeepSpeed branch without constructing an engine."""

    def __init__(self):
        self.weight = torch.nn.Parameter(torch.tensor(2.0))

    def forward(self, _batch):
        return {"action_loss": self.weight.square()}

    def backward(self, loss):
        loss.backward()

    def is_gradient_accumulation_boundary(self):
        return True

    def step(self):
        pass


class _UntouchableMetrics(dict):
    """Raise if a skipped logging call tries to inspect or mutate metrics."""

    @staticmethod
    def _unexpected_access(*_args, **_kwargs):
        raise AssertionError("metrics were touched on a non-logging step")

    __contains__ = _unexpected_access
    __getitem__ = _unexpected_access
    __iter__ = _unexpected_access
    __len__ = _unexpected_access
    __repr__ = _unexpected_access
    __setitem__ = _unexpected_access
    get = _unexpected_access
    items = _unexpected_access
    keys = _unexpected_access
    values = _unexpected_access


def _make_trainer(*, completed_steps=1, logging_frequency=10):
    trainer = cotrain.VLAMTrainer.__new__(cotrain.VLAMTrainer)
    trainer.completed_steps = completed_steps
    trainer.config = SimpleNamespace(
        trainer=_AttrDict(
            enable_detailed_timing=False,
            gradient_clipping=None,
            logging_frequency=logging_frequency,
            log_full_metrics=False,
            loss_scale={},
        )
    )
    return trainer


def test_train_step_defers_metric_materialization_until_logging(monkeypatch):
    trainer = _make_trainer()
    trainer.model = _DeepSpeedLikeModel()

    @contextmanager
    def no_autocast(*_args, **_kwargs):
        yield

    # Keep this unit test CPU-only; autocast behavior is unrelated to metric
    # lifetime and is covered by the real training path.
    monkeypatch.setattr(cotrain.torch, "autocast", no_autocast)

    metrics = trainer._train_step({})

    assert torch.is_tensor(metrics["action_loss"])
    assert torch.is_tensor(metrics["loss"])
    assert metrics["action_loss"].grad_fn is None
    assert metrics["loss"].grad_fn is None
    assert not metrics["action_loss"].requires_grad
    assert not metrics["loss"].requires_grad
    torch.testing.assert_close(metrics["action_loss"], torch.tensor(4.0))
    torch.testing.assert_close(metrics["loss"], torch.tensor(4.0))
    torch.testing.assert_close(trainer.model.weight.grad, torch.tensor(4.0))
    assert metrics["is_update_step"] is True


def test_log_metrics_returns_before_touching_metrics_on_skipped_step(monkeypatch):
    trainer = _make_trainer(completed_steps=3, logging_frequency=10)

    def unexpected_rank_lookup():
        raise AssertionError("distributed rank was queried on a non-logging step")

    monkeypatch.setattr(cotrain.dist, "get_rank", unexpected_rank_lookup)

    assert trainer._log_metrics(_UntouchableMetrics()) is None
