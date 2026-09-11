from __future__ import annotations

import math

import torch
from torch import nn


class HistorySinusoidalEncoding(nn.Module):
    """Add explicit oldest-to-newest positions to a causal state history."""

    def __init__(self, hidden_size: int, max_history_len: int = 8):
        super().__init__()
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if max_history_len <= 0:
            raise ValueError(f"max_history_len must be positive, got {max_history_len}")

        positions = torch.arange(max_history_len, dtype=torch.float32).unsqueeze(1)
        even_dims = torch.arange(0, hidden_size, 2, dtype=torch.float32)
        frequencies = torch.exp(-math.log(10000.0) * even_dims / hidden_size)
        encoding = torch.zeros(max_history_len, hidden_size, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        if hidden_size > 1:
            encoding[:, 1::2] = torch.cos(
                positions * frequencies[: encoding[:, 1::2].shape[1]]
            )

        # Fixed encoding keeps existing checkpoint state_dicts load-compatible.
        self.register_buffer("encoding", encoding, persistent=False)
        self.max_history_len = max_history_len

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(
                "State history tokens must have shape [batch, time, hidden], "
                f"got {tuple(tokens.shape)}"
            )
        history_len = tokens.shape[1]
        if history_len > self.max_history_len:
            raise ValueError(
                f"State history has {history_len} steps, but the configured maximum "
                f"is {self.max_history_len}."
            )

        # Short inputs are right-aligned so a one-step input is always "current".
        start = self.max_history_len - history_len
        positions = self.encoding[start : start + history_len]
        return tokens + positions.to(device=tokens.device, dtype=tokens.dtype).unsqueeze(0)
