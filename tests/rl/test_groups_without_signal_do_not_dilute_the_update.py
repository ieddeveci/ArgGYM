"""A group scored all 0 or all 1 adds no tokens to the loss's denominator.

With 8 completions per prompt and strict 0/1 success, about half the groups
came out uniform; their zero advantages gave no gradient, but their tokens
still divided every update.
"""
from __future__ import annotations

import torch

from rl.train import signal_tokens


def test_only_completions_with_an_advantage_count():
    rewards = torch.tensor([[1.0, 1.0], [0.0, 0.0], [1.0, 0.0]])  # all 1, all 0, mixed
    advantages = (rewards - rewards.mean(dim=1, keepdim=True)).flatten()
    mask = torch.tensor([[1, 1, 1, 0]] * 6)  # 3 loss tokens each
    assert signal_tokens(advantages, mask).item() == 6


def test_a_masked_truncated_completion_counts_nothing():
    advantages = torch.tensor([0.5, -0.5, 0.0])  # the third returned None
    mask = torch.tensor([[1, 1, 0], [1, 0, 0], [0, 0, 0]])
    assert signal_tokens(advantages, mask).item() == 3


def test_a_batch_with_no_signal_counts_zero():
    assert signal_tokens(torch.zeros(4), torch.ones(4, 5, dtype=torch.long)).item() == 0
