"""
RLVR — RL with Verifiable Rewards (GRPO variant using binary r_v on GSM8K).

This is a thin wrapper around grpo.py that swaps the learned RM for
the verifiable reward checker from data/gsm8k.py.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from config import RLVRConfig
from alignment.grpo import GroupRollout


@torch.no_grad()
def rlvr_rollout(
    policy: nn.Module,
    ref_model: nn.Module,
    prompt_batch: dict,         # includes gold_answer
    policy_tokenizer,
    cfg: RLVRConfig,
    device: str = "cuda",
) -> GroupRollout:
    """GRPO-style group rollout with verifiable reward instead of RM.

    The reward for each completion is r_v ∈ {0, 1}.
    """
    raise NotImplementedError("Implement in Phase 8")


def rlvr_step(
    policy: nn.Module,
    ref_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    group_data: GroupRollout,
    cfg: RLVRConfig,
) -> dict:
    """One RLVR gradient step (reuses GRPO mechanics).

    Returns dict: {loss, kl, pass_rate, degenerate_frac, mean_length}.
    """
    raise NotImplementedError("Implement in Phase 8")
