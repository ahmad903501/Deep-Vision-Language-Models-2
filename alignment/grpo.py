"""
Group Relative Policy Optimization (GRPO).

Critic-free online alignment.  For each prompt, sample K completions,
compute group-relative advantages, and apply clipped surrogate (eq. 12).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from config import GRPOConfig


# ---------------------------------------------------------------------------
# GroupRollout dataclass
# ---------------------------------------------------------------------------

@dataclass
class GroupRollout:
    """Stores data from a GRPO group rollout."""
    prompt_ids: torch.Tensor            # (B, P)
    response_ids: torch.Tensor          # (B*K, T)
    old_logprobs: torch.Tensor          # (B*K, T)
    ref_logprobs: torch.Tensor          # (B*K, T)
    rewards: torch.Tensor               # (B*K,)
    advantages: torch.Tensor            # (B*K,)  — group-relative, broadcast to tokens
    response_mask: torch.Tensor         # (B*K, T)


# ---------------------------------------------------------------------------
# Group rollout
# ---------------------------------------------------------------------------

@torch.no_grad()
def group_rollout(
    policy: nn.Module,
    ref_model: nn.Module,
    rm_model: nn.Module,
    rm_tokenizer,
    prompt_batch: dict,
    policy_tokenizer,
    cfg: GRPOConfig,
    device: str = "cuda",
) -> GroupRollout:
    """Sample K completions per prompt and compute group-relative advantages.

    Returns GroupRollout with all data needed for grpo_step.
    """
    raise NotImplementedError("Implement in Phase 7")


# ---------------------------------------------------------------------------
# GRPO update step
# ---------------------------------------------------------------------------

def grpo_step(
    policy: nn.Module,
    ref_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    group_data: GroupRollout,
    cfg: GRPOConfig,
) -> dict:
    """One GRPO gradient step.

    Returns dict: {loss, kl, clip_frac, degenerate_frac, grad_norm}.
    """
    raise NotImplementedError("Implement in Phase 7")
