"""
Proximal Policy Optimization (PPO) for LLM alignment.

Components:
  - rollout()      : generate responses, cache log-probs / values / rewards
  - compute_gae()  : Generalized Advantage Estimation (eq. 8)
  - ppo_step()     : clipped surrogate + critic loss + KL (eq. 9)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from config import PPOConfig


# ---------------------------------------------------------------------------
# RolloutBatch dataclass — shared buffer between rollout and update
# ---------------------------------------------------------------------------

@dataclass
class RolloutBatch:
    """Stores everything collected during a PPO rollout."""
    prompt_ids: torch.Tensor           # (B, P)
    response_ids: torch.Tensor         # (B, T)
    old_logprobs: torch.Tensor         # (B, T)  per-token log π_old(y_t|s_t)
    ref_logprobs: torch.Tensor         # (B, T)  per-token log π_ref(y_t|s_t)
    values: torch.Tensor               # (B, T)  V_old(s_t)
    rewards: torch.Tensor              # (B,)    sequence-level r_ψ(x,y)
    response_mask: torch.Tensor        # (B, T)  1 for real tokens, 0 for pad


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

@torch.no_grad()
def rollout(
    policy: nn.Module,
    ref_model: nn.Module,
    value_model: nn.Module,
    rm_model: nn.Module,
    rm_tokenizer,
    prompt_batch: dict,
    policy_tokenizer,
    cfg: PPOConfig,
    device: str = "cuda",
) -> RolloutBatch:
    """Generate responses and collect (log-probs, values, rewards).

    All models run under torch.no_grad().
    """
    raise NotImplementedError("Implement in Phase 5")


# ---------------------------------------------------------------------------
# GAE
# ---------------------------------------------------------------------------

def compute_gae(
    rewards_per_token: torch.Tensor,   # (B, T)
    values: torch.Tensor,              # (B, T)
    response_mask: torch.Tensor,       # (B, T)
    gamma: float = 1.0,
    lam: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE advantages and λ-returns.

    Args:
        rewards_per_token: per-token shaped rewards (r^task at last token + r^KL at all tokens)
        values: V_old(s_t)
        response_mask: 1 for valid response tokens
        gamma: discount factor (1.0 = no discounting)
        lam: GAE λ

    Returns:
        advantages: (B, T)  — standardized batch-wide
        returns:    (B, T)  — V^GAE targets (detached)
    """
    raise NotImplementedError("Implement in Phase 5")


# ---------------------------------------------------------------------------
# PPO update step
# ---------------------------------------------------------------------------

def ppo_step(
    policy: nn.Module,
    value_model: nn.Module,
    policy_optimizer: torch.optim.Optimizer,
    value_optimizer: torch.optim.Optimizer,
    batch: RolloutBatch,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    cfg: PPOConfig,
) -> dict:
    """Run one PPO update (multiple mini-batch epochs over the rollout).

    Returns dict with: policy_loss, value_loss, kl, clip_frac, grad_norm.
    """
    raise NotImplementedError("Implement in Phase 5")
