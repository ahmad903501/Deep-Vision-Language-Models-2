"""
Direct Preference Optimization (DPO).

Offline, reward-model-free alignment using equation (10):
  ℓ_DPO = -log σ( β (Δ_θ - Δ_ref) )
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import DPOConfig


# ---------------------------------------------------------------------------
# Sequence log-probability
# ---------------------------------------------------------------------------

def sequence_log_prob(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    response_start: torch.Tensor,
) -> torch.Tensor:
    """Compute log π(y | x) = Σ_{t=t_start}^{T} log π(y_t | x, y_{<t}).

    Args:
        model:          CausalLM (policy or reference)
        input_ids:      (B, L) — full prompt+response, left-padded
        attention_mask:  (B, L)
        response_start: (B,) — index where response tokens begin

    Returns: (B,) scalar log-probability per example
    """
    raise NotImplementedError("Implement in Phase 6")


# ---------------------------------------------------------------------------
# DPO loss
# ---------------------------------------------------------------------------

def dpo_loss(
    policy: nn.Module,
    ref_model: nn.Module,
    batch: dict,
    beta: float,
) -> tuple[torch.Tensor, dict]:
    """Compute the DPO loss for a batch of preference pairs.

    Args:
        policy:    trainable CausalLM (LoRA-adapted)
        ref_model: frozen reference (LoRA disabled or separate copy)
        batch:     output of PolicyCollator.collate_dpo()
        beta:      KL penalty coefficient

    Returns:
        loss:    scalar DPO loss
        metrics: dict with {loss, reward_margin_z, pref_accuracy}
    """
    raise NotImplementedError("Implement in Phase 6")
