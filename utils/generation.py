"""
Generation helper: generate text and collect per-token log-probabilities.

Used by PPO rollout, GRPO rollout, and RLVR rollout.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def generate_with_logprobs(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
    pad_token_id: int = 0,
) -> dict:
    """Generate a response and return per-token log-probs.

    Steps:
      1. Call model.generate() to produce response tokens.
      2. Re-run a single teacher-forced forward pass over (prompt + response)
         to extract log-probs for each generated token.

    Returns dict with:
        response_ids:   (B, T) generated token ids (response only)
        logprobs:       (B, T) log π(y_t | s_t) for each response token
        response_mask:  (B, T) 1 where token is real, 0 for padding
    """
    raise NotImplementedError("Implement in Phase 5")


@torch.no_grad()
def get_per_token_logprobs(
    model: nn.Module,
    full_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    response_start: int | torch.Tensor,
) -> torch.Tensor:
    """Teacher-forced forward pass to extract per-token log-probs.

    Args:
        model:          CausalLM
        full_ids:       (B, L) — prompt + response concatenated
        attention_mask: (B, L)
        response_start: int or (B,) — index where response tokens start

    Returns: (B, T) log-probabilities for response tokens only
    """
    raise NotImplementedError("Implement in Phase 5")
