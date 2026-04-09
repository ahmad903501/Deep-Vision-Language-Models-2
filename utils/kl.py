"""
KL divergence helpers.

Two modes:
  - full-vocab:  exact KL using full softmax distributions (VRAM-heavy)
  - mc:          Monte Carlo sampled-token approximation (cheap, unbiased)

Configurable via config.grpo.kl_mode / config.rlvr.kl_mode.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def kl_full_vocab(
    policy_logits: torch.Tensor,
    ref_logits: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Exact per-token KL using full vocabulary distributions.

    KL_t = Σ_v π_θ(v|s_t) [log π_θ(v|s_t) - log π_ref(v|s_t)]

    Args:
        policy_logits: (B, T, V) raw logits from policy
        ref_logits:    (B, T, V) raw logits from reference
        mask:          (B, T) binary mask for valid tokens

    Returns: (B, T) per-token KL values (masked)
    """
    p = F.softmax(policy_logits, dim=-1)
    log_p = F.log_softmax(policy_logits, dim=-1)
    log_q = F.log_softmax(ref_logits, dim=-1)
    kl = (p * (log_p - log_q)).sum(dim=-1)   # (B, T)
    return kl * mask


def kl_mc_approx(
    policy_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Monte Carlo sampled-token KL approximation.

    KL_t ≈ log π_θ(y_t|s_t) - log π_ref(y_t|s_t)

    This is an unbiased estimate of the true KL but has high variance
    for tokens with low probability under π_θ.

    Args:
        policy_logprobs: (B, T) log π_θ for sampled tokens
        ref_logprobs:    (B, T) log π_ref for the same tokens
        mask:            (B, T) binary mask

    Returns: (B, T) per-token approximate KL (masked)
    """
    return (policy_logprobs - ref_logprobs) * mask


def compute_kl(
    policy_logprobs_or_logits: torch.Tensor,
    ref_logprobs_or_logits: torch.Tensor,
    mask: torch.Tensor,
    mode: str = "mc",
) -> torch.Tensor:
    """Dispatch to the appropriate KL function.

    Args:
        mode: "full" for exact KL (inputs are logits), "mc" for MC approx (inputs are logprobs)
    """
    if mode == "full":
        return kl_full_vocab(policy_logprobs_or_logits, ref_logprobs_or_logits, mask)
    elif mode == "mc":
        return kl_mc_approx(policy_logprobs_or_logits, ref_logprobs_or_logits, mask)
    else:
        raise ValueError(f"Unknown KL mode: {mode!r}. Use 'full' or 'mc'.")
