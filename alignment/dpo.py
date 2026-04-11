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
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # (B, L, V)

    # Shift: logits[t] predicts token[t+1], so for token at position t
    # we need logits at position t-1.
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)   # (B, L-1, V)
    targets = input_ids[:, 1:]                               # (B, L-1)

    # Gather the log-prob of the actual token at each position
    per_token_logp = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)  # (B, L-1)

    # Build a mask: 1 for response tokens, 0 for prompt/pad.
    # response_start[i] is the position where response begins in input_ids.
    # In the shifted log_probs tensor, position t corresponds to predicting
    # input_ids[t+1]. So response tokens in shifted space start at
    # response_start[i] - 1 (predicting token at response_start[i]).
    B, L_minus1 = per_token_logp.shape
    positions = torch.arange(L_minus1, device=input_ids.device).unsqueeze(0)  # (1, L-1)
    resp_mask = (positions >= (response_start.unsqueeze(1) - 1)).float()      # (B, L-1)

    # Also mask out padding (attention_mask[:, 1:] covers the target positions)
    resp_mask = resp_mask * attention_mask[:, 1:].float()

    # Sum log-probs over response tokens → scalar per example
    return (per_token_logp * resp_mask).sum(dim=1)  # (B,)


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
    # Unpack batch (from PolicyCollator.collate_dpo)
    c_ids = batch["chosen_input_ids"]
    c_mask = batch["chosen_attention_mask"]
    r_ids = batch["rejected_input_ids"]
    r_mask = batch["rejected_attention_mask"]
    c_start = batch["chosen_response_start"]
    r_start = batch["rejected_response_start"]

    # Fuse chosen+rejected into one forward pass per model to halve peak VRAM
    cat_ids = torch.cat([c_ids, r_ids], dim=0)        # (2B, L)
    cat_mask = torch.cat([c_mask, r_mask], dim=0)      # (2B, L)
    cat_start = torch.cat([c_start, r_start], dim=0)   # (2B,)
    B = c_ids.shape[0]

    # Policy log-probs (single forward pass, keeps grad graph)
    pi_logp = sequence_log_prob(policy, cat_ids, cat_mask, cat_start)
    pi_logp_chosen, pi_logp_rejected = pi_logp[:B], pi_logp[B:]

    # Reference log-probs (single forward pass, no gradients)
    with torch.no_grad():
        ref_logp = sequence_log_prob(ref_model, cat_ids, cat_mask, cat_start)
        ref_logp_chosen, ref_logp_rejected = ref_logp[:B], ref_logp[B:]

    # DPO implicit reward margin:
    #   z = β * ( (log π_θ(y+|x) - log π_θ(y-|x)) - (log π_ref(y+|x) - log π_ref(y-|x)) )
    delta_policy = pi_logp_chosen - pi_logp_rejected      # (B,)
    delta_ref = ref_logp_chosen - ref_logp_rejected        # (B,)
    z = beta * (delta_policy - delta_ref)                  # (B,)

    # Loss = -log σ(z)   (Eq. 10)
    loss = -F.logsigmoid(z).mean()

    # Metrics
    with torch.no_grad():
        pref_acc = (z > 0).float().mean().item()
        reward_margin = z.mean().item()

    metrics = {
        "loss": loss.item(),
        "reward_margin_z": reward_margin,
        "pref_accuracy": pref_acc,
    }

    return loss, metrics
