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
    B = input_ids.shape[0]
    prompt_len = input_ids.shape[1]

    # 1. Generate responses via sampling
    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=pad_token_id,
    )  # (B, P+T') where T' varies per sequence

    # 2. Extract response portion and build mask
    response_ids = output_ids[:, prompt_len:]  # (B, T')
    T = response_ids.shape[1]

    # Build response mask (1 for real tokens, 0 for pad)
    response_mask = (response_ids != pad_token_id).long()

    # 3. Get per-token log-probs via teacher-forced forward pass
    full_mask = torch.cat([
        attention_mask,
        response_mask,
    ], dim=1)

    logprobs = get_per_token_logprobs(
        model, output_ids, full_mask, prompt_len
    )  # (B, T)

    return {
        "response_ids": response_ids,
        "logprobs": logprobs,
        "response_mask": response_mask,
    }


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
    outputs = model(input_ids=full_ids, attention_mask=attention_mask)
    logits = outputs.logits  # (B, L, V)

    # Shift: logits at position t predict token at position t+1
    # For response starting at index `response_start`, we need logits
    # from positions [response_start-1, ..., L-2] to predict tokens
    # [response_start, ..., L-1]
    if isinstance(response_start, int):
        # All sequences share the same prompt length
        shift_logits = logits[:, response_start - 1 : -1, :]  # (B, T, V)
        target_ids = full_ids[:, response_start:]              # (B, T)
    else:
        raise ValueError("Per-sequence response_start not yet supported")

    log_probs = F.log_softmax(shift_logits, dim=-1)  # (B, T, V)

    # Gather log-prob of the actual generated token
    token_logprobs = log_probs.gather(
        2, target_ids.unsqueeze(-1)
    ).squeeze(-1)  # (B, T)

    return token_logprobs
