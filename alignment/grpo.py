"""
Group Relative Policy Optimization (GRPO).

Critic-free online alignment. For each prompt, sample K completions,
compute group-relative advantages, and apply clipped surrogate (eq. 12).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import GRPOConfig
from utils.generation import generate_with_logprobs, get_per_token_logprobs
from alignment.reward_model import score_with_rm


# ---------------------------------------------------------------------------
# GroupRollout dataclass
# ---------------------------------------------------------------------------

@dataclass
class GroupRollout:
    """Stores data from a GRPO group rollout."""
    prompt_ids: torch.Tensor             # (B*K, P)
    prompt_mask: torch.Tensor            # (B*K, P)
    response_ids: torch.Tensor           # (B*K, T)
    old_logprobs: torch.Tensor           # (B*K, T)
    ref_logprobs: torch.Tensor           # (B*K, T)
    rewards: torch.Tensor                # (B*K,)
    advantages: torch.Tensor             # (B*K,)  — group-relative, broadcast to tokens
    response_mask: torch.Tensor          # (B*K, T)


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

    For B prompts, generates B*K responses. Each group of K shares the same prompt.
    Advantages are normalized within each group (mean=0, std=1).
    """
    policy.eval()
    ref_model.eval()
    rm_model.eval()

    input_ids = prompt_batch["input_ids"].to(device)       # (B, P)
    attention_mask = prompt_batch["attention_mask"].to(device)
    B, P = input_ids.shape
    K = cfg.K

    # Repeat each prompt K times: (B, P) -> (B*K, P)
    input_ids_rep = input_ids.repeat_interleave(K, dim=0)
    attn_mask_rep = attention_mask.repeat_interleave(K, dim=0)

    # 1. Generate K responses per prompt + old log-probs
    max_new = getattr(cfg, "max_new_tokens", 128)
    gen_out = generate_with_logprobs(
        model=policy,
        input_ids=input_ids_rep,
        attention_mask=attn_mask_rep,
        max_new_tokens=max_new,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        pad_token_id=policy_tokenizer.pad_token_id,
    )
    response_ids = gen_out["response_ids"]         # (B*K, T)
    old_logprobs = gen_out["logprobs"]             # (B*K, T)
    response_mask = gen_out["response_mask"]       # (B*K, T)

    # 2. Reference log-probs
    full_ids = torch.cat([input_ids_rep, response_ids], dim=1)
    full_mask = torch.cat([attn_mask_rep, response_mask], dim=1)
    ref_logprobs = get_per_token_logprobs(ref_model, full_ids, full_mask, P)

    # 3. RM scores (Safe-Decode Implementation)
    # CRITICAL: We use skip_special_tokens=False because the RM needs the context,
    # and it prevents the crash caused by empty strings if the model only outputs special tokens.
    texts = policy_tokenizer.batch_decode(full_ids, skip_special_tokens=False)
    
    # Safety fallback: ensure no string is empty or purely whitespace
    texts = [t if (t and len(t.strip()) > 0) else "Invalid empty response" for t in texts]
    
    rewards = score_with_rm(rm_model, rm_tokenizer, texts, device=device).to(device)  # (B*K,)

    # 4. Group-relative advantages
    rewards_grouped = rewards.view(B, K)                   # (B, K)
    group_mean = rewards_grouped.mean(dim=1, keepdim=True) # (B, 1)
    group_std = rewards_grouped.std(dim=1, keepdim=True)   # (B, 1)
    
    # Use 1e-4 for numerical stability in case all rewards in a group are identical
    advantages_grouped = (rewards_grouped - group_mean) / (group_std + 1e-4)  # (B, K)
    advantages = advantages_grouped.view(B * K)            # (B*K,)

    return GroupRollout(
        prompt_ids=input_ids_rep,
        prompt_mask=attn_mask_rep,
        response_ids=response_ids,
        old_logprobs=old_logprobs,
        ref_logprobs=ref_logprobs,
        rewards=rewards,
        advantages=advantages,
        response_mask=response_mask,
    )


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
    """One GRPO gradient step (Eq. 12)."""
    policy.train()
    optimizer.zero_grad()

    mask = group_data.response_mask
    n_valid = mask.sum().clamp(min=1)
    P = group_data.prompt_ids.shape[1]

    # Full sequence for forward pass
    full_ids = torch.cat([group_data.prompt_ids, group_data.response_ids], dim=1)
    full_mask = torch.cat([group_data.prompt_mask, group_data.response_mask], dim=1)

    # Current policy log-probs
    outputs = policy(input_ids=full_ids, attention_mask=full_mask)
    logits = outputs.logits[:, P - 1 : -1, :]   # (B*K, T, V)

    log_probs = F.log_softmax(logits, dim=-1)
    current_logprobs = log_probs.gather(
        2, group_data.response_ids.unsqueeze(-1)
    ).squeeze(-1)                                 # (B*K, T)

    # Importance-sampling ratio
    ratio = torch.exp(current_logprobs - group_data.old_logprobs.detach())

    # Broadcast per-sequence advantage to per-token
    adv = group_data.advantages.detach().unsqueeze(1) * mask   # (B*K, T)

    # Clipped surrogate (Eq. 12)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - cfg.epsilon, 1.0 + cfg.epsilon) * adv
    surrogate_loss = -torch.min(unclipped, clipped)
    surrogate_loss = (surrogate_loss * mask).sum() / n_valid

    # KL penalty (MC approximation: log π_θ - log π_ref)
    kl_per_token = (current_logprobs - group_data.ref_logprobs.detach()) * mask
    kl = kl_per_token.sum() / n_valid

    # Combined loss
    loss = surrogate_loss + cfg.beta * kl
    loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(policy.parameters(), max_norm=cfg.max_grad_norm)
    optimizer.step()

    # Monitoring metrics
    with torch.no_grad():
        clip_frac = ((ratio.detach() - 1.0).abs() > cfg.epsilon).float()
        clip_frac = (clip_frac * mask).sum() / n_valid

        # Degenerate fraction: groups where all K rewards are identical (std ≈ 0)
        BK = group_data.rewards.shape[0]
        K = cfg.K
        B = BK // K
        rewards_grouped = group_data.rewards.view(B, K)
        degenerate_frac = (rewards_grouped.std(dim=1) < 1e-6).float().mean()

        gn = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm)

    return {
        "loss": loss.item(),
        "kl": kl.item(),
        "clip_frac": clip_frac.item(),
        "degenerate_frac": degenerate_frac.item(),
        "grad_norm": gn,
        "reward_mean": group_data.rewards.mean().item(),
        "reward_std": group_data.rewards.std().item(),
    }