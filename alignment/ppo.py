"""
Proximal Policy Optimization (PPO) for LLM alignment.

Components:
  - rollout()        : generate responses, cache log-probs / values / rewards
  - shape_rewards()  : per-token reward shaping (Eq. 7)
  - compute_gae()    : Generalized Advantage Estimation (Eq. 8)
  - ppo_step()       : clipped surrogate + critic loss + KL (Eq. 9)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import PPOConfig
from utils.generation import generate_with_logprobs, get_per_token_logprobs
from alignment.reward_model import score_with_rm


# ---------------------------------------------------------------------------
# RolloutBatch dataclass — shared buffer between rollout and update
# ---------------------------------------------------------------------------

@dataclass
class RolloutBatch:
    """Stores everything collected during a PPO rollout."""
    prompt_ids: torch.Tensor           # (B, P)
    prompt_mask: torch.Tensor          # (B, P)  attention mask for left-padded prompts
    response_ids: torch.Tensor         # (B, T)
    old_logprobs: torch.Tensor         # (B, T)  per-token log π_old(y_t|s_t)
    ref_logprobs: torch.Tensor         # (B, T)  per-token log π_ref(y_t|s_t)
    values: torch.Tensor               # (B, T)  V_old(s_t)
    rewards: torch.Tensor              # (B,)    sequence-level r_ψ(x,y)
    response_mask: torch.Tensor        # (B, T)  1 for real tokens, 0 for pad


# ---------------------------------------------------------------------------
# Reward shaping (Eq. 7)
# ---------------------------------------------------------------------------

def shape_rewards(
    old_logprobs: torch.Tensor,     # (B, T)
    ref_logprobs: torch.Tensor,     # (B, T)
    task_rewards: torch.Tensor,     # (B,)
    response_mask: torch.Tensor,    # (B, T)
    beta: float = 0.1,
) -> torch.Tensor:
    """Per-token reward shaping: r_{i,t} = r^task·1[t=T_i] - β(log π_old - log π_ref)."""
    # KL penalty at every token
    kl_penalty = old_logprobs - ref_logprobs  # (B, T)
    shaped = -beta * kl_penalty               # (B, T)

    # Add task reward at last real token of each sequence
    seq_lens = response_mask.sum(dim=1).long()          # (B,)
    batch_idx = torch.arange(shaped.shape[0], device=shaped.device)
    shaped[batch_idx, seq_lens - 1] += task_rewards

    # Zero out padding
    return shaped * response_mask


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
    policy.eval()
    ref_model.eval()
    value_model.eval()
    rm_model.eval()

    input_ids = prompt_batch["input_ids"].to(device)
    attention_mask = prompt_batch["attention_mask"].to(device)
    prompt_len = input_ids.shape[1]

    # 1. Generate responses and collect old log-probs
    gen_out = generate_with_logprobs(
        model=policy,
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=getattr(cfg, "max_new_tokens", 128),
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        pad_token_id=policy_tokenizer.pad_token_id,
    )
    response_ids = gen_out["response_ids"]      # (B, T)
    old_logprobs = gen_out["logprobs"]           # (B, T)
    response_mask = gen_out["response_mask"]     # (B, T)

    # 2. Build full sequence (prompt + response)
    full_ids = torch.cat([input_ids, response_ids], dim=1)
    full_mask = torch.cat([attention_mask, response_mask], dim=1)

    # 3. Reference model log-probs
    ref_logprobs = get_per_token_logprobs(
        ref_model, full_ids, full_mask, prompt_len,
    )  # (B, T)

    # 4. Value estimates for response tokens
    all_values = value_model(full_ids, full_mask)   # (B, P+T)
    values = all_values[:, prompt_len:]              # (B, T)

    # 5. Sequence-level RM score
    texts = policy_tokenizer.batch_decode(full_ids, skip_special_tokens=True)
    rewards = score_with_rm(
        rm_model, rm_tokenizer, texts, device=device,
    ).to(device)  # (B,)

    return RolloutBatch(
        prompt_ids=input_ids,
        prompt_mask=attention_mask,
        response_ids=response_ids,
        old_logprobs=old_logprobs,
        ref_logprobs=ref_logprobs,
        values=values.detach(),
        rewards=rewards,
        response_mask=response_mask,
    )


# ---------------------------------------------------------------------------
# GAE (Eq. 8)
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
        advantages: (B, T)  — **raw** (not standardized; caller normalizes)
        returns:    (B, T)  — V^GAE targets (detached)
    """
    B, T = rewards_per_token.shape
    advantages = torch.zeros_like(rewards_per_token)
    last_gae = torch.zeros(B, device=rewards_per_token.device)

    for t in reversed(range(T)):
        # V(s_{t+1}): 0 at the terminal step, else masked next value
        if t == T - 1:
            next_value = torch.zeros(B, device=values.device)
        else:
            next_value = values[:, t + 1] * response_mask[:, t + 1]

        delta = rewards_per_token[:, t] + gamma * next_value - values[:, t]
        last_gae = delta + gamma * lam * last_gae
        # Reset accumulation for padded positions
        last_gae = last_gae * response_mask[:, t]
        advantages[:, t] = last_gae

    returns = (values + advantages).detach()
    return advantages, returns


# ---------------------------------------------------------------------------
# PPO update step (Eq. 9)
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
    B, T = batch.response_ids.shape
    prompt_len = batch.prompt_ids.shape[1]
    mask = batch.response_mask
    n_valid = mask.sum()

    # Normalize advantages batch-wide
    valid_advs = advantages[mask.bool()]
    adv_mean = valid_advs.mean()
    adv_std = valid_advs.std()
    norm_adv = ((advantages - adv_mean) / (adv_std + 1e-8)) * mask

    # Reconstruct full sequence for forward passes
    full_ids = torch.cat([batch.prompt_ids, batch.response_ids], dim=1)
    full_mask = torch.cat([batch.prompt_mask, batch.response_mask], dim=1)

    totals = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "kl": 0.0,
        "clip_frac": 0.0,
        "grad_norm": 0.0,
    }

    for _epoch in range(cfg.ppo_epochs):
        # ---------- Policy update ----------
        policy.train()
        policy_optimizer.zero_grad()

        outputs = policy(input_ids=full_ids, attention_mask=full_mask)
        logits = outputs.logits[:, prompt_len - 1 : -1, :]  # (B, T, V)

        log_probs = F.log_softmax(logits, dim=-1)           # (B, T, V)
        current_logprobs = log_probs.gather(
            2, batch.response_ids.unsqueeze(-1),
        ).squeeze(-1)                                        # (B, T)

        # Entropy: H(π) = -Σ π log π
        probs = torch.exp(log_probs)
        entropy_per_token = -(probs * log_probs).sum(dim=-1)  # (B, T)
        entropy = (entropy_per_token * mask).sum() / n_valid

        # Importance-sampling ratio
        ratio = torch.exp(current_logprobs - batch.old_logprobs.detach())

        # Clipped surrogate objective (Eq. 9)
        unclipped = ratio * norm_adv.detach()
        clipped = torch.clamp(ratio, 1.0 - cfg.epsilon, 1.0 + cfg.epsilon) * norm_adv.detach()
        clip_loss = -torch.min(unclipped, clipped)
        clip_loss = (clip_loss * mask).sum() / n_valid

        # Approximate KL (old → current)
        approx_kl = (batch.old_logprobs.detach() - current_logprobs)
        approx_kl = (approx_kl * mask).sum() / n_valid

        # Clip fraction (monitoring)
        clip_frac = ((ratio.detach() - 1.0).abs() > cfg.epsilon).float()
        clip_frac = (clip_frac * mask).sum() / n_valid

        # Combined policy loss
        policy_loss = clip_loss - cfg.entropy_coef * entropy + cfg.kl_coef * approx_kl
        policy_loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        policy_optimizer.step()

        # ---------- Value update ----------
        value_model.train()
        value_optimizer.zero_grad()

        current_values = value_model(full_ids, full_mask)[:, prompt_len:]  # (B, T)
        value_loss = ((current_values - returns.detach()) ** 2 * mask).sum() / n_valid
        (cfg.value_loss_coef * value_loss).backward()
        nn.utils.clip_grad_norm_(value_model.parameters(), max_norm=1.0)
        value_optimizer.step()

        # Accumulate metrics
        totals["policy_loss"] += clip_loss.item()
        totals["value_loss"] += value_loss.item()
        totals["kl"] += approx_kl.item()
        totals["clip_frac"] += clip_frac.item()
        gn = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
        totals["grad_norm"] += gn

    n = cfg.ppo_epochs
    return {k: v / n for k, v in totals.items()}
