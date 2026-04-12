"""
Proximal Policy Optimization (PPO) for LLM alignment.

Components:
  - shape_rewards()  : per-token reward shaping (Eq. 7)
  - rollout()        : generate responses, cache log-probs / values / rewards
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
    prompt_mask: torch.Tensor          # (B, P) attention mask for prompts
    response_ids: torch.Tensor         # (B, T)
    old_logprobs: torch.Tensor         # (B, T)  per-token log π_old(y_t|s_t)
    ref_logprobs: torch.Tensor         # (B, T)  per-token log π_ref(y_t|s_t)
    values: torch.Tensor               # (B, T)  V_old(s_t)
    rewards: torch.Tensor              # (B,)    sequence-level r_ψ(x,y)
    response_mask: torch.Tensor        # (B, T)  1 for real tokens, 0 for pad


# ---------------------------------------------------------------------------
# Reward shaping  (Eq. 7)
# ---------------------------------------------------------------------------

def shape_rewards(
    old_logprobs: torch.Tensor,        # (B, T)
    ref_logprobs: torch.Tensor,        # (B, T)
    task_rewards: torch.Tensor,        # (B,)
    response_mask: torch.Tensor,       # (B, T)
    beta: float = 0.1,
) -> torch.Tensor:
    """Per-token reward: r_{i,t} = -β(log π_old - log π_ref) + r^task · 1[t=T_i]."""
    kl_penalty = old_logprobs - ref_logprobs         # (B, T)
    shaped = -beta * kl_penalty                      # (B, T)

    # Add task reward at last real token of each sequence
    seq_lens = response_mask.sum(dim=1).long()       # (B,)
    batch_idx = torch.arange(shaped.shape[0], device=shaped.device)
    shaped[batch_idx, seq_lens - 1] += task_rewards

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

    # 1. Generate responses + old log-probs from policy
    gen_out = generate_with_logprobs(
        model=policy,
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=getattr(cfg, "max_new_tokens", 128),
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        pad_token_id=policy_tokenizer.pad_token_id,
        # If your model continues to hallucinate turns, uncomment the line below:
        # stop_strings=["Human:", "###"], tokenizer=policy_tokenizer 
    )
    response_ids = gen_out["response_ids"]       # (B, T)
    old_logprobs = gen_out["logprobs"]           # (B, T)
    response_mask = gen_out["response_mask"]     # (B, T)

    # 2. Full sequence for forward passes
    full_ids = torch.cat([input_ids, response_ids], dim=1)
    full_mask = torch.cat([attention_mask, response_mask], dim=1)

    # 3. Reference model log-probs
    ref_logprobs = get_per_token_logprobs(
        ref_model, full_ids, full_mask, prompt_len
    )

    # 4. Value estimates for response tokens
    all_values = value_model(full_ids, full_mask)     # (B, P+T)
    values = all_values[:, prompt_len:]               # (B, T)

    # 5. Sequence-level RM score (Safe-Decode Implementation)
    # Using skip_special_tokens=False ensures the Reward Model receives 
    # the structural context and prevents zero-length tensor crashes.
    texts = policy_tokenizer.batch_decode(full_ids, skip_special_tokens=False)
    
    # Safety fallback: ensure no string is empty or purely whitespace
    texts = [t if (t and len(t.strip()) > 0) else "Invalid empty response" for t in texts]

    rewards = score_with_rm(
        rm_model, rm_tokenizer, texts, device=device
    ).to(device)

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
# GAE  (Eq. 8)
# ---------------------------------------------------------------------------

def compute_gae(
    rewards_per_token: torch.Tensor,   # (B, T)
    values: torch.Tensor,              # (B, T)
    response_mask: torch.Tensor,       # (B, T)
    gamma: float = 1.0,
    lam: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE advantages and λ-returns."""
    B, T = rewards_per_token.shape
    advantages = torch.zeros_like(rewards_per_token)
    last_gae = torch.zeros(B, device=rewards_per_token.device)

    for t in reversed(range(T)):
        if t == T - 1:
            next_value = torch.zeros(B, device=values.device)
        else:
            next_value = values[:, t + 1] * response_mask[:, t + 1]

        delta = rewards_per_token[:, t] + gamma * next_value - values[:, t]
        last_gae = delta + gamma * lam * last_gae
        last_gae = last_gae * response_mask[:, t]
        advantages[:, t] = last_gae

    returns = (values + advantages).detach()
    return advantages, returns


# ---------------------------------------------------------------------------
# PPO update step  (Eq. 9)
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
    mask = batch.response_mask
    n_valid = mask.sum().clamp(min=1)
    prompt_len = batch.prompt_ids.shape[1]
    B = batch.prompt_ids.shape[0]
    
    # Normalize advantages
    valid_advs = advantages[mask.bool()]
    adv_mean = valid_advs.mean()
    adv_std = valid_advs.std()
    norm_adv = ((advantages - adv_mean) / (adv_std + 1e-8)) * mask

    totals = {"policy_loss": 0.0, "value_loss": 0.0, "kl": 0.0, "clip_frac": 0.0, "grad_norm": 0.0}
    
    # CS Major Move: Use a small mini-batch size for the update
    # Adjust this based on VRAM (2 is safe, 4 is faster)
    mbs = 2 

    for _epoch in range(cfg.ppo_epochs):
        for i in range(0, B, mbs):
            # Slice the batch
            m_ids = torch.cat([batch.prompt_ids[i:i+mbs], batch.response_ids[i:i+mbs]], dim=1)
            m_mask = torch.cat([batch.prompt_mask[i:i+mbs], batch.response_mask[i:i+mbs]], dim=1)
            m_resp_mask = batch.response_mask[i:i+mbs]
            m_old_logprobs = batch.old_logprobs[i:i+mbs]
            m_norm_adv = norm_adv[i:i+mbs]
            m_returns = returns[i:i+mbs]
            m_resp_ids = batch.response_ids[i:i+mbs]
            
            m_valid_tokens = m_resp_mask.sum().clamp(min=1)

            # -------------------- Policy --------------------
            policy.train()
            policy_optimizer.zero_grad()
            logits = policy(input_ids=m_ids, attention_mask=m_mask).logits[:, prompt_len - 1 : -1, :]
            
            log_probs = F.log_softmax(logits, dim=-1)
            current_logprobs = log_probs.gather(2, m_resp_ids.unsqueeze(-1)).squeeze(-1)

            ratio = torch.exp(current_logprobs - m_old_logprobs.detach())
            surr1 = ratio * m_norm_adv.detach()
            surr2 = torch.clamp(ratio, 1.0 - cfg.epsilon, 1.0 + cfg.epsilon) * m_norm_adv.detach()
            
            approx_kl = (m_old_logprobs.detach() - current_logprobs)
            policy_loss = -torch.min(surr1, surr2).sum() / m_valid_tokens
            policy_loss += cfg.kl_coef * (approx_kl * m_resp_mask).sum() / m_valid_tokens
            
            policy_loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            policy_optimizer.step()

            # -------------------- Value --------------------
            value_model.train()
            value_optimizer.zero_grad()
            current_values = value_model(m_ids, m_mask)[:, prompt_len:]
            value_loss = ((current_values - m_returns.detach()) ** 2 * m_resp_mask).sum() / m_valid_tokens
            (cfg.value_loss_coef * value_loss).backward()
            nn.utils.clip_grad_norm_(value_model.parameters(), cfg.max_grad_norm)
            value_optimizer.step()

            # Logging accumulation
            totals["policy_loss"] += policy_loss.item() / (B/mbs)
            totals["value_loss"] += value_loss.item() / (B/mbs)
            totals["kl"] += (approx_kl * m_resp_mask).sum().item() / n_valid

    return {k: v / cfg.ppo_epochs for k, v in totals.items()}