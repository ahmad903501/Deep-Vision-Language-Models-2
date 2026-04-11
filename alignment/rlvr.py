"""
RLVR — RL with Verifiable Rewards (GRPO variant using binary r_v on GSM8K).

This is a thin wrapper around grpo.py that swaps the learned RM for
the verifiable reward checker from data/gsm8k.py.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import RLVRConfig
from alignment.grpo import GroupRollout
from utils.generation import generate_with_logprobs, get_per_token_logprobs
from data.gsm8k import verifiable_reward


# ---------------------------------------------------------------------------
# RLVR rollout  (GRPO-style, binary reward from answer checker)
# ---------------------------------------------------------------------------

@torch.no_grad()
def rlvr_rollout(
    policy: nn.Module,
    ref_model: nn.Module,
    prompt_batch: dict,         # includes gold_answer
    policy_tokenizer,
    cfg: RLVRConfig,
    device: str = "cuda",
) -> GroupRollout:
    """GRPO-style group rollout with verifiable reward instead of RM.

    The reward for each completion is r_v ∈ {0, 1}.

    prompt_batch must contain:
        input_ids:      (B, P)
        attention_mask:  (B, P)
        gold_answers:    list[float] of length B
    """
    policy.eval()
    ref_model.eval()

    input_ids = prompt_batch["input_ids"].to(device)       # (B, P)
    attention_mask = prompt_batch["attention_mask"].to(device)
    gold_answers = prompt_batch["gold_answers"]             # list of floats, len B
    B, P = input_ids.shape
    K = cfg.K

    # Repeat each prompt K times: (B, P) -> (B*K, P)
    input_ids_rep = input_ids.repeat_interleave(K, dim=0)
    attn_mask_rep = attention_mask.repeat_interleave(K, dim=0)

    # 1. Generate K responses per prompt + old log-probs
    gen_out = generate_with_logprobs(
        model=policy,
        input_ids=input_ids_rep,
        attention_mask=attn_mask_rep,
        max_new_tokens=cfg.max_new_tokens,
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

    # 3. Verifiable reward: decode each response, check against gold
    response_texts = policy_tokenizer.batch_decode(response_ids, skip_special_tokens=True)
    # gold_answers[i] corresponds to prompt i; repeat K times
    gold_repeated = [g for g in gold_answers for _ in range(K)]  # len B*K
    rewards = torch.tensor(
        [verifiable_reward(text, gold) for text, gold in zip(response_texts, gold_repeated)],
        dtype=torch.float32, device=device,
    )  # (B*K,)

    # 4. Group-relative advantages: normalize within each group of K
    rewards_grouped = rewards.view(B, K)                      # (B, K)
    group_mean = rewards_grouped.mean(dim=1, keepdim=True)    # (B, 1)
    group_std = rewards_grouped.std(dim=1, keepdim=True)      # (B, 1)
    advantages_grouped = (rewards_grouped - group_mean) / (group_std + 1e-8)  # (B, K)
    advantages = advantages_grouped.view(B * K)               # (B*K,)

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
# RLVR update step  (reuses GRPO mechanics)
# ---------------------------------------------------------------------------

def rlvr_step(
    policy: nn.Module,
    ref_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    group_data: GroupRollout,
    cfg: RLVRConfig,
) -> dict:
    """One RLVR gradient step (reuses GRPO mechanics).

    Returns dict: {loss, kl, clip_frac, pass_rate, degenerate_frac,
                   mean_length, grad_norm, reward_mean, reward_std}.
    """
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

    # Clipped surrogate
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

        # Pass rate: fraction of completions with r_v = 1
        pass_rate = (group_data.rewards > 0.5).float().mean().item()

        # Degenerate fraction: groups where all K rewards are identical
        BK = group_data.rewards.shape[0]
        K = cfg.K
        B = BK // K
        rewards_grouped = group_data.rewards.view(B, K)
        degenerate_frac = (rewards_grouped.std(dim=1) < 1e-6).float().mean().item()

        # Mean response length (in tokens)
        mean_length = (group_data.response_mask.sum(dim=1).float().mean().item())

        gn = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm)

    return {
        "loss": loss.item(),
        "kl": kl.item(),
        "clip_frac": clip_frac.item(),
        "pass_rate": pass_rate,
        "degenerate_frac": degenerate_frac,
        "mean_length": mean_length,
        "reward_mean": group_data.rewards.mean().item(),
        "reward_std": group_data.rewards.std().item(),
        "grad_norm": gn,
    }
