"""
Evaluation utilities for Task C7 (ablations) and Task C8 (final evaluation).

Provides:
  - compute_win_rate:     RM win-rate vs SFT baseline
  - compute_kl_from_ref:  Monte Carlo KL(π_θ || π_ref) over held-out prompts
  - generate_greedy:      Greedy decode helper
  - generate_sample_table: Side-by-side responses from multiple models + RM scores
  - gsm8k_pass_at_1:      Greedy accuracy on GSM8K test problems

Usage:
    python eval.py
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from config import Config
from alignment.reward_model import score_with_rm


# ---------------------------------------------------------------------------
# Greedy generation helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_greedy(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int = 128,
    max_prompt_length: int = 512,
    device: str = "cuda",
) -> list[str]:
    """Generate greedy responses for a list of prompts.

    Returns list of response strings (prompt portion stripped).
    """
    model.eval()
    tokenizer.padding_side = "left"
    responses = []

    for prompt in prompts:
        enc = tokenizer(
            prompt, return_tensors="pt", truncation=True,
            max_length=max_prompt_length,
        ).to(device)
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        gen_ids = out[0, enc["input_ids"].shape[1]:]
        responses.append(tokenizer.decode(gen_ids, skip_special_tokens=True))

    return responses


# ---------------------------------------------------------------------------
# RM win-rate vs SFT
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_win_rate(
    aligned_model,
    sft_model,
    rm_model,
    rm_tokenizer,
    prompts: list[str],
    policy_tokenizer,
    max_new_tokens: int = 128,
    device: str = "cuda",
) -> dict:
    """Fraction of prompts where aligned model scores higher than SFT.

    Returns dict with win_rate, aligned_scores (list), sft_scores (list).
    """
    aligned_responses = generate_greedy(
        aligned_model, policy_tokenizer, prompts,
        max_new_tokens=max_new_tokens, device=device,
    )
    sft_responses = generate_greedy(
        sft_model, policy_tokenizer, prompts,
        max_new_tokens=max_new_tokens, device=device,
    )

    aligned_texts = [p + " " + r for p, r in zip(prompts, aligned_responses)]
    sft_texts = [p + " " + r for p, r in zip(prompts, sft_responses)]

    aligned_scores = score_with_rm(rm_model, rm_tokenizer, aligned_texts, device=device)
    sft_scores = score_with_rm(rm_model, rm_tokenizer, sft_texts, device=device)

    wins = (aligned_scores > sft_scores).float().mean().item()

    return {
        "win_rate": wins,
        "aligned_scores": aligned_scores.tolist(),
        "sft_scores": sft_scores.tolist(),
        "aligned_responses": aligned_responses,
        "sft_responses": sft_responses,
    }


# ---------------------------------------------------------------------------
# KL from reference (Monte Carlo approximation)
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_kl_from_ref(
    policy,
    ref_model,
    prompts: list[str],
    tokenizer,
    max_new_tokens: int = 128,
    max_prompt_length: int = 512,
    device: str = "cuda",
) -> float:
    """Monte Carlo KL estimate: E_x[ KL(π_θ(·|x) || π_ref(·|x)) ].

    For each prompt, generate a response from policy, then compute
    KL ≈ (1/T) Σ_t [log π_θ(y_t|...) - log π_ref(y_t|...)].
    Average over prompts.
    """
    policy.eval()
    ref_model.eval()
    tokenizer.padding_side = "left"

    kl_estimates = []

    for prompt in prompts:
        enc = tokenizer(
            prompt, return_tensors="pt", truncation=True,
            max_length=max_prompt_length,
        ).to(device)
        P = enc["input_ids"].shape[1]

        # Generate from policy
        out = policy.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
        )
        full_ids = out  # (1, P+T)
        resp_ids = out[:, P:]
        if resp_ids.shape[1] == 0:
            continue

        resp_mask = (resp_ids != tokenizer.pad_token_id).float()
        full_mask = torch.cat([enc["attention_mask"].float(), resp_mask], dim=1)

        # Policy log-probs
        pi_out = policy(input_ids=full_ids, attention_mask=full_mask)
        pi_logits = pi_out.logits[:, P - 1:-1, :]
        pi_logp = F.log_softmax(pi_logits, dim=-1)
        pi_token_logp = pi_logp.gather(2, resp_ids.unsqueeze(-1)).squeeze(-1)

        # Ref log-probs
        ref_out = ref_model(input_ids=full_ids, attention_mask=full_mask)
        ref_logits = ref_out.logits[:, P - 1:-1, :]
        ref_logp = F.log_softmax(ref_logits, dim=-1)
        ref_token_logp = ref_logp.gather(2, resp_ids.unsqueeze(-1)).squeeze(-1)

        # KL per token = log π_θ - log π_ref, averaged over response
        kl_per_token = (pi_token_logp - ref_token_logp) * resp_mask
        n_tokens = resp_mask.sum().clamp(min=1)
        kl_estimates.append((kl_per_token.sum() / n_tokens).item())

    return sum(kl_estimates) / max(len(kl_estimates), 1)


# ---------------------------------------------------------------------------
# Sample response table
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_sample_table(
    models_dict: dict,
    prompts: list[str],
    tokenizer,
    rm_model,
    rm_tokenizer,
    max_new_tokens: int = 128,
    device: str = "cuda",
) -> list[dict]:
    """Generate side-by-side responses from multiple models.

    Args:
        models_dict: {"SFT": model, "PPO": model, "DPO": model, ...}

    Returns list of dicts, one per prompt, each with keys:
        prompt, and for each method: response_<method>, rm_score_<method>
    """
    rows = []
    for prompt in prompts:
        row = {"prompt": prompt}
        for name, model in models_dict.items():
            resp = generate_greedy(
                model, tokenizer, [prompt],
                max_new_tokens=max_new_tokens, device=device,
            )[0]
            full_text = prompt + " " + resp
            score = score_with_rm(
                rm_model, rm_tokenizer, [full_text], device=device,
            ).item()
            row[f"response_{name}"] = resp
            row[f"rm_score_{name}"] = score
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# GSM8K pass@1
# ---------------------------------------------------------------------------

@torch.no_grad()
def gsm8k_pass_at_1(
    model,
    tokenizer,
    test_data: list[dict],
    max_new_tokens: int = 256,
    max_prompt_length: int = 512,
    device: str = "cuda",
) -> dict:
    """Fraction of GSM8K test problems solved correctly (greedy).

    test_data: list of dicts with 'prompt' and 'gold_answer' keys.
    Returns dict with pass_at_1, n_correct, n_total.
    """
    from data.gsm8k import extract_answer, verifiable_reward

    model.eval()
    tokenizer.padding_side = "left"
    n_correct, n_total = 0, 0

    for item in test_data:
        enc = tokenizer(
            item["prompt"], return_tensors="pt", truncation=True,
            max_length=max_prompt_length,
        ).to(device)
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        gen_ids = out[0, enc["input_ids"].shape[1]:]
        resp = tokenizer.decode(gen_ids, skip_special_tokens=True)
        n_total += 1
        if verifiable_reward(resp, item["gold_answer"]) == 1:
            n_correct += 1

    return {
        "pass_at_1": n_correct / max(n_total, 1),
        "n_correct": n_correct,
        "n_total": n_total,
    }


# ---------------------------------------------------------------------------
# DPO preference accuracy on test pairs
# ---------------------------------------------------------------------------

@torch.no_grad()
def dpo_test_pref_accuracy(
    policy,
    ref_model,
    test_loader,
    beta: float,
    device: str = "cuda",
) -> float:
    """Compute preference accuracy on test preference pairs.

    Uses dpo_loss in eval mode to check if the model ranks chosen > rejected.
    """
    from alignment.dpo import dpo_loss

    policy.eval()
    ref_model.eval()
    accs = []
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        _, metrics = dpo_loss(policy, ref_model, batch, beta)
        accs.append(metrics["pref_accuracy"])
    return sum(accs) / max(len(accs), 1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    cfg = Config()
    torch.manual_seed(cfg.seed)
    print("[eval] Run evaluation from the notebook (Phase 9).")


if __name__ == "__main__":
    main()
