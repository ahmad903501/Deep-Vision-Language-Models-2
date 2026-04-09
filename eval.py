"""
Entry point: Evaluation (Task C8).

Computes:
  - RM win-rate vs SFT baseline
  - KL from π_ref
  - Sample response table
  - GSM8K pass@1 (for RLVR)
  - Resource table

Usage:
    python eval.py
"""
import torch
from config import Config


def compute_win_rate(
    aligned_model,
    sft_model,
    rm_model,
    rm_tokenizer,
    prompts: list[str],
    policy_tokenizer,
    max_new_tokens: int = 128,
    device: str = "cuda",
) -> float:
    """Fraction of prompts where aligned model scores higher than SFT."""
    raise NotImplementedError("Implement in Phase 9")


def compute_kl_from_ref(
    policy,
    ref_model,
    prompts: list[str],
    tokenizer,
    max_new_tokens: int = 128,
    device: str = "cuda",
) -> float:
    """Monte Carlo KL estimate over held-out prompts."""
    raise NotImplementedError("Implement in Phase 9")


def generate_sample_table(
    models_dict: dict,
    prompts: list[str],
    tokenizer,
    rm_model,
    rm_tokenizer,
    max_new_tokens: int = 128,
) -> list[dict]:
    """Generate side-by-side responses from multiple models."""
    raise NotImplementedError("Implement in Phase 9")


def gsm8k_pass_at_1(
    model,
    tokenizer,
    test_data: list[dict],
    max_new_tokens: int = 256,
    device: str = "cuda",
) -> float:
    """Fraction of GSM8K test problems solved correctly (greedy)."""
    raise NotImplementedError("Implement in Phase 9")


def main():
    cfg = Config()
    torch.manual_seed(cfg.seed)
    print("[eval] Evaluation not yet implemented.")


if __name__ == "__main__":
    main()
