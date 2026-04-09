"""
Entry point: PPO / DPO / GRPO / RLVR training (Task C3–C6).

Usage:
    python train_rl.py --method ppo
    python train_rl.py --method dpo
    python train_rl.py --method grpo
    python train_rl.py --method rlvr
"""
import argparse

import torch
from config import Config


def run_ppo(cfg: Config):
    """PPO training loop (Task C3)."""
    raise NotImplementedError("Implement in Phase 5")


def run_dpo(cfg: Config):
    """DPO training loop (Task C4)."""
    raise NotImplementedError("Implement in Phase 6")


def run_grpo(cfg: Config):
    """GRPO training loop (Task C5)."""
    raise NotImplementedError("Implement in Phase 7")


def run_rlvr(cfg: Config):
    """RLVR training loop (Task C6)."""
    from utils.checkpoint import load_checkpoint_for_rlvr

    # Hard guard: must start from clean SFT
    ckpt_path = f"{cfg.checkpoint_dir}/sft"
    load_checkpoint_for_rlvr(ckpt_path)
    raise NotImplementedError("Implement in Phase 8")


METHODS = {
    "ppo": run_ppo,
    "dpo": run_dpo,
    "grpo": run_grpo,
    "rlvr": run_rlvr,
}


def main():
    parser = argparse.ArgumentParser(description="RL alignment training")
    parser.add_argument(
        "--method", type=str, required=True, choices=list(METHODS.keys()),
        help="Alignment method to run",
    )
    args = parser.parse_args()

    cfg = Config()
    torch.manual_seed(cfg.seed)

    print(f"[train_rl] Running method: {args.method}")
    METHODS[args.method](cfg)


if __name__ == "__main__":
    main()
