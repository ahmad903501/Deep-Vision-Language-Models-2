"""
Checkpoint save / load utilities.

Each checkpoint stores:
  - model state (LoRA adapter weights)
  - optimizer state
  - meta.json with source_method, step, metrics

RLVR has a hard guard: it MUST start from a clean SFT checkpoint.
"""
from __future__ import annotations

import json
import os
from typing import Optional

import torch
import torch.nn as nn


RLVR_FORBIDDEN_TAGS = {"ppo", "dpo", "grpo"}


def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    path: str,
    source_method: str,
    step: int = 0,
    metrics: dict | None = None,
):
    """Save LoRA adapter weights + metadata.

    Args:
        model: PeftModel (will call save_pretrained)
        optimizer: optional optimizer to save
        path: directory to save into
        source_method: one of "sft", "ppo", "dpo", "grpo", "rlvr"
        step: current training step
        metrics: optional dict of metrics to record
    """
    os.makedirs(path, exist_ok=True)

    # Save adapter weights
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(path)
    else:
        torch.save(model.state_dict(), os.path.join(path, "model.pt"))

    if optimizer is not None:
        torch.save(optimizer.state_dict(), os.path.join(path, "optimizer.pt"))

    meta = {
        "source_method": source_method,
        "step": step,
        "metrics": metrics or {},
    }
    with open(os.path.join(path, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[checkpoint] Saved {source_method} checkpoint at step {step} → {path}")


def load_checkpoint(path: str) -> dict:
    """Load checkpoint metadata.  Model loading is handled by PEFT.

    Returns the meta dict.
    """
    meta_path = os.path.join(path, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"No meta.json at {path}")
    with open(meta_path) as f:
        return json.load(f)


def load_checkpoint_for_rlvr(path: str) -> dict:
    """Load checkpoint for RLVR, enforcing that it's a clean SFT checkpoint.

    Raises ValueError if the checkpoint was produced by PPO/DPO/GRPO.
    """
    meta = load_checkpoint(path)
    source = meta.get("source_method", "unknown")
    if source in RLVR_FORBIDDEN_TAGS:
        raise ValueError(
            f"RLVR must start from a clean SFT checkpoint, "
            f"but this checkpoint was produced by '{source}'. "
            f"Use the SFT checkpoint from Task C2 instead."
        )
    return meta
