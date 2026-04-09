"""
Supervised Fine-Tuning (SFT) on chosen responses.

Loss: cross-entropy on response tokens only (prompt tokens masked with -100).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import SFTConfig


def train_sft(
    model: nn.Module,
    train_loader: DataLoader,
    eval_loader: DataLoader | None,
    cfg: SFTConfig,
    device: str = "cuda",
) -> dict:
    """Run 1-epoch SFT.  The collator already masks prompt tokens in labels.

    Returns dict of metrics.
    """
    raise NotImplementedError("Implement in Phase 4")
