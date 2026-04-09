"""
Reward model training and scoring.

Architecture: AutoModelForSequenceClassification (Llama-3.2-1B backbone).
Loss: Bradley-Terry margin ranking + L2 regularization on reward magnitudes.
Scalar extraction: AutoModelForSequenceClassification pools internally at the
last non-pad token when config.pad_token_id is set.  We also provide a standalone
extraction helper for custom architectures / testing.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from config import RMConfig


# ---------------------------------------------------------------------------
# Scalar extraction  (CRITICAL: handles right-padding correctly)
# ---------------------------------------------------------------------------

def extract_reward_at_last_real_token(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    pad_token_id: int,
) -> torch.Tensor:
    """Extract a scalar from the last non-pad token position in a sequence.

    This is a standalone utility for manual extraction from raw hidden states
    shaped (B, seq_len, D).  AutoModelForSequenceClassification already does
    this pooling internally when config.pad_token_id is set — in that case,
    its output is (B, num_labels) and you can use it directly.

    Args:
        logits:       (B, seq_len, D) — e.g., hidden states projected to scalars
        input_ids:    (B, seq_len) — right-padded
        pad_token_id: id of the padding token

    Returns: (B,) or (B, D) values at the last real token position
    """
    non_pad = (input_ids != pad_token_id).long()       # (B, L)
    last_real_idx = non_pad.sum(dim=1) - 1             # (B,)
    batch_idx = torch.arange(logits.size(0), device=logits.device)

    if logits.dim() == 3:
        return logits[batch_idx, last_real_idx, 0]     # (B,)
    elif logits.dim() == 2:
        return logits[batch_idx, last_real_idx]        # (B,)
    else:
        raise ValueError(f"Expected 2D or 3D logits, got {logits.dim()}D")


def get_rm_scores(
    rm_model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Forward pass through the RM.  Returns (B,) scalar rewards.

    AutoModelForSequenceClassification with num_labels=1 returns (B, 1).
    We squeeze to (B,).
    """
    outputs = rm_model(input_ids=input_ids, attention_mask=attention_mask)
    return outputs.logits.squeeze(-1)  # (B,)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_reward_model(
    rm_model: nn.Module,
    train_loader: DataLoader,
    eval_loader: DataLoader | None,
    cfg: RMConfig,
    device: str = "cuda",
) -> dict:
    """Train the reward model for 1 epoch with Bradley-Terry loss.

    Returns dict of metrics: {train_loss, train_acc, eval_loss, eval_acc}.
    """
    raise NotImplementedError("Implement in Phase 3")


# ---------------------------------------------------------------------------
# Scoring helper (used by PPO / GRPO / eval)
# ---------------------------------------------------------------------------

@torch.no_grad()
def score_with_rm(
    rm_model: nn.Module,
    rm_tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    max_length: int = 1024,
    device: str = "cuda",
) -> torch.Tensor:
    """Score a list of full (prompt+response) strings with the frozen RM.

    Returns: (N,) tensor of scalar rewards.
    """
    raise NotImplementedError("Implement in Phase 3")
