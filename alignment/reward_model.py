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
from tqdm.auto import tqdm
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


def get_rm_scores_headonly(
    rm_model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Head-only forward: backbone runs under no_grad, only score head gets gradients.

    This avoids building a computation graph through the entire frozen backbone,
    making backward() essentially free (only ~2048 params in the linear head).
    """
    # 1) Forward through frozen backbone — no graph needed
    with torch.no_grad():
        backbone = rm_model.model  # LlamaModel inside AutoModelForSequenceClassification
        transformer_outputs = backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden_states = transformer_outputs[0]  # (B, seq_len, hidden_dim)

    # 2) Detach so backward stops here, then cast dtype for score head
    hidden_states = hidden_states.detach().to(rm_model.score.weight.dtype)

    # 3) Forward through the trainable score head
    logits = rm_model.score(hidden_states)  # (B, seq_len, 1)

    # 4) Pool at the last non-pad token (same logic as AutoModelForSequenceClassification)
    batch_size = input_ids.shape[0]
    sequence_lengths = (attention_mask.sum(dim=1) - 1).long()  # last real token idx
    pooled = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]
    return pooled.squeeze(-1)  # (B,)


# ---------------------------------------------------------------------------
# Bradley-Terry loss with L2 regularization
# ---------------------------------------------------------------------------

def rm_loss_fn(
    r_chosen: torch.Tensor,
    r_rejected: torch.Tensor,
    lambda_reg: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Bradley-Terry margin ranking loss + L2 reward regularization.

    L = -E[log σ(r⁺ - r⁻)] + λ_reg · E[(r⁺)² + (r⁻)²]

    Args:
        r_chosen:   (B,) scalar rewards for chosen responses
        r_rejected: (B,) scalar rewards for rejected responses
        lambda_reg: L2 penalty weight on reward magnitudes

    Returns:
        (total_loss, pref_accuracy) where pref_accuracy = fraction r⁺ > r⁻
    """
    margin = r_chosen - r_rejected
    bt_loss = -F.logsigmoid(margin).mean()
    reg_loss = lambda_reg * (r_chosen.pow(2).mean() + r_rejected.pow(2).mean())
    total_loss = bt_loss + reg_loss

    with torch.no_grad():
        acc = (margin > 0).float().mean()

    return total_loss, acc


# ---------------------------------------------------------------------------
# Precompute hidden states (one-time backbone pass)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _precompute_embeddings(
    rm_model: nn.Module,
    data_loader: DataLoader,
    device: str = "cuda",
    desc: str = "[RM] Precomputing embeddings",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the frozen backbone once over the dataset and cache the pooled hidden state
    at the last non-pad token for each chosen and rejected sequence.

    Returns:
        chosen_embeds:   (N, hidden_dim) on CPU
        rejected_embeds: (N, hidden_dim) on CPU
    """
    rm_model.eval()
    backbone = rm_model.model  # LlamaModel inside AutoModelForSequenceClassification

    all_chosen = []
    all_rejected = []

    for batch in tqdm(data_loader, desc=desc):
        for key_prefix, accumulator in [("chosen", all_chosen), ("rejected", all_rejected)]:
            ids = batch[f"{key_prefix}_input_ids"].to(device)
            mask = batch[f"{key_prefix}_attention_mask"].to(device)

            out = backbone(input_ids=ids, attention_mask=mask)
            hidden = out[0]  # (B, seq_len, hidden_dim)

            # Pool at last real token
            seq_lengths = mask.sum(dim=1) - 1  # (B,)
            pooled = hidden[torch.arange(hidden.size(0), device=device), seq_lengths]
            accumulator.append(pooled.cpu().float())  # store in fp32 on CPU

    return torch.cat(all_chosen), torch.cat(all_rejected)


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
    """Train the reward model head on precomputed backbone embeddings.

    Strategy: Since only the (hidden_dim → 1) score head is trainable and
    the backbone is frozen, we precompute hidden states in one pass (~20 min),
    then train the head on cached embeddings (seconds).

    Returns dict of metrics: {train_loss, train_acc, eval_loss, eval_acc}.
    """

    # =====================================================================
    # Phase 1: Precompute embeddings (one forward pass through backbone)
    # =====================================================================
    print("[RM] Phase 1/2: Precomputing backbone embeddings (one-time cost)...")
    train_chosen_emb, train_rejected_emb = _precompute_embeddings(
        rm_model, train_loader, device, desc="[RM] Precomputing train embeddings"
    )
    print(f"  Cached {train_chosen_emb.shape[0]} pairs, dim={train_chosen_emb.shape[1]}")

    eval_chosen_emb, eval_rejected_emb = None, None
    if eval_loader is not None:
        eval_chosen_emb, eval_rejected_emb = _precompute_embeddings(
            rm_model, eval_loader, device, desc="[RM] Precomputing eval embeddings"
        )
        print(f"  Cached {eval_chosen_emb.shape[0]} eval pairs")

    # =====================================================================
    # Phase 2: Train the score head on cached embeddings
    # =====================================================================
    print("[RM] Phase 2/2: Training score head on cached embeddings...")
    score_head = rm_model.score  # nn.Linear(hidden_dim, 1)
    score_head.train()

    optimizer = torch.optim.AdamW(
        score_head.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    n_train = train_chosen_emb.shape[0]
    total_steps = (n_train // cfg.batch_size) * cfg.epochs
    step_count = 0

    def lr_lambda(current_step: int) -> float:
        if current_step < cfg.warmup_steps:
            return current_step / max(1, cfg.warmup_steps)
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    running_loss = 0.0
    running_acc = 0.0
    metrics: dict = {}

    for epoch in range(cfg.epochs):
        # Shuffle training pairs
        perm = torch.randperm(n_train)
        train_chosen_emb = train_chosen_emb[perm]
        train_rejected_emb = train_rejected_emb[perm]

        n_batches = n_train // cfg.batch_size
        pbar = tqdm(range(n_batches), desc=f"[RM] Head training epoch {epoch+1}/{cfg.epochs}")
        for i in pbar:
            start = i * cfg.batch_size
            end = start + cfg.batch_size

            c_emb = train_chosen_emb[start:end].to(device).to(score_head.weight.dtype)
            r_emb = train_rejected_emb[start:end].to(device).to(score_head.weight.dtype)

            r_chosen = score_head(c_emb).squeeze(-1)     # (B,)
            r_rejected = score_head(r_emb).squeeze(-1)    # (B,)

            loss, acc = rm_loss_fn(r_chosen, r_rejected, cfg.lambda_reg)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            running_acc += acc.item()
            step_count += 1

            pbar.set_postfix(loss=loss.item(), acc=acc.item(), lr=f"{scheduler.get_last_lr()[0]:.2e}")

            if step_count % cfg.log_every == 0:
                avg_loss = running_loss / cfg.log_every
                avg_acc = running_acc / cfg.log_every
                tqdm.write(
                    f"  [RM] step {step_count}/{total_steps} | "
                    f"loss={avg_loss:.4f} | acc={avg_acc:.4f} | "
                    f"lr={scheduler.get_last_lr()[0]:.2e}"
                )
                running_loss = 0.0
                running_acc = 0.0
        pbar.close()

    # --- Final train metrics ---
    metrics["train_loss"] = loss.item()
    metrics["train_steps"] = step_count

    # --- Eval on cached embeddings ---
    if eval_chosen_emb is not None:
        score_head.eval()
        with torch.no_grad():
            all_r_chosen, all_r_rejected = [], []
            n_eval = eval_chosen_emb.shape[0]
            total_loss, total_acc, n_batches = 0.0, 0.0, 0
            for i in range(0, n_eval, cfg.batch_size):
                c_emb = eval_chosen_emb[i:i+cfg.batch_size].to(device).to(score_head.weight.dtype)
                r_emb = eval_rejected_emb[i:i+cfg.batch_size].to(device).to(score_head.weight.dtype)
                rc = score_head(c_emb).squeeze(-1)
                rr = score_head(r_emb).squeeze(-1)
                loss, acc = rm_loss_fn(rc, rr, cfg.lambda_reg)
                total_loss += loss.item()
                total_acc += acc.item()
                n_batches += 1
                all_r_chosen.append(rc.cpu())
                all_r_rejected.append(rr.cpu())

            metrics["eval_loss"] = total_loss / max(n_batches, 1)
            metrics["eval_acc"] = total_acc / max(n_batches, 1)
            metrics["r_chosen_all"] = torch.cat(all_r_chosen)
            metrics["r_rejected_all"] = torch.cat(all_r_rejected)
            metrics["r_chosen_mean"] = metrics["r_chosen_all"].mean().item()
            metrics["r_rejected_mean"] = metrics["r_rejected_all"].mean().item()
            print(
                f"  [RM] Eval | loss={metrics['eval_loss']:.4f} | acc={metrics['eval_acc']:.4f} | "
                f"r+_mean={metrics['r_chosen_mean']:.3f} | "
                f"r-_mean={metrics['r_rejected_mean']:.3f}"
            )

    # Freeze after training
    rm_model.eval()
    for p in rm_model.parameters():
        p.requires_grad = False

    return metrics


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_rm(
    rm_model: nn.Module,
    eval_loader: DataLoader,
    lambda_reg: float = 1e-3,
    device: str = "cuda",
) -> tuple[float, float, torch.Tensor, torch.Tensor]:
    """Evaluate RM preference accuracy on a dataset.

    Returns:
        (avg_loss, avg_acc, r_chosen_all, r_rejected_all)
        where r_*_all are (N,) tensors for histogram plotting.
    """
    rm_model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0
    all_r_chosen = []
    all_r_rejected = []

    for batch in tqdm(eval_loader, desc="[RM] Evaluating"):
        chosen_ids = batch["chosen_input_ids"].to(device)
        chosen_mask = batch["chosen_attention_mask"].to(device)
        rejected_ids = batch["rejected_input_ids"].to(device)
        rejected_mask = batch["rejected_attention_mask"].to(device)

        r_chosen = get_rm_scores(rm_model, chosen_ids, chosen_mask)
        r_rejected = get_rm_scores(rm_model, rejected_ids, rejected_mask)

        loss, acc = rm_loss_fn(r_chosen, r_rejected, lambda_reg)
        total_loss += loss.item()
        total_acc += acc.item()
        n_batches += 1

        all_r_chosen.append(r_chosen.cpu())
        all_r_rejected.append(r_rejected.cpu())

    avg_loss = total_loss / max(n_batches, 1)
    avg_acc = total_acc / max(n_batches, 1)
    return avg_loss, avg_acc, torch.cat(all_r_chosen), torch.cat(all_r_rejected)


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
    batch_size: int = 8,
) -> torch.Tensor:
    """Score a list of full (prompt+response) strings with the frozen RM.

    Returns: (N,) tensor of scalar rewards on CPU.
    """
    rm_model.eval()
    rm_tokenizer.padding_side = "right"
    all_scores = []

    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        enc = rm_tokenizer(
            chunk,
            padding="longest",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        scores = get_rm_scores(rm_model, ids, mask)
        all_scores.append(scores.cpu())

    return torch.cat(all_scores)
