"""
Supervised Fine-Tuning (SFT) on chosen responses.

Loss: cross-entropy on response tokens only (prompt tokens masked with -100).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import SFTConfig


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_sft(
    model: nn.Module,
    eval_loader: DataLoader,
    device: str = "cuda",
) -> float:
    """Compute average per-token perplexity on the eval set."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in tqdm(eval_loader, desc="[SFT] Evaluating"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        total_loss += outputs.loss.item()
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    perplexity = math.exp(min(avg_loss, 100))  # cap to avoid overflow
    model.train()
    return perplexity


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

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
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    # Linear warmup then constant LR
    total_steps = (len(train_loader) * cfg.epochs) // cfg.grad_accum_steps
    def lr_lambda(step: int) -> float:
        if step < cfg.warmup_steps:
            return step / max(1, cfg.warmup_steps)
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    metrics: dict = {}
    global_step = 0
    running_loss = 0.0
    optimizer.zero_grad()

    for epoch in range(cfg.epochs):
        pbar = tqdm(train_loader, desc=f"[SFT] Epoch {epoch+1}/{cfg.epochs}")
        for micro_step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss / cfg.grad_accum_steps
            loss.backward()

            running_loss += loss.item()

            if (micro_step + 1) % cfg.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Logging
                if global_step % cfg.log_every == 0:
                    avg_loss = running_loss / cfg.log_every
                    tqdm.write(
                        f"  [SFT] step {global_step}/{total_steps} | "
                        f"loss={avg_loss:.4f} | "
                        f"ppl={math.exp(min(avg_loss * cfg.grad_accum_steps, 100)):.2f} | "
                        f"lr={scheduler.get_last_lr()[0]:.2e}"
                    )
                    running_loss = 0.0

                # Eval
                if eval_loader is not None and global_step % cfg.eval_every == 0:
                    ppl = evaluate_sft(model, eval_loader, device)
                    tqdm.write(f"  [SFT] step {global_step} | eval_ppl={ppl:.2f}")

            pbar.set_postfix(
                loss=outputs.loss.item(),
                step=global_step,
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )
        pbar.close()

        # Handle leftover micro-steps (if dataset size % grad_accum != 0)
        if (micro_step + 1) % cfg.grad_accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

    # --- Final eval ---
    metrics["train_steps"] = global_step
    if eval_loader is not None:
        final_ppl = evaluate_sft(model, eval_loader, device)
        metrics["eval_perplexity"] = final_ppl
        print(f"  [SFT] Final eval perplexity: {final_ppl:.2f}")

    return metrics
