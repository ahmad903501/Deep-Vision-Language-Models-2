from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from src.config.schema import ExperimentConfig
from src.models.reward_scorer import reward_scores_from_batch


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def reward_model_loss(
    chosen_scores: torch.Tensor,
    rejected_scores: torch.Tensor,
    lambda_reg: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    pref_loss = -F.logsigmoid(chosen_scores - rejected_scores).mean()
    reg_loss = lambda_reg * ((chosen_scores.square() + rejected_scores.square()).mean())
    loss = pref_loss + reg_loss

    metrics = {
        "loss": float(loss.detach().item()),
        "pref_loss": float(pref_loss.detach().item()),
        "reg_loss": float(reg_loss.detach().item()),
        "pref_accuracy": float((chosen_scores > rejected_scores).float().mean().item()),
    }
    return loss, metrics


def train_reward_model(
    rm_model,
    rm_train_loader,
    config: ExperimentConfig,
    device: str | torch.device,
) -> dict[str, float]:
    device = torch.device(device)
    rm_model.to(device)
    rm_model.train()

    if rm_model.config.pad_token_id is None:
        raise ValueError("rm.config.pad_token_id is not set.")

    optimizer = AdamW(
        rm_model.parameters(),
        lr=config.rm_train.learning_rate,
        weight_decay=config.rm_train.weight_decay,
    )

    global_step = 0
    last_metrics: dict[str, float] = {}

    for epoch in range(config.rm_train.epochs):
        for batch in rm_train_loader:
            global_step += 1
            batch = _to_device(batch, device)

            chosen_scores = reward_scores_from_batch(
                rm_model,
                input_ids=batch["chosen_input_ids"],
                attention_mask=batch["chosen_attention_mask"],
                provided_last_non_pad_idx=batch["chosen_last_non_pad_idx"],
            )
            rejected_scores = reward_scores_from_batch(
                rm_model,
                input_ids=batch["rejected_input_ids"],
                attention_mask=batch["rejected_attention_mask"],
                provided_last_non_pad_idx=batch["rejected_last_non_pad_idx"],
            )

            loss, metrics = reward_model_loss(
                chosen_scores,
                rejected_scores,
                lambda_reg=config.rm_train.lambda_reg,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rm_model.parameters(), config.rm_train.max_grad_norm)
            optimizer.step()

            if global_step % config.rm_train.log_every == 0:
                print(
                    f"[RM][epoch={epoch+1} step={global_step}] "
                    f"loss={metrics['loss']:.4f} "
                    f"pref_acc={metrics['pref_accuracy']:.4f}"
                )

            last_metrics = metrics

    return {
        "global_steps": float(global_step),
        "final_loss": float(last_metrics.get("loss", 0.0)),
        "final_pref_accuracy": float(last_metrics.get("pref_accuracy", 0.0)),
    }


@torch.no_grad()
def evaluate_reward_model(
    rm_model,
    rm_eval_loader,
    device: str | torch.device,
) -> dict[str, float | list[float]]:
    device = torch.device(device)
    rm_model.to(device)
    rm_model.eval()

    pref_correct = 0
    total = 0
    chosen_all = []
    rejected_all = []

    for batch in rm_eval_loader:
        batch = _to_device(batch, device)

        chosen_scores = reward_scores_from_batch(
            rm_model,
            input_ids=batch["chosen_input_ids"],
            attention_mask=batch["chosen_attention_mask"],
            provided_last_non_pad_idx=batch["chosen_last_non_pad_idx"],
        )
        rejected_scores = reward_scores_from_batch(
            rm_model,
            input_ids=batch["rejected_input_ids"],
            attention_mask=batch["rejected_attention_mask"],
            provided_last_non_pad_idx=batch["rejected_last_non_pad_idx"],
        )

        pref_correct += int((chosen_scores > rejected_scores).sum().item())
        total += int(chosen_scores.shape[0])
        chosen_all.append(chosen_scores.detach().float().cpu())
        rejected_all.append(rejected_scores.detach().float().cpu())

    if total == 0:
        return {
            "pref_accuracy": 0.0,
            "chosen_mean": 0.0,
            "rejected_mean": 0.0,
            "chosen_hist": [],
            "rejected_hist": [],
        }

    chosen_cat = torch.cat(chosen_all)
    rejected_cat = torch.cat(rejected_all)

    chist = torch.histc(chosen_cat, bins=20, min=float(chosen_cat.min()), max=float(chosen_cat.max()))
    rhist = torch.histc(rejected_cat, bins=20, min=float(rejected_cat.min()), max=float(rejected_cat.max()))

    return {
        "pref_accuracy": float(pref_correct / total),
        "chosen_mean": float(chosen_cat.mean().item()),
        "rejected_mean": float(rejected_cat.mean().item()),
        "chosen_hist": chist.tolist(),
        "rejected_hist": rhist.tolist(),
    }


def save_reward_artifact(rm_model, rm_tokenizer, output_dir: str) -> str:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rm_model.save_pretrained(out)
    rm_tokenizer.save_pretrained(out)
    return str(out)
