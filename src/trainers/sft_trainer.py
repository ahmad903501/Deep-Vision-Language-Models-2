from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW

from src.config.schema import ExperimentConfig
from src.models.lora_utils import freeze_model
from src.utils.checkpoints import build_manifest, save_checkpoint_manifest


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def train_sft(
    policy_model,
    sft_train_loader,
    config: ExperimentConfig,
    device: str | torch.device,
) -> dict[str, float]:
    device = torch.device(device)
    policy_model.to(device)
    policy_model.train()

    optimizer = AdamW(
        policy_model.parameters(),
        lr=config.sft_train.learning_rate,
        weight_decay=config.sft_train.weight_decay,
    )

    grad_accum = max(1, int(config.sft_train.grad_accum_steps))

    global_step = 0
    optimizer_step = 0
    last_loss = 0.0

    for epoch in range(config.sft_train.epochs):
        optimizer.zero_grad(set_to_none=True)

        for batch in sft_train_loader:
            global_step += 1
            batch = _to_device(batch, device)

            outputs = policy_model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )

            loss = outputs.loss / grad_accum
            loss.backward()

            if global_step % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), config.sft_train.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1

            last_loss = float(outputs.loss.detach().item())

            if global_step % 20 == 0:
                print(f"[SFT][epoch={epoch+1} step={global_step}] loss={last_loss:.4f}")

    return {
        "global_steps": float(global_step),
        "optimizer_steps": float(optimizer_step),
        "final_loss": float(last_loss),
    }


@torch.no_grad()
def evaluate_sft_perplexity(
    policy_model,
    sft_eval_loader,
    device: str | torch.device,
    max_batches: int | None = None,
) -> float:
    device = torch.device(device)
    policy_model.to(device)
    policy_model.eval()

    losses = []
    for idx, batch in enumerate(sft_eval_loader):
        if max_batches is not None and idx >= max_batches:
            break

        batch = _to_device(batch, device)
        outputs = policy_model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        losses.append(float(outputs.loss.detach().item()))

    if not losses:
        return float("inf")

    mean_loss = torch.tensor(losses).mean()
    ppl = torch.exp(mean_loss)
    return float(ppl.item())


@torch.no_grad()
def generate_sft_samples(
    policy_model,
    tokenizer,
    prompts: list[str],
    device: str | torch.device,
    max_new_tokens: int,
) -> list[str]:
    device = torch.device(device)
    policy_model.to(device)
    policy_model.eval()

    inputs = tokenizer(prompts, padding=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    out = policy_model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=max_new_tokens,
    )

    decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
    return decoded


def build_reference_from_sft(policy_model):
    # Separate object with no shared gradient state.
    reference = copy.deepcopy(policy_model)
    freeze_model(reference)
    return reference


def save_sft_artifacts(
    policy_model,
    tokenizer,
    output_dir: str,
    base_checkpoint_name: str,
) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    trainable_dir = out / "pi_theta_init"
    reference_dir = out / "pi_ref"

    trainable_dir.mkdir(parents=True, exist_ok=True)
    reference_dir.mkdir(parents=True, exist_ok=True)

    # Save trainable SFT policy.
    policy_model.save_pretrained(trainable_dir)
    tokenizer.save_pretrained(trainable_dir)

    # Save a separate frozen reference snapshot.
    reference_model = build_reference_from_sft(policy_model)
    reference_model.save_pretrained(reference_dir)
    tokenizer.save_pretrained(reference_dir)

    manifest = build_manifest(
        stage="sft",
        base_checkpoint=base_checkpoint_name,
        data_domain="hh-rlhf-harmless",
        parent_stages=["pretrain"],
        notes="SFT warm-up artifacts for pi_ref and pi_theta_init.",
    )
    save_checkpoint_manifest(str(reference_dir / "manifest.json"), manifest)
    save_checkpoint_manifest(str(trainable_dir / "manifest.json"), manifest)

    return {
        "pi_theta_init": str(trainable_dir),
        "pi_ref": str(reference_dir),
    }
