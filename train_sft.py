"""
Entry point: Supervised Fine-Tuning (Task C2).

Usage:
    python train_sft.py
"""
import torch
from config import Config
from data.hh_rlhf import load_hh_rlhf, SFTDataset
from data.collators import PolicyCollator
from model.loader import load_policy_model, apply_lora
from alignment.sft import train_sft
from utils.checkpoint import save_checkpoint
from utils.logging import MetricLogger


def main():
    cfg = Config()
    torch.manual_seed(cfg.seed)

    # --- Data ---
    print("[C2] Loading HH-RLHF data...")
    train_data = load_hh_rlhf(subset=cfg.data.dataset_subset, split="train")
    test_data = load_hh_rlhf(subset=cfg.data.dataset_subset, split="test")

    train_ds = SFTDataset(train_data, max_samples=cfg.sft.max_train_samples)
    test_ds = SFTDataset(test_data)

    # --- Model ---
    print("[C2] Loading policy model...")
    model, tokenizer = load_policy_model(cfg.model)
    model.gradient_checkpointing_enable()
    model = apply_lora(model, cfg.lora, task_type="CAUSAL_LM")

    # --- Collator + DataLoader ---
    collator = PolicyCollator(tokenizer, max_length=cfg.data.max_seq_length)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg.sft.batch_size, shuffle=True,
        collate_fn=collator.collate_sft,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=cfg.sft.batch_size, shuffle=False,
        collate_fn=collator.collate_sft,
    )

    # --- Train ---
    print("[C2] Training SFT...")
    metrics = train_sft(model, train_loader, test_loader, cfg.sft)

    # --- Save as both π_ref (frozen) and π_θ^(0) (trainable) ---
    save_checkpoint(model, None, f"{cfg.checkpoint_dir}/sft", source_method="sft")
    print("[C2] Done. SFT checkpoint saved.")


if __name__ == "__main__":
    main()
