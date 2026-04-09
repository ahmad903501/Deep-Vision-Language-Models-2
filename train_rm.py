"""
Entry point: Reward Model training (Task C1).

Usage:
    python train_rm.py
"""
import torch
from config import Config
from data.hh_rlhf import load_hh_rlhf, RMDataset
from data.collators import RMCollator
from model.loader import load_rm_backbone, apply_lora
from alignment.reward_model import train_reward_model
from utils.checkpoint import save_checkpoint
from utils.logging import MetricLogger


def main():
    cfg = Config()
    torch.manual_seed(cfg.seed)

    # --- Data ---
    print("[C1] Loading HH-RLHF data...")
    train_data = load_hh_rlhf(subset=cfg.data.dataset_subset, split="train")
    test_data = load_hh_rlhf(subset=cfg.data.dataset_subset, split="test")

    train_ds = RMDataset(train_data, max_samples=cfg.rm.max_train_samples)
    test_ds = RMDataset(test_data)

    # --- Model ---
    print("[C1] Loading reward model backbone...")
    rm_model, rm_tokenizer = load_rm_backbone(cfg.model, num_labels=1)
    rm_model = apply_lora(rm_model, cfg.lora, task_type="SEQ_CLS")

    # --- Collator + DataLoader ---
    collator = RMCollator(rm_tokenizer, max_length=cfg.data.max_seq_length)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg.rm.batch_size, shuffle=True, collate_fn=collator,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=cfg.rm.batch_size, shuffle=False, collate_fn=collator,
    )

    # --- Train ---
    print("[C1] Training reward model...")
    metrics = train_reward_model(rm_model, train_loader, test_loader, cfg.rm)

    # --- Save ---
    save_checkpoint(rm_model, None, f"{cfg.checkpoint_dir}/rm", source_method="rm")
    print(f"[C1] Done. Test accuracy: {metrics.get('eval_acc', 'N/A')}")


if __name__ == "__main__":
    main()
