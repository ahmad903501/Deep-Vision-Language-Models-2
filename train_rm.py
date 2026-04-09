from __future__ import annotations

import os
import random

import torch

from src.config import default_experiment_config
from src.data import build_dataloaders, load_hh_harmless_dataset, parse_hh_split
from src.models import create_model_bundle, load_reward_model
from src.trainers import evaluate_reward_model, save_reward_artifact, train_reward_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    cfg = default_experiment_config()
    set_seed(cfg.seed)

    dataset = load_hh_harmless_dataset(
        dataset_name=cfg.data.hh_dataset_name,
        harmless_config=cfg.data.hh_harmless_config,
    )

    train_triples = parse_hh_split(dataset["train"], skip_invalid=False)
    test_triples = parse_hh_split(dataset["test"], skip_invalid=False)

    bundle = create_model_bundle(cfg)

    train_loaders = build_dataloaders(train_triples, bundle.policy_tokenizer, bundle.rm_tokenizer, cfg.data)
    test_loaders = build_dataloaders(test_triples, bundle.policy_tokenizer, bundle.rm_tokenizer, cfg.data)

    rm_model = load_reward_model(cfg, rm_tokenizer=bundle.rm_tokenizer)

    # Required: ensure RM pad token id matches right-padded RM tokenizer.
    rm_model.config.pad_token_id = bundle.rm_tokenizer.pad_token_id
    if rm_model.config.pad_token_id != bundle.rm_tokenizer.pad_token_id:
        raise ValueError("RM config pad_token_id mismatch with RM tokenizer.")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_metrics = train_reward_model(rm_model, train_loaders["rm"], cfg, device=device)
    eval_metrics = evaluate_reward_model(rm_model, test_loaders["rm"], device=device)

    out_dir = os.path.join("artifacts", "rm_model")
    saved_path = save_reward_artifact(rm_model, bundle.rm_tokenizer, out_dir)

    print("\n=== RM TRAIN METRICS ===")
    print(train_metrics)
    print("\n=== RM EVAL METRICS ===")
    print(eval_metrics)
    print("RM adaptation mode:", cfg.rm_train.adaptation_mode)
    print("\nSaved RM artifact:", saved_path)


if __name__ == "__main__":
    main()
