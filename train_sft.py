from __future__ import annotations

import os
import random

import torch

from src.config import default_experiment_config
from src.data import build_dataloaders, load_hh_harmless_dataset, parse_hh_split
from src.models import create_model_bundle
from src.trainers import (
    evaluate_sft_perplexity,
    generate_sft_samples,
    save_sft_artifacts,
    train_sft,
)


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

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_metrics = train_sft(bundle.policy_model, train_loaders["sft"], cfg, device=device)
    ppl = evaluate_sft_perplexity(
        bundle.policy_model,
        test_loaders["sft"],
        device=device,
        max_batches=20,
    )

    prompts = [tri.prompt for tri in test_triples[:5]]
    samples = generate_sft_samples(
        bundle.policy_model,
        bundle.policy_tokenizer,
        prompts,
        device=device,
        max_new_tokens=cfg.sft_train.sample_gen_max_new_tokens,
    )

    artifacts = save_sft_artifacts(
        bundle.policy_model,
        bundle.policy_tokenizer,
        output_dir=os.path.join("artifacts", "sft"),
        base_checkpoint_name=cfg.model.policy_model_name,
    )

    print("\n=== SFT TRAIN METRICS ===")
    print(train_metrics)
    print("Held-out perplexity:", ppl)

    print("\n=== SFT SAMPLE OUTPUTS ===")
    for i, sample in enumerate(samples, start=1):
        print(f"\n--- sample {i} ---")
        print(sample)

    print("\nSaved SFT artifacts:")
    print(artifacts)


if __name__ == "__main__":
    main()
