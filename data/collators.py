"""
Collators that handle tokenization + padding with the correct padding side.

CRITICAL DESIGN RULE:
  - PolicyCollator  → padding_side = "left"   (decoder-only generation)
  - RMCollator      → padding_side = "right"  (sequence classification)

The padding_side is SET inside the collator, not assumed from the tokenizer's
global state.  This prevents silent bugs when a shared tokenizer object is
reused across dataloaders.
"""
from __future__ import annotations

from typing import Optional

import torch
from transformers import PreTrainedTokenizerBase


class PolicyCollator:
    """Collator for SFT, DPO, and prompt-only batches.

    Enforces left-padding so generated tokens align at the right edge.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 1024,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

    # -- SFT batches ----------------------------------------------------------

    # -- SFT batches ----------------------------------------------------------

    def collate_sft(self, batch: list[dict]) -> dict:
        """Collate SFT examples (prompt + chosen response + EOS).

        Returns:
            input_ids, attention_mask, labels (prompt tokens masked with -100)
        """
        self.tokenizer.padding_side = "left"  # Required for decoder-only generation [cite: 259]
        
        # We append the eos_token here so the model learns to terminate generation.
        full_texts = [ex["prompt"] + " " + ex["response"] + self.tokenizer.eos_token for ex in batch]
        prompts = [ex["prompt"] for ex in batch]

        encoded = self.tokenizer(
            full_texts,
            padding="longest",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        # We tokenize prompts separately to find the length for masking [cite: 305, 306]
        prompt_encoded = self.tokenizer(
            prompts,
            padding=False,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
        )

        labels = encoded["input_ids"].clone()
        for i, plen in enumerate(len(ids) for ids in prompt_encoded["input_ids"]):
            # For left-padding, the actual content starts after the padding tokens.
            # total_len = number of non-pad tokens in this specific sequence
            total_len = encoded["attention_mask"][i].sum().item()
            # pad_len = number of leading [PAD] tokens in the batch tensor
            pad_len = encoded["input_ids"].size(1) - total_len
            
            # CRITICAL: Mask everything from the start through the padding and the prompt tokens.
            # Only the response and the appended EOS token remain unmasked for loss calculation[cite: 305, 306].
            labels[i, : pad_len + plen] = -100

        # Also ensure explicit padding tokens are masked to prevent gradient interference
        labels[encoded["attention_mask"] == 0] = -100

        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": labels,
        }

    # -- Prompt-only batches ---------------------------------------------------

    def collate_prompts(self, batch: list[dict]) -> dict:
        """Collate prompt-only examples for PPO/GRPO rollout."""
        self.tokenizer.padding_side = "left"
        prompts = [ex["prompt"] for ex in batch]
        encoded = self.tokenizer(
            prompts,
            padding="longest",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }

    # -- DPO batches -----------------------------------------------------------

    def collate_dpo(self, batch: list[dict]) -> dict:
        """Collate DPO examples.

        Tokenizes prompt+chosen and prompt+rejected separately.
        Returns input_ids/mask for both, plus response_start indices.
        """
        self.tokenizer.padding_side = "left"

        chosen_texts = [ex["prompt"] + " " + ex["chosen"] for ex in batch]
        rejected_texts = [ex["prompt"] + " " + ex["rejected"] for ex in batch]
        prompts = [ex["prompt"] for ex in batch]

        chosen_enc = self.tokenizer(
            chosen_texts, padding="longest", truncation=True,
            max_length=self.max_length, return_tensors="pt",
        )
        rejected_enc = self.tokenizer(
            rejected_texts, padding="longest", truncation=True,
            max_length=self.max_length, return_tensors="pt",
        )
        prompt_enc = self.tokenizer(
            prompts, padding=False, truncation=True,
            max_length=self.max_length, add_special_tokens=False,
        )

        # Compute response_start index for each example
        # With left-padding: response_start = pad_len + prompt_token_len
        chosen_starts = []
        rejected_starts = []
        for i, plen in enumerate(len(ids) for ids in prompt_enc["input_ids"]):
            c_total = chosen_enc["attention_mask"][i].sum().item()
            c_pad = chosen_enc["input_ids"].size(1) - c_total
            chosen_starts.append(c_pad + plen)

            r_total = rejected_enc["attention_mask"][i].sum().item()
            r_pad = rejected_enc["input_ids"].size(1) - r_total
            rejected_starts.append(r_pad + plen)

        return {
            "chosen_input_ids": chosen_enc["input_ids"],
            "chosen_attention_mask": chosen_enc["attention_mask"],
            "rejected_input_ids": rejected_enc["input_ids"],
            "rejected_attention_mask": rejected_enc["attention_mask"],
            "chosen_response_start": torch.tensor(chosen_starts, dtype=torch.long),
            "rejected_response_start": torch.tensor(rejected_starts, dtype=torch.long),
        }

    # -- GSM8K batches (for RLVR) ---------------------------------------------

    def collate_gsm8k(self, batch: list[dict]) -> dict:
        """Collate GSM8K examples for RLVR rollout.

        Each item has keys: prompt, gold_answer.
        Returns input_ids, attention_mask, gold_answers (list[float]).
        """
        self.tokenizer.padding_side = "left"
        prompts = [ex["prompt"] for ex in batch]
        encoded = self.tokenizer(
            prompts,
            padding="longest",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "gold_answers": [ex["gold_answer"] for ex in batch],
        }


class RMCollator:
    """Collator for reward model training/inference.

    Enforces right-padding.  The scalar is extracted from the last *non-pad*
    token position (handled in reward_model.py, not here).
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 1024,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: list[dict]) -> dict:
        """Collate RM preference pairs.

        Each item has keys: prompt, chosen, rejected.
        """
        self.tokenizer.padding_side = "right"

        chosen_texts = [ex["prompt"] + " " + ex["chosen"] for ex in batch]
        rejected_texts = [ex["prompt"] + " " + ex["rejected"] for ex in batch]

        chosen_enc = self.tokenizer(
            chosen_texts, padding="longest", truncation=True,
            max_length=self.max_length, return_tensors="pt",
        )
        rejected_enc = self.tokenizer(
            rejected_texts, padding="longest", truncation=True,
            max_length=self.max_length, return_tensors="pt",
        )

        return {
            "chosen_input_ids": chosen_enc["input_ids"],
            "chosen_attention_mask": chosen_enc["attention_mask"],
            "rejected_input_ids": rejected_enc["input_ids"],
            "rejected_attention_mask": rejected_enc["attention_mask"],
        }
