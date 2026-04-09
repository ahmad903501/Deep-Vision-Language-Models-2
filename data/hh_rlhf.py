"""
Anthropic HH-RLHF dataset loading, parsing, and Dataset classes.

The raw dataset has `chosen` and `rejected` fields, each containing the full
conversation as a single string with "\n\nHuman:" and "\n\nAssistant:" turns.
We parse these into (prompt, chosen_response, rejected_response) triples.
"""
from __future__ import annotations

import re
from typing import Optional

from datasets import load_dataset
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_hh_rlhf_example(example: dict) -> dict:
    """Parse a single HH-RLHF example into (prompt, chosen, rejected).

    The chosen/rejected fields look like:
        "\\n\\nHuman: ...\\n\\nAssistant: ...\\n\\nHuman: ...\\n\\nAssistant: ..."
    We split at the *last* "\\n\\nAssistant:" boundary so that:
        prompt  = everything up to and including the last "\\n\\nAssistant:"
        response = the text after that boundary

    Returns dict with keys: prompt, chosen, rejected
    """
    chosen_text: str = example["chosen"]
    rejected_text: str = example["rejected"]

    prompt = _extract_prompt(chosen_text)
    chosen_response = _extract_response(chosen_text)
    rejected_response = _extract_response(rejected_text)

    return {
        "prompt": prompt,
        "chosen": chosen_response,
        "rejected": rejected_response,
    }


def _extract_prompt(conversation: str) -> str:
    """Return everything up to and including the last '\\n\\nAssistant:' marker."""
    marker = "\n\nAssistant:"
    idx = conversation.rfind(marker)
    if idx == -1:
        return conversation
    return conversation[: idx + len(marker)]


def _extract_response(conversation: str) -> str:
    """Return the text after the last '\\n\\nAssistant:' marker, stripped."""
    marker = "\n\nAssistant:"
    idx = conversation.rfind(marker)
    if idx == -1:
        return ""
    return conversation[idx + len(marker) :].strip()


# ---------------------------------------------------------------------------
# Dataset loading helper
# ---------------------------------------------------------------------------

def load_hh_rlhf(subset: str = "harmless-base", split: str = "train"):
    """Load the HH-RLHF dataset from HuggingFace and parse it.

    Returns a list of dicts with keys: prompt, chosen, rejected.
    """
    ds = load_dataset("Anthropic/hh-rlhf", data_dir=subset, split=split)
    parsed = [parse_hh_rlhf_example(ex) for ex in ds]
    return parsed


# ---------------------------------------------------------------------------
# PyTorch Datasets
# ---------------------------------------------------------------------------

class SFTDataset(Dataset):
    """Dataset for supervised fine-tuning on chosen responses.

    Each item returns raw text strings; tokenization happens in the collator.
    """

    def __init__(self, data: list[dict], max_samples: Optional[int] = None):
        self.data = data[:max_samples] if max_samples else data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> dict:
        item = self.data[idx]
        return {
            "prompt": item["prompt"],
            "response": item["chosen"],
        }


class RMDataset(Dataset):
    """Dataset for reward model training on preference pairs.

    Each item returns (prompt, chosen_response, rejected_response) as raw text.
    """

    def __init__(self, data: list[dict], max_samples: Optional[int] = None):
        self.data = data[:max_samples] if max_samples else data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> dict:
        item = self.data[idx]
        return {
            "prompt": item["prompt"],
            "chosen": item["chosen"],
            "rejected": item["rejected"],
        }


class DPODataset(Dataset):
    """Dataset for DPO training.  Returns raw text triples."""

    def __init__(self, data: list[dict], max_samples: Optional[int] = None):
        self.data = data[:max_samples] if max_samples else data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> dict:
        item = self.data[idx]
        return {
            "prompt": item["prompt"],
            "chosen": item["chosen"],
            "rejected": item["rejected"],
        }


class PromptDataset(Dataset):
    """Dataset that yields prompts only (for PPO / GRPO rollouts)."""

    def __init__(self, data: list[dict], max_samples: Optional[int] = None):
        self.data = data[:max_samples] if max_samples else data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> dict:
        return {"prompt": self.data[idx]["prompt"]}
