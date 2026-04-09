"""
GSM8K dataset loading, answer extraction, and verifiable reward.
"""
from __future__ import annotations

import re
from typing import Optional

from datasets import load_dataset
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

_ANSWER_PATTERNS = [
    re.compile(r"####\s*([+-]?[\d,]+\.?\d*)"),          # GSM8K gold format
    re.compile(r"[Tt]he answer is\s*[:\s]*([+-]?[\d,]+\.?\d*)"),
    re.compile(r"\\boxed\{([+-]?[\d,]+\.?\d*)\}"),      # LaTeX boxed
]

# Fallback: last number in the string
_LAST_NUMBER = re.compile(r"([+-]?[\d,]+\.?\d*)\s*$")


def extract_answer(text: str) -> Optional[float]:
    """Parse the final numeric answer from a model-generated (or gold) solution.

    Tries, in order:
      1. "#### <number>"
      2. "The answer is <number>"
      3. "\\boxed{<number>}"
      4. Last number in the string

    Returns None if no valid number is found.
    """
    for pattern in _ANSWER_PATTERNS:
        m = pattern.search(text)
        if m:
            return _parse_number(m.group(1))

    # Fallback: last number
    m = _LAST_NUMBER.search(text.strip())
    if m:
        return _parse_number(m.group(1))

    return None


def _parse_number(s: str) -> Optional[float]:
    """Convert a string like '1,234' or '-3.5' to float."""
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


def verifiable_reward(generated: str, gold_answer: float) -> int:
    """Return 1 if extracted answer matches gold, else 0."""
    extracted = extract_answer(generated)
    if extracted is None:
        return 0
    # Compare with tolerance for floating point
    return int(abs(extracted - gold_answer) < 1e-5)


# ---------------------------------------------------------------------------
# Gold answer extraction from GSM8K ground-truth
# ---------------------------------------------------------------------------

def extract_gold_answer(solution: str) -> float:
    """Extract the numeric answer from a GSM8K ground-truth solution string.

    GSM8K solutions always end with '#### <number>'.
    """
    m = re.search(r"####\s*([+-]?[\d,]+\.?\d*)", solution)
    if m is None:
        raise ValueError(f"Could not extract gold answer from: {solution!r}")
    return float(m.group(1).replace(",", ""))


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

GSM8K_PROMPT_TEMPLATE = (
    "Solve the following math problem step by step.\n"
    "At the end, write your final answer as a single number.\n\n"
    "Problem: {question}\nSolution:"
)


def format_gsm8k_prompt(question: str) -> str:
    return GSM8K_PROMPT_TEMPLATE.format(question=question)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class GSM8KDataset(Dataset):
    """GSM8K dataset for RLVR training or evaluation."""

    def __init__(self, split: str = "train"):
        ds = load_dataset("openai/gsm8k", "main", split=split)
        self.data = []
        for ex in ds:
            gold = extract_gold_answer(ex["answer"])
            self.data.append({
                "prompt": format_gsm8k_prompt(ex["question"]),
                "gold_answer": gold,
                "raw_question": ex["question"],
                "raw_solution": ex["answer"],
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> dict:
        return self.data[idx]
