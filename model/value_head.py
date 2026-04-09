"""
Value model for PPO: Llama-3.2-1B backbone + scalar head.

The value backbone is architecturally separate from the policy to avoid
gradient interference.  The scalar head is a single nn.Linear(d_model, 1)
initialized with near-zero weights.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

from config import ModelConfig, LoRAConfig
from model.loader import _get_bnb_config, apply_lora


class ValueModel(nn.Module):
    """Llama backbone + linear scalar head for PPO critic."""

    def __init__(self, backbone, hidden_size: int):
        super().__init__()
        self.backbone = backbone
        self.value_head = nn.Linear(hidden_size, 1, bias=False)
        # Near-zero init to avoid corrupting early training
        nn.init.normal_(self.value_head.weight, std=0.01)

    def forward(self, input_ids, attention_mask):
        """Return per-token scalar values.

        Returns: (B, T) tensor of value estimates.
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden = outputs.hidden_states[-1]        # (B, T, d_model)
        # Cast to match value_head dtype (avoids bf16 vs f32 mismatch)
        values = self.value_head(hidden.to(self.value_head.weight.dtype)).squeeze(-1)
        return values


def load_value_model(
    model_cfg: ModelConfig,
    lora_cfg: LoRAConfig | None = None,
    freeze_backbone: bool = False,
) -> tuple:
    """Load a ValueModel from the Llama backbone.

    Args:
        model_cfg: Model configuration.
        lora_cfg: If provided, apply LoRA to the backbone.
        freeze_backbone: If True, freeze the backbone and only train the head.

    Returns: (value_model, tokenizer)
    """
    dtype = getattr(torch, model_cfg.torch_dtype)
    bnb = _get_bnb_config(model_cfg)

    backbone = AutoModel.from_pretrained(
        model_cfg.value_model_name,
        torch_dtype=dtype,
        quantization_config=bnb,
        device_map="auto" if bnb else None,
        resume_download=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_cfg.value_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    hidden_size = backbone.config.hidden_size

    if freeze_backbone:
        for p in backbone.parameters():
            p.requires_grad = False
    elif lora_cfg is not None:
        backbone = apply_lora(backbone, lora_cfg, task_type="CAUSAL_LM")

    value_model = ValueModel(backbone, hidden_size)

    # Cast value head to match backbone dtype (avoids bf16 vs f32 mismatch)
    value_model.value_head = value_model.value_head.to(dtype)

    if bnb is None:
        value_model = value_model.to(model_cfg.device)

    return value_model, tokenizer
