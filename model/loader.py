"""
Model loading utilities: policy, reward backbone, LoRA, frozen reference.
"""
from __future__ import annotations

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType

from config import ModelConfig, LoRAConfig


# ---------------------------------------------------------------------------
# Quantization helper
# ---------------------------------------------------------------------------

def _get_bnb_config(model_cfg: ModelConfig) -> BitsAndBytesConfig | None:
    if model_cfg.load_in_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    if model_cfg.load_in_8bit:
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


# ---------------------------------------------------------------------------
# Policy model (CausalLM)
# ---------------------------------------------------------------------------

def load_policy_model(
    model_cfg: ModelConfig,
) -> tuple:
    """Load the policy CausalLM and its tokenizer.

    Returns (model, tokenizer).
    """
    dtype = getattr(torch, model_cfg.torch_dtype)
    bnb = _get_bnb_config(model_cfg)

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.policy_model_name,
        dtype=dtype,
        quantization_config=bnb,
        device_map="auto" if bnb else None,
    )
    if bnb is None:
        model = model.to(model_cfg.device)

    tokenizer = AutoTokenizer.from_pretrained(model_cfg.policy_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    return model, tokenizer


# ---------------------------------------------------------------------------
# Reward / Value backbone (SequenceClassification)
# ---------------------------------------------------------------------------

def load_rm_backbone(
    model_cfg: ModelConfig,
    num_labels: int = 1,
) -> tuple:
    """Load the RM backbone as a sequence classifier + tokenizer.

    Used for both the reward model (Task C1) and value model backbone (Task C3).
    The scalar head is the classification layer (d_model → 1).
    """
    dtype = getattr(torch, model_cfg.torch_dtype)
    bnb = _get_bnb_config(model_cfg)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_cfg.rm_model_name,
        num_labels=num_labels,
        dtype=dtype,
        quantization_config=bnb,
        device_map="auto" if bnb else None,
    )
    if bnb is None:
        model = model.to(model_cfg.device)

    tokenizer = AutoTokenizer.from_pretrained(model_cfg.rm_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    # RM uses right-padding by default; set explicitly
    tokenizer.padding_side = "right"

    # Ensure the model knows the pad token id
    model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------

def apply_lora(
    model,
    lora_cfg: LoRAConfig,
    task_type: str = "CAUSAL_LM",
):
    """Wrap a model with LoRA adapters and return the PeftModel.

    After this call:
      - model.enable_input_require_grads() is called (needed for gradient
        checkpointing + PEFT).
      - Only LoRA parameters are trainable.
    """
    peft_task = TaskType.CAUSAL_LM if task_type == "CAUSAL_LM" else TaskType.SEQ_CLS

    config = LoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        target_modules=lora_cfg.target_modules,
        lora_dropout=lora_cfg.lora_dropout,
        bias=lora_cfg.bias,
        task_type=peft_task,
    )
    model.enable_input_require_grads()
    peft_model = get_peft_model(model, config)
    peft_model.print_trainable_parameters()
    return peft_model


# ---------------------------------------------------------------------------
# Frozen reference model
# ---------------------------------------------------------------------------

def create_reference_model(peft_model):
    """Create a frozen reference by disabling LoRA adapters.

    The returned model shares the base weights but LoRA deltas are zeroed out,
    giving the behaviour of π_ref (pre-LoRA fine-tuning).

    Usage — context-manager approach (preferred, saves VRAM):
        with peft_model.disable_adapter():
            ref_logits = peft_model(...)   # acts as π_ref

    Usage — separate frozen copy (needed when policy trains in parallel):
        ref = create_reference_model(policy)
        ref_logits = ref(...)
    """
    import copy
    ref_model = copy.deepcopy(peft_model)
    # Disable LoRA permanently on the copy
    ref_model.disable_adapter_layers()
    # Freeze all parameters
    for param in ref_model.parameters():
        param.requires_grad = False
    ref_model.eval()
    return ref_model


# ---------------------------------------------------------------------------
# Diagnostic helpers
# ---------------------------------------------------------------------------

def print_model_info(model, label: str = "Model"):
    """Print parameter counts and GPU memory usage."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    print(f"\n{'=' * 50}")
    print(f"  {label}")
    print(f"  Total params:     {total:>12,}")
    print(f"  Trainable params: {trainable:>12,}")
    print(f"  Frozen params:    {frozen:>12,}")
    print(f"  Trainable %:      {100 * trainable / total:.2f}%")

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"  GPU mem allocated: {allocated:.2f} GB")
        print(f"  GPU mem reserved:  {reserved:.2f} GB")
    print(f"{'=' * 50}\n")
