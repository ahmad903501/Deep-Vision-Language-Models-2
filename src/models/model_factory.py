from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config.schema import ExperimentConfig
from src.models.lora_utils import apply_lora, freeze_model


DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


@dataclass
class ModelBundle:
    policy_model: torch.nn.Module
    reference_model: torch.nn.Module
    llama_backbone_model: torch.nn.Module
    policy_tokenizer: object
    rm_tokenizer: object


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name not in DTYPE_MAP:
        raise ValueError(f"Unsupported dtype '{dtype_name}'.")
    return DTYPE_MAP[dtype_name]


def _make_tokenizer(model_name: str, padding_side: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.padding_side = padding_side
    if tokenizer.eos_token is None:
        raise ValueError(f"Tokenizer for {model_name} has no eos_token.")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_causallm(model_name: str, dtype: torch.dtype, trust_remote_code: bool):
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=trust_remote_code,
    )


def _enable_checkpointing_for_peft(model) -> None:
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()


def create_model_bundle(config: ExperimentConfig) -> ModelBundle:
    dtype = _resolve_dtype(config.model.torch_dtype)

    policy_tokenizer = _make_tokenizer(config.model.policy_model_name, padding_side="left")
    rm_tokenizer = _make_tokenizer(config.model.llama_backbone_name, padding_side="right")

    policy_base = _load_causallm(
        config.model.policy_model_name,
        dtype=dtype,
        trust_remote_code=config.model.trust_remote_code,
    )
    policy_model = apply_lora(
        policy_base,
        rank=config.lora.rank,
        alpha=config.lora.alpha,
        dropout=config.lora.dropout,
        target_modules=config.lora.target_modules,
    )

    if config.model.use_gradient_checkpointing:
        _enable_checkpointing_for_peft(policy_model)

    reference_model = _load_causallm(
        config.model.policy_model_name,
        dtype=dtype,
        trust_remote_code=config.model.trust_remote_code,
    )
    freeze_model(reference_model)

    llama_backbone_model = _load_causallm(
        config.model.llama_backbone_name,
        dtype=dtype,
        trust_remote_code=config.model.trust_remote_code,
    )

    return ModelBundle(
        policy_model=policy_model,
        reference_model=reference_model,
        llama_backbone_model=llama_backbone_model,
        policy_tokenizer=policy_tokenizer,
        rm_tokenizer=rm_tokenizer,
    )


def parameter_counts(model) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def memory_allocated_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / (1024**2)
