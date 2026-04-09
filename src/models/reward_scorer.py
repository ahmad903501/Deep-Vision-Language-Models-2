from __future__ import annotations

import torch
from transformers import AutoModelForSequenceClassification

from src.config.schema import ExperimentConfig
from src.models.lora_utils import apply_lora


DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name not in DTYPE_MAP:
        raise ValueError(f"Unsupported dtype '{dtype_name}'.")
    dtype = DTYPE_MAP[dtype_name]
    if dtype == torch.bfloat16 and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        # P100/T4 class GPUs do not support bf16 efficiently; fp16 is lower-memory and faster there.
        return torch.float16
    return dtype


def _unwrap_sequence_classifier_model(model):
    current = model
    for _ in range(6):
        if hasattr(current, "score") or hasattr(current, "classifier"):
            return current

        if hasattr(current, "base_model"):
            nxt = getattr(current, "base_model")
            if nxt is not current:
                current = nxt
                continue

        if hasattr(current, "model"):
            nxt = getattr(current, "model")
            if nxt is not current:
                current = nxt
                continue

        break

    raise ValueError("Could not locate sequence-classification model with a known head.")


def _backbone_model(seq_cls_model):
    if hasattr(seq_cls_model, "model"):
        return seq_cls_model.model
    return None


def _classification_head(model):
    seq_cls_model = _unwrap_sequence_classifier_model(model)
    if hasattr(seq_cls_model, "score"):
        return seq_cls_model.score
    if hasattr(seq_cls_model, "classifier"):
        return seq_cls_model.classifier
    raise ValueError("Reward model has no known sequence-classification head ('score'/'classifier').")


def _set_head_only_trainable(model) -> None:
    for param in model.parameters():
        param.requires_grad = False

    head = _classification_head(model)
    for param in head.parameters():
        param.requires_grad = True


def configure_reward_trainable_params(model, adaptation_mode: str) -> None:
    mode = adaptation_mode.lower().strip()
    if mode == "head_only":
        _set_head_only_trainable(model)
    elif mode == "lora":
        # For LoRA path, get_peft_model controls trainable params.
        return
    else:
        raise ValueError(
            f"Unsupported RM adaptation_mode '{adaptation_mode}'. Use 'lora' or 'head_only'."
        )


def last_non_pad_indices(attention_mask: torch.Tensor) -> torch.Tensor:
    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must have shape [batch, seq_len].")
    # Clamp so fully padded rows do not produce negative indices.
    return torch.clamp(attention_mask.sum(dim=1) - 1, min=0)


def gather_last_token_hidden(last_hidden_state: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if last_hidden_state.ndim != 3:
        raise ValueError("last_hidden_state must have shape [batch, seq_len, hidden].")
    if indices.ndim != 1:
        raise ValueError("indices must have shape [batch].")

    batch_size = last_hidden_state.shape[0]
    row_ids = torch.arange(batch_size, device=last_hidden_state.device)
    return last_hidden_state[row_ids, indices]


def reward_scores_from_batch(
    rm_model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    provided_last_non_pad_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    seq_cls_model = _unwrap_sequence_classifier_model(rm_model)
    backbone = _backbone_model(seq_cls_model)

    if backbone is not None:
        # Lower-memory path: only request last_hidden_state from backbone.
        outputs = backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state
    else:
        # Fallback for tests/dummy models that expose only forward(..., output_hidden_states=True).
        outputs = seq_cls_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = outputs.hidden_states[-1]

    indices = provided_last_non_pad_idx
    if indices is None:
        indices = last_non_pad_indices(attention_mask)

    indices = indices.to(hidden.device)
    last_hidden = gather_last_token_hidden(hidden, indices)

    # We explicitly score the last non-pad token for right-padded batches.
    head = _classification_head(seq_cls_model)
    scores = head(last_hidden)
    if scores.ndim == 2 and scores.shape[1] == 1:
        scores = scores[:, 0]
    return scores


def load_reward_model(
    config: ExperimentConfig,
    rm_tokenizer,
):
    dtype = _resolve_dtype(config.model.torch_dtype)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model.llama_backbone_name,
        num_labels=1,
        torch_dtype=dtype,
        trust_remote_code=config.model.trust_remote_code,
    )

    model.config.pad_token_id = rm_tokenizer.pad_token_id
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    mode = config.rm_train.adaptation_mode.lower().strip()
    if mode == "lora":
        model = apply_lora(
            model,
            rank=config.lora.rank,
            alpha=config.lora.alpha,
            dropout=config.lora.dropout,
            target_modules=config.lora.target_modules,
            task_type="SEQ_CLS",
        )
    elif mode == "head_only":
        configure_reward_trainable_params(model, mode)
    else:
        raise ValueError(
            f"Unsupported RM adaptation_mode '{config.rm_train.adaptation_mode}'. "
            "Use 'lora' or 'head_only'."
        )

    if config.model.use_gradient_checkpointing:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    return model
