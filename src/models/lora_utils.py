from peft import LoraConfig, TaskType, get_peft_model


def _resolve_task_type(task_type: str) -> TaskType:
    mapping = {
        "CAUSAL_LM": TaskType.CAUSAL_LM,
        "SEQ_CLS": TaskType.SEQ_CLS,
    }
    if task_type not in mapping:
        raise ValueError(f"Unsupported LoRA task type '{task_type}'.")
    return mapping[task_type]


def make_lora_config(
    rank: int,
    alpha: int,
    dropout: float,
    target_modules: list[str],
    task_type: str = "CAUSAL_LM",
) -> LoraConfig:
    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        task_type=_resolve_task_type(task_type),
        bias="none",
    )


def apply_lora(
    model,
    rank: int,
    alpha: int,
    dropout: float,
    target_modules: list[str],
    task_type: str = "CAUSAL_LM",
):
    config = make_lora_config(rank, alpha, dropout, target_modules, task_type=task_type)
    lora_model = get_peft_model(model, config)
    return lora_model


def disable_adapter_layers_if_available(model) -> None:
    if hasattr(model, "disable_adapter_layers"):
        model.disable_adapter_layers()
    elif hasattr(model, "disable_adapters"):
        model.disable_adapters()


def freeze_model(model) -> None:
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
