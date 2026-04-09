from peft import LoraConfig, TaskType, get_peft_model


def make_lora_config(rank: int, alpha: int, dropout: float, target_modules: list[str]) -> LoraConfig:
    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )


def apply_lora(model, rank: int, alpha: int, dropout: float, target_modules: list[str]):
    config = make_lora_config(rank, alpha, dropout, target_modules)
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
