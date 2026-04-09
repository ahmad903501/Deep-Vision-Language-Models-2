from dataclasses import dataclass, field
from typing import List


@dataclass
class LoRAConfig:
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])


@dataclass
class ModelConfig:
    policy_model_name: str = "HuggingFaceTB/SmolLM2-360M"
    llama_backbone_name: str = "meta-llama/Llama-3.2-1B-Instruct"
    torch_dtype: str = "bfloat16"
    use_4bit: bool = False
    use_8bit: bool = False
    trust_remote_code: bool = True
    use_gradient_checkpointing: bool = True


@dataclass
class DataConfig:
    hh_dataset_name: str = "Anthropic/hh-rlhf"
    hh_harmless_config: str = "harmless-base"
    max_seq_len: int = 1024
    sft_batch_size: int = 8
    rm_batch_size: int = 8
    dpo_batch_size: int = 8


@dataclass
class ExperimentConfig:
    seed: int = 42
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    data: DataConfig = field(default_factory=DataConfig)
