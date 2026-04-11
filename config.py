"""
Centralized configuration for all PA2 tasks.
Every hyperparameter lives here — no magic numbers scattered in code.
"""
from dataclasses import dataclass, field
from typing import Literal, Optional


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    policy_model_name: str = "/kaggle/input/models/ahmadjawwad903/smollm2-360-base/transformers/default/1/model2files"
    rm_model_name: str = "/kaggle/input/models/qwen-lm/qwen2.5/transformers/0.5b-instruct/1"
    value_model_name: str = "/kaggle/input/models/qwen-lm/qwen2.5/transformers/0.5b-instruct/1"
    torch_dtype: str = "bfloat16"
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    device: str = "cuda"


# ---------------------------------------------------------------------------
# LoRA configuration
# ---------------------------------------------------------------------------
@dataclass
class LoRAConfig:
    r: int = 8
    lora_alpha: int = 16
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


# ---------------------------------------------------------------------------
# Data / tokenization
# ---------------------------------------------------------------------------
@dataclass
class DataConfig:
    dataset_name: str = "Anthropic/hh-rlhf"
    dataset_subset: str = "harmless-base"
    gsm8k_name: str = "openai/gsm8k"
    gsm8k_config: str = "main"
    max_seq_length: int = 1024
    max_prompt_length: int = 512
    max_new_tokens: int = 128
    max_new_tokens_rlvr: int = 256
    num_workers: int = 0


# ---------------------------------------------------------------------------
# SFT
# ---------------------------------------------------------------------------
@dataclass
class SFTConfig:
    epochs: int = 1
    batch_size: int = 8
    grad_accum_steps: int = 4
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    log_every: int = 10
    eval_every: int = 100
    max_train_samples: Optional[int] = None  # None = use all


# ---------------------------------------------------------------------------
# Reward model training
# ---------------------------------------------------------------------------
@dataclass
class RMConfig:
    epochs: int = 1
    batch_size: int = 8
    lr: float = 1e-3
    weight_decay: float = 0.01
    warmup_steps: int = 50
    lambda_reg: float = 1e-3
    log_every: int = 10
    target_accuracy: float = 0.60
    max_train_samples: Optional[int] = 8000  # Head-only: 2k params → 1000 steps is plenty


# ---------------------------------------------------------------------------
# PPO
# ---------------------------------------------------------------------------
@dataclass
class PPOConfig:
    num_steps: int = 200
    prompts_per_step: int = 8
    ppo_epochs: int = 4
    beta: float = 0.1          # KL penalty coefficient
    epsilon: float = 0.2       # clipping parameter
    gamma: float = 1.0         # discount (no discounting within response)
    lam: float = 0.95          # GAE lambda
    lr_policy: float = 1e-5
    lr_value: float = 1e-5
    max_grad_norm: float = 1.0
    max_new_tokens: int = 128  # max response length during rollout
    temperature: float = 0.7
    top_p: float = 0.9
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    kl_coef: float = 0.1
    eval_every: int = 25
    eval_prompts: int = 200
    log_every: int = 1


# ---------------------------------------------------------------------------
# DPO
# ---------------------------------------------------------------------------
@dataclass
class DPOConfig:
    epochs: int = 1
    batch_size: int = 2
    grad_accum_steps: int = 16
    beta: float = 0.1
    lr: float = 1e-5
    weight_decay: float = 0.01
    warmup_steps: int = 50
    eval_every: int = 25
    eval_prompts: int = 200
    log_every: int = 1
    max_train_samples: Optional[int] = None


# ---------------------------------------------------------------------------
# GRPO
# ---------------------------------------------------------------------------
@dataclass
class GRPOConfig:
    num_steps: int = 200
    prompts_per_step: int = 8
    K: int = 4                 # completions per prompt
    beta: float = 0.1
    epsilon: float = 0.2
    lr: float = 1e-5
    max_grad_norm: float = 1.0
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    kl_mode: Literal["full", "mc"] = "mc"
    eval_every: int = 25
    eval_prompts: int = 200
    log_every: int = 1


# ---------------------------------------------------------------------------
# RLVR
# ---------------------------------------------------------------------------
@dataclass
class RLVRConfig:
    num_steps: int = 300
    prompts_per_step: int = 8
    K: int = 4
    beta: float = 0.05
    epsilon: float = 0.2
    lr: float = 1e-5
    max_grad_norm: float = 1.0
    temperature: float = 0.7
    top_p: float = 0.9
    kl_mode: Literal["full", "mc"] = "mc"
    eval_every: int = 25
    log_every: int = 1


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@dataclass
class EvalConfig:
    num_prompts: int = 200
    temperature: float = 0.0   # greedy for eval
    max_new_tokens: int = 128
    sample_table_size: int = 5


# ---------------------------------------------------------------------------
# Top-level config aggregator
# ---------------------------------------------------------------------------
@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    data: DataConfig = field(default_factory=DataConfig)
    sft: SFTConfig = field(default_factory=SFTConfig)
    rm: RMConfig = field(default_factory=RMConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    dpo: DPOConfig = field(default_factory=DPOConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    rlvr: RLVRConfig = field(default_factory=RLVRConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    seed: int = 42
    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"
