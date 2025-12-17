"""
Centralized configuration defaults for niels-gpt.

This is the SINGLE SOURCE OF TRUTH for all model, training, and data defaults.
All other configs (JSON files, schemas) must inherit from these values.

Training target: "surly website concierge" assistant on M4 MacBook Air (16GB)
Expected pipeline duration: ~8 hours (17k pretrain + 6k sft steps)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


# ----------------------------- Model Defaults ----------------------------- #


@dataclass(frozen=True)
class ModelDefaults:
    """Model architecture defaults (v1 configuration)."""

    # V (vocab size) is REQUIRED - no default since it depends on tokenizer
    T: int = 512  # sequence length / context window
    C: int = 384  # embedding/channel dimension
    L: int = 8  # number of transformer layers
    H: int = 6  # number of attention heads
    d_ff: int = 1152  # feed-forward hidden dimension (3×C)
    dropout: float = 0.1  # dropout rate
    rope_theta: float = 10000.0  # RoPE positional encoding base


# ----------------------------- Training Defaults ----------------------------- #


@dataclass(frozen=True)
class TrainDefaults:
    """Base training hyperparameters (common across pretrain/sft)."""

    seed: int = 42
    B: int = 16  # micro batch size
    eval_every: int = 200  # evaluation cadence (steps)
    eval_steps: int = 50  # number of batches per evaluation
    log_every: int = 50  # logging cadence (steps)
    ckpt_every: int = 1000  # checkpoint save cadence (steps)
    grad_clip: float = 1.0  # gradient clipping value
    accum_steps: int = 1  # gradient accumulation steps

    # AMP (automatic mixed precision)
    amp: bool = True  # enable AMP by default on MPS
    amp_dtype: Literal["fp16", "bf16"] = "fp16"
    activation_checkpointing: bool = False  # memory optimization (off by default)


@dataclass(frozen=True)
class PretrainDefaults:
    """Pretrain-specific hyperparameters."""

    total_steps: int = 17000  # ~6 hours on M4 Air
    base_lr: float = 3e-4  # peak learning rate
    warmup_steps: int = 340  # 2% of total_steps
    min_lr: float = 3e-5  # minimum LR (cosine decay floor)

    # Optimizer
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.1

    # Dataset mixing (pretrain sources)
    # These must sum to 1.0
    wiki_weight: float = 0.72  # 80% of non-primer
    roam_weight: float = 0.18  # 20% of non-primer
    primer_weight: float = 0.10  # 10% synthetic primer data


@dataclass(frozen=True)
class SFTDefaults:
    """SFT-specific hyperparameters."""

    total_steps: int = 6000  # ~2 hours on M4 Air
    base_lr: float = 1e-4  # lower LR for fine-tuning
    warmup_steps: int = 120  # 2% of total_steps
    min_lr: float = 1e-5  # minimum LR (cosine decay floor)

    # Optimizer
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.05  # lighter regularization for SFT

    # SFT-specific settings
    assistant_only_loss: bool = True  # mask out system+user tokens
    pack_sequences: bool = True  # pack multiple conversations per window

    # Dataset mixing (SFT sources)
    # These must sum to 1.0
    oasst1_weight: float = 0.60  # Open Assistant dataset
    dolly15k_weight: float = 0.20  # Databricks Dolly
    primer_weight: float = 0.20  # Synthetic primer conversations


# ----------------------------- Data Defaults ----------------------------- #


@dataclass(frozen=True)
class DataDefaults:
    """Data loading and caching defaults."""

    # Cache directories (relative to REPO_ROOT)
    pretrain_cache_dir: str = "data/cache/streams"
    sft_cache_dir: str = "data/cache/sft"

    # Validation settings
    pretrain_val_source: str = "wiki"
    sft_val_source: str = "wiki"  # can be "wiki" or "sft"

    # SFT data handling
    allow_missing_idx: bool = False  # auto-generate missing .idx.npy files


# ----------------------------- Eval & Checkpointing Defaults ----------------------------- #


@dataclass(frozen=True)
class EvalDefaults:
    """Evaluation and checkpointing defaults."""

    save_best: bool = True  # save best checkpoint based on val loss
    keep_periodic: bool = True  # keep periodic checkpoints (every ckpt_every)


# ----------------------------- Profile Presets ----------------------------- #


@dataclass(frozen=True)
class PretrainProfile:
    """Complete pretrain configuration profile."""

    # Model
    model: ModelDefaults = ModelDefaults()

    # Training
    total_steps: int = PretrainDefaults.total_steps
    base_lr: float = PretrainDefaults.base_lr
    warmup_steps: int = PretrainDefaults.warmup_steps
    min_lr: float = PretrainDefaults.min_lr
    beta1: float = PretrainDefaults.beta1
    beta2: float = PretrainDefaults.beta2
    weight_decay: float = PretrainDefaults.weight_decay

    # Common training settings
    seed: int = TrainDefaults.seed
    B: int = TrainDefaults.B
    eval_every: int = TrainDefaults.eval_every
    eval_steps: int = TrainDefaults.eval_steps
    log_every: int = TrainDefaults.log_every
    ckpt_every: int = TrainDefaults.ckpt_every
    grad_clip: float = TrainDefaults.grad_clip
    accum_steps: int = TrainDefaults.accum_steps
    amp: bool = TrainDefaults.amp
    amp_dtype: Literal["fp16", "bf16"] = TrainDefaults.amp_dtype
    activation_checkpointing: bool = TrainDefaults.activation_checkpointing

    # Data
    wiki_weight: float = PretrainDefaults.wiki_weight
    roam_weight: float = PretrainDefaults.roam_weight
    primer_weight: float = PretrainDefaults.primer_weight
    val_source: str = DataDefaults.pretrain_val_source
    cache_dir: str = DataDefaults.pretrain_cache_dir


@dataclass(frozen=True)
class SFTProfile:
    """Complete SFT configuration profile."""

    # Model (inherits from pretrain)
    model: ModelDefaults = ModelDefaults()

    # Training
    total_steps: int = SFTDefaults.total_steps
    base_lr: float = SFTDefaults.base_lr
    warmup_steps: int = SFTDefaults.warmup_steps
    min_lr: float = SFTDefaults.min_lr
    beta1: float = SFTDefaults.beta1
    beta2: float = SFTDefaults.beta2
    weight_decay: float = SFTDefaults.weight_decay

    # Common training settings
    seed: int = TrainDefaults.seed
    B: int = TrainDefaults.B
    eval_every: int = TrainDefaults.eval_every
    eval_steps: int = TrainDefaults.eval_steps
    log_every: int = TrainDefaults.log_every
    ckpt_every: int = TrainDefaults.ckpt_every
    grad_clip: float = TrainDefaults.grad_clip
    accum_steps: int = TrainDefaults.accum_steps
    amp: bool = TrainDefaults.amp
    amp_dtype: Literal["fp16", "bf16"] = TrainDefaults.amp_dtype
    activation_checkpointing: bool = TrainDefaults.activation_checkpointing

    # SFT-specific
    assistant_only_loss: bool = SFTDefaults.assistant_only_loss
    pack_sequences: bool = SFTDefaults.pack_sequences

    # Data
    oasst1_weight: float = SFTDefaults.oasst1_weight
    dolly15k_weight: float = SFTDefaults.dolly15k_weight
    primer_weight: float = SFTDefaults.primer_weight
    val_source: str = DataDefaults.sft_val_source
    cache_dir: str = DataDefaults.sft_cache_dir
    streams_cache_dir: str = DataDefaults.pretrain_cache_dir
    allow_missing_idx: bool = DataDefaults.allow_missing_idx


# ----------------------------- Profile Constants ----------------------------- #

# These are the named profiles accessible via CLI --profile flag
DEFAULT_PRETRAIN = PretrainProfile()
DEFAULT_SFT = SFTProfile()


# ----------------------------- Helper Functions ----------------------------- #


def get_pretrain_sources_dict() -> dict[str, float]:
    """Get default pretrain dataset mixing dictionary."""
    return {
        "wiki": PretrainDefaults.wiki_weight,
        "roam": PretrainDefaults.roam_weight,
        "primer": PretrainDefaults.primer_weight,
    }


def get_sft_sources_dict() -> dict[str, float]:
    """Get default SFT dataset mixing dictionary."""
    return {
        "oasst1": SFTDefaults.oasst1_weight,
        "dolly": SFTDefaults.dolly15k_weight,
        "primer": SFTDefaults.primer_weight,
    }
