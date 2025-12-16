"""Checkpoint save/load utilities with config validation."""

import torch

from niels_gpt.model.gpt import GPT


def save_checkpoint(
    path: str,
    *,
    model_cfg: dict,
    train_cfg: dict,
    model: GPT,
    optimizer: torch.optim.Optimizer | None,
    step: int,
    best_val_loss: float | None,
) -> None:
    """
    Save checkpoint to disk.

    Args:
        path: Path to save checkpoint (.pt file)
        model_cfg: Model configuration dict (for architecture reconstruction)
        train_cfg: Training configuration dict (for resuming training)
        model: GPT model instance
        optimizer: Optimizer instance (or None to skip saving optimizer state)
        step: Current training step
        best_val_loss: Best validation loss so far (or None if not available)
    """
    checkpoint = {
        "model_cfg": model_cfg,
        "train_cfg": train_cfg,
        "model_state": model.state_dict(),
        "step": step,
    }

    if optimizer is not None:
        checkpoint["optimizer_state"] = optimizer.state_dict()

    if best_val_loss is not None:
        checkpoint["best_val_loss"] = best_val_loss

    torch.save(checkpoint, path)


def load_checkpoint(path: str, *, device: str) -> dict:
    """
    Load checkpoint from disk.

    Args:
        path: Path to checkpoint file
        device: Device to map tensors to

    Returns:
        Dict with keys:
            - model_cfg: dict
            - train_cfg: dict
            - model_state: state_dict
            - optimizer_state: state_dict or None
            - step: int
            - best_val_loss: float or None
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    return {
        "model_cfg": checkpoint["model_cfg"],
        "train_cfg": checkpoint["train_cfg"],
        "model_state": checkpoint["model_state"],
        "optimizer_state": checkpoint.get("optimizer_state", None),
        "step": checkpoint["step"],
        "best_val_loss": checkpoint.get("best_val_loss", None),
    }


def validate_model_config_match(loaded_cfg: dict, current_cfg: dict) -> None:
    """
    Validate that loaded model config matches current config for shape-critical fields.

    Args:
        loaded_cfg: Model config from checkpoint
        current_cfg: Current model config

    Raises:
        ValueError: If shape-critical fields don't match, with a clear diff message
    """
    # Shape-critical fields that must match
    shape_fields = ["V", "T", "C", "L", "H", "d_ff", "dropout", "rope_theta"]

    mismatches = []
    for field in shape_fields:
        loaded_val = loaded_cfg.get(field)
        current_val = current_cfg.get(field)
        if loaded_val != current_val:
            mismatches.append(f"  {field}: loaded={loaded_val}, current={current_val}")

    if mismatches:
        raise ValueError(
            "Model config mismatch (shape-critical fields differ):\n"
            + "\n".join(mismatches)
            + "\n\nCannot resume training with incompatible model architecture."
        )
