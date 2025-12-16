"""Evaluation utilities for computing loss on validation streams."""

import torch
import torch.nn.functional as F

from niels_gpt.batching import get_batch
from niels_gpt.model.gpt import GPT


@torch.no_grad()
def eval_loss_on_stream(
    model: GPT,
    *,
    stream: bytes,
    B: int,
    T: int,
    device: str,
    eval_steps: int,
    seed: int,
) -> float:
    """
    Compute average cross-entropy loss on a single validation stream.

    Args:
        model: GPT model to evaluate
        stream: Byte stream to evaluate on
        B: Batch size
        T: Context length
        device: Device to run evaluation on
        eval_steps: Number of batches to evaluate
        seed: Random seed for deterministic batch sampling

    Returns:
        Average cross-entropy loss over eval_steps batches

    Note:
        - Model mode is temporarily set to eval during evaluation, then restored
        - Mode is restored even if evaluation raises an exception (via try/finally)
        - Uses a local torch.Generator seeded with seed for deterministic sampling
        - Generator is created on CPU device (as required by torch.randint/multinomial)
    """
    # Save original model mode and set to eval
    was_training = model.training
    model.eval()

    try:
        # Create deterministic generator on CPU (required by torch random ops)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)

        # Wrap stream in dict for get_batch interface
        sources = {"eval": stream}
        p = {"eval": 1.0}

        total_loss = 0.0

        for _ in range(eval_steps):
            # Sample batch deterministically
            x, y = get_batch(sources, p=p, B=B, T=T, device=device, generator=generator)

            # Forward pass
            logits = model(x)  # (B, T, V)

            # Compute cross-entropy loss
            # Flatten to (B*T, V) and (B*T,)
            B_cur, T_cur, V = logits.shape
            logits_flat = logits.view(B_cur * T_cur, V)
            targets_flat = y.view(B_cur * T_cur)

            loss = F.cross_entropy(logits_flat, targets_flat)
            total_loss += loss.item()

        avg_loss = total_loss / eval_steps
        return avg_loss
    finally:
        # Restore original model mode even if an exception occurred
        if was_training:
            model.train()
