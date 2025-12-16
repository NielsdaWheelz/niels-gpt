"""Learning rate schedule: linear warmup + cosine decay."""

import math


def lr_at_step(
    step: int,
    total_steps: int,
    *,
    base_lr: float,
    warmup_steps: int,
    min_lr: float,
) -> float:
    """
    Compute learning rate at a given step using warmup + cosine decay schedule.

    Args:
        step: Current step in [0, total_steps-1]
        total_steps: Total training steps
        base_lr: Peak learning rate (reached at end of warmup)
        warmup_steps: Number of warmup steps (linear warmup from 0 to base_lr)
        min_lr: Minimum learning rate (reached at end of cosine decay)

    Returns:
        Learning rate at given step, clamped to [min_lr, base_lr]

    Schedule:
        - step < warmup_steps: linear warmup from 0 -> base_lr
        - step >= warmup_steps: cosine decay from base_lr -> min_lr
    """
    # Warmup phase: linear increase from 0 to base_lr
    if step < warmup_steps:
        return base_lr * step / warmup_steps

    # Cosine decay phase
    # Compute progress through decay phase
    decay_steps = total_steps - warmup_steps
    decay_progress = (step - warmup_steps) / decay_steps

    # Cosine decay from 1.0 to 0.0
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * decay_progress))

    # Scale from base_lr to min_lr
    lr = min_lr + (base_lr - min_lr) * cosine_decay

    # Clamp to valid range
    return max(min_lr, min(base_lr, lr))
