"""Tests for learning rate schedule."""

import pytest

from niels_gpt.lr_schedule import lr_at_step


def test_lr_at_step_warmup_start():
    """LR should be 0 (or extremely close) at step 0."""
    lr = lr_at_step(
        step=0,
        total_steps=1000,
        base_lr=1e-3,
        warmup_steps=100,
        min_lr=1e-4,
    )
    assert abs(lr - 0.0) < 1e-9, f"Expected lr ~0.0 at step 0, got {lr}"


def test_lr_at_step_warmup_end():
    """LR should equal base_lr at end of warmup."""
    base_lr = 1e-3
    warmup_steps = 100

    lr = lr_at_step(
        step=warmup_steps,
        total_steps=1000,
        base_lr=base_lr,
        warmup_steps=warmup_steps,
        min_lr=1e-4,
    )
    assert abs(lr - base_lr) < 1e-9, f"Expected lr ~{base_lr} at step {warmup_steps}, got {lr}"


def test_lr_at_step_decay_end():
    """LR should be close to min_lr at final step."""
    min_lr = 1e-4
    total_steps = 1000

    lr = lr_at_step(
        step=total_steps - 1,
        total_steps=total_steps,
        base_lr=1e-3,
        warmup_steps=100,
        min_lr=min_lr,
    )
    # Allow some tolerance for cosine decay approximation
    assert abs(lr - min_lr) < 1e-6, f"Expected lr ~{min_lr} at final step, got {lr}"


def test_lr_warmup_monotonic_increasing():
    """LR should be monotonically non-decreasing during warmup."""
    total_steps = 1000
    base_lr = 1e-3
    warmup_steps = 100
    min_lr = 1e-4

    prev_lr = 0.0
    for step in range(warmup_steps + 1):
        lr = lr_at_step(
            step=step,
            total_steps=total_steps,
            base_lr=base_lr,
            warmup_steps=warmup_steps,
            min_lr=min_lr,
        )
        assert lr >= prev_lr, f"LR decreased during warmup at step {step}: {prev_lr} -> {lr}"
        prev_lr = lr


def test_lr_decay_monotonic_decreasing():
    """LR should be non-increasing after warmup (cosine decay)."""
    total_steps = 1000
    base_lr = 1e-3
    warmup_steps = 100
    min_lr = 1e-4

    prev_lr = base_lr
    for step in range(warmup_steps, total_steps):
        lr = lr_at_step(
            step=step,
            total_steps=total_steps,
            base_lr=base_lr,
            warmup_steps=warmup_steps,
            min_lr=min_lr,
        )
        assert lr <= prev_lr + 1e-9, f"LR increased during decay at step {step}: {prev_lr} -> {lr}"
        prev_lr = lr


def test_lr_clamping():
    """LR should be clamped to [min_lr, base_lr] range after warmup."""
    total_steps = 1000
    base_lr = 1e-3
    warmup_steps = 100
    min_lr = 1e-4

    # During warmup, LR can be below min_lr (starts at 0)
    for step in range(warmup_steps):
        lr = lr_at_step(
            step=step,
            total_steps=total_steps,
            base_lr=base_lr,
            warmup_steps=warmup_steps,
            min_lr=min_lr,
        )
        assert 0.0 <= lr <= base_lr, f"LR {lr} outside valid range [0, {base_lr}] at warmup step {step}"

    # After warmup, LR should be in [min_lr, base_lr]
    for step in range(warmup_steps, total_steps):
        lr = lr_at_step(
            step=step,
            total_steps=total_steps,
            base_lr=base_lr,
            warmup_steps=warmup_steps,
            min_lr=min_lr,
        )
        assert min_lr <= lr <= base_lr, f"LR {lr} outside valid range [{min_lr}, {base_lr}] at step {step}"


def test_lr_schedule_realistic_params():
    """Test with realistic training parameters."""
    total_steps = 20000
    base_lr = 3e-4
    warmup_steps = 200
    min_lr = 3e-5

    # Check key points
    lr_start = lr_at_step(0, total_steps, base_lr=base_lr, warmup_steps=warmup_steps, min_lr=min_lr)
    lr_warmup_end = lr_at_step(warmup_steps, total_steps, base_lr=base_lr, warmup_steps=warmup_steps, min_lr=min_lr)
    lr_mid = lr_at_step(total_steps // 2, total_steps, base_lr=base_lr, warmup_steps=warmup_steps, min_lr=min_lr)
    lr_end = lr_at_step(total_steps - 1, total_steps, base_lr=base_lr, warmup_steps=warmup_steps, min_lr=min_lr)

    assert abs(lr_start - 0.0) < 1e-9
    assert abs(lr_warmup_end - base_lr) < 1e-9
    assert min_lr <= lr_mid <= base_lr
    assert abs(lr_end - min_lr) < 1e-6
