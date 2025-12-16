"""Training entrypoint: train GPT with AdamW + warmup+cosine LR."""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.nn.utils as nn_utils

from niels_gpt.checkpoint import (
    load_checkpoint,
    save_checkpoint,
    validate_model_config_match,
)
from niels_gpt.config import load_config_from_json, to_dict
from niels_gpt.device import get_device
from niels_gpt.eval import eval_loss_on_stream
from niels_gpt.lr_schedule import lr_at_step
from niels_gpt.model.gpt import GPT
from niels_gpt.paths import CHECKPOINT_DIR, ensure_dirs
from niels_gpt.batching import get_batch
from niels_gpt.streams import StreamBuildConfig, build_sources


def main() -> None:
    """Main training entrypoint."""
    parser = argparse.ArgumentParser(description="Train GPT model")
    parser.add_argument("--config", required=True, help="Path to config JSON file")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--device", default=None, help="Device to train on (default: auto-detect)")
    args = parser.parse_args()

    # Ensure output directories exist
    ensure_dirs()

    # Load config
    model_cfg, train_cfg = load_config_from_json(args.config)

    # Select device
    device = args.device if args.device else get_device()
    print(f"Using device: {device}")

    # Set seed for reproducibility
    torch.manual_seed(train_cfg.seed)

    # Build data sources
    print("Building data sources...")
    stream_cfg = StreamBuildConfig(
        seed=train_cfg.seed,
        allow_missing_sources=True,
        required_sources=("wiki",),  # Only wiki is strictly required
    )
    train_sources, val_sources = build_sources(stream_cfg)

    # Filter sources to only those available
    p_train_filtered = {}
    missing_sources = []
    for source, prob in train_cfg.p_train.items():
        if source in train_sources:
            p_train_filtered[source] = prob
        else:
            missing_sources.append(source)

    # Warn about missing sources
    if missing_sources:
        print(f"WARNING: Missing training sources (will renormalize): {missing_sources}")
        print(f"  Original p_train: {train_cfg.p_train}")

    # Renormalize probabilities if some sources are missing
    total_prob = sum(p_train_filtered.values())
    if total_prob > 0:
        p_train_filtered = {k: v / total_prob for k, v in p_train_filtered.items()}
    else:
        raise RuntimeError("No training sources available")

    # Ensure wiki is present
    if "wiki" not in train_sources:
        raise RuntimeError("Wiki source is required for training but was not built successfully")

    print(f"Training sources: {list(train_sources.keys())}")
    print(f"Effective p_train: {p_train_filtered}")
    print(f"Validation sources: {list(val_sources.keys())}")

    # Initialize or resume model
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        ckpt = load_checkpoint(args.resume, device=device)

        # Validate config compatibility
        validate_model_config_match(ckpt["model_cfg"], to_dict(model_cfg))

        # Reconstruct model
        model = GPT(model_cfg).to(device)
        model.load_state_dict(ckpt["model_state"])

        # Reconstruct optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_cfg.base_lr,  # Will be overridden by scheduler
            betas=(0.9, 0.95),
            weight_decay=0.1,
        )
        if ckpt["optimizer_state"] is not None:
            optimizer.load_state_dict(ckpt["optimizer_state"])

        start_step = ckpt["step"]
        best_val_loss = ckpt["best_val_loss"]

        print(f"Resumed from step {start_step}")
        if best_val_loss is not None:
            print(f"Best val loss so far: {best_val_loss:.4f}")
    else:
        print("Initializing new model...")
        model = GPT(model_cfg).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_cfg.base_lr,  # Will be overridden by scheduler
            betas=(0.9, 0.95),
            weight_decay=0.1,
        )

        start_step = 0
        best_val_loss = None

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Training loop
    print(f"\nStarting training for {train_cfg.total_steps} steps...")
    print(f"Eval every {train_cfg.eval_every} steps, log every {train_cfg.log_every} steps")

    # Create training generator for deterministic batch sampling
    train_gen = torch.Generator(device="cpu")
    train_gen.manual_seed(train_cfg.seed)

    model.train()

    for step in range(start_step, train_cfg.total_steps):
        # Set learning rate for this step
        lr = lr_at_step(
            step,
            train_cfg.total_steps,
            base_lr=train_cfg.base_lr,
            warmup_steps=train_cfg.warmup_steps,
            min_lr=train_cfg.min_lr,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Sample batch deterministically
        x, y = get_batch(
            train_sources,
            p=p_train_filtered,
            B=train_cfg.B,
            T=model_cfg.T,
            device=device,
            generator=train_gen,
        )

        # Forward pass
        logits = model(x)  # (B, T, V)

        # Compute loss
        B_cur, T_cur, V = logits.shape
        logits_flat = logits.view(B_cur * T_cur, V)
        targets_flat = y.view(B_cur * T_cur)
        loss = F.cross_entropy(logits_flat, targets_flat)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        nn_utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)

        # Optimizer step
        optimizer.step()

        # Logging
        if (step + 1) % train_cfg.log_every == 0:
            print(f"Step {step + 1}/{train_cfg.total_steps} | loss: {loss.item():.4f} | lr: {lr:.6f}")

        # Evaluation
        if (step + 1) % train_cfg.eval_every == 0:
            if "wiki" in val_sources:
                val_loss = eval_loss_on_stream(
                    model,
                    stream=val_sources["wiki"],
                    B=train_cfg.B,
                    T=model_cfg.T,
                    device=device,
                    eval_steps=train_cfg.eval_steps,
                    seed=train_cfg.seed,
                )
                print(f"  val_wiki_loss: {val_loss:.4f}")

                # Track best validation loss
                if best_val_loss is None or val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_path = CHECKPOINT_DIR / "best.pt"
                    save_checkpoint(
                        str(best_path),
                        model_cfg=to_dict(model_cfg),
                        train_cfg=to_dict(train_cfg),
                        model=model,
                        optimizer=optimizer,
                        step=step + 1,
                        best_val_loss=best_val_loss,
                    )
                    print(f"  Saved best checkpoint (val_loss={best_val_loss:.4f})")

        # Checkpointing
        if (step + 1) % train_cfg.ckpt_every == 0:
            # Save periodic checkpoint
            periodic_path = CHECKPOINT_DIR / f"step_{step + 1:07d}.pt"
            save_checkpoint(
                str(periodic_path),
                model_cfg=to_dict(model_cfg),
                train_cfg=to_dict(train_cfg),
                model=model,
                optimizer=optimizer,
                step=step + 1,
                best_val_loss=best_val_loss,
            )
            print(f"  Saved periodic checkpoint: {periodic_path.name}")

            # Save/overwrite latest checkpoint
            latest_path = CHECKPOINT_DIR / "latest.pt"
            save_checkpoint(
                str(latest_path),
                model_cfg=to_dict(model_cfg),
                train_cfg=to_dict(train_cfg),
                model=model,
                optimizer=optimizer,
                step=step + 1,
                best_val_loss=best_val_loss,
            )

    # Save final checkpoint
    print("\nTraining complete! Saving final checkpoint...")
    final_path = CHECKPOINT_DIR / "latest.pt"
    save_checkpoint(
        str(final_path),
        model_cfg=to_dict(model_cfg),
        train_cfg=to_dict(train_cfg),
        model=model,
        optimizer=optimizer,
        step=train_cfg.total_steps,
        best_val_loss=best_val_loss,
    )
    print(f"Final checkpoint saved: {final_path}")

    if best_val_loss is not None:
        print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
