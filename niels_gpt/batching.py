"""Batching: sample mixed-source batches deterministically."""

from __future__ import annotations

from typing import Dict, Tuple

import torch

BytesSources = Dict[str, bytes]


def get_batch(
    sources: BytesSources,
    *,
    p: Dict[str, float],
    B: int,
    T: int,
    device: str,
    generator: torch.Generator | None = None,
) -> Tuple[torch.LongTensor, torch.LongTensor]:
    """
    sources: dict source_name -> bytes stream
    p: dict source_name -> prob; must sum to 1.0 (within 1e-6) and keys must match sources
    behavior:
      - sample source choice for each of B items via torch.multinomial(probs, B, replacement=True, generator=generator)
      - for each chosen source:
          - sample i via torch.randint(0, len(src)-(T+1)+1, (1,), generator=generator)
          - chunk = src[i:i+T+1]
          - x = chunk[:-1], y = chunk[1:]
      - return x,y as (B,T) int64 on device
    invariants:
      - values in [0..255]
      - y[:, :-1] == x[:, 1:]
    errors:
      - if any p key missing from sources or any source missing from p: raise
      - if abs(sum(p)-1) > 1e-6: raise
      - if any source len < T+1: raise listing sources and lengths
    determinism:
      - if generator is provided, use it for multinomial + randint; no python random.
    """
    # Validate keys
    p_keys = set(p.keys())
    source_keys = set(sources.keys())

    if p_keys != source_keys:
        missing_from_sources = p_keys - source_keys
        missing_from_p = source_keys - p_keys
        error_parts = []
        if missing_from_sources:
            error_parts.append(f"p keys missing from sources: {sorted(missing_from_sources)}")
        if missing_from_p:
            error_parts.append(f"source keys missing from p: {sorted(missing_from_p)}")
        raise ValueError("; ".join(error_parts))

    # Validate probabilities sum to 1
    total_prob = sum(p.values())
    if abs(total_prob - 1.0) > 1e-6:
        raise ValueError(f"probabilities must sum to 1.0, got {total_prob}")

    # Validate source lengths
    too_short = []
    for source_name, stream in sources.items():
        if len(stream) < T + 1:
            too_short.append(f"{source_name}: {len(stream)} bytes")

    if too_short:
        raise ValueError(
            f"Source(s) too short for T={T} (need at least {T+1} bytes):\n  "
            + "\n  ".join(too_short)
        )

    # Prepare for sampling
    # Sort keys to ensure deterministic ordering
    source_names = sorted(sources.keys())
    probs = torch.tensor([p[name] for name in source_names], dtype=torch.float32)

    # Sample source indices for each batch item
    source_indices = torch.multinomial(
        probs, num_samples=B, replacement=True, generator=generator
    )

    # Build batch
    x_list = []
    y_list = []

    for idx in source_indices:
        source_name = source_names[idx.item()]
        stream = sources[source_name]

        # Sample random start position
        max_start = len(stream) - (T + 1)
        start_idx = torch.randint(0, max_start + 1, (1,), generator=generator).item()

        # Extract chunk
        chunk = stream[start_idx : start_idx + T + 1]

        # Convert to tensor directly on target device
        chunk_tensor = torch.tensor(list(chunk), dtype=torch.int64, device=device)

        # Split into x and y
        x_item = chunk_tensor[:-1]
        y_item = chunk_tensor[1:]

        x_list.append(x_item)
        y_list.append(y_item)

    # Stack into batch tensors (already on device)
    x = torch.stack(x_list, dim=0)
    y = torch.stack(y_list, dim=0)

    return x, y
