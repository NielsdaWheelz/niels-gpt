"""KV-cache infrastructure for efficient autoregressive decoding."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from niels_gpt.model.gpt import GPT


@dataclass
class KVCache:
    """
    Key-value cache for transformer decoder inference.

    Stores pre-computed keys and values for all layers to avoid recomputation
    during autoregressive generation.

    Attributes:
        k: Key cache, shape (L, B, H, T_max, D).
        v: Value cache, shape (L, B, H, T_max, D).
        t: Number of timesteps currently cached (0 <= t <= T_max).
    """

    k: torch.Tensor  # (L, B, H, T_max, D)
    v: torch.Tensor  # (L, B, H, T_max, D)
    t: int  # number of cached timesteps (0..T_max)

    @property
    def L(self) -> int:
        """Number of layers."""
        return self.k.shape[0]

    @property
    def T_max(self) -> int:
        """Maximum sequence length (model context size)."""
        return self.k.shape[3]


def allocate_kv_cache(
    *,
    L: int,
    B: int,
    H: int,
    T_max: int,
    D: int,
    device: str,
    dtype: torch.dtype,
) -> KVCache:
    """
    Allocate a zero-initialized KV cache.

    Args:
        L: Number of transformer layers.
        B: Batch size.
        H: Number of attention heads.
        T_max: Maximum sequence length (context window).
        D: Head dimension (C // H).
        device: Device to allocate on ("cpu", "mps", "cuda").
        dtype: Data type for cache tensors (should match inference dtype).

    Returns:
        KVCache with t=0 and zero-initialized k/v tensors.
    """
    k = torch.zeros(L, B, H, T_max, D, device=device, dtype=dtype)
    v = torch.zeros(L, B, H, T_max, D, device=device, dtype=dtype)
    return KVCache(k=k, v=v, t=0)


@torch.no_grad()
def prefill(
    model: "GPT",
    prompt_ids: torch.LongTensor,  # (B, t0)
    cache: KVCache,
    *,
    trace_layer: int | None = None,
    return_attn_row: bool = False,
) -> tuple[torch.Tensor, KVCache, dict | None]:
    """
    Run the model over the full prompt and fill the KV cache.

    This function processes the entire prompt in one forward pass, computing
    all key-value pairs and storing them in the cache for subsequent decode steps.

    INVARIANT: model must be in eval mode (model.eval()) before calling.
    This function does NOT call model.eval() - caller is responsible.

    Args:
        model: GPT model in eval mode.
        prompt_ids: Prompt token IDs, shape (B, t0), dtype int64.
        cache: KVCache to fill (must have cache.t == 0).
        trace_layer: Optional layer index to trace attention (0 <= trace_layer < L).
        return_attn_row: If True and trace_layer is set, return attention row
                         for the last prompt token.

    Returns:
        logits: (B, t0, V) output logits for the prompt.
        cache: Same cache object with cache.t == t0 and k/v filled.
        trace: Optional dict with:
            - "layer": int (trace_layer value)
            - "attn_row": (B, H, t0) attention probs for last prompt token

    Raises:
        AssertionError: If batch size mismatches, cache not empty, or prompt
                        length exceeds T_max.
    """
    # Validate inputs
    assert prompt_ids.ndim == 2, f"prompt_ids must be 2D (B, t0), got {prompt_ids.shape}"
    B, t0 = prompt_ids.shape

    assert B == cache.k.shape[1], (
        f"batch size mismatch: prompt B={B} must equal cache B={cache.k.shape[1]}"
    )
    assert cache.t == 0, f"cache must be empty (t=0) before prefill, got t={cache.t}"
    assert t0 <= cache.T_max, (
        f"prompt length {t0} exceeds T_max {cache.T_max}"
    )

    # Token embeddings + dropout
    h = model.tok_emb(prompt_ids)  # (B, t0, C)
    h = model.drop(h)

    # Process through transformer blocks, filling cache
    attn_row = None
    for layer_idx, block in enumerate(model.blocks):
        if trace_layer is not None and layer_idx == trace_layer:
            h, attn_row = block.prefill(
                h, cache=cache, layer_idx=layer_idx, return_attn_row=True
            )
        else:
            h, _ = block.prefill(h, cache=cache, layer_idx=layer_idx, return_attn_row=False)

    # Update cache position counter
    cache.t = t0

    # Validate cache state
    assert cache.t == t0, f"cache.t should be {t0} after prefill, got {cache.t}"

    # Final layer norm
    h = model.ln_f(h)  # (B, t0, C)

    # LM head
    logits = model.lm_head(h)  # (B, t0, V)

    # Build trace dict if requested
    trace = None
    if trace_layer is not None and return_attn_row and attn_row is not None:
        trace = {
            "layer": trace_layer,
            "attn_row": attn_row,  # (B, H, t0)
        }

    return logits, cache, trace


@torch.no_grad()
def decode_step(
    model: "GPT",
    last_token_ids: torch.LongTensor,  # (B, 1)
    cache: KVCache,
    *,
    trace_layer: int | None = None,
    return_attn_row: bool = False,
) -> tuple[torch.Tensor, KVCache, dict | None]:
    """
    Process exactly one new token and append its K/V to the cache.

    This function takes a single token, computes its key-value pair, appends
    it to the cache, and computes the output logits by attending to all
    previously cached keys and values.

    INVARIANT: model must be in eval mode (model.eval()) before calling.
    This function does NOT call model.eval() - caller is responsible.

    Args:
        model: GPT model in eval mode.
        last_token_ids: Single token IDs, shape (B, 1), dtype int64.
        cache: KVCache with t timesteps already filled.
        trace_layer: Optional layer index to trace attention (0 <= trace_layer < L).
        return_attn_row: If True and trace_layer is set, return attention row
                         for this new token.

    Returns:
        logits: (B, 1, V) output logits for the new token.
        cache: Same cache object with cache.t increased by 1.
        trace: Optional dict with:
            - "layer": int (trace_layer value)
            - "attn_row": (B, H, t+1) attention probs for new token attending
                          to all cached keys including itself

    Raises:
        AssertionError: If input shape is wrong, batch size mismatches, or
                        cache overflow (t >= T_max).
        ValueError: If cache overflow during generation.
    """
    # Validate inputs
    assert last_token_ids.ndim == 2, f"last_token_ids must be 2D, got {last_token_ids.shape}"
    assert last_token_ids.shape[1] == 1, (
        f"decode_step requires exactly (B, 1) input, got {last_token_ids.shape}"
    )
    B = last_token_ids.shape[0]

    assert B == cache.k.shape[1], (
        f"batch size mismatch: input B={B} must equal cache B={cache.k.shape[1]}"
    )
    assert cache.t < cache.T_max, (
        f"cache overflow: t={cache.t} must be < T_max={cache.T_max}"
    )

    # Current position for this new token
    pos = cache.t

    # Token embeddings + dropout
    h = model.tok_emb(last_token_ids)  # (B, 1, C)
    h = model.drop(h)

    # Process through transformer blocks
    attn_row = None
    for layer_idx, block in enumerate(model.blocks):
        if trace_layer is not None and layer_idx == trace_layer:
            h, attn_row = block.decode_step(
                h, cache=cache, layer_idx=layer_idx, pos=pos, return_attn_row=True
            )
        else:
            h, _ = block.decode_step(
                h, cache=cache, layer_idx=layer_idx, pos=pos, return_attn_row=False
            )

    # Update cache position counter (after all blocks have written to position pos)
    cache.t = pos + 1

    # Validate cache state
    assert cache.t <= cache.T_max, (
        f"cache.t={cache.t} exceeded T_max={cache.T_max} after decode_step"
    )

    # Final layer norm
    h = model.ln_f(h)  # (B, 1, C)

    # LM head
    logits = model.lm_head(h)  # (B, 1, V)

    # Build trace dict if requested
    trace = None
    if trace_layer is not None and return_attn_row and attn_row is not None:
        trace = {
            "layer": trace_layer,
            "attn_row": attn_row,  # (B, H, pos+1)
        }

    return logits, cache, trace
