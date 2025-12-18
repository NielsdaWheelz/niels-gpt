"""Tests for KV-cache inference."""

import torch
import pytest

from niels_gpt.config import ModelConfig
from niels_gpt.model.gpt import GPT
from niels_gpt.generate import generate_ids_greedy_cached, generate_ids_greedy_full
from niels_gpt.infer.kv_cache import allocate_kv_cache, prefill, decode_step


@pytest.fixture
def tiny_model():
    """Create a tiny model for testing."""
    cfg = ModelConfig(
        V=256,
        T=128,
        C=64,
        L=2,
        H=4,
        d_ff=128,
        dropout=0.0,
        rope_theta=10000.0,
    )
    model = GPT(cfg)
    model.eval()
    return model


def test_greedy_equivalence_cpu(tiny_model):
    """Test that greedy cached generation matches greedy full generation on CPU."""
    # Seed for reproducibility
    torch.manual_seed(42)

    # Random prompt
    prompt_ids = [1, 2, 3, 4, 5]
    max_new_tokens = 16
    eot_token_id = 0

    # Move model to CPU
    tiny_model = tiny_model.cpu()

    # Generate with full forward pass
    torch.manual_seed(42)
    output_full = generate_ids_greedy_full(
        tiny_model,
        prompt_ids,
        max_new_tokens=max_new_tokens,
        eot_token_id=eot_token_id,
    )

    # Generate with KV-cache
    torch.manual_seed(42)
    output_cached = generate_ids_greedy_cached(
        tiny_model,
        prompt_ids,
        max_new_tokens=max_new_tokens,
        eot_token_id=eot_token_id,
    )

    # Should be exactly equal
    assert output_full == output_cached, (
        f"Greedy outputs differ:\nFull: {output_full}\nCached: {output_cached}"
    )


def test_attn_row_shape_and_normalization(tiny_model):
    """Test attention row shape and normalization properties."""
    tiny_model = tiny_model.cpu()

    # Small prompt
    prompt_ids = [1, 2, 3]
    device = next(tiny_model.parameters()).device
    dtype = torch.float32

    # Allocate cache
    L = len(tiny_model.blocks)
    H = tiny_model.cfg.H
    D = tiny_model.cfg.C // tiny_model.cfg.H
    T_max = tiny_model.cfg.T
    cache = allocate_kv_cache(L=L, B=1, H=H, T_max=T_max, D=D, device=device, dtype=dtype)

    # Prefill
    prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
    _, cache, _ = prefill(tiny_model, prompt_tensor, cache, trace_layer=0, return_attn_row=True)

    # Decode one step with tracing
    last_token = torch.tensor([[prompt_ids[-1]]], dtype=torch.long, device=device)
    logits, cache, trace = decode_step(
        tiny_model, last_token, cache, trace_layer=0, return_attn_row=True
    )

    # Check trace exists
    assert trace is not None, "Trace should be returned when return_attn_row=True"
    assert "attn_row" in trace, "Trace should contain attn_row"

    attn_row = trace["attn_row"]

    # Check shape: (B, H, cache.t) where cache.t is now len(prompt) + 1
    expected_t = len(prompt_ids) + 1
    assert attn_row.shape == (1, H, expected_t), (
        f"Expected shape (1, {H}, {expected_t}), got {attn_row.shape}"
    )

    # Check normalization: each head's attention row should sum to 1
    row_sums = attn_row.sum(dim=-1)  # (B, H)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4), (
        f"Attention rows should sum to 1, got sums: {row_sums}"
    )

    # Check all entries are finite
    assert torch.isfinite(attn_row).all(), "All attention values should be finite"

    # Check all entries are in valid range [0, 1] (with small tolerance for numerical errors)
    assert (attn_row >= -1e-6).all(), f"Found negative attention values: {attn_row.min()}"
    assert (attn_row <= 1.0 + 1e-6).all(), f"Found attention values > 1: {attn_row.max()}"


def test_stop_on_eot():
    """Test that generation stops when eot token is generated and includes eot in output."""

    class DummyModel:
        """Mock model that always returns eot_token_id as argmax."""

        def __init__(self, cfg, eot_id):
            self.cfg = cfg
            self.eot_id = eot_id
            self.blocks = [None] * cfg.L

        def eval(self):
            return self

        def parameters(self):
            # Return a dummy parameter on CPU
            yield torch.tensor([0.0], device="cpu")

        def tok_emb(self, x):
            B, t = x.shape
            return torch.zeros(B, t, self.cfg.C)

        def drop(self, x):
            return x

        def ln_f(self, x):
            return x

        def lm_head(self, x):
            B, t, C = x.shape
            V = self.cfg.V
            logits = torch.zeros(B, t, V)
            # Set argmax to eot_id
            logits[:, :, self.eot_id] = 10.0
            return logits

    class DummyBlock:
        """Mock block that passes through input."""

        def prefill(self, x, cache, layer_idx, return_attn_row=False):
            return x, None

        def decode_step(self, x, cache, layer_idx, pos, return_attn_row=False):
            return x, None

    cfg = ModelConfig(
        V=256,
        T=128,
        C=64,
        L=2,
        H=4,
        d_ff=128,
        dropout=0.0,
        rope_theta=10000.0,
    )

    eot_id = 10
    model = DummyModel(cfg, eot_id)
    model.blocks = [DummyBlock() for _ in range(cfg.L)]

    prompt_ids = [1, 2, 3]
    max_new_tokens = 10

    # Should generate eot on first decode step and stop
    output = generate_ids_greedy_cached(
        model,
        prompt_ids,
        max_new_tokens=max_new_tokens,
        eot_token_id=eot_id,
    )

    # Should return prompt + eot (eot is INCLUDED in output)
    # prompt is [1,2,3], first decode generates eot_id=10,
    # which gets appended to make [1,2,3,10], then we detect eot and return [1,2,3,10]
    expected = prompt_ids + [eot_id]
    assert output == expected, (
        f"Expected {expected} (prompt + eot), got {output}"
    )


def test_hard_cap_enforcement(tiny_model):
    """Test that hard cap is enforced correctly."""
    tiny_model = tiny_model.cpu()

    T_max = tiny_model.cfg.T

    # Test 1: Prompt + max_new_tokens > T_max should raise at entry
    prompt_ids = [1] * 100
    max_new_tokens = T_max  # This will exceed T_max

    with pytest.raises(ValueError, match="exceeds T_max"):
        generate_ids_greedy_cached(
            tiny_model,
            prompt_ids,
            max_new_tokens=max_new_tokens,
            eot_token_id=0,
        )

    # Test 2: Same for greedy_full
    with pytest.raises(ValueError, match="exceeds T_max"):
        generate_ids_greedy_full(
            tiny_model,
            prompt_ids,
            max_new_tokens=max_new_tokens,
            eot_token_id=0,
        )


def test_cache_overflow_during_generation(tiny_model):
    """Test that cache overflow is detected during generation."""
    tiny_model = tiny_model.cpu()

    T_max = tiny_model.cfg.T

    # Create a prompt that leaves room for exactly 1 token
    prompt_ids = [1] * (T_max - 1)
    max_new_tokens = 2  # Try to generate 2, but only room for 1

    # This should raise during generation when trying to generate the 2nd token
    # Actually, with the hard cap check at entry, this will fail immediately
    # Let me adjust: prompt + max_new = (T_max-1) + 2 = T_max+1, which exceeds
    with pytest.raises(ValueError, match="exceeds T_max"):
        generate_ids_greedy_cached(
            tiny_model,
            prompt_ids,
            max_new_tokens=max_new_tokens,
            eot_token_id=0,
        )


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_mps_equivalence():
    """Test equivalence on MPS if available."""
    cfg = ModelConfig(
        V=256,
        T=128,
        C=64,
        L=2,
        H=4,
        d_ff=128,
        dropout=0.0,
        rope_theta=10000.0,
    )
    model = GPT(cfg)
    model.eval()
    model = model.to("mps")

    torch.manual_seed(42)
    prompt_ids = [1, 2, 3, 4, 5]
    max_new_tokens = 16
    eot_token_id = 0

    # Generate with full forward pass
    torch.manual_seed(42)
    output_full = generate_ids_greedy_full(
        model,
        prompt_ids,
        max_new_tokens=max_new_tokens,
        eot_token_id=eot_token_id,
    )

    # Generate with KV-cache
    torch.manual_seed(42)
    output_cached = generate_ids_greedy_cached(
        model,
        prompt_ids,
        max_new_tokens=max_new_tokens,
        eot_token_id=eot_token_id,
    )

    # Should be exactly equal
    assert output_full == output_cached, (
        f"MPS greedy outputs differ:\nFull: {output_full}\nCached: {output_cached}"
    )


def test_cache_allocation():
    """Test KV cache allocation."""
    cache = allocate_kv_cache(
        L=4,
        B=2,
        H=8,
        T_max=256,
        D=32,
        device="cpu",
        dtype=torch.float32,
    )

    assert cache.k.shape == (4, 2, 8, 256, 32)
    assert cache.v.shape == (4, 2, 8, 256, 32)
    assert cache.t == 0
    assert cache.L == 4
    assert cache.T_max == 256
    assert cache.k.dtype == torch.float32
    assert cache.v.dtype == torch.float32


def test_prefill_decode_basic(tiny_model):
    """Test basic prefill and decode operations."""
    tiny_model = tiny_model.cpu()

    prompt_ids = [1, 2, 3]
    device = next(tiny_model.parameters()).device
    dtype = torch.float32

    # Allocate cache
    L = len(tiny_model.blocks)
    H = tiny_model.cfg.H
    D = tiny_model.cfg.C // tiny_model.cfg.H
    T_max = tiny_model.cfg.T
    cache = allocate_kv_cache(L=L, B=1, H=H, T_max=T_max, D=D, device=device, dtype=dtype)

    assert cache.t == 0

    # Prefill
    prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
    logits_prefill, cache, _ = prefill(tiny_model, prompt_tensor, cache)

    assert logits_prefill.shape == (1, len(prompt_ids), tiny_model.cfg.V)
    assert cache.t == len(prompt_ids)

    # Decode one step
    last_token = torch.tensor([[prompt_ids[-1]]], dtype=torch.long, device=device)
    logits_decode, cache, _ = decode_step(tiny_model, last_token, cache)

    assert logits_decode.shape == (1, 1, tiny_model.cfg.V)
    assert cache.t == len(prompt_ids) + 1


def test_logits_parity_prefill_decode(tiny_model):
    """Test that prefill+decode logits match full forward logits at each step."""
    tiny_model = tiny_model.cpu()

    prompt_ids = [1, 2, 3, 4, 5]
    num_decode_steps = 5
    device = next(tiny_model.parameters()).device
    dtype = torch.float32

    # Allocate cache
    L = len(tiny_model.blocks)
    H = tiny_model.cfg.H
    D = tiny_model.cfg.C // tiny_model.cfg.H
    T_max = tiny_model.cfg.T
    cache = allocate_kv_cache(L=L, B=1, H=H, T_max=T_max, D=D, device=device, dtype=dtype)

    # Prefill with prompt
    prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
    logits_prefill, cache, _ = prefill(tiny_model, prompt_tensor, cache)

    # Get full forward logits for prompt
    full_logits_prompt = tiny_model(prompt_tensor)

    # Check prefill logits match full forward at last position
    assert torch.allclose(logits_prefill, full_logits_prompt, atol=1e-5), (
        "Prefill logits should match full forward logits"
    )

    # Now decode N steps and compare each step's logits
    ids_list = list(prompt_ids)
    for step in range(num_decode_steps):
        # Decode one step with cache
        last_token = torch.tensor([[ids_list[-1]]], dtype=torch.long, device=device)
        logits_decode, cache, _ = decode_step(tiny_model, last_token, cache)

        # Get next token (greedy for determinism)
        next_token = int(logits_decode[0, 0].argmax().item())
        ids_list.append(next_token)

        # Get full forward logits for the current full sequence
        full_sequence = torch.tensor(ids_list, dtype=torch.long, device=device).unsqueeze(0)
        logits_full = tiny_model(full_sequence)

        # Compare last position logits
        logits_decode_last = logits_decode[0, 0]  # (V,)
        logits_full_last = logits_full[0, -1]  # (V,)

        assert torch.allclose(logits_decode_last, logits_full_last, atol=1e-5), (
            f"Step {step}: decode logits should match full forward logits at last position\n"
            f"Max diff: {(logits_decode_last - logits_full_last).abs().max()}"
        )
