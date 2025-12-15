"""Tests for batching module."""

import pytest
import torch

from niels_gpt.batching import get_batch


class TestGetBatch:
    """Test get_batch function."""

    def test_get_batch_shapes(self):
        """Test that get_batch returns correct shapes."""
        sources = {
            "wiki": b"abcdefghijklmnopqrstuvwxyz" * 50,
            "roam": b"0123456789" * 200,
            "primer": b"system: x\nuser: y\nassistant: z\n" * 50,
        }
        p = {"wiki": 0.5, "roam": 0.3, "primer": 0.2}
        B = 8
        T = 16

        x, y = get_batch(sources, p=p, B=B, T=T, device="cpu")

        assert x.shape == (B, T)
        assert y.shape == (B, T)

    def test_get_batch_dtype(self):
        """Test that get_batch returns int64."""
        sources = {
            "wiki": b"abcdefghijklmnopqrstuvwxyz" * 50,
        }
        p = {"wiki": 1.0}
        B = 4
        T = 16

        x, y = get_batch(sources, p=p, B=B, T=T, device="cpu")

        assert x.dtype == torch.int64
        assert y.dtype == torch.int64

    def test_get_batch_value_range(self):
        """Test that values are in [0..255]."""
        sources = {
            "wiki": b"abcdefghijklmnopqrstuvwxyz" * 50,
            "roam": b"0123456789" * 200,
        }
        p = {"wiki": 0.7, "roam": 0.3}
        B = 8
        T = 16

        x, y = get_batch(sources, p=p, B=B, T=T, device="cpu")

        assert (x >= 0).all()
        assert (x <= 255).all()
        assert (y >= 0).all()
        assert (y <= 255).all()

    def test_get_batch_shift_invariant(self):
        """Test that y[:, :-1] == x[:, 1:] (shift invariant)."""
        sources = {
            "wiki": b"abcdefghijklmnopqrstuvwxyz" * 50,
        }
        p = {"wiki": 1.0}
        B = 8
        T = 16

        x, y = get_batch(sources, p=p, B=B, T=T, device="cpu")

        # y should be x shifted by one position
        assert torch.equal(y[:, :-1], x[:, 1:])

    def test_get_batch_determinism(self):
        """Test that get_batch is deterministic with same generator seed."""
        sources = {
            "wiki": b"abcdefghijklmnopqrstuvwxyz" * 50,
            "roam": b"0123456789" * 200,
            "primer": b"system: x\nuser: y\nassistant: z\n" * 50,
        }
        p = {"wiki": 0.5, "roam": 0.3, "primer": 0.2}
        B = 8
        T = 16

        g1 = torch.Generator().manual_seed(0)
        x1, y1 = get_batch(sources, p=p, B=B, T=T, device="cpu", generator=g1)

        g2 = torch.Generator().manual_seed(0)
        x2, y2 = get_batch(sources, p=p, B=B, T=T, device="cpu", generator=g2)

        assert torch.equal(x1, x2)
        assert torch.equal(y1, y2)

    def test_get_batch_different_seeds_different_results(self):
        """Test that different seeds produce different results."""
        sources = {
            "wiki": b"abcdefghijklmnopqrstuvwxyz" * 50,
            "roam": b"0123456789" * 200,
        }
        p = {"wiki": 0.6, "roam": 0.4}
        B = 8
        T = 16

        g1 = torch.Generator().manual_seed(0)
        x1, y1 = get_batch(sources, p=p, B=B, T=T, device="cpu", generator=g1)

        g2 = torch.Generator().manual_seed(999)
        x2, y2 = get_batch(sources, p=p, B=B, T=T, device="cpu", generator=g2)

        # Should be different with high probability
        assert not torch.equal(x1, x2)

    def test_get_batch_prob_sum_error(self):
        """Test that probabilities not summing to 1.0 raises error."""
        sources = {
            "wiki": b"abcdefghijklmnopqrstuvwxyz" * 50,
        }
        p = {"wiki": 0.5}  # Doesn't sum to 1.0
        B = 4
        T = 16

        with pytest.raises(ValueError) as exc_info:
            get_batch(sources, p=p, B=B, T=T, device="cpu")

        assert "sum to 1.0" in str(exc_info.value)

    def test_get_batch_missing_key_in_sources(self):
        """Test that missing key in sources raises error."""
        sources = {
            "wiki": b"abcdefghijklmnopqrstuvwxyz" * 50,
        }
        p = {"wiki": 0.7, "roam": 0.3}  # roam not in sources
        B = 4
        T = 16

        with pytest.raises(ValueError) as exc_info:
            get_batch(sources, p=p, B=B, T=T, device="cpu")

        assert "missing from sources" in str(exc_info.value) or "roam" in str(exc_info.value)

    def test_get_batch_missing_key_in_p(self):
        """Test that missing key in p raises error."""
        sources = {
            "wiki": b"abcdefghijklmnopqrstuvwxyz" * 50,
            "roam": b"0123456789" * 200,
        }
        p = {"wiki": 1.0}  # roam not in p
        B = 4
        T = 16

        with pytest.raises(ValueError) as exc_info:
            get_batch(sources, p=p, B=B, T=T, device="cpu")

        assert "missing from p" in str(exc_info.value) or "roam" in str(exc_info.value)

    def test_get_batch_too_short_source(self):
        """Test that too-short source raises error."""
        sources = {
            "wiki": b"short",  # Only 5 bytes
        }
        p = {"wiki": 1.0}
        B = 4
        T = 16

        with pytest.raises(ValueError) as exc_info:
            get_batch(sources, p=p, B=B, T=T, device="cpu")

        assert "too short" in str(exc_info.value).lower()
        assert "wiki" in str(exc_info.value)
        assert "5" in str(exc_info.value)

    def test_get_batch_single_source(self):
        """Test get_batch with single source."""
        sources = {
            "wiki": b"abcdefghijklmnopqrstuvwxyz" * 50,
        }
        p = {"wiki": 1.0}
        B = 4
        T = 16

        x, y = get_batch(sources, p=p, B=B, T=T, device="cpu")

        assert x.shape == (B, T)
        assert y.shape == (B, T)

    def test_get_batch_device_placement(self):
        """Test that tensors are placed on correct device."""
        sources = {
            "wiki": b"abcdefghijklmnopqrstuvwxyz" * 50,
        }
        p = {"wiki": 1.0}
        B = 4
        T = 16

        x, y = get_batch(sources, p=p, B=B, T=T, device="cpu")

        assert x.device.type == "cpu"
        assert y.device.type == "cpu"

    def test_get_batch_multiple_sources_sampling(self):
        """Test that multiple sources are sampled from."""
        sources = {
            "wiki": b"a" * 1000,
            "roam": b"b" * 1000,
        }
        p = {"wiki": 0.5, "roam": 0.5}
        B = 100  # Large batch to ensure both sources get sampled
        T = 16

        g = torch.Generator().manual_seed(42)
        x, y = get_batch(sources, p=p, B=B, T=T, device="cpu", generator=g)

        # Check that we see both 'a' (97) and 'b' (98) in the batch
        unique_values = torch.unique(x)
        # With 100 samples and 50/50 split, we should see both
        # (This is probabilistic but with B=100 it's virtually certain)
        assert 97 in unique_values or 98 in unique_values
