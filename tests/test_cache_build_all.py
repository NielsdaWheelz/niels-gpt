"""Tests for cache build-all command (PR-04).

Tests verify:
1. Per-format validators catch missing artifacts
2. Primer builder is invoked by build-all
3. Only sources in default mixes are built
4. Missing caches raise FileNotFoundError listing the missing paths
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from niels_gpt.cache import cli as cache_cli
from niels_gpt.cache.cli import (
    PRETRAIN_VALIDATORS,
    SFT_VALIDATORS,
    _validate_caches,
    _validate_pretrain_sharded,
    _validate_sft_paired_bins,
    build_all,
)


class FakeTokenizer:
    """Minimal fake tokenizer for testing cache builds without real tokenizer."""

    def __init__(self, model_path: str):
        self.vocab_size = 100
        self.model_path = model_path
        self._expected_special_token_ids = {"sys": 3, "usr": 4, "asst": 5, "eot": 6}

    def special_token_ids(self):
        return self._expected_special_token_ids

    def encode(self, text: str) -> list[int]:
        """Simple char-to-id encoding for tests."""
        if not text:
            return []
        return [10 + ord(c) % 80 for c in text]


def _create_fake_tokenizer_file(tmp_path: Path) -> str:
    """Create a fake tokenizer file for testing."""
    tok_path = tmp_path / "tokenizer" / "spm.model"
    tok_path.parent.mkdir(parents=True, exist_ok=True)
    tok_path.write_bytes(b"fake tokenizer content")
    return str(tok_path)


def _create_stub_pretrain_cache(cache_root: Path, source: str) -> None:
    """Create a valid pretrain cache structure (sharded format)."""
    src_dir = cache_root / source
    train_dir = src_dir / "train"
    val_dir = src_dir / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    # Write at least one shard per split
    (train_dir / "shard_00000.bin").write_bytes(b"\x00" * 100)
    (val_dir / "shard_00000.bin").write_bytes(b"\x00" * 100)
    meta = {
        "dataset_name": source,
        "train_tokens": 50,
        "val_tokens": 50,
        "stub": True,
    }
    with open(src_dir / "meta.json", "w") as f:
        json.dump(meta, f)


def _create_stub_sft_cache(cache_root: Path, source: str) -> None:
    """Create a valid SFT cache structure (paired bins format)."""
    src_dir = cache_root / source
    src_dir.mkdir(parents=True, exist_ok=True)
    # Required files for SFT paired bins format
    (src_dir / "train_input_ids.bin").write_bytes(b"\x00" * 100)
    (src_dir / "train_labels.bin").write_bytes(b"\x00" * 100)
    (src_dir / "train_idx.npy").write_bytes(b"\x00" * 100)
    (src_dir / "val_input_ids.bin").write_bytes(b"\x00" * 100)
    (src_dir / "val_labels.bin").write_bytes(b"\x00" * 100)
    (src_dir / "val_idx.npy").write_bytes(b"\x00" * 100)
    meta = {
        "dataset_name": source,
        "train_examples": 1,
        "val_examples": 1,
        "stub": True,
    }
    with open(src_dir / "meta.json", "w") as f:
        json.dump(meta, f)


# ============================================================================
# Tests for per-format validators
# ============================================================================


def test_validate_pretrain_sharded_missing_all():
    """Test pretrain sharded validator catches all missing artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        src_dir = Path(tmpdir) / "wikitext"
        src_dir.mkdir()
        # Empty dir - everything missing
        missing = _validate_pretrain_sharded(src_dir)
        assert any("meta.json" in m for m in missing)
        assert any("train" in m for m in missing)
        assert any("val" in m for m in missing)


def test_validate_pretrain_sharded_missing_shards():
    """Test pretrain sharded validator catches missing shards in dirs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        src_dir = Path(tmpdir) / "wikitext"
        (src_dir / "train").mkdir(parents=True)
        (src_dir / "val").mkdir(parents=True)
        (src_dir / "meta.json").write_text("{}")
        # Dirs exist but no shards
        missing = _validate_pretrain_sharded(src_dir)
        assert any("train" in m and "shard" in m for m in missing)
        assert any("val" in m and "shard" in m for m in missing)


def test_validate_pretrain_sharded_valid():
    """Test pretrain sharded validator passes valid cache."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_root = Path(tmpdir)
        _create_stub_pretrain_cache(cache_root, "wikitext")
        missing = _validate_pretrain_sharded(cache_root / "wikitext")
        assert missing == []


def test_validate_sft_paired_bins_missing_all():
    """Test SFT paired bins validator catches all missing artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        src_dir = Path(tmpdir) / "primer"
        src_dir.mkdir()
        missing = _validate_sft_paired_bins(src_dir)
        assert any("meta.json" in m for m in missing)
        assert any("train_input_ids.bin" in m for m in missing)
        assert any("train_labels.bin" in m for m in missing)
        assert any("train_idx.npy" in m for m in missing)
        assert any("val_input_ids.bin" in m for m in missing)
        assert any("val_labels.bin" in m for m in missing)
        assert any("val_idx.npy" in m for m in missing)


def test_validate_sft_paired_bins_partial():
    """Test SFT paired bins validator catches partially missing files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        src_dir = Path(tmpdir) / "primer"
        src_dir.mkdir()
        # Only create some files
        (src_dir / "meta.json").write_text("{}")
        (src_dir / "train_input_ids.bin").write_bytes(b"")
        missing = _validate_sft_paired_bins(src_dir)
        assert "meta.json" not in str(missing)
        assert "train_input_ids.bin" not in str(missing)
        assert any("train_labels.bin" in m for m in missing)
        assert any("val_input_ids.bin" in m for m in missing)


def test_validate_sft_paired_bins_valid():
    """Test SFT paired bins validator passes valid cache."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_root = Path(tmpdir)
        _create_stub_sft_cache(cache_root, "primer")
        missing = _validate_sft_paired_bins(cache_root / "primer")
        assert missing == []


# ============================================================================
# Tests for _validate_caches with validator mapping
# ============================================================================


def test_validate_caches_pretrain_missing():
    """Test _validate_caches raises for missing pretrain caches."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_root = Path(tmpdir)
        with pytest.raises(FileNotFoundError) as exc_info:
            _validate_caches(cache_root, ["wikitext", "roam"], PRETRAIN_VALIDATORS)
        msg = str(exc_info.value)
        assert "wikitext" in msg
        assert "roam" in msg


def test_validate_caches_sft_missing():
    """Test _validate_caches raises for missing SFT caches."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_root = Path(tmpdir)
        with pytest.raises(FileNotFoundError) as exc_info:
            _validate_caches(cache_root, ["primer", "oasst1"], SFT_VALIDATORS)
        msg = str(exc_info.value)
        assert "primer" in msg
        assert "oasst1" in msg


def test_validate_caches_pretrain_valid():
    """Test _validate_caches passes when pretrain caches exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_root = Path(tmpdir)
        _create_stub_pretrain_cache(cache_root, "wikitext")
        _create_stub_pretrain_cache(cache_root, "roam")
        # Should not raise
        _validate_caches(cache_root, ["wikitext", "roam"], PRETRAIN_VALIDATORS)


def test_validate_caches_sft_valid():
    """Test _validate_caches passes when SFT caches exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_root = Path(tmpdir)
        _create_stub_sft_cache(cache_root, "primer")
        _create_stub_sft_cache(cache_root, "oasst1")
        # Should not raise
        _validate_caches(cache_root, ["primer", "oasst1"], SFT_VALIDATORS)


# ============================================================================
# Tests for build_all behavior
# ============================================================================


def test_build_all_calls_primer_builder(monkeypatch, tmp_path):
    """Test that build_all invokes the primer SFT builder."""
    cache_dir = tmp_path / "cache"
    primer_jsonl = tmp_path / "data" / "primer.jsonl"

    # Create test primer.jsonl
    primer_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(primer_jsonl, "w") as f:
        example = {
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi there"},
            ]
        }
        f.write(json.dumps(example) + "\n")

    tok_path = _create_fake_tokenizer_file(tmp_path)
    fake_tokenizer = FakeTokenizer(tok_path)
    primer_builder_called = []

    def fake_ensure_tokenizer(cache_dir):
        return fake_tokenizer

    def fake_build_primer_sft_cache(primer_jsonl_path, out_dir, *, tokenizer, val_frac, seed, t_max):
        """Fake primer builder that creates the expected structure."""
        primer_builder_called.append({
            "primer_jsonl": primer_jsonl_path,
            "out_dir": out_dir,
        })
        # Create valid SFT paired bins cache structure
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        (out_path / "train_input_ids.bin").write_bytes(b"\x00")
        (out_path / "train_labels.bin").write_bytes(b"\x00")
        (out_path / "train_idx.npy").write_bytes(b"\x00")
        (out_path / "val_input_ids.bin").write_bytes(b"\x00")
        (out_path / "val_labels.bin").write_bytes(b"\x00")
        (out_path / "val_idx.npy").write_bytes(b"\x00")
        meta = {"dataset_name": "primer", "train_examples": 1, "val_examples": 0}
        with open(out_path / "meta.json", "w") as f:
            json.dump(meta, f)

    def fake_build_pretrain_cache(*args, **kwargs):
        """No-op pretrain builder."""
        pass

    def fake_build_sft_cache(*args, **kwargs):
        """No-op SFT builder."""
        pass

    def fake_load_dataset(*args, **kwargs):
        """Fake dataset loader that returns a mock."""
        mock_ds = mock.MagicMock()
        mock_ds.info = mock.MagicMock()
        mock_ds.info.dataset_revision = "test"
        return mock_ds

    # Create stub caches for default pretrain sources
    pretrain_dir = cache_dir / "pretrain"
    sft_dir = cache_dir / "sft"

    # Default pretrain: fineweb_edu, wikitext, roam
    for src in ["fineweb_edu", "wikitext", "roam"]:
        _create_stub_pretrain_cache(pretrain_dir, src)

    # Default SFT: dolly15k, oasst1 (primer created by fake builder)
    for src in ["oasst1", "dolly15k"]:
        _create_stub_sft_cache(sft_dir, src)

    # Monkeypatch
    monkeypatch.setattr(cache_cli, "_ensure_tokenizer", fake_ensure_tokenizer)
    monkeypatch.setattr(cache_cli, "build_primer_sft_cache", fake_build_primer_sft_cache)
    monkeypatch.setattr(cache_cli, "build_pretrain_cache", fake_build_pretrain_cache)
    monkeypatch.setattr(cache_cli, "build_sft_cache", fake_build_sft_cache)
    monkeypatch.setattr(cache_cli, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(cache_cli, "iter_fineweb_edu", lambda **kwargs: iter([]))
    monkeypatch.setattr(cache_cli, "iter_wikitext", lambda **kwargs: iter([]))
    monkeypatch.setattr(cache_cli, "iter_dolly_sft", lambda **kwargs: iter([]))
    monkeypatch.setattr(cache_cli, "iter_oasst1_sft", lambda **kwargs: iter([]))
    monkeypatch.setattr(cache_cli, "list_roam_paths", lambda path: [])
    monkeypatch.setattr(cache_cli, "REPO_ROOT", tmp_path)

    build_all(
        cache_dir=str(cache_dir),
        seed=42,
        fineweb_train_tokens=1000,
        fineweb_val_tokens=100,
        shard_bytes=1024,
        roam_dir=None,
    )

    # Verify primer builder was called
    assert len(primer_builder_called) == 1, "primer builder should be called exactly once"
    assert str(primer_jsonl) in primer_builder_called[0]["primer_jsonl"]

    # Verify primer cache exists after build
    assert (sft_dir / "primer" / "meta.json").exists()


def test_build_all_missing_primer_cache_raises(monkeypatch, tmp_path):
    """Test that build_all raises FileNotFoundError when primer cache is missing after build."""
    cache_dir = tmp_path / "cache"
    primer_jsonl = tmp_path / "data" / "primer.jsonl"

    # Create test primer.jsonl
    primer_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(primer_jsonl, "w") as f:
        example = {
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi there"},
            ]
        }
        f.write(json.dumps(example) + "\n")

    tok_path = _create_fake_tokenizer_file(tmp_path)
    fake_tokenizer = FakeTokenizer(tok_path)

    def fake_ensure_tokenizer(cache_dir):
        return fake_tokenizer

    def fake_build_primer_sft_cache_noop(primer_jsonl_path, out_dir, *, tokenizer, val_frac, seed, t_max):
        """No-op primer builder - does NOT create cache structure."""
        pass

    def fake_build_pretrain_cache(*args, **kwargs):
        pass

    def fake_build_sft_cache(*args, **kwargs):
        pass

    def fake_load_dataset(*args, **kwargs):
        mock_ds = mock.MagicMock()
        mock_ds.info = mock.MagicMock()
        mock_ds.info.dataset_revision = "test"
        return mock_ds

    # Create stub caches for all default sources EXCEPT primer
    pretrain_dir = cache_dir / "pretrain"
    sft_dir = cache_dir / "sft"

    for src in ["fineweb_edu", "wikitext", "roam"]:
        _create_stub_pretrain_cache(pretrain_dir, src)

    for src in ["oasst1", "dolly15k"]:
        _create_stub_sft_cache(sft_dir, src)

    # Monkeypatch
    monkeypatch.setattr(cache_cli, "_ensure_tokenizer", fake_ensure_tokenizer)
    monkeypatch.setattr(cache_cli, "build_primer_sft_cache", fake_build_primer_sft_cache_noop)
    monkeypatch.setattr(cache_cli, "build_pretrain_cache", fake_build_pretrain_cache)
    monkeypatch.setattr(cache_cli, "build_sft_cache", fake_build_sft_cache)
    monkeypatch.setattr(cache_cli, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(cache_cli, "iter_fineweb_edu", lambda **kwargs: iter([]))
    monkeypatch.setattr(cache_cli, "iter_wikitext", lambda **kwargs: iter([]))
    monkeypatch.setattr(cache_cli, "iter_dolly_sft", lambda **kwargs: iter([]))
    monkeypatch.setattr(cache_cli, "iter_oasst1_sft", lambda **kwargs: iter([]))
    monkeypatch.setattr(cache_cli, "list_roam_paths", lambda path: [])
    monkeypatch.setattr(cache_cli, "REPO_ROOT", tmp_path)

    # Run build_all - should raise FileNotFoundError because primer cache is missing
    with pytest.raises(FileNotFoundError) as exc_info:
        build_all(
            cache_dir=str(cache_dir),
            seed=42,
            fineweb_train_tokens=1000,
            fineweb_val_tokens=100,
            shard_bytes=1024,
            roam_dir=None,
        )

    # Check error message lists the missing primer paths
    msg = str(exc_info.value)
    assert "primer" in msg
    assert "meta.json" in msg


def test_build_all_missing_primer_jsonl_causes_validation_failure(monkeypatch, tmp_path):
    """Test that build_all fails validation when primer.jsonl is missing (primer build fails).

    When primer.jsonl is missing, the primer build fails (caught/printed), but then
    validation fails because the primer cache doesn't exist.
    """
    cache_dir = tmp_path / "cache"
    # Note: NOT creating primer.jsonl

    tok_path = _create_fake_tokenizer_file(tmp_path)
    fake_tokenizer = FakeTokenizer(tok_path)

    def fake_ensure_tokenizer(cache_dir):
        return fake_tokenizer

    def fake_build_pretrain_cache(*args, **kwargs):
        pass

    def fake_build_sft_cache(*args, **kwargs):
        pass

    def fake_load_dataset(*args, **kwargs):
        mock_ds = mock.MagicMock()
        mock_ds.info = mock.MagicMock()
        mock_ds.info.dataset_revision = "test"
        return mock_ds

    # Create stub caches for pretrain and other SFT sources
    pretrain_dir = cache_dir / "pretrain"
    sft_dir = cache_dir / "sft"

    for src in ["fineweb_edu", "wikitext", "roam"]:
        _create_stub_pretrain_cache(pretrain_dir, src)

    for src in ["oasst1", "dolly15k"]:
        _create_stub_sft_cache(sft_dir, src)

    # Monkeypatch
    monkeypatch.setattr(cache_cli, "_ensure_tokenizer", fake_ensure_tokenizer)
    monkeypatch.setattr(cache_cli, "build_pretrain_cache", fake_build_pretrain_cache)
    monkeypatch.setattr(cache_cli, "build_sft_cache", fake_build_sft_cache)
    monkeypatch.setattr(cache_cli, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(cache_cli, "iter_fineweb_edu", lambda **kwargs: iter([]))
    monkeypatch.setattr(cache_cli, "iter_wikitext", lambda **kwargs: iter([]))
    monkeypatch.setattr(cache_cli, "iter_dolly_sft", lambda **kwargs: iter([]))
    monkeypatch.setattr(cache_cli, "iter_oasst1_sft", lambda **kwargs: iter([]))
    monkeypatch.setattr(cache_cli, "list_roam_paths", lambda path: [])
    monkeypatch.setattr(cache_cli, "REPO_ROOT", tmp_path)

    # Run build_all - should raise FileNotFoundError at validation because primer cache is missing
    with pytest.raises(FileNotFoundError) as exc_info:
        build_all(
            cache_dir=str(cache_dir),
            seed=42,
            fineweb_train_tokens=1000,
            fineweb_val_tokens=100,
            shard_bytes=1024,
            roam_dir=None,
        )

    msg = str(exc_info.value)
    # Validation error mentions primer cache files are missing
    assert "primer" in msg.lower()
    assert "meta.json" in msg.lower() or "input_ids" in msg.lower()


def test_build_all_only_builds_mix_sources(monkeypatch, tmp_path):
    """Test that build_all only builds sources in the default mixes."""
    cache_dir = tmp_path / "cache"
    primer_jsonl = tmp_path / "data" / "primer.jsonl"

    # Create test primer.jsonl
    primer_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(primer_jsonl, "w") as f:
        example = {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hey"}]}
        f.write(json.dumps(example) + "\n")

    tok_path = _create_fake_tokenizer_file(tmp_path)
    fake_tokenizer = FakeTokenizer(tok_path)

    # Track which builders are called
    builders_called = []

    def fake_ensure_tokenizer(cache_dir):
        return fake_tokenizer

    def make_fake_builder(name):
        def fake_builder(*args, **kwargs):
            builders_called.append(name)
        return fake_builder

    def fake_build_primer_sft_cache(primer_jsonl_path, out_dir, *, tokenizer, val_frac, seed, t_max):
        builders_called.append("primer")
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        for f in ["train_input_ids.bin", "train_labels.bin", "train_idx.npy",
                  "val_input_ids.bin", "val_labels.bin", "val_idx.npy"]:
            (out_path / f).write_bytes(b"\x00")
        with open(out_path / "meta.json", "w") as f:
            json.dump({"dataset_name": "primer", "train_examples": 1, "val_examples": 0}, f)

    def fake_build_sft_cache(examples, out_dir, *, tokenizer, val_frac, seed, source_name=None):
        # Determine which SFT source based on out_dir
        out_path = Path(out_dir)
        src_name = out_path.name
        builders_called.append(src_name)
        out_path.mkdir(parents=True, exist_ok=True)
        for f in ["train_input_ids.bin", "train_labels.bin", "train_idx.npy",
                  "val_input_ids.bin", "val_labels.bin", "val_idx.npy"]:
            (out_path / f).write_bytes(b"\x00")
        with open(out_path / "meta.json", "w") as f:
            json.dump({"dataset_name": src_name, "train_examples": 1, "val_examples": 0}, f)

    def fake_load_dataset(*args, **kwargs):
        mock_ds = mock.MagicMock()
        mock_ds.info = mock.MagicMock()
        mock_ds.info.dataset_revision = "test"
        return mock_ds

    # Create valid pretrain stub caches
    pretrain_dir = cache_dir / "pretrain"
    for src in ["fineweb_edu", "wikitext", "roam"]:
        _create_stub_pretrain_cache(pretrain_dir, src)

    # Monkeypatch
    monkeypatch.setattr(cache_cli, "_ensure_tokenizer", fake_ensure_tokenizer)
    monkeypatch.setattr(cache_cli, "build_primer_sft_cache", fake_build_primer_sft_cache)
    monkeypatch.setattr(cache_cli, "build_pretrain_cache", make_fake_builder("pretrain"))
    monkeypatch.setattr(cache_cli, "build_sft_cache", fake_build_sft_cache)
    monkeypatch.setattr(cache_cli, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(cache_cli, "iter_fineweb_edu", lambda **kwargs: iter([]))
    monkeypatch.setattr(cache_cli, "iter_wikitext", lambda **kwargs: iter([]))
    monkeypatch.setattr(cache_cli, "iter_dolly_sft", lambda **kwargs: iter([]))
    monkeypatch.setattr(cache_cli, "iter_oasst1_sft", lambda **kwargs: iter([]))
    monkeypatch.setattr(cache_cli, "list_roam_paths", lambda path: [])
    monkeypatch.setattr(cache_cli, "REPO_ROOT", tmp_path)

    build_all(
        cache_dir=str(cache_dir),
        seed=42,
        fineweb_train_tokens=1000,
        fineweb_val_tokens=100,
        shard_bytes=1024,
        roam_dir=None,
    )

    # Verify SFT builders called are exactly those in default mix_sft
    sft_built = [b for b in builders_called if b in ["primer", "oasst1", "dolly15k"]]
    assert set(sft_built) == {"primer", "oasst1", "dolly15k"}, f"Expected default SFT sources, got {sft_built}"

    # Verify gutenberg is NOT built (not in default mix_pretrain)
    assert "gutenberg" not in builders_called, "gutenberg should not be built (not in default mix)"
