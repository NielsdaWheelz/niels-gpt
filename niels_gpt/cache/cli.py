"""CLI for building token caches."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
import sys
from typing import Callable

import torch
from datasets import load_dataset

from niels_gpt.data.dolly import iter_dolly_sft
from niels_gpt.data.fineweb_edu import iter_fineweb_edu
from niels_gpt.data.oasst1 import iter_oasst1_sft
from niels_gpt.data.primer_sft import build_primer_sft_cache
from niels_gpt.data.roam import list_roam_paths, load_texts
from niels_gpt.data.wikitext import iter_wikitext
from niels_gpt.paths import REPO_ROOT, ROAM_DIR
from niels_gpt.settings import default_settings
from niels_gpt.tokenizer import DEFAULT_TOKENIZER_PATH, load_tokenizer

from .build_cache import build_pretrain_cache, build_sft_cache
from .formats import DEFAULT_SHARD_BYTES, TOKEN_DTYPE
from .meta import read_meta, sha256_file, write_meta

FINEWEB_DATASET_ID = "HuggingFaceFW/fineweb-edu"
FINEWEB_CONFIG = "CC-MAIN-2024-10"
DEFAULT_FINEWEB_TRAIN_TOKENS = 200_000_000
DEFAULT_FINEWEB_VAL_TOKENS = 5_000_000
DEFAULT_SFT_VAL_FRAC = 0.10
DEFAULT_SHUFFLE_BUFFER = 10_000


def _ensure_tokenizer(cache_dir: Path):
    tokenizer_model_path = DEFAULT_TOKENIZER_PATH
    tokenizer_meta_path = tokenizer_model_path.with_name("tokenizer_meta.json")
    if not tokenizer_model_path.exists():
        print(f"error: tokenizer not found at {tokenizer_model_path}")
        print("run tokenizer training first (PR-01)")
        sys.exit(1)

    tokenizer_cache_dir = cache_dir / "tokenizer"
    tokenizer_cache_dir.mkdir(parents=True, exist_ok=True)
    cached_model = tokenizer_cache_dir / "spm.model"
    cached_meta = tokenizer_cache_dir / "tokenizer_meta.json"
    # Always copy tokenizer to cache (ensures cache uses current tokenizer)
    shutil.copy2(tokenizer_model_path, cached_model)
    if tokenizer_meta_path.exists():
        shutil.copy2(tokenizer_meta_path, cached_meta)

    expected_ids = None
    if tokenizer_meta_path.exists():
        try:
            meta = read_meta(str(tokenizer_meta_path))
            expected_ids = meta.get("special_tokens") or meta.get("special_token_ids")
        except Exception:
            expected_ids = None

    tokenizer = load_tokenizer(str(cached_model), expected_special_token_ids=expected_ids)
    meta = {
        "dataset_name": "tokenizer",
        "dataset_config": None,
        "split_rule": "copy of artifacts/tokenizer/v2",
        "tokenizer_sha256": sha256_file(str(tokenizer_model_path)),
        "vocab_size": tokenizer.vocab_size,
        "special_token_ids": tokenizer.special_token_ids(),
        "token_dtype": TOKEN_DTYPE,
    }
    write_meta(str(tokenizer_cache_dir / "meta.json"), meta)
    return tokenizer


def _split_indices(count: int, *, val_frac: float, seed: int) -> tuple[list[int], list[int]]:
    if count == 0:
        return [], []
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(count, generator=gen).tolist()
    num_val = 0
    if count > 1:
        num_val = max(1, int(count * val_frac))
    val_set = set(perm[:num_val])
    train_indices = [i for i in perm if i not in val_set]
    val_indices = [i for i in perm if i in val_set]
    return train_indices, val_indices


def _token_count(texts: list[str], tokenizer) -> int:
    return sum(len(tokenizer.encode(t)) for t in texts)


def _update_meta(meta_path: Path, updates: dict) -> None:
    meta = read_meta(str(meta_path)) if meta_path.exists() else {}
    meta.update(updates)
    write_meta(str(meta_path), meta)


# ============================================================================
# Cache validators (pr-04): check actual artifacts, not just meta.json
# ============================================================================


def _validate_pretrain_sharded(source_dir: Path) -> list[str]:
    """
    Validate pretrain cache with sharded format.
    Expected: meta.json, train/shard_*.bin (>=1), val/shard_*.bin (>=1)
    Returns list of missing paths.
    """
    missing = []
    meta_path = source_dir / "meta.json"
    if not meta_path.exists():
        missing.append(str(meta_path))

    train_dir = source_dir / "train"
    val_dir = source_dir / "val"

    if not train_dir.exists():
        missing.append(str(train_dir))
    elif not list(train_dir.glob("shard_*.bin")):
        missing.append(f"{train_dir}/shard_*.bin (no shards found)")

    if not val_dir.exists():
        missing.append(str(val_dir))
    elif not list(val_dir.glob("shard_*.bin")):
        missing.append(f"{val_dir}/shard_*.bin (no shards found)")

    return missing


def _validate_sft_paired_bins(source_dir: Path) -> list[str]:
    """
    Validate SFT cache with paired bins format.
    Expected: meta.json, {train,val}_input_ids.bin, {train,val}_labels.bin, {train,val}_idx.npy
    Returns list of missing paths.
    """
    missing = []
    required_files = [
        "meta.json",
        "train_input_ids.bin",
        "train_labels.bin",
        "train_idx.npy",
        "val_input_ids.bin",
        "val_labels.bin",
        "val_idx.npy",
    ]
    for fname in required_files:
        fpath = source_dir / fname
        if not fpath.exists():
            missing.append(str(fpath))
    return missing


# Source -> validator mapping
PRETRAIN_VALIDATORS: dict[str, Callable[[Path], list[str]]] = {
    "fineweb_edu": _validate_pretrain_sharded,
    "wikitext": _validate_pretrain_sharded,
    "roam": _validate_pretrain_sharded,
    "gutenberg": _validate_pretrain_sharded,
}

SFT_VALIDATORS: dict[str, Callable[[Path], list[str]]] = {
    "dolly15k": _validate_sft_paired_bins,
    "oasst1": _validate_sft_paired_bins,
    "primer": _validate_sft_paired_bins,
}


def _validate_caches(
    cache_root: Path,
    sources: list[str],
    validators: dict[str, Callable[[Path], list[str]]],
) -> None:
    """
    Validate caches for given sources using per-source validators.
    Raises FileNotFoundError with multi-line list of missing paths.
    """
    all_missing: list[str] = []
    for src in sources:
        validator = validators.get(src)
        if validator is None:
            all_missing.append(f"{src}: no validator registered")
            continue
        src_dir = cache_root / src
        missing = validator(src_dir)
        all_missing.extend(missing)

    if all_missing:
        msg = "missing required cache paths:\n" + "\n".join(f"  - {p}" for p in all_missing)
        raise FileNotFoundError(msg)


# ============================================================================
# Per-source builders (called only if source is in mix)
# ============================================================================


def _build_fineweb_edu(
    pretrain_dir: Path,
    tokenizer,
    *,
    seed: int,
    train_tokens: int,
    val_tokens: int,
    shard_bytes: int,
) -> None:
    """Build fineweb-edu cache."""
    print("\n=== building fineweb-edu cache ===")
    fineweb_dir = pretrain_dir / "fineweb_edu"
    fineweb_dir.mkdir(exist_ok=True)

    ds_info = load_dataset(
        FINEWEB_DATASET_ID,
        name=FINEWEB_CONFIG,
        split="train",
        streaming=True,
    ).info
    dataset_revision = getattr(ds_info, "dataset_revision", None)

    texts = (
        sample.text
        for sample in iter_fineweb_edu(
            name=FINEWEB_CONFIG,
            split="train",
            streaming=True,
            shuffle=True,
            shuffle_buffer_size=DEFAULT_SHUFFLE_BUFFER,
            seed=seed,
        )
    )
    build_pretrain_cache(
        texts,
        str(fineweb_dir),
        tokenizer=tokenizer,
        max_train_tokens=train_tokens,
        max_val_tokens=val_tokens,
        shard_bytes=shard_bytes,
        seed=seed,
        shuffle_buffer=DEFAULT_SHUFFLE_BUFFER,
        source_name=FINEWEB_DATASET_ID,
        source_config=FINEWEB_CONFIG,
        streaming=True,
    )
    _update_meta(
        fineweb_dir / "meta.json",
        {
            "dataset_name": FINEWEB_DATASET_ID,
            "dataset_config": FINEWEB_CONFIG,
            "split_rule": "hf streaming shuffle buffer 10000; first N tokens to val",
            "streaming": True,
            "dataset_revision": dataset_revision,
            "shuffle_buffer": DEFAULT_SHUFFLE_BUFFER,
            "seed": seed,
            "train_tokens_target": train_tokens,
            "val_tokens_target": val_tokens,
        },
    )
    meta = read_meta(str(fineweb_dir / "meta.json"))
    print(f"✓ fineweb-edu: {meta['train_tokens']} train, {meta['val_tokens']} val tokens")


def _build_wikitext(pretrain_dir: Path, tokenizer, *, seed: int, shard_bytes: int) -> None:
    """Build wikitext cache."""
    print("\n=== building wikitext cache ===")
    wikitext_dir = pretrain_dir / "wikitext"
    wikitext_dir.mkdir(exist_ok=True)

    val_texts = [sample.text for sample in iter_wikitext(config="wikitext-103-raw-v1", split="validation")]
    train_texts = [sample.text for sample in iter_wikitext(config="wikitext-103-raw-v1", split="train")]
    val_tokens = _token_count(val_texts, tokenizer)
    train_tokens = _token_count(train_texts, tokenizer)

    build_pretrain_cache(
        list(val_texts) + list(train_texts),
        str(wikitext_dir),
        tokenizer=tokenizer,
        max_train_tokens=train_tokens,
        max_val_tokens=val_tokens,
        shard_bytes=shard_bytes,
        seed=seed,
        shuffle_buffer=None,
        source_name="wikitext",
        source_config="wikitext-103-raw-v1",
        streaming=False,
    )
    _update_meta(
        wikitext_dir / "meta.json",
        {
            "dataset_name": "wikitext",
            "dataset_config": "wikitext-103-raw-v1",
            "split_rule": "hf validation->val, train->train",
            "streaming": False,
        },
    )
    meta = read_meta(str(wikitext_dir / "meta.json"))
    print(f"✓ wikitext: {meta['train_tokens']} train, {meta['val_tokens']} val tokens")


def _build_roam(
    pretrain_dir: Path,
    tokenizer,
    *,
    seed: int,
    shard_bytes: int,
    roam_dir: str | None,
) -> None:
    """Build roam cache (creates stub if no files found)."""
    print("\n=== building roam cache ===")
    roam_cache_dir = pretrain_dir / "roam"
    roam_cache_dir.mkdir(exist_ok=True)

    roam_root = Path(roam_dir) if roam_dir is not None else ROAM_DIR
    paths = list_roam_paths(str(roam_root)) if roam_root.exists() else []

    if not paths:
        # Create stub with empty shards to pass validation
        (roam_cache_dir / "train").mkdir(exist_ok=True)
        (roam_cache_dir / "val").mkdir(exist_ok=True)
        # Write empty shard files so validation passes
        (roam_cache_dir / "train" / "shard_00000.bin").write_bytes(b"")
        (roam_cache_dir / "val" / "shard_00000.bin").write_bytes(b"")
        _update_meta(
            roam_cache_dir / "meta.json",
            {
                "dataset_name": "roam",
                "dataset_config": None,
                "split_rule": "no files found",
                "stub": True,
                "token_dtype": TOKEN_DTYPE,
                "seed": seed,
                "train_tokens": 0,
                "val_tokens": 0,
                "tokenizer_sha256": sha256_file(tokenizer.model_path),
                "vocab_size": tokenizer.vocab_size,
                "special_token_ids": tokenizer.special_token_ids(),
            },
        )
        print("⚠ roam: stub created (no files found)")
    else:
        train_idx, val_idx = _split_indices(len(paths), val_frac=0.1, seed=seed)
        train_paths = [paths[i] for i in train_idx]
        val_paths = [paths[i] for i in val_idx]
        train_texts = load_texts(train_paths)
        val_texts = load_texts(val_paths)
        val_tokens = _token_count(val_texts, tokenizer)
        train_tokens = _token_count(train_texts, tokenizer)

        build_pretrain_cache(
            list(val_texts) + list(train_texts),
            str(roam_cache_dir),
            tokenizer=tokenizer,
            max_train_tokens=train_tokens,
            max_val_tokens=val_tokens,
            shard_bytes=shard_bytes,
            seed=seed,
            shuffle_buffer=None,
            source_name="roam",
            source_config=None,
            streaming=False,
        )
        _update_meta(
            roam_cache_dir / "meta.json",
            {
                "dataset_name": "roam",
                "dataset_config": None,
                "split_rule": f"by file list, val_frac=0.1, seed={seed}",
                "num_files": len(paths),
                "streaming": False,
            },
        )
        meta = read_meta(str(roam_cache_dir / "meta.json"))
        print(f"✓ roam: {meta['train_tokens']} train, {meta['val_tokens']} val tokens from {len(paths)} files")


def _build_gutenberg(pretrain_dir: Path, tokenizer, *, seed: int) -> None:
    """Build gutenberg stub cache."""
    print("\n=== building gutenberg cache (stub) ===")
    gutenberg_dir = pretrain_dir / "gutenberg"
    (gutenberg_dir / "train").mkdir(parents=True, exist_ok=True)
    (gutenberg_dir / "val").mkdir(parents=True, exist_ok=True)
    # Write empty shard files so validation passes
    (gutenberg_dir / "train" / "shard_00000.bin").write_bytes(b"")
    (gutenberg_dir / "val" / "shard_00000.bin").write_bytes(b"")
    _update_meta(
        gutenberg_dir / "meta.json",
        {
            "dataset_name": "gutenberg",
            "dataset_config": None,
            "split_rule": "not built",
            "token_dtype": TOKEN_DTYPE,
            "seed": seed,
            "train_tokens": 0,
            "val_tokens": 0,
            "tokenizer_sha256": sha256_file(tokenizer.model_path),
            "vocab_size": tokenizer.vocab_size,
            "special_token_ids": tokenizer.special_token_ids(),
            "stub": True,
        },
    )
    print("✓ gutenberg: stub created")


def _build_dolly15k(sft_dir: Path, tokenizer, *, seed: int) -> None:
    """Build dolly15k SFT cache."""
    print("\n=== building dolly15k SFT cache ===")
    dolly_dir = sft_dir / "dolly15k"
    dolly_dir.mkdir(exist_ok=True)

    examples = (
        [{"role": msg.role, "content": msg.content} for msg in sample.messages]
        for sample in iter_dolly_sft(split="train", seed=seed, shuffle=False)
    )
    build_sft_cache(
        list(examples),
        str(dolly_dir),
        tokenizer=tokenizer,
        val_frac=DEFAULT_SFT_VAL_FRAC,
        seed=seed,
        source_name="databricks/databricks-dolly-15k",
    )
    _update_meta(
        dolly_dir / "meta.json",
        {
            "dataset_name": "databricks/databricks-dolly-15k",
            "dataset_config": None,
            "split_rule": f"torch randperm val_frac={DEFAULT_SFT_VAL_FRAC}, seed={seed}",
        },
    )
    meta = read_meta(str(dolly_dir / "meta.json"))
    print(f"✓ dolly15k: {meta['train_examples']} train, {meta['val_examples']} val examples")


def _build_oasst1(sft_dir: Path, tokenizer, *, seed: int) -> None:
    """Build oasst1 SFT cache."""
    print("\n=== building oasst1 SFT cache ===")
    oasst_dir = sft_dir / "oasst1"
    oasst_dir.mkdir(exist_ok=True)

    examples = (
        [{"role": msg.role, "content": msg.content} for msg in sample.messages]
        for sample in iter_oasst1_sft(split="train", seed=seed, shuffle_trees=False)
    )
    build_sft_cache(
        list(examples),
        str(oasst_dir),
        tokenizer=tokenizer,
        val_frac=DEFAULT_SFT_VAL_FRAC,
        seed=seed,
        source_name="OpenAssistant/oasst1",
    )
    _update_meta(
        oasst_dir / "meta.json",
        {
            "dataset_name": "OpenAssistant/oasst1",
            "dataset_config": None,
            "split_rule": f"torch randperm val_frac={DEFAULT_SFT_VAL_FRAC}, seed={seed}",
        },
    )
    meta = read_meta(str(oasst_dir / "meta.json"))
    print(f"✓ oasst1: {meta['train_examples']} train, {meta['val_examples']} val examples")


def _build_primer(sft_dir: Path, tokenizer, *, seed: int) -> None:
    """Build primer SFT cache."""
    print("\n=== building primer SFT cache ===")
    primer_dir = sft_dir / "primer"
    primer_dir.mkdir(exist_ok=True)

    primer_jsonl_path = REPO_ROOT / "data" / "primer.jsonl"
    if not primer_jsonl_path.exists():
        raise FileNotFoundError(f"missing primer jsonl: {primer_jsonl_path}")

    build_primer_sft_cache(
        str(primer_jsonl_path),
        str(primer_dir),
        tokenizer=tokenizer,
        val_frac=DEFAULT_SFT_VAL_FRAC,
        seed=seed,
        t_max=1024,
    )
    meta = read_meta(str(primer_dir / "meta.json"))
    print(f"✓ primer: {meta['train_examples']} train, {meta['val_examples']} val examples")


# ============================================================================
# build_all: builds ONLY sources required by default mixes
# ============================================================================


def build_all(
    *,
    cache_dir: str,
    seed: int,
    fineweb_train_tokens: int,
    fineweb_val_tokens: int,
    shard_bytes: int,
    roam_dir: str | None,
) -> None:
    """
    Build all caches required by default settings mixes.

    Only builds sources that appear in default mix_pretrain and mix_sft.
    Validates that all required caches exist at end.
    """

    def _shutdown_arrow():
        """Try to tear down pyarrow threadpools proactively to avoid exit hangs."""
        try:
            import pyarrow as pa  # type: ignore

            pool = pa.default_io_pool()
            if hasattr(pool, "shutdown"):
                pool.shutdown(wait=False)
        except Exception as exc:  # pragma: no cover - best-effort cleanup
            print(f"warning: pyarrow shutdown failed: {exc}", file=sys.stderr)

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    tokenizer = _ensure_tokenizer(cache_path)

    # Load default settings and determine sources from mixes
    settings = default_settings()
    pretrain_sources = list(settings.data.mix_pretrain.keys())
    sft_sources = list(settings.data.mix_sft.keys())

    # Print resolved plan
    print("\n=== build-all resolved plan ===")
    print(f"pretrain cache root: {cache_path / 'pretrain'}")
    print(f"pretrain sources to build: {pretrain_sources}")
    print(f"sft cache root: {cache_path / 'sft'}")
    print(f"sft sources to build: {sft_sources}")

    pretrain_dir = cache_path / "pretrain"
    pretrain_dir.mkdir(exist_ok=True)
    sft_dir = cache_path / "sft"
    sft_dir.mkdir(exist_ok=True)

    # Build only pretrain sources in mix
    for src in pretrain_sources:
        try:
            if src == "fineweb_edu":
                _build_fineweb_edu(
                    pretrain_dir,
                    tokenizer,
                    seed=seed,
                    train_tokens=fineweb_train_tokens,
                    val_tokens=fineweb_val_tokens,
                    shard_bytes=shard_bytes,
                )
            elif src == "wikitext":
                _build_wikitext(pretrain_dir, tokenizer, seed=seed, shard_bytes=shard_bytes)
            elif src == "roam":
                _build_roam(pretrain_dir, tokenizer, seed=seed, shard_bytes=shard_bytes, roam_dir=roam_dir)
            elif src == "gutenberg":
                _build_gutenberg(pretrain_dir, tokenizer, seed=seed)
            else:
                print(f"⚠ unknown pretrain source: {src} (skipped)")
        except Exception as exc:
            print(f"✗ {src} failed: {exc}")

    # Build only SFT sources in mix
    for src in sft_sources:
        try:
            if src == "dolly15k":
                _build_dolly15k(sft_dir, tokenizer, seed=seed)
            elif src == "oasst1":
                _build_oasst1(sft_dir, tokenizer, seed=seed)
            elif src == "primer":
                _build_primer(sft_dir, tokenizer, seed=seed)
            else:
                print(f"⚠ unknown sft source: {src} (skipped)")
        except Exception as exc:
            print(f"✗ {src} failed: {exc}")

    # Validate caches for default mixes exist
    print("\n=== validating default caches ===")
    _validate_caches(pretrain_dir, pretrain_sources, PRETRAIN_VALIDATORS)
    print(f"✓ pretrain caches validated: {pretrain_sources}")
    _validate_caches(sft_dir, sft_sources, SFT_VALIDATORS)
    print(f"✓ sft caches validated: {sft_sources}")

    _shutdown_arrow()
    print(f"\n✓ cache build complete: {cache_dir}")


def main():
    parser = argparse.ArgumentParser(description="Build token caches for pretrain and SFT datasets")
    parser.add_argument("command", choices=["build-all", "build-sft-primer"], help="Command to run")
    parser.add_argument("--cache-dir", type=str, default="cache", help="Cache directory (default: cache)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--fineweb-train-tokens",
        type=int,
        default=DEFAULT_FINEWEB_TRAIN_TOKENS,
        help="FineWeb-Edu training token budget (default: 200M)",
    )
    parser.add_argument(
        "--fineweb-val-tokens",
        type=int,
        default=DEFAULT_FINEWEB_VAL_TOKENS,
        help="FineWeb-Edu validation token budget (default: 5M)",
    )
    parser.add_argument(
        "--shard-bytes",
        type=int,
        default=DEFAULT_SHARD_BYTES,
        help="Shard size in bytes (default: 128MB)",
    )
    parser.add_argument(
        "--roam-dir",
        type=str,
        default=None,
        help="Roam data directory (default: .roam-data)",
    )
    parser.add_argument(
        "--primer-jsonl",
        type=str,
        default=str(REPO_ROOT / "data" / "primer.jsonl"),
        help="Path to primer.jsonl file (default: data/primer.jsonl)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory for SFT cache (default: cache/sft for build-sft-primer)",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=DEFAULT_SFT_VAL_FRAC,
        help=f"Validation fraction (default: {DEFAULT_SFT_VAL_FRAC})",
    )
    parser.add_argument(
        "--t-max",
        type=int,
        default=1024,
        help="Maximum sequence length for truncation (default: 1024)",
    )

    args = parser.parse_args()
    if args.command == "build-all":
        build_all(
            cache_dir=args.cache_dir,
            seed=args.seed,
            fineweb_train_tokens=args.fineweb_train_tokens,
            fineweb_val_tokens=args.fineweb_val_tokens,
            shard_bytes=args.shard_bytes,
            roam_dir=args.roam_dir,
        )
    elif args.command == "build-sft-primer":
        cache_path = Path(args.cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)

        tokenizer = _ensure_tokenizer(cache_path)

        out_dir = args.out_dir or str(cache_path / "sft" / "primer")

        primer_path = Path(args.primer_jsonl)
        if not primer_path.exists():
            print(f"error: primer.jsonl not found at {primer_path}")
            print("create a primer.jsonl file with chat messages in the specified format")
            sys.exit(1)

        print(f"\n=== building primer SFT cache ===")
        print(f"source: {primer_path}")
        print(f"output: {out_dir}")
        print(f"val_frac: {args.val_frac}")
        print(f"seed: {args.seed}")
        print(f"t_max: {args.t_max}")

        build_primer_sft_cache(
            str(primer_path),
            out_dir,
            tokenizer=tokenizer,
            val_frac=args.val_frac,
            seed=args.seed,
            t_max=args.t_max,
        )


if __name__ == "__main__":
    main()
