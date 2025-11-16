#!/usr/bin/env python
"""Reshuffle latent shards using a global permutation with streaming writes.

The script materialises a random permutation of every sample in the corpus,
loads shards on-demand, and emits uniformly sized output shards without
reloading data unnecessarily. Output tensors are cloned before serialisation,
ensuring each shard owns its storage and stays at the expected size (~32 MB for
4096×4×32×32 float16 latents).

Example::

    python scripts/reshuffle_latent_shards.py \
        --input-root cache/latents/imagenet21k_resized/imagenet21k_train \
        --output-root cache/latents/imagenet21k_resized/imagenet21k_train_shuffled_v2 \
        --chunk-size 4096 \
        --group-size 64 \
        --max-cache-shards 128 \
        --drop-tail
"""

from __future__ import annotations

import argparse
import math
from collections import OrderedDict
from pathlib import Path

import torch

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is optional
    def tqdm(iterable, **_: object):
        return iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reshuffle latent shard .pt files")
    parser.add_argument("--input-root", required=True, help="Directory with existing shards (read-only)")
    parser.add_argument("--output-root", required=True, help="Directory to write shuffled shards")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=4096,
        help="Number of samples per output shard (default: 4096)",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=64,
        help="Buffer multiple of chunk-size to process per step (default: 64)",
    )
    parser.add_argument(
        "--max-cache-shards",
        type=int,
        default=128,
        help="Maximum number of source shards to keep in memory concurrently",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: torch default RNG)",
    )
    parser.add_argument(
        "--prefix",
        default="shard",
        help="Filename prefix for output shards (default: shard)",
    )
    parser.add_argument(
        "--drop-tail",
        action="store_true",
        help="If set, drop any leftover samples smaller than chunk-size",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in the output directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print planned actions without writing output shards",
    )
    return parser.parse_args()


def scan_shards(
    root: Path,
) -> tuple[list[Path], list[int], list[int], torch.dtype, tuple[int, ...], float | None, int | None, int | None]:
    shard_paths = sorted(root.glob("*.pt"))
    if not shard_paths:
        raise FileNotFoundError(f"No .pt shards found under {root}")

    lengths: list[int] = []
    offsets: list[int] = []
    total = 0
    label_min: int | None = None
    label_max: int | None = None

    first_payload = torch.load(shard_paths[0], map_location="cpu")
    latents = first_payload["latents"]
    if not isinstance(latents, torch.Tensor) or latents.ndim < 2:
        raise ValueError(f"Shard {shard_paths[0]} missing latents tensor")
    latent_dtype = latents.dtype
    latent_shape = tuple(latents.shape[1:])
    scaling_factor = first_payload.get("scaling_factor")
    scaling_factor = None if scaling_factor is None else float(scaling_factor)

    del first_payload

    for path in tqdm(shard_paths, desc="scanning shards", unit="shard"):
        payload = torch.load(path, map_location="cpu")
        latents = payload["latents"]
        labels = payload["labels"]
        if not isinstance(latents, torch.Tensor) or latents.ndim < 2:
            raise ValueError(f"Shard {path} missing latents tensor")
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.long)
        labels = labels.long()
        if latents.dtype != latent_dtype:
            raise ValueError(f"Shard {path} has dtype {latents.dtype}, expected {latent_dtype}")
        if tuple(latents.shape[1:]) != latent_shape:
            raise ValueError(f"Shard {path} latent shape {tuple(latents.shape[1:])} != {latent_shape}")
        shard_scaling = payload.get("scaling_factor")
        if shard_scaling is not None:
            shard_scaling = float(shard_scaling)
            if scaling_factor is None:
                scaling_factor = shard_scaling
            elif not math.isclose(shard_scaling, scaling_factor, rel_tol=1e-6):
                raise ValueError(f"Shard {path} scaling_factor {shard_scaling} != {scaling_factor}")

        length = latents.shape[0]
        total += length
        lengths.append(length)
        offsets.append(total)

        shard_label_min = int(labels.min())
        shard_label_max = int(labels.max())
        label_min = shard_label_min if label_min is None else min(label_min, shard_label_min)
        label_max = shard_label_max if label_max is None else max(label_max, shard_label_max)
        del payload

    return shard_paths, lengths, offsets, latent_dtype, latent_shape, scaling_factor, label_min, label_max


def build_source_index(shard_lengths: list[int]) -> torch.Tensor:
    total = sum(shard_lengths)
    mapping = torch.empty((total, 2), dtype=torch.int32)
    cursor = 0
    for shard_idx, length in enumerate(shard_lengths):
        next_cursor = cursor + length
        mapping[cursor:next_cursor, 0] = shard_idx
        mapping[cursor:next_cursor, 1] = torch.arange(length, dtype=torch.int32)
        cursor = next_cursor
    return mapping


class ShardCache:
    """LRU cache that keeps recently used shards in memory."""

    def __init__(self, paths: list[Path], dtype: torch.dtype, max_items: int) -> None:
        self.paths = paths
        self.dtype = dtype
        self.max_items = max(1, max_items)
        self._store: OrderedDict[int, tuple[torch.Tensor, torch.Tensor]] = OrderedDict()

    def _evict_if_needed(self) -> None:
        while len(self._store) > self.max_items:
            shard_idx, (latents, labels) = self._store.popitem(last=False)
            del latents
            del labels

    def get(self, shard_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if shard_idx in self._store:
            latents, labels = self._store.pop(shard_idx)
            self._store[shard_idx] = (latents, labels)
            return latents, labels

        payload = torch.load(self.paths[shard_idx], map_location="cpu")
        latents = payload["latents"].to(dtype=self.dtype, copy=False).contiguous()
        labels = payload["labels"]
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.long)
        labels = labels.long().contiguous()
        self._store[shard_idx] = (latents, labels)
        self._evict_if_needed()
        return latents, labels


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    if not input_root.exists():
        raise FileNotFoundError(f"Input root {input_root} does not exist")
    output_root.mkdir(parents=True, exist_ok=True)

    if not args.overwrite:
        existing = list(output_root.glob("*.pt"))
        if existing:
            raise FileExistsError(
                f"Output directory {output_root} already contains {len(existing)} shard(s); "
                "use --overwrite to replace them"
            )

    if args.seed is not None:
        torch.manual_seed(args.seed)
        generator = torch.Generator().manual_seed(args.seed)
    else:
        generator = torch.Generator()
        generator.manual_seed(torch.seed())

    (
        shard_paths,
        shard_lengths,
        shard_offsets,
        latent_dtype,
        latent_shape,
        scaling_factor,
        label_min,
        label_max,
    ) = scan_shards(input_root)

    total_samples = shard_offsets[-1]
    chunk_size = max(1, int(args.chunk_size))
    group_size = max(1, int(args.group_size))
    if group_size > len(shard_paths):
        group_size = len(shard_paths)

    print(f"Discovered {len(shard_paths)} input shards containing {total_samples} samples")
    print(f"Latent dtype: {latent_dtype}, latent shape: {latent_shape}")
    if label_min is not None and label_max is not None:
        print(f"Label range: [{label_min}, {label_max}] ({label_max - label_min + 1} distinct levels if dense)")
    if scaling_factor is None:
        print("Warning: shards do not include scaling_factor metadata")

    tail_samples = total_samples % chunk_size
    if tail_samples and not args.drop_tail:
        raise ValueError(
            "Total samples not divisible by chunk-size. "
            "Re-run with --drop-tail to discard the remainder, or choose a chunk-size that divides the dataset."
        )

    effective_total = total_samples - tail_samples
    if effective_total <= 0:
        raise RuntimeError("No samples remain after applying drop-tail policy")

    buffer_multiplier = max(1, group_size)
    block_size = chunk_size * buffer_multiplier
    total_outputs = effective_total // chunk_size
    print(
        f"Streaming permutation with block size {block_size} samples "
        f"({buffer_multiplier} chunk(s) per block)"
    )

    if args.dry_run:
        planned = total_outputs
        message = f"Dry run: would write {planned} shard(s) × {chunk_size} samples"
        if tail_samples:
            message += f" (dropping {tail_samples} leftover)"
        print(message)
        return

    source_index = build_source_index(shard_lengths)
    perm = torch.randperm(total_samples, generator=generator)
    perm = perm[:effective_total]

    cache = ShardCache(shard_paths, latent_dtype, args.max_cache_shards)

    carry_latents: torch.Tensor | None = None
    carry_labels: torch.Tensor | None = None
    written_samples = 0
    output_index = 0

    for block_start in tqdm(range(0, effective_total, block_size), desc="writing blocks", unit="block"):
        block_end = min(block_start + block_size, effective_total)
        block_indices = perm[block_start:block_end]
        block_pairs = source_index.index_select(0, block_indices)
        block_len = block_pairs.shape[0]

        latents_block = torch.empty((block_len, *latent_shape), dtype=latent_dtype)
        labels_block = torch.empty((block_len,), dtype=torch.long)

        shard_ids = torch.unique(block_pairs[:, 0]).tolist()
        for shard_id in shard_ids:
            positions = (block_pairs[:, 0] == shard_id).nonzero(as_tuple=False).squeeze(1).to(torch.long)
            local_indices = block_pairs.index_select(0, positions)[:, 1].to(torch.long)
            latents, labels = cache.get(shard_id)
            latents_block.index_copy_(0, positions, latents.index_select(0, local_indices))
            labels_block.index_copy_(0, positions, labels.index_select(0, local_indices))

        block_perm = torch.randperm(block_len, generator=generator)
        latents_block = latents_block.index_select(0, block_perm)
        labels_block = labels_block.index_select(0, block_perm)

        if carry_latents is not None:
            if carry_labels is None:
                raise RuntimeError("carry_labels missing while carry_latents is populated")
            latents_block = torch.cat([carry_latents, latents_block], dim=0)
            labels_block = torch.cat([carry_labels, labels_block], dim=0)
            carry_latents = None
            carry_labels = None

        total_in_buffer = latents_block.shape[0]
        cursor = 0
        while cursor + chunk_size <= total_in_buffer:
            latents_slice = latents_block[cursor : cursor + chunk_size].clone()
            labels_slice = labels_block[cursor : cursor + chunk_size].clone()
            payload: dict[str, torch.Tensor | float] = {
                "latents": latents_slice,
                "labels": labels_slice,
            }
            if scaling_factor is not None:
                payload["scaling_factor"] = scaling_factor
            out_path = output_root / f"{args.prefix}{output_index:05d}.pt"
            torch.save(payload, out_path)
            written_samples += chunk_size
            output_index += 1
            cursor += chunk_size

        if cursor < total_in_buffer:
            carry_latents = latents_block[cursor:].clone()
            carry_labels = labels_block[cursor:].clone()

        del latents_block
        del labels_block

    if carry_latents is not None:
        leftover = carry_latents.shape[0]
        if leftover and not args.drop_tail:
            raise RuntimeError(
                f"Shuffle completed with {leftover} leftover samples. "
                "Re-run with --drop-tail or choose a chunk-size that divides the dataset."
            )
        written_samples += leftover
        output_index += leftover // chunk_size

    print(
        "Completed reshuffle: "
        f"wrote {output_index} shard(s) containing {written_samples} samples to {output_root}"
    )
    if tail_samples:
        print(f"Dropped tail samples: {tail_samples} (from original total {total_samples})")


if __name__ == "__main__":
    main()
