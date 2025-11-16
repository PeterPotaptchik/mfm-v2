#!/usr/bin/env python
"""Inspect class distributions inside ImageNet latent shards.

Usage:
    python scripts/check_latent_shard_classes.py --root cache/latents/train
"""

import argparse
from collections import Counter
from pathlib import Path

import torch


def load_shard(path: Path) -> Counter:
    payload = torch.load(path, map_location="cpu")
    labels = payload["labels"].tolist()
    return Counter(labels)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise class counts per latent shard.")
    parser.add_argument("--root", required=True, help="Directory containing *.pt shards")
    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="Show the top/bottom-k classes per shard (default: 5)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit the number of shards to inspect (0 = all)",
    )
    args = parser.parse_args()

    root = Path(args.root)
    shard_paths = sorted(root.glob("*.pt"))
    if args.limit:
        shard_paths = shard_paths[: args.limit]

    if not shard_paths:
        raise SystemExit(f"No shard files found under {root}")

    global_counts: Counter[int] = Counter()
    shard_summaries: list[tuple[str, int, Counter[int]]] = []

    for shard_path in shard_paths:
        counts = load_shard(shard_path)
        total = sum(counts.values())
        global_counts.update(counts)
        shard_summaries.append((shard_path.name, total, counts))

    num_classes = len(global_counts)
    total_samples = sum(global_counts.values())

    print(f"Scanned {len(shard_summaries)} shards under {root}")
    print(f"Total samples: {total_samples}")
    print(f"Classes observed: {num_classes}")
    print("Global per-class sample counts (top 10):")
    for cls, n in global_counts.most_common(10):
        print(f"  class {cls:>4}: {n}")

    print("\nPer-shard summaries:")
    for name, total, counts in shard_summaries:
        top = counts.most_common(args.topk)
        bottom = counts.most_common()[:-args.topk - 1:-1]
        print(f"- {name}: {total} samples")
        print("  Top classes:")
        for cls, n in top:
            frac = n / total
            print(f"    class {cls:>4}: {n} ({frac:.2%})")
        print("  Bottom classes:")
        for cls, n in bottom:
            frac = n / total
            print(f"    class {cls:>4}: {n} ({frac:.2%})")
        print()


if __name__ == "__main__":
    main()
