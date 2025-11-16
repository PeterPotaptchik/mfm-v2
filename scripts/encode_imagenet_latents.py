#!/usr/bin/env python
"""Encode ImageNet images into Stable Diffusion VAE latents and save sharded tensors.

Usage (single node, 4 GPUs):
    torchrun --nproc_per_node=4 scripts/encode_imagenet_latents.py \
        --data-dir /path/to/imagenet21k_resized/raw \
        --split imagenet21k_resized/imagenet21k_train \
        --output-dir /path/to/output/latents \
        --autoencoder stabilityai/sd-vae-ft-mse \
        --pixel-resolution 256 \
        --batch-size 64 \
        --shard-size 2048

The script mirrors the ImageNet folder hierarchy in the output directory by writing
sharded `.pt` files (torch.save dictionaries) containing latents, labels, and
originating image paths. Run separately for each split (train, val, etc.).
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder

try:
    from diffusers import AutoencoderKL
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "diffusers is required; install with `pip install diffusers` before running this script."
    ) from exc


class ImageFolderWithPaths(ImageFolder):
    """ImageFolder that also returns the original file path."""

    def __getitem__(self, index: int):
        image, label = super().__getitem__(index)
        path = self.samples[index][0]
        return image, label, path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=str, required=True, help="Root directory containing the raw images.")
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        help="Relative path from data-dir to the split (e.g., imagenet21k_resized/imagenet21k_train).",
    )
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to store latent shards.")
    parser.add_argument(
        "--autoencoder",
        type=str,
        default="stabilityai/sd-vae-ft-mse",
        help="Stable Diffusion VAE identifier or local path.",
    )
    parser.add_argument("--pixel-resolution", type=int, default=256, help="Resize/Crop resolution for input images.")
    parser.add_argument("--batch-size", type=int, default=64, help="Per-process batch size for encoding.")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of dataloader workers per process.")
    parser.add_argument(
        "--dtype",
        type=str,
        choices=("fp32", "fp16", "bf16"),
        default="fp16",
        help="Precision to store latents on disk (affects storage size).",
    )
    parser.add_argument(
        "--sample-mode",
        action="store_true",
        help="Sample from the VAE posterior instead of using the mean (mode).",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=2048,
        help="Number of images per saved shard (set 0 to write one file per image).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing shards (otherwise skip shards that already exist).",
    )
    parser.add_argument(
        "--save-paths",
        action="store_true",
        help="Include original relative paths in the shard payload for traceability.",
    )
    return parser.parse_args()


def init_distributed() -> Tuple[int, int]:
    """Initialise torch.distributed if launched via torchrun."""
    if "RANK" not in os.environ:
        return 0, 1

    torch.distributed.init_process_group(backend="nccl")
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    return rank, world_size


def main() -> None:
    args = parse_args()
    rank, world_size = init_distributed()
    device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")

    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    storage_dtype = dtype_map[args.dtype]

    if rank == 0:
        print(f"Using device {device}, world_size={world_size}")
        print(f"Encoding split '{args.split}' from {args.data_dir} -> {args.output_dir}")

    transform = transforms.Compose(
        [
            transforms.Resize(args.pixel_resolution),
            transforms.CenterCrop(args.pixel_resolution),
            transforms.ToTensor(),
        ]
    )

    dataset_root = os.path.join(args.data_dir, args.split)
    dataset = ImageFolderWithPaths(dataset_root, transform=transform)

    sampler = (
        DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=False if sampler else True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )

    if sampler is not None:
        sampler.set_epoch(0)

    autoencoder = AutoencoderKL.from_pretrained(args.autoencoder, torch_dtype=torch.float16)
    autoencoder = autoencoder.to(device)
    autoencoder.eval()
    scaling_factor = getattr(autoencoder.config, "scaling_factor", 0.18215)

    output_root = pathlib.Path(args.output_dir) / args.split
    output_root.mkdir(parents=True, exist_ok=True)

    shard_latents: List[torch.Tensor] = []
    shard_labels: List[int] = []
    shard_paths: List[str] = []
    shard_idx = 0

    def save_shard() -> None:
        nonlocal shard_idx
        if not shard_latents:
            return
        latents_tensor = torch.stack(shard_latents).to(storage_dtype)
        labels_tensor = torch.tensor(shard_labels, dtype=torch.int64)
        payload: Dict[str, torch.Tensor | List[str]] = {
            "latents": latents_tensor,
            "labels": labels_tensor,
            "scaling_factor": torch.tensor(scaling_factor, dtype=torch.float32),
        }
        if args.save_paths:
            payload["paths"] = shard_paths.copy()
        shard_name = f"rank{rank:02d}_shard{shard_idx:05d}.pt"
        shard_path = output_root / shard_name
        if shard_path.exists() and not args.overwrite:
            if rank == 0:
                print(f"Skipping existing shard {shard_path}")
        else:
            torch.save(payload, shard_path)
            if rank == 0:
                print(f"Wrote {shard_path} with {len(shard_latents)} samples")
        shard_latents.clear()
        shard_labels.clear()
        shard_paths.clear()
        shard_idx += 1

    with torch.inference_mode():
        for step, (images, labels, paths) in enumerate(loader):
            images = images.to(device, dtype=torch.float16)
            images = images * 2.0 - 1.0
            posterior = autoencoder.encode(images)
            if args.sample_mode:
                latents = posterior.latent_dist.sample()
            else:
                latents = posterior.latent_dist.mode()
            latents = latents * scaling_factor
            latents = latents.to(torch.float32).cpu()

            for latent, label, path in zip(latents, labels, paths):
                shard_latents.append(latent)
                shard_labels.append(int(label))
                if args.save_paths:
                    rel_path = os.path.relpath(path, dataset_root)
                    shard_paths.append(rel_path)

                if args.shard_size > 0 and len(shard_latents) >= args.shard_size:
                    save_shard()

            if rank == 0 and step % 50 == 0:
                processed = step * args.batch_size * world_size
                total = len(dataset)
                print(f"Processed approx. {processed}/{total} images")

    save_shard()

    if world_size > 1:
        torch.distributed.barrier()

    if rank == 0:
        print("Encoding complete.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
