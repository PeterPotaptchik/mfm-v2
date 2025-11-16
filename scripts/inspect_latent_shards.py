#!/usr/bin/env python
"""Randomly visualize decoded latents from precomputed shards.

Example usage:
    python scripts/inspect_latent_shards.py \
        --latents-root cache/latents/imagenet21k_resized/imagenet21k_train \
        --autoencoder stabilityai/sd-vae-ft-mse \
        --num-images 16

The script loads random latent shards, decodes a handful of samples through
the VAE, and displays them in a matplotlib grid for a quick sanity check.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms

try:
    from diffusers import AutoencoderKL
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "diffusers is required. Install with `pip install diffusers` before running this script."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latents-root", type=str, required=True, help="Directory containing latent shard .pt files.")
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Root directory with original images (same root used when encoding).",
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        help="Split subdirectory relative to data-dir (e.g., imagenet21k_resized/imagenet21k_train).",
    )
    parser.add_argument("--autoencoder", type=str, default="stabilityai/sd-vae-ft-mse", help="Autoencoder identifier or path.")
    parser.add_argument("--device", type=str, default="cuda", help="Device for decoding (cuda, cuda:0, cpu, etc.).")
    parser.add_argument("--num-images", type=int, default=16, help="Number of images to sample and display.")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for reproducibility.")
    parser.add_argument(
        "--pixel-resolution",
        type=int,
        default=256,
        help="Resize/Crop resolution to match encoded inputs (default: 256).",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="decoded_latents.png",
        help="Path to save the visualization image (default: decoded_latents.png).",
    )
    return parser.parse_args()


def load_random_latents(latent_dir: Path, max_images: int) -> tuple[torch.Tensor, List[str], float]:
    shard_paths: List[Path] = sorted(p for p in latent_dir.glob("*.pt"))
    if not shard_paths:
        raise FileNotFoundError(f"No .pt shards found under {latent_dir}")

    accumulated = []
    remaining = max_images
    collected_paths: List[str] = []
    scaling_factor: float | None = None
    random.shuffle(shard_paths)
    for shard in shard_paths:
        payload = torch.load(shard, map_location="cpu")
        latents = payload["latents"]
        shard_paths_rel = payload.get("paths")
        if shard_paths_rel is None:
            raise KeyError(
                f"Shard {shard} does not contain 'paths'. Re-run encoding with --save-paths to enable comparisons."
            )
        shard_scaling = payload.get("scaling_factor")
        shard_scaling = float(shard_scaling.item()) if torch.is_tensor(shard_scaling) else float(shard_scaling)
        if scaling_factor is None:
            scaling_factor = shard_scaling
        elif not torch.isclose(torch.tensor(scaling_factor), torch.tensor(shard_scaling), atol=1e-6):
            raise ValueError("Inconsistent scaling_factor across shards.")

        num_take = min(remaining, latents.shape[0])
        indices = torch.randperm(latents.shape[0])[:num_take]
        accumulated.append(latents[indices])
        collected_paths.extend([shard_paths_rel[i] for i in indices.tolist()])
        remaining -= num_take
        if remaining <= 0:
            break
    if not accumulated:
        raise RuntimeError("Failed to sample any latents; check shard contents.")
    assert scaling_factor is not None
    return torch.cat(accumulated, dim=0), collected_paths[:max_images], scaling_factor


def decode_images(autoencoder: AutoencoderKL, latents: torch.Tensor, device: torch.device, scaling_factor: float) -> torch.Tensor:
    latents = latents.to(device, dtype=torch.float32)
    with torch.inference_mode():
        decoded = autoencoder.decode(latents / scaling_factor).sample
    decoded = decoded.clamp(-1.0, 1.0)
    decoded = (decoded + 1.0) / 2.0
    return decoded.cpu()


def plot_side_by_side(originals: torch.Tensor, reconstructions: torch.Tensor, save_path: str) -> None:
    num_images = originals.shape[0]
    originals = originals.permute(0, 2, 3, 1).numpy()
    reconstructions = reconstructions.permute(0, 2, 3, 1).numpy()

    fig, axes = plt.subplots(num_images, 2, figsize=(6, 3 * num_images))
    if num_images == 1:
        axes = axes.reshape(1, 2)
    for idx in range(num_images):
        axes[idx, 0].imshow(originals[idx])
        axes[idx, 0].set_title("Original")
        axes[idx, 0].axis("off")
        axes[idx, 1].imshow(reconstructions[idx])
        axes[idx, 1].set_title("Decoded")
        axes[idx, 1].axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved visualization to {save_path}")
    plt.show()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    latent_dir = Path(args.latents_root)
    data_root = Path(args.data_dir) / args.split
    device = torch.device(args.device)

    print(f"Loading autoencoder '{args.autoencoder}' on {device}")
    vae = AutoencoderKL.from_pretrained(args.autoencoder, torch_dtype=torch.float32)
    vae = vae.to(device)
    vae.eval()

    print(f"Sampling {args.num_images} latents from {latent_dir}")
    latents, rel_paths, scaling_factor = load_random_latents(latent_dir, args.num_images)

    transform = transforms.Compose(
        [
            transforms.Resize(args.pixel_resolution),
            transforms.CenterCrop(args.pixel_resolution),
            transforms.ToTensor(),
        ]
    )
    originals = []
    for rel_path in rel_paths:
        img_path = data_root / rel_path
        if not img_path.exists():
            raise FileNotFoundError(f"Original image not found: {img_path}")
        with Image.open(img_path) as img:
            originals.append(transform(img.convert("RGB")))
    originals_tensor = torch.stack(originals)

    print("Decoding latents...")
    pixels = decode_images(vae, latents, device, scaling_factor)

    print("Rendering original vs decoded pairs...")
    plot_side_by_side(originals_tensor, pixels, args.save)


if __name__ == "__main__":
    main()
