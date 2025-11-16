#!/usr/bin/env python
"""Visualize original images alongside their decoded latents for sanity checking."""
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms

try:
    from diffusers import AutoencoderKL
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install diffusers to decode latents (pip install diffusers).") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latents-root", type=str, required=True, help="Directory containing latent shard .pt files.")
    parser.add_argument("--data-dir", type=str, required=True, help="Root directory with the original images.")
    parser.add_argument("--split", type=str, required=True, help="Relative split path (e.g. imagenet21k_resized/imagenet21k_train).")
    parser.add_argument("--autoencoder", type=str, default="stabilityai/sd-vae-ft-mse", help="Autoencoder identifier or local path.")
    parser.add_argument("--device", type=str, default="cuda", help="Device for decoding (cuda, cuda:0, cpu, ...).")
    parser.add_argument("--num-images", type=int, default=8, help="Number of samples to visualize.")
    parser.add_argument("--pixel-resolution", type=int, default=256, help="Resize/Crop resolution that was used during encoding.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for reproducibility.")
    parser.add_argument("--save", type=str, default="decoded_pairs.png", help="Path to save the side-by-side grid.")
    return parser.parse_args()


def list_shards(latents_root: Path) -> List[Path]:
    shards = sorted(latents_root.glob("*.pt"))
    if not shards:
        raise FileNotFoundError(f"No .pt shards found under {latents_root}")
    return shards


def sample_latents(shards: List[Path], num_images: int, rng: random.Random) -> Tuple[torch.Tensor, List[str], float]:
    picked_latents: List[torch.Tensor] = []
    picked_paths: List[str] = []
    scaling_factor: float | None = None
    remaining = num_images
    while remaining > 0:
        shard_path = rng.choice(shards)
        payload = torch.load(shard_path, map_location="cpu")
        latents = payload["latents"].to(torch.float32)
        paths = payload.get("paths")
        if paths is None:
            raise KeyError(f"Shard {shard_path} has no 'paths'; re-run encoding with --save-paths.")
        shard_scaling = payload.get("scaling_factor", 1.0)
        shard_scaling = float(shard_scaling.item()) if torch.is_tensor(shard_scaling) else float(shard_scaling)
        if scaling_factor is None:
            scaling_factor = shard_scaling
        elif abs(scaling_factor - shard_scaling) > 1e-6:
            raise ValueError(f"Scaling factor mismatch across shards: {scaling_factor} vs {shard_scaling}")

        indices = torch.randperm(latents.shape[0])[: min(remaining, latents.shape[0])]
        picked_latents.append(latents[indices])
        picked_paths.extend(paths[i] for i in indices.tolist())
        remaining -= indices.numel()
    assert scaling_factor is not None
    stacked = torch.cat(picked_latents, dim=0)[:num_images]
    return stacked, picked_paths[:num_images], scaling_factor


def load_originals(data_root: Path, rel_paths: List[str], pixel_resolution: int) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.Resize(pixel_resolution),
            transforms.CenterCrop(pixel_resolution),
            transforms.ToTensor(),
        ]
    )
    images = []
    for rel_path in rel_paths:
        img_path = data_root / rel_path
        if not img_path.exists():
            raise FileNotFoundError(f"Original image not found: {img_path}")
        with Image.open(img_path) as img:
            images.append(transform(img.convert("RGB")))
    return torch.stack(images)


def decode_latents(autoencoder: AutoencoderKL, latents: torch.Tensor, scaling_factor: float, device: torch.device) -> torch.Tensor:
    latents = latents.to(device)
    with torch.inference_mode():
        decoded = autoencoder.decode(latents / scaling_factor).sample
    decoded = decoded.clamp(-1.0, 1.0)
    decoded = (decoded + 1.0) / 2.0
    return decoded.cpu()


def plot_side_by_side(originals: torch.Tensor, reconstructions: torch.Tensor, save_path: str) -> None:
    num_images = originals.shape[0]
    originals_np = originals.permute(0, 2, 3, 1).numpy()
    recon_np = reconstructions.permute(0, 2, 3, 1).numpy()
    fig, axes = plt.subplots(num_images, 2, figsize=(6, 3 * num_images))
    if num_images == 1:
        axes = axes.reshape(1, 2)
    for idx in range(num_images):
        axes[idx, 0].imshow(originals_np[idx])
        axes[idx, 0].set_title("Original")
        axes[idx, 0].axis("off")
        axes[idx, 1].imshow(recon_np[idx])
        axes[idx, 1].set_title("Decoded")
        axes[idx, 1].axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved comparison grid to {save_path}")
    plt.show()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    latents_root = Path(args.latents_root) / args.split
    data_root = Path(args.data_dir) / args.split

    shards = list_shards(latents_root)
    latents, rel_paths, scaling_factor = sample_latents(shards, args.num_images, rng)

    originals = load_originals(data_root, rel_paths, args.pixel_resolution)

    device = torch.device(args.device)
    print(f"Loading autoencoder '{args.autoencoder}' on {device}")
    autoencoder = AutoencoderKL.from_pretrained(args.autoencoder, torch_dtype=torch.float32).to(device)
    autoencoder.eval()

    recon = decode_latents(autoencoder, latents, scaling_factor, device)
    plot_side_by_side(originals, recon, args.save)


if __name__ == "__main__":
    main()
