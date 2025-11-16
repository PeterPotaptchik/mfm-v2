import os
import bisect
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import lightning as pl
from torch.utils.data import DataLoader, Dataset, IterableDataset, random_split
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10
from distcfm.data.toy import ToyDataModule

from diffusers import AutoencoderKL

from typing import Iterator, Sequence

TOY_DATASETS = ["8gaussians", "gmm", "checkerboard"]

def get_data_module(cfg):
    """Factory function to instantiate a data module from its config."""
    print("Loading data module for dataset:", cfg.dataset.name)
    print("Data directory:", cfg.data_dir)
    if cfg.dataset.name == "mnist":
        return MNISTDataModule(cfg)
    elif cfg.dataset.name == "cifar10":
        return CIFAR10DataModule(cfg)
    elif cfg.dataset.name in TOY_DATASETS:
        return ToyDataModule(cfg)
    elif cfg.dataset.name == "imagenet":
        return ImageNetDataModule(cfg)
    elif cfg.dataset.name == "imagenet_latent":
        return ImageNetPrecomputedLatentDataModule(cfg)

def image_scaler(data): 
    """Assumes data is in [0, 1] range and scales to [-1, 1]."""
    return (data - 0.5) * 2

def inverse_image_scaler(data):
    """Inverse of image_scaler, scales from [-1, 1] back to [0, 1]."""
    return (data / 2) + 0.5

class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.data_dir = cfg.data_dir
        self.batch_size = cfg.trainer.batch_size
        self.scaler = image_scaler
        self.inverse_scaler = inverse_image_scaler
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(), 
                transforms.Lambda(self.scaler),  # Scale to [-1, 1]
            ]
        )
        self.dims = (1, 28, 28)

    def prepare_data(self): # only called on main process
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None): # called on every process
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        if stage == "test" or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform,
                                    download=True,)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, shuffle=True,
                          num_workers=4, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, 
                          num_workers=4, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size,
                          num_workers=4, pin_memory=True)

class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.data_dir = cfg.data_dir
        self.batch_size = cfg.trainer.batch_size
        self.scaler = image_scaler
        self.inverse_scaler = inverse_image_scaler
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(self.scaler),  # Scale to [-1, 1]
            ]
        )
        self.dims = (3, 32, 32)

    def prepare_data(self): # only called on main process
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None): # called on every process
        if stage == "fit" or stage is None:
            cifar_full = CIFAR10(self.data_dir, train=True, transform=self.transform, download=True)
            self.cifar_train, self.cifar_val = random_split(cifar_full, [45000, 5000])

        if stage == "test" or stage is None:
            self.cifar_test = CIFAR10(self.data_dir, train=False, transform=self.transform, download=True)

    def train_dataloader(self):
        return DataLoader(self.cifar_train, batch_size=self.batch_size, shuffle=True,
                          num_workers=4, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.cifar_val, batch_size=self.batch_size,
                          num_workers=4, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.cifar_test, batch_size=self.batch_size,
                          num_workers=4, pin_memory=True)

class ImageNetDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.data_dir = cfg.data_dir
        self.batch_size = cfg.trainer.batch_size
        self.num_workers = cfg.trainer.num_workers 
        self.resolution = cfg.dataset.img_resolution
        self.scaler = image_scaler
        self.inverse_scaler = inverse_image_scaler
        # Standard ImageNet normalization, equivalent to scaling
        self.transform = transforms.Compose(
            [
                transforms.Resize(self.resolution),
                transforms.CenterCrop(self.resolution),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.dims = (3, self.resolution, self.resolution)

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        # This method is called on every process (every GPU).
        # ImageNet is typically split into 'train' and 'val' directories.
        if stage == "fit" or stage is None:
            train_dir = os.path.join(self.data_dir, "train")
            val_dir = os.path.join(self.data_dir, "val")
            self.imagenet_train = ImageFolder(train_dir, transform=self.transform)
            self.imagenet_val = ImageFolder(val_dir, transform=self.transform)

        if stage == "test" or stage is None:
            val_dir = os.path.join(self.data_dir, "val")
            # The validation set is commonly used as the test set for ImageNet
            self.imagenet_test = ImageFolder(val_dir, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(
            self.imagenet_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.imagenet_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.imagenet_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
        )

@dataclass
class _LatentShardMetadata:
    shard_paths: list[Path]
    shard_lengths: list[int]
    shard_offsets: list[int]
    total_length: int
    latent_shape: tuple[int, ...]
    scaling_factor: float | None


def _inspect_latent_shards(root: Path) -> _LatentShardMetadata:
    if not root.exists():
        raise FileNotFoundError(f"Latent shard directory not found: {root}")

    shard_paths = sorted(root.glob("*.pt"))
    if not shard_paths:
        raise FileNotFoundError(f"No latent shards (*.pt) found under {root}")

    shard_lengths: list[int] = []
    shard_offsets: list[int] = []
    latent_shape: tuple[int, ...] | None = None
    scaling_factor: float | None = None

    first_payload = torch.load(shard_paths[0], map_location="cpu")
    first_latents = first_payload["latents"]
    base_length = first_latents.shape[0]
    latent_shape = tuple(first_latents.shape[1:])
    scaling_val = first_payload.get("scaling_factor")
    if scaling_val is not None:
        scaling_factor = float(scaling_val)
    del first_latents
    del first_payload

    base_size = shard_paths[0].stat().st_size

    total = 0
    for idx, shard_path in enumerate(shard_paths):
        if idx == 0:
            length = base_length
        else:
            current_size = shard_path.stat().st_size
            if current_size == base_size:
                length = base_length
            else:
                payload = torch.load(shard_path, map_location="cpu")
                latents = payload["latents"]
                length = latents.shape[0]
                if scaling_factor is None and payload.get("scaling_factor") is not None:
                    scaling_factor = float(payload["scaling_factor"])
                del payload
        total += length
        shard_lengths.append(length)
        shard_offsets.append(total)

    if latent_shape is None:
        raise RuntimeError(f"Failed to infer latent shape from shards under {root}")

    return _LatentShardMetadata(
        shard_paths=shard_paths,
        shard_lengths=shard_lengths,
        shard_offsets=shard_offsets,
        total_length=total,
        latent_shape=latent_shape,
        scaling_factor=scaling_factor,
    )


def _load_shard_payload(shard_path: Path, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, float | None]:
    payload = torch.load(shard_path, map_location="cpu")
    latents = payload["latents"].to(dtype=dtype, copy=False)
    labels = payload["labels"].long()
    scaling_factor = payload.get("scaling_factor")
    return latents.contiguous(), labels.contiguous(), None if scaling_factor is None else float(scaling_factor)


class LatentShardDataset(Dataset):
    """Dataset that lazily loads VAE latents stored in sharded .pt files."""

    def __init__(self, root: str | Path, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.root = Path(root)
        self.dtype = dtype

        metadata = _inspect_latent_shards(self.root)
        self.shard_paths = metadata.shard_paths
        self.shard_lengths = metadata.shard_lengths
        self.shard_offsets = metadata.shard_offsets
        self.total_length = metadata.total_length
        self.latent_shape = metadata.latent_shape
        self.scaling_factor = metadata.scaling_factor

        self._cache_index: int | None = None
        self._cache_latents: torch.Tensor | None = None
        self._cache_labels: torch.Tensor | None = None

    def __len__(self) -> int:
        return self.total_length

    def _load_shard(self, shard_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if shard_idx == self._cache_index and self._cache_latents is not None:
            return self._cache_latents, self._cache_labels  # type: ignore[return-value]

        shard_path = self.shard_paths[shard_idx]
        latents, labels, scaling_factor = _load_shard_payload(shard_path, self.dtype)
        if scaling_factor is not None and self.scaling_factor is None:
            self.scaling_factor = scaling_factor
        self._cache_index = shard_idx
        self._cache_latents = latents
        self._cache_labels = labels
        return latents, labels

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0 or index >= self.total_length:
            raise IndexError(f"Index {index} out of range for dataset of size {self.total_length}")

        shard_idx = bisect.bisect_right(self.shard_offsets, index)
        prev_offset = 0 if shard_idx == 0 else self.shard_offsets[shard_idx - 1]
        local_index = index - prev_offset
        latents, labels = self._load_shard(shard_idx)
        latent = latents[local_index]
        label = labels[local_index]
        return latent, label


def _distributed_rank_and_world() -> tuple[int, int]:
    """Best-effort helper to learn the distributed rank/world size from env."""

    rank: int | None = None
    world_size: int | None = None

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        try:
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        except RuntimeError:
            rank = None
            world_size = None

    if rank is None:
        rank_env = os.environ.get("RANK")
        if rank_env is not None and rank_env.isdigit():
            rank = int(rank_env)
        else:
            rank = 0

    if world_size is None:
        world_env = os.environ.get("WORLD_SIZE")
        if world_env is not None and world_env.isdigit():
            world_size = int(world_env)
        else:
            world_size = 1

    if world_size < 1:
        world_size = 1

    return rank, world_size


#TODO: Shuffle latents before to avoid same index shard stuff

class LatentShardIterableDataset(IterableDataset):
    """Iterable dataset that ensures each worker loads a disjoint set of shards."""

    def __init__(
        self,
        root: str | Path,
        dtype: torch.dtype = torch.float32,
        *,
        shuffle: bool = True,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.dtype = dtype
        self.shuffle = shuffle

        metadata = _inspect_latent_shards(self.root)
        self.shard_paths = metadata.shard_paths
        self.shard_lengths = metadata.shard_lengths
        self.total_length = metadata.total_length
        self.latent_shape = metadata.latent_shape
        self.scaling_factor = metadata.scaling_factor
        rank, world_size = _distributed_rank_and_world()
        self._rank = rank
        self._world_size = world_size

    def __len__(self) -> int:
        if self._world_size is None or self._world_size < 1:
            world_size = 1
        else:
            world_size = self._world_size

        usable_shards = (len(self.shard_paths) // world_size) * world_size
        if usable_shards == 0:
            return self.total_length // world_size

        shard_lengths = self.shard_lengths[:usable_shards]
        per_rank = usable_shards // world_size
        per_rank_lengths = [
            sum(
                shard_lengths[r * per_rank + offset]
                for offset in range(per_rank)
            )
            for r in range(world_size)
        ]
        return min(per_rank_lengths)

    def _shard_indices(
        self,
        worker_id: int,
        num_workers: int,
        generator: torch.Generator,
        *,
        rank: int,
        world_size: int,
    ) -> Sequence[int]:
        indices = list(range(len(self.shard_paths)))
        if self.shuffle:
            perm = torch.randperm(len(indices), generator=generator).tolist()
            indices = perm
        if world_size < 1:
            world_size = 1

        usable_shards = (len(indices) // world_size) * world_size
        if usable_shards == 0:
            usable_shards = len(indices)
        indices = indices[:usable_shards]

        shards_per_rank = usable_shards // world_size
        rank_start = rank * shards_per_rank
        rank_end = min(rank_start + shards_per_rank, len(indices))
        rank_indices = indices[rank_start:rank_end]

        if num_workers <= 1:
            return rank_indices

        return rank_indices[worker_id::num_workers]

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
            base_seed = torch.initial_seed()
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            base_seed = worker_info.seed

        rank, world_size = self._rank, self._world_size
        if rank is None or world_size is None:
            rank, world_size = _distributed_rank_and_world()
            self._rank = rank
            self._world_size = world_size

        generator = torch.Generator()
        generator.manual_seed(base_seed)

        for shard_idx in self._shard_indices(
            worker_id,
            num_workers,
            generator,
            rank=rank,
            world_size=world_size,
        ):
            latents, labels, scaling_factor = _load_shard_payload(self.shard_paths[shard_idx], self.dtype)
            if scaling_factor is not None and self.scaling_factor is None:
                self.scaling_factor = scaling_factor

            sample_order: Sequence[int]
            if self.shuffle:
                perm = torch.randperm(latents.shape[0], generator=generator).tolist()
                sample_order = perm
            else:
                sample_order = range(latents.shape[0])

            for local_index in sample_order:
                yield latents[local_index], labels[local_index]

            del latents
            del labels

class ImageNetPrecomputedLatentDataModule(pl.LightningDataModule):
    """Lightning datamodule for precomputed ImageNet latents stored as shards."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.latents_root = Path(cfg.data_dir)
        self.batch_size = cfg.trainer.batch_size
        self.num_workers = cfg.trainer.num_workers
        self.train_split = getattr(cfg.dataset, "train_split", "train")
        self.val_split = getattr(cfg.dataset, "val_split", "val")
        self.latent_dtype = getattr(cfg.dataset, "latent_dtype", "float32")
        self.img_channels = getattr(cfg.dataset, "img_channels", 4)
        self.img_resolution = getattr(cfg.dataset, "img_resolution", 32)
        self.scaler = lambda x: x
        self.scaling_factor: float | None = None
        self.autoencoder_id = getattr(cfg.dataset, "autoencoder", "stabilityai/sd-vae-ft-mse")
        self.autoencoder_dtype = torch.float32
        self.autoencoder: AutoencoderKL | None = None
        self.decode_chunk_size = getattr(cfg.dataset, "decode_chunk_size", 16)
        self.inverse_scaler = self._inverse_scaler
        self.train_dataset: LatentShardIterableDataset | None = None
        self.val_dataset: LatentShardDataset | None = None
        self.test_dataset: LatentShardDataset | None = None
        self.dims = (self.img_channels, self.img_resolution, self.img_resolution)

    def _dtype_from_str(self, dtype_str: str) -> torch.dtype:
        mapping = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        if dtype_str not in mapping:
            raise ValueError(f"Unsupported latent_dtype '{dtype_str}'")
        return mapping[dtype_str]

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        dtype = self._dtype_from_str(self.latent_dtype)
        if self.autoencoder is None:
            self.autoencoder = AutoencoderKL.from_pretrained(self.autoencoder_id, torch_dtype=self.autoencoder_dtype)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.autoencoder = self.autoencoder.to(device)
            self.autoencoder.eval()
        if stage in (None, "fit"):
            train_dir = self.latents_root / self.train_split
            val_dir = self.latents_root / self.val_split
            self.train_dataset = LatentShardIterableDataset(train_dir, dtype=dtype, shuffle=True)
            self.val_dataset = LatentShardDataset(val_dir, dtype=dtype)
            self.dims = self.train_dataset.latent_shape
            if self.train_dataset.scaling_factor is None:
                raise RuntimeError("Latent shards missing scaling_factor metadata")
            self.scaling_factor = float(self.train_dataset.scaling_factor)
        if stage in (None, "test"):
            val_dir = self.latents_root / self.val_split
            self.test_dataset = LatentShardDataset(val_dir, dtype=dtype)
            if self.test_dataset.scaling_factor is None:
                raise RuntimeError("Latent shards missing scaling_factor metadata")
            self.scaling_factor = float(self.test_dataset.scaling_factor)

    def train_dataloader(self):
        if self.train_dataset is None:
            raise RuntimeError("DataModule.setup() must be called before requesting dataloaders.")
        dl_kwargs = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": True,
            "persistent_workers": self.num_workers > 0,
            "drop_last": True,
        }
        if self.num_workers > 0:
            dl_kwargs["prefetch_factor"] = 4
        return DataLoader(self.train_dataset, shuffle=False, **dl_kwargs)

    def val_dataloader(self):
        if self.val_dataset is None:
            raise RuntimeError("DataModule.setup() must be called before requesting dataloaders.")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        if self.test_dataset is None:
            raise RuntimeError("DataModule.setup() must be called before requesting dataloaders.")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def _inverse_scaler(self, latents) -> torch.Tensor:
        if self.autoencoder is None:
            raise RuntimeError("Autoencoder not initialised; call setup() first.")
        if not torch.is_tensor(latents):
            latents = torch.from_numpy(latents)

        if latents.dim() < 3:
            raise ValueError(f"Expected latent tensor with at least 3 dims, got shape {latents.shape}")

        device = next(self.autoencoder.parameters()).device
        leading_shape = latents.shape[:-3]
        latents_flat = latents.reshape(-1, *latents.shape[-3:])
        latents_flat = latents_flat.to(device=device, dtype=self.autoencoder_dtype)

        decoded_chunks: list[torch.Tensor] = []
        chunk_size = max(1, int(self.decode_chunk_size))
        with torch.inference_mode():
            for chunk in latents_flat.split(chunk_size, dim=0):
                decoded = self.autoencoder.decode(chunk / self.scaling_factor).sample
                decoded_chunks.append(decoded)

        decoded_flat = torch.cat(decoded_chunks, dim=0).to(dtype=torch.float32)
        decoded_flat = decoded_flat.clamp(-1.0, 1.0)
        decoded_flat = (decoded_flat + 1.0) / 2.0
        decoded = decoded_flat.reshape(*leading_shape, *decoded_flat.shape[-3:])
        return decoded.cpu()

