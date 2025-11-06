import os
import torch
from torch.utils.data import DataLoader, DistributedSampler, Dataset
import torchvision.transforms as T
import torchvision.datasets as datasets
from torchvision.transforms import InterpolationMode as IM
from PIL import Image

from config import TrainingConfig


def get_train_transforms(cfg: TrainingConfig):
    return T.Compose([
        T.RandomResizedCrop(cfg.image_size, interpolation=IM.BICUBIC, antialias=True),
        T.RandomHorizontalFlip(),
        T.RandAugment(num_ops=cfg.randaugment_num_ops, magnitude=cfg.randaugment_magnitude),
        T.ToTensor(),
        T.Normalize(mean=cfg.mean, std=cfg.std),
        T.RandomErasing(p=cfg.random_erasing_prob, scale=(0.02, 0.33),
                        ratio=(0.3, 3.3), value=0),
    ])


def get_val_transforms(cfg: TrainingConfig):
    return T.Compose([
        T.Resize(cfg.resize_size, interpolation=IM.BICUBIC, antialias=True),
        T.CenterCrop(cfg.image_size),
        T.ToTensor(),
        T.Normalize(mean=cfg.mean, std=cfg.std),
    ])


class ParquetDataset(Dataset):
    """
    Dataset wrapper for Hugging Face parquet datasets.
    Supports lazy loading and efficient streaming of parquet files.
    """
    def __init__(self, parquet_data_dir, pattern, transform=None):
        """
        Args:
            parquet_data_dir: Directory containing parquet files
            pattern: Glob pattern for parquet files (e.g., "train-*.parquet")
            transform: Torchvision transforms to apply to images
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "The 'datasets' library is required to use parquet datasets. "
                "Install it with: pip install datasets"
            )

        self.transform = transform

        # Load parquet dataset from Hugging Face
        parquet_files = os.path.join(parquet_data_dir, pattern)

        # Load the dataset using Hugging Face datasets library
        # This will automatically handle multiple parquet files matching the pattern
        self.dataset = load_dataset(
            "parquet",
            data_files=parquet_files,
            split="train",  # Hugging Face uses 'train' split name for all data
            keep_in_memory=False  # Stream from disk to save memory
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get the item from the Hugging Face dataset
        item = self.dataset[idx]

        # Extract image and label
        # Hugging Face datasets typically store images as PIL Images
        image = item['image']

        # Convert to RGB if needed (handles grayscale, RGBA, etc.)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Get label (might be stored as 'label' or 'labels' depending on dataset)
        label = item.get('label', item.get('labels', 0))

        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)

        return image, label


def get_dataloaders(cfg: TrainingConfig, rank=None, world_size=None):
    train_tfms = get_train_transforms(cfg)
    val_tfms = get_val_transforms(cfg)

    # Choose dataset type based on configuration
    if cfg.use_parquet:
        if cfg.parquet_data_dir is None:
            raise ValueError(
                "parquet_data_dir must be specified when use_parquet=True"
            )

        # Use Hugging Face parquet datasets
        train_ds = ParquetDataset(
            cfg.parquet_data_dir,
            cfg.parquet_train_pattern,
            transform=train_tfms
        )
        val_ds = ParquetDataset(
            cfg.parquet_data_dir,
            cfg.parquet_val_pattern,
            transform=val_tfms
        )
    else:
        # Use traditional ImageFolder datasets
        train_ds = datasets.ImageFolder(cfg.train_dir, transform=train_tfms)
        val_ds = datasets.ImageFolder(cfg.val_dir, transform=val_tfms)

    if rank is not None and world_size is not None:
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
        )
        val_sampler = DistributedSampler(
            val_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
        )
        shuffle = False
    else:
        train_sampler = None
        val_sampler = None
        shuffle = True

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers,
        prefetch_factor=cfg.prefetch_factor,
        drop_last=True
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers,
        prefetch_factor=cfg.prefetch_factor,
        drop_last=False
    )

    return train_dl, val_dl, train_sampler, val_sampler
