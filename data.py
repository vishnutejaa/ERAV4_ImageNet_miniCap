import torch
from torch.utils.data import DataLoader, DistributedSampler
import torchvision.transforms as T
import torchvision.datasets as datasets
from torchvision.transforms import InterpolationMode as IM

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


def get_dataloaders(cfg: TrainingConfig, rank=None, world_size=None):
    train_tfms = get_train_transforms(cfg)
    val_tfms = get_val_transforms(cfg)

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
