import os
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    # Training
    epochs: int = 100
    batch_size: int = 128
    num_workers: int = 8
    grad_clip: float = 1.0
    seed: int = 42

    # Model
    num_classes: int = 1000

    # Optimizer
    max_lr: float = 0.23
    momentum: float = 0.95
    weight_decay: float = 1e-4
    nesterov: bool = True

    # Scheduler (OneCycleLR)
    pct_start: float = 0.3
    div_factor: float = 25.0
    final_div_factor: float = 100.0
    base_momentum: float = 0.85
    max_momentum: float = 0.95

    # Loss
    label_smoothing: float = 0.1

    # Augmentation
    mixup_cutmix_alpha: float = 0.2
    cutmix_prob: float = 0.5
    randaugment_num_ops: int = 2
    randaugment_magnitude: int = 9
    random_erasing_prob: float = 0.25

    # Data
    image_size: int = 224
    resize_size: int = 256
    mean: tuple = (0.485, 0.456, 0.406)
    std: tuple = (0.229, 0.224, 0.225)
    train_dir: str = "data/train"
    val_dir: str = "data/val"

    # DataLoader
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2

    # DDP
    backend: str = "nccl"
    find_unused_parameters: bool = False

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_freq: int = 10

    # Logging
    log_freq: int = 50

    # AMP
    use_amp: bool = True


def get_config():
    return TrainingConfig()
