import os
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    # Training
    epochs: int = 300  # ⚠️ CRITICAL: 100 epochs won't reach 78%. Use 300-600 for SOTA
    batch_size: int = 128
    num_workers: int = 8
    grad_clip: float = 1.0
    seed: int = 42

    # Model
    num_classes: int = 1000

    # Optimizer
    max_lr: float = 0.4  # Tuned for 300 epochs (run lr_finder to verify)
    momentum: float = 0.9  # Standard for longer training
    weight_decay: float = 1e-4
    nesterov: bool = True

    # Scheduler (OneCycleLR) - Adjusted for 300 epochs
    pct_start: float = 0.3
    div_factor: float = 25.0
    final_div_factor: float = 1e4  # More aggressive decay for longer training
    base_momentum: float = 0.85
    max_momentum: float = 0.95

    # Loss
    label_smoothing: float = 0.1  # Can try 0.15-0.2 for even stronger regularization

    # Augmentation - OPTIMIZED FOR 78%+ ACCURACY ON IMAGENET-1K (1.2M IMAGES)
    mixup_cutmix_alpha: float = 1.0  # ✅ CRITICAL: Changed from 0.2 to 1.0 (SOTA setting)
    cutmix_prob: float = 0.5
    randaugment_num_ops: int = 2
    randaugment_magnitude: int = 10  # Increased from 9 to 10 for 1.2M images
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
    log_freq: int = 50  # Log every N batches
    log_dir: str = "logs"
    experiment_name: str = None  # Auto-generated if None (timestamp)

    # AMP
    use_amp: bool = True


def get_config():
    return TrainingConfig()
