import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np


def mixup_cutmix(imgs, targets, alpha=1.0, cutmix_prob=0.5):
    """
    MixUp/CutMix augmentation optimized for ImageNet-1k (1.2M images).

    CRITICAL FOR 78%+ ACCURACY:
    - Use alpha=0.8-1.0 (NOT 0.2) for stronger regularization on large datasets
    - Higher alpha = more aggressive mixing = better generalization
    - CutMix is generally better than MixUp for ImageNet

    Args:
        imgs: Batch of images [B, C, H, W]
        targets: Target labels [B]
        alpha: Beta distribution parameter (0.8-1.0 for SOTA)
        cutmix_prob: Probability of CutMix vs MixUp (0.5 = balanced)

    Returns:
        mixed_imgs, targets, targets_permuted, lambda
    """
    if random.random() < cutmix_prob:
        # CutMix: Cut and paste patches between images
        lam = np.random.beta(alpha, alpha)
        B, C, H, W = imgs.size()

        # Compute bounding box size based on lambda
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        # Random center point
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        # Bounding box coordinates
        x1 = np.clip(cx - cut_w // 2, 0, W)
        y1 = np.clip(cy - cut_h // 2, 0, H)
        x2 = np.clip(cx + cut_w // 2, 0, W)
        y2 = np.clip(cy + cut_h // 2, 0, H)

        # Apply CutMix
        perm = torch.randperm(B, device=imgs.device)
        imgs[:, :, y1:y2, x1:x2] = imgs[perm, :, y1:y2, x1:x2]

        # Adjust lambda based on actual cut area (handles boundary cases)
        lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
        return imgs, targets, targets[perm], lam
    else:
        # MixUp: Linear interpolation between images
        lam = np.random.beta(alpha, alpha)
        perm = torch.randperm(imgs.size(0), device=imgs.device)
        mixed_imgs = lam * imgs + (1 - lam) * imgs[perm]
        return mixed_imgs, targets, targets[perm], lam


def mixed_ce(logits, y1, y2, lam, criterion):
    """
    Mixed cross-entropy loss for MixUp/CutMix.

    PERFORMANCE FIX:
    - Pass pre-created criterion to avoid overhead (was creating new loss every call)
    - With 1.2M images, this saves significant computation time

    Args:
        logits: Model predictions [B, num_classes]
        y1: Original targets [B]
        y2: Mixed targets [B]
        lam: Mixing coefficient (scalar)
        criterion: Pre-created CrossEntropyLoss with label_smoothing

    Returns:
        Weighted cross-entropy loss
    """
    return lam * criterion(logits, y1) + (1 - lam) * criterion(logits, y2)


def mixup_cutmix_accuracy(logits, y, y_a, lam):
    """
    Compute weighted accuracy for MixUp/CutMix training.

    This is an approximation since the true label is a mixture.
    Only use for monitoring - validation uses standard accuracy.

    Args:
        logits: Model predictions [B, num_classes]
        y: Original targets [B]
        y_a: Mixed targets [B]
        lam: Mixing coefficient (scalar)

    Returns:
        Weighted correct predictions (float)
    """
    pred = logits.argmax(dim=1)
    correct_y = (pred == y).float()
    correct_ya = (pred == y_a).float()
    weighted = lam * correct_y + (1.0 - lam) * correct_ya
    return weighted.sum().item()


# ADVANCED: Progressive augmentation strength
class ProgressiveAugmentation:
    """
    Optional: Gradually increase augmentation strength during training.

    For 78%+: Start with moderate augmentation, increase over epochs.
    This helps initial convergence while maintaining strong regularization.
    """
    def __init__(self, initial_alpha=0.4, final_alpha=1.0, warmup_epochs=30):
        self.initial_alpha = initial_alpha
        self.final_alpha = final_alpha
        self.warmup_epochs = warmup_epochs

    def get_alpha(self, epoch):
        if epoch < self.warmup_epochs:
            progress = epoch / self.warmup_epochs
            return self.initial_alpha + (self.final_alpha - self.initial_alpha) * progress
        return self.final_alpha


# ADVANCED: Test-time augmentation helper
@torch.no_grad()
def tta_predict(model, image, num_augs=5):
    """
    Test-time augmentation for inference.

    Can boost validation accuracy by 0.5-1.0% for 78%+ target.
    Apply multiple augmented versions and average predictions.
    """
    model.eval()
    predictions = []

    # Original image
    predictions.append(model(image))

    # Horizontal flip
    predictions.append(model(torch.flip(image, dims=[3])))

    # Multi-crop if num_augs > 2
    if num_augs > 2:
        for _ in range(num_augs - 2):
            # Random crop + flip
            # (Implement based on your needs)
            pass

    return torch.stack(predictions).mean(dim=0)
