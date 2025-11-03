import torch
import torch.nn as nn
import random


def mixup_cutmix(imgs, targets, alpha=0.2, cutmix_prob=0.5):
    if random.random() < cutmix_prob:
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
        B, C, H, W = imgs.size()
        rx, ry = random.randint(0, W - 1), random.randint(0, H - 1)
        rw, rh = int(W * (1 - lam) ** 0.5), int(H * (1 - lam) ** 0.5)
        x1, y1 = max(rx - rw // 2, 0), max(ry - rh // 2, 0)
        x2, y2 = min(rx + rw // 2, W), min(ry + rh // 2, H)
        perm = torch.randperm(B, device=imgs.device)
        imgs[:, :, y1:y2, x1:x2] = imgs[perm, :, y1:y2, x1:x2]
        lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
        return imgs, targets, targets[perm], lam
    else:
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
        perm = torch.randperm(imgs.size(0), device=imgs.device)
        return lam * imgs + (1 - lam) * imgs[perm], targets, targets[perm], lam


def mixed_ce(logits, y1, y2, lam, smoothing=0.1):
    ce = nn.CrossEntropyLoss(label_smoothing=smoothing)
    return lam * ce(logits, y1) + (1 - lam) * ce(logits, y2)


def mixup_cutmix_accuracy(logits, y, y_a, lam):
    pred = logits.argmax(dim=1)
    correct_y = (pred == y).float()
    correct_ya = (pred == y_a).float()
    weighted = lam * correct_y + (1.0 - lam) * correct_ya
    return weighted.sum().item()
