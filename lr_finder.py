"""
Learning Rate Finder for ImageNet-1k + ResNet-50
Optimized for SGD with OneCycleLR scheduler

Usage:
    python lr_finder.py --min_lr 1e-7 --max_lr 10 --num_iter 300
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

from config import get_config
from model import resnet50
from data import get_dataloaders
from utils import set_seed


class LRFinder:
    """
    Learning Rate Range Test for finding optimal learning rate.

    Based on the paper "Cyclical Learning Rates for Training Neural Networks"
    and the fastai implementation.
    """

    def __init__(self, model, optimizer, criterion, device, use_amp=True):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.use_amp = use_amp

        # Save initial states
        self.model_state = deepcopy(model.state_dict())
        self.optimizer_state = deepcopy(optimizer.state_dict())

        # Results storage
        self.lrs = []
        self.losses = []
        self.smoothed_losses = []

        # Best loss tracking
        self.best_loss = float('inf')

    def range_test(
        self,
        train_loader,
        min_lr=1e-7,
        max_lr=10,
        num_iter=300,
        smooth_f=0.05,
        diverge_th=5.0
    ):
        """
        Perform learning rate range test.

        Args:
            train_loader: Training data loader
            min_lr: Minimum learning rate to test
            max_lr: Maximum learning rate to test
            num_iter: Number of iterations to run
            smooth_f: Smoothing factor for loss (0-1, higher = more smoothing)
            diverge_th: Stop if loss > diverge_th * best_loss

        Returns:
            lrs: List of learning rates tested
            losses: List of losses at each LR
        """
        print("\n" + "="*80)
        print("LEARNING RATE RANGE TEST")
        print("="*80)
        print(f"Min LR: {min_lr:.2e}")
        print(f"Max LR: {max_lr:.2e}")
        print(f"Iterations: {num_iter}")
        print(f"Smoothing: {smooth_f}")
        print(f"Divergence threshold: {diverge_th}x")
        print("="*80 + "\n")

        # Set model to training mode
        self.model.train()

        # Calculate LR multiplication factor
        lr_mult = (max_lr / min_lr) ** (1 / num_iter)

        # Set initial LR
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = min_lr

        # Setup AMP
        scaler = GradScaler(enabled=self.use_amp)

        # Get data iterator
        data_iter = iter(train_loader)

        # Progress bar
        pbar = tqdm(range(num_iter), desc="LR Finder")

        for iteration in pbar:
            # Get batch
            try:
                images, targets = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                images, targets = next(data_iter)

            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            # Forward pass
            self.optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="cuda", enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

            # Backward pass
            if self.use_amp:
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            # Get current LR
            current_lr = self.optimizer.param_groups[0]['lr']

            # Record
            loss_val = loss.item()
            self.lrs.append(current_lr)
            self.losses.append(loss_val)

            # Smooth loss
            if iteration == 0:
                smoothed_loss = loss_val
            else:
                smoothed_loss = smooth_f * loss_val + (1 - smooth_f) * self.smoothed_losses[-1]
            self.smoothed_losses.append(smoothed_loss)

            # Update best loss
            if smoothed_loss < self.best_loss:
                self.best_loss = smoothed_loss

            # Check for divergence
            if smoothed_loss > diverge_th * self.best_loss:
                print(f"\n[Early Stop] Loss diverged at iteration {iteration}")
                print(f"Current loss: {smoothed_loss:.4f}")
                print(f"Best loss: {self.best_loss:.4f}")
                print(f"LR at divergence: {current_lr:.2e}")
                break

            # Update progress bar
            pbar.set_postfix({
                'lr': f'{current_lr:.2e}',
                'loss': f'{smoothed_loss:.4f}',
                'best': f'{self.best_loss:.4f}'
            })

            # Increase learning rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= lr_mult

        print("\nLR Range Test completed!")
        return np.array(self.lrs), np.array(self.smoothed_losses)

    def plot(self, skip_start=10, skip_end=5, log_lr=True, suggest=True):
        """
        Plot the learning rate range test results.

        Args:
            skip_start: Skip first N points (unstable)
            skip_end: Skip last N points (diverged)
            log_lr: Use log scale for LR axis
            suggest: Show suggested LR values
        """
        if len(self.lrs) == 0:
            print("No data to plot. Run range_test() first.")
            return

        # Prepare data
        lrs = np.array(self.lrs)
        losses = np.array(self.smoothed_losses)

        # Skip points
        if skip_end > 0:
            lrs = lrs[skip_start:-skip_end]
            losses = losses[skip_start:-skip_end]
        else:
            lrs = lrs[skip_start:]
            losses = losses[skip_start:]

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot
        ax.plot(lrs, losses, linewidth=2, color='#2E86AB', label='Smoothed Loss')

        if log_lr:
            ax.set_xscale('log')

        ax.set_xlabel('Learning Rate (log scale)' if log_lr else 'Learning Rate', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Learning Rate Finder - ImageNet-1k ResNet-50', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')

        if suggest:
            # Find suggestions
            suggestions = self.suggest_lr(skip_start, skip_end)

            # Plot suggestions
            for name, lr_val, color in [
                ('Steepest Gradient', suggestions['steepest'], '#E63946'),
                ('Minimum Loss', suggestions['min_loss'], '#06A77D'),
                ('Min/10 (Conservative)', suggestions['conservative'], '#F77F00')
            ]:
                ax.axvline(lr_val, color=color, linestyle='--', alpha=0.7, linewidth=1.5, label=name)

            ax.legend(loc='best', fontsize=10)

        plt.tight_layout()

        # Save plot
        save_path = 'lr_finder_result.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")

        plt.show()

    def suggest_lr(self, skip_start=10, skip_end=5):
        """
        Suggest optimal learning rates based on the test results.

        Returns:
            dict with suggested LR values
        """
        if len(self.lrs) == 0:
            print("No data available. Run range_test() first.")
            return None

        # Prepare data
        lrs = np.array(self.lrs)
        losses = np.array(self.smoothed_losses)

        # Skip points
        if skip_end > 0:
            lrs = lrs[skip_start:-skip_end]
            losses = losses[skip_start:-skip_end]
        else:
            lrs = lrs[skip_start:]
            losses = losses[skip_start:]

        # Method 1: Steepest gradient (fastest decrease)
        log_lrs = np.log10(lrs)
        gradients = np.gradient(losses, log_lrs)
        min_gradient_idx = int(np.argmin(gradients))
        lr_steepest = float(lrs[min_gradient_idx])

        # Method 2: Minimum loss
        min_loss_idx = int(np.argmin(losses))
        lr_min = float(lrs[min_loss_idx])

        # Method 3: Conservative (min loss / 10)
        lr_conservative = lr_min / 10.0

        # Method 4: OneCycleLR recommended (steepest / 4 to steepest * 10)
        lr_onecycle_max = lr_steepest * 6  # Recommended max_lr for OneCycleLR

        print("\n" + "="*80)
        print("LEARNING RATE SUGGESTIONS")
        print("="*80)
        print(f"\n1. Steepest Gradient Method:")
        print(f"   LR = {lr_steepest:.2e}")
        print(f"   → Use this as starting point")

        print(f"\n2. Minimum Loss Method:")
        print(f"   LR = {lr_min:.2e}")
        print(f"   → This is the peak performance in the test")

        print(f"\n3. Conservative (Min/10):")
        print(f"   LR = {lr_conservative:.2e}")
        print(f"   → Safe choice for stable training")

        print(f"\n4. OneCycleLR Recommendation:")
        print(f"   max_lr = {lr_onecycle_max:.2e}")
        print(f"   → Suggested for OneCycleLR scheduler")
        print(f"   → Policy: Start at max_lr/25, peak at max_lr, end at max_lr/10000")

        print(f"\n" + "="*80)
        print("RECOMMENDED FOR YOUR IMAGENET TRAINING:")
        print("="*80)
        print(f"\nFor 100 epochs with SGD + OneCycleLR:")
        print(f"  max_lr = {lr_onecycle_max:.2e}  (in config.py)")
        print(f"  div_factor = 25.0")
        print(f"  final_div_factor = 10000.0")
        print(f"  pct_start = 0.3  (30% warmup)")
        print(f"\nInitial LR: {lr_onecycle_max/25:.2e}")
        print(f"Peak LR: {lr_onecycle_max:.2e}")
        print(f"Final LR: {lr_onecycle_max/25/10000:.2e}")
        print("="*80 + "\n")

        return {
            'steepest': lr_steepest,
            'min_loss': lr_min,
            'conservative': lr_conservative,
            'onecycle_max': lr_onecycle_max
        }

    def reset(self):
        """Reset model and optimizer to initial state."""
        self.model.load_state_dict(self.model_state)
        self.optimizer.load_state_dict(self.optimizer_state)
        self.lrs = []
        self.losses = []
        self.smoothed_losses = []
        self.best_loss = float('inf')
        print("Model and optimizer reset to initial state.")


def main():
    parser = argparse.ArgumentParser(description='Learning Rate Finder for ImageNet-1k ResNet-50')
    parser.add_argument('--min_lr', type=float, default=1e-7, help='Minimum LR to test')
    parser.add_argument('--max_lr', type=float, default=10.0, help='Maximum LR to test')
    parser.add_argument('--num_iter', type=int, default=300, help='Number of iterations')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--smooth_f', type=float, default=0.05, help='Loss smoothing factor')
    parser.add_argument('--diverge_th', type=float, default=5.0, help='Divergence threshold')
    parser.add_argument('--skip_start', type=int, default=10, help='Skip first N points in plot')
    parser.add_argument('--skip_end', type=int, default=5, help='Skip last N points in plot')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Get config
    cfg = get_config()
    cfg.batch_size = args.batch_size

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("\n" + "="*80)
    print("LEARNING RATE FINDER SETUP")
    print("="*80)
    print(f"Device: {device}")
    print(f"Batch size: {cfg.batch_size}")
    print(f"Workers: {cfg.num_workers}")
    print(f"Use AMP: {cfg.use_amp}")
    print("="*80 + "\n")

    # Load data (only training set needed)
    print("Loading ImageNet-1k training data...")
    train_dl, _, _, _ = get_dataloaders(cfg, rank=None, world_size=None)
    print(f"Training samples: {len(train_dl.dataset):,}")
    print(f"Batches: {len(train_dl):,}\n")

    # Create model
    print("Creating ResNet-50 model...")
    model = resnet50(num_classes=cfg.num_classes).to(device)
    model = model.to(memory_format=torch.channels_last)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB\n")

    # Create optimizer (same as training)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.min_lr,  # Will be overridden by LR finder
        momentum=0.9,
        nesterov=True,
        weight_decay=1e-4
    )

    # Create LR Finder
    lr_finder = LRFinder(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        use_amp=cfg.use_amp
    )

    # Run range test
    print("Starting LR range test...\n")
    lrs, losses = lr_finder.range_test(
        train_loader=train_dl,
        min_lr=args.min_lr,
        max_lr=args.max_lr,
        num_iter=args.num_iter,
        smooth_f=args.smooth_f,
        diverge_th=args.diverge_th
    )

    # Get suggestions
    suggestions = lr_finder.suggest_lr(
        skip_start=args.skip_start,
        skip_end=args.skip_end
    )

    # Plot results
    print("\nGenerating plot...")
    lr_finder.plot(
        skip_start=args.skip_start,
        skip_end=args.skip_end,
        log_lr=True,
        suggest=True
    )

    # Save results to file
    results_file = 'lr_finder_results.txt'
    with open(results_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("LEARNING RATE FINDER RESULTS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Steepest Gradient LR: {suggestions['steepest']:.2e}\n")
        f.write(f"Minimum Loss LR: {suggestions['min_loss']:.2e}\n")
        f.write(f"Conservative LR: {suggestions['conservative']:.2e}\n")
        f.write(f"OneCycleLR max_lr: {suggestions['onecycle_max']:.2e}\n\n")
        f.write("="*80 + "\n")
        f.write("RECOMMENDED CONFIG.PY SETTINGS:\n")
        f.write("="*80 + "\n\n")
        f.write(f"max_lr: float = {suggestions['onecycle_max']:.2e}\n")
        f.write(f"div_factor: float = 25.0\n")
        f.write(f"final_div_factor: float = 10000.0\n")
        f.write(f"pct_start: float = 0.3\n\n")

    print(f"\nResults saved to: {results_file}")
    print("\n✅ LR Finder completed successfully!")
    print("\nNext steps:")
    print("  1. Check the plot: lr_finder_result.png")
    print("  2. Update config.py with the suggested max_lr")
    print("  3. Run full training: python train.py --world_size=4")
    print()


if __name__ == '__main__':
    main()
