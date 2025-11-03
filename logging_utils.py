import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime
import logging
import torch
from torch.utils.tensorboard import SummaryWriter


class Logger:
    """
    Comprehensive logging system for training.

    Features:
    - Console output with colors
    - File logging
    - TensorBoard integration
    - JSON metrics export
    - Training statistics
    """

    def __init__(self, log_dir="logs", experiment_name=None, rank=0):
        self.rank = rank
        self.is_main = (rank == 0)

        if not self.is_main:
            return

        # Create experiment directory
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.log_dir = Path(log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Setup file logging
        log_file = self.log_dir / "training.log"
        self.file_logger = logging.getLogger("training")
        self.file_logger.setLevel(logging.INFO)

        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)

        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        self.file_logger.addHandler(fh)
        self.file_logger.addHandler(ch)

        # TensorBoard writer
        self.tb_writer = SummaryWriter(log_dir=str(self.log_dir / "tensorboard"))

        # Metrics storage
        self.metrics = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': [],
            'epoch_time': []
        }

        # Timing
        self.epoch_start_time = None
        self.training_start_time = time.time()

        self.info("=" * 80)
        self.info(f"Logging initialized: {self.log_dir}")
        self.info("=" * 80)

    def info(self, message):
        """Log info message"""
        if self.is_main:
            self.file_logger.info(message)

    def warning(self, message):
        """Log warning message"""
        if self.is_main:
            self.file_logger.warning(message)

    def error(self, message):
        """Log error message"""
        if self.is_main:
            self.file_logger.error(message)

    def log_config(self, config):
        """Log training configuration"""
        if not self.is_main:
            return

        self.info("\n" + "=" * 80)
        self.info("TRAINING CONFIGURATION")
        self.info("=" * 80)

        config_dict = vars(config) if not isinstance(config, dict) else config

        for key, value in sorted(config_dict.items()):
            self.info(f"  {key:.<35} {value}")

        # Save config to JSON
        config_file = self.log_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)

        self.info("=" * 80 + "\n")

    def log_model_info(self, model):
        """Log model architecture and parameters"""
        if not self.is_main:
            return

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        self.info("\n" + "=" * 80)
        self.info("MODEL INFORMATION")
        self.info("=" * 80)
        self.info(f"  Total parameters.......... {total_params:,}")
        self.info(f"  Trainable parameters...... {trainable_params:,}")
        self.info(f"  Non-trainable parameters.. {total_params - trainable_params:,}")
        self.info(f"  Model size (MB)........... {total_params * 4 / 1024 / 1024:.2f}")
        self.info("=" * 80 + "\n")

    def start_epoch(self):
        """Mark start of epoch"""
        if self.is_main:
            self.epoch_start_time = time.time()

    def log_epoch(self, epoch, train_loss, train_acc, val_loss, val_acc, lr):
        """Log epoch results"""
        if not self.is_main:
            return

        # Calculate epoch time
        epoch_time = time.time() - self.epoch_start_time
        total_time = time.time() - self.training_start_time

        # Store metrics
        self.metrics['train_loss'].append(train_loss)
        self.metrics['train_acc'].append(train_acc)
        self.metrics['val_loss'].append(val_loss)
        self.metrics['val_acc'].append(val_acc)
        self.metrics['learning_rate'].append(lr)
        self.metrics['epoch_time'].append(epoch_time)

        # TensorBoard logging
        self.tb_writer.add_scalar('Loss/train', train_loss, epoch)
        self.tb_writer.add_scalar('Loss/val', val_loss, epoch)
        self.tb_writer.add_scalar('Accuracy/train', train_acc, epoch)
        self.tb_writer.add_scalar('Accuracy/val', val_acc, epoch)
        self.tb_writer.add_scalar('Learning_Rate', lr, epoch)
        self.tb_writer.add_scalar('Time/epoch', epoch_time, epoch)

        # Console logging with nice formatting
        self.info("")
        self.info("=" * 100)
        self.info(f"EPOCH {epoch:3d} SUMMARY")
        self.info("=" * 100)
        self.info(f"  Train Loss: {train_loss:.4f}  |  Train Acc: {train_acc:6.2f}%  |  LR: {lr:.6f}")
        self.info(f"  Val Loss:   {val_loss:.4f}  |  Val Acc:   {val_acc:6.2f}%")
        self.info(f"  Epoch Time: {self._format_time(epoch_time)}  |  Total Time: {self._format_time(total_time)}")

        # Best model tracking
        if len(self.metrics['val_acc']) == 1 or val_acc > max(self.metrics['val_acc'][:-1]):
            self.info(f"  🏆 NEW BEST VALIDATION ACCURACY: {val_acc:.2f}%")

        self.info("=" * 100)
        self.info("")

        # Save metrics to JSON every epoch
        self._save_metrics()

    def log_batch(self, epoch, batch_idx, total_batches, loss, acc, lr):
        """Log batch progress (called periodically during training)"""
        if not self.is_main:
            return

        # Calculate progress
        progress = 100.0 * batch_idx / total_batches

        # Estimate time remaining
        if self.epoch_start_time:
            elapsed = time.time() - self.epoch_start_time
            if batch_idx > 0:
                time_per_batch = elapsed / batch_idx
                remaining_batches = total_batches - batch_idx
                eta = time_per_batch * remaining_batches
                eta_str = self._format_time(eta)
            else:
                eta_str = "N/A"
        else:
            eta_str = "N/A"

        self.info(
            f"Epoch [{epoch:3d}] [{batch_idx:4d}/{total_batches:4d}] ({progress:5.1f}%) | "
            f"Loss: {loss:.4f} | Acc: {acc:6.2f}% | LR: {lr:.6f} | ETA: {eta_str}"
        )

    def log_gpu_stats(self):
        """Log GPU memory statistics"""
        if not self.is_main or not torch.cuda.is_available():
            return

        for i in range(torch.cuda.device_count()):
            memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
            memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3

            self.info(
                f"GPU {i}: {memory_allocated:.2f}GB / {memory_total:.2f}GB allocated | "
                f"{memory_reserved:.2f}GB reserved"
            )

    def plot_metrics(self):
        """Generate and save training plots"""
        if not self.is_main:
            return

        try:
            import matplotlib.pyplot as plt

            epochs = range(1, len(self.metrics['train_loss']) + 1)

            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))

            # Loss plot
            axes[0, 0].plot(epochs, self.metrics['train_loss'], 'b-', label='Train Loss')
            axes[0, 0].plot(epochs, self.metrics['val_loss'], 'r-', label='Val Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Training and Validation Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)

            # Accuracy plot
            axes[0, 1].plot(epochs, self.metrics['train_acc'], 'b-', label='Train Acc')
            axes[0, 1].plot(epochs, self.metrics['val_acc'], 'r-', label='Val Acc')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy (%)')
            axes[0, 1].set_title('Training and Validation Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True)

            # Learning rate plot
            axes[1, 0].plot(epochs, self.metrics['learning_rate'], 'g-')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_title('Learning Rate Schedule')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True)

            # Epoch time plot
            axes[1, 1].plot(epochs, self.metrics['epoch_time'], 'm-')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Time (seconds)')
            axes[1, 1].set_title('Epoch Duration')
            axes[1, 1].grid(True)

            plt.tight_layout()
            plt.savefig(self.log_dir / 'training_curves.png', dpi=150)
            plt.close()

            self.info(f"Training curves saved to {self.log_dir / 'training_curves.png'}")

        except ImportError:
            self.warning("matplotlib not installed, skipping plot generation")

    def _save_metrics(self):
        """Save metrics to JSON file"""
        metrics_file = self.log_dir / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)

    def _format_time(self, seconds):
        """Format seconds into human-readable time"""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.2f}h"

    def log_final_summary(self, best_epoch, best_val_acc):
        """Log final training summary"""
        if not self.is_main:
            return

        total_time = time.time() - self.training_start_time

        self.info("\n" + "=" * 100)
        self.info("TRAINING COMPLETED")
        self.info("=" * 100)
        self.info(f"  Total Training Time....... {self._format_time(total_time)}")
        self.info(f"  Best Epoch................ {best_epoch}")
        self.info(f"  Best Validation Accuracy.. {best_val_acc:.2f}%")
        self.info(f"  Final Train Accuracy...... {self.metrics['train_acc'][-1]:.2f}%")
        self.info(f"  Final Val Accuracy........ {self.metrics['val_acc'][-1]:.2f}%")
        self.info(f"  Average Epoch Time........ {self._format_time(sum(self.metrics['epoch_time']) / len(self.metrics['epoch_time']))}")
        self.info("=" * 100)

        # Generate plots
        self.plot_metrics()

    def close(self):
        """Close logger and cleanup"""
        if self.is_main:
            self.tb_writer.close()
            self._save_metrics()


class ProgressTracker:
    """Track training progress with ETA estimation"""

    def __init__(self, total_epochs, batches_per_epoch):
        self.total_epochs = total_epochs
        self.batches_per_epoch = batches_per_epoch
        self.start_time = time.time()
        self.epoch_times = []

    def update(self, epoch, batch=None):
        """Update progress"""
        if batch is None:
            # Epoch completed
            elapsed = time.time() - self.start_time
            self.epoch_times.append(elapsed / epoch)

    def get_eta(self, current_epoch):
        """Get estimated time remaining"""
        if not self.epoch_times:
            return "Unknown"

        avg_time_per_epoch = sum(self.epoch_times) / len(self.epoch_times)
        remaining_epochs = self.total_epochs - current_epoch
        eta_seconds = avg_time_per_epoch * remaining_epochs

        hours = int(eta_seconds // 3600)
        minutes = int((eta_seconds % 3600) // 60)

        if hours > 0:
            return f"{hours}h {minutes}m"
        else:
            return f"{minutes}m"
