"""
Quick test script to demonstrate the logging system.
Run this to see how the logging looks without running full training.
"""

import torch
import time
from config import get_config
from logging_utils import Logger
from model import resnet50

def main():
    print("\n" + "="*80)
    print("LOGGING SYSTEM DEMONSTRATION")
    print("="*80 + "\n")

    # Initialize config
    cfg = get_config()
    cfg.experiment_name = "logging_demo"
    cfg.epochs = 5  # Just demo 5 epochs

    # Initialize logger
    logger = Logger(log_dir=cfg.log_dir, experiment_name=cfg.experiment_name, rank=0)

    # Log configuration
    logger.info("Demonstrating logging system...")
    logger.log_config(cfg)

    # Create model and log info
    model = resnet50(num_classes=cfg.num_classes)
    logger.log_model_info(model)

    # Log system info
    logger.info(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        logger.info(f"CUDA available: True")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
        logger.log_gpu_stats()
    else:
        logger.info(f"CUDA available: False (CPU only)")

    # Simulate training
    logger.info("\nSimulating training...\n")

    for epoch in range(1, cfg.epochs + 1):
        logger.start_epoch()

        # Simulate some batches
        for batch in range(0, 100, 10):
            time.sleep(0.1)  # Simulate processing
            logger.log_batch(
                epoch=epoch,
                batch_idx=batch,
                total_batches=100,
                loss=7.0 - epoch * 0.5 - batch * 0.01,
                acc=epoch * 5 + batch * 0.1,
                lr=0.01 * (1 + epoch * 0.1)
            )

        # Simulate epoch end
        train_loss = 7.0 - epoch * 0.5
        train_acc = epoch * 10
        val_loss = 6.8 - epoch * 0.45
        val_acc = epoch * 9.5
        lr = 0.01 * (1 + epoch * 0.1)

        logger.log_epoch(epoch, train_loss, train_acc, val_loss, val_acc, lr)

        time.sleep(0.5)

    # Final summary
    logger.log_final_summary(best_epoch=3, best_val_acc=28.5)
    logger.close()

    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE!")
    print("="*80)
    print(f"\nCheck your logs at: {logger.log_dir}")
    print("\nGenerated files:")
    print(f"  - {logger.log_dir}/training.log")
    print(f"  - {logger.log_dir}/config.json")
    print(f"  - {logger.log_dir}/metrics.json")
    print(f"  - {logger.log_dir}/training_curves.png")
    print(f"  - {logger.log_dir}/tensorboard/")
    print("\nTo view TensorBoard:")
    print(f"  tensorboard --logdir={cfg.log_dir}")
    print("  Then open: http://localhost:6006")
    print()

if __name__ == "__main__":
    main()
