import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from config import get_config
from model import resnet50
from data import get_dataloaders
from augmentation import mixup_cutmix, mixed_ce, mixup_cutmix_accuracy
from utils import set_seed, save_checkpoint, load_checkpoint, AverageMeter, reduce_tensor
from logging_utils import Logger


def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    dist.destroy_process_group()


def train_epoch(model, device, train_loader, optimizer, criterion, scheduler, scaler, cfg, epoch, rank, world_size, logger=None):
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    if rank == 0:
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
    else:
        pbar = train_loader

    total_batches = len(train_loader)

    for batch_idx, (data, target) in enumerate(pbar):
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type="cuda", dtype=torch.float16, enabled=cfg.use_amp):
            data, target, target_a, lam = mixup_cutmix(data, target, alpha=cfg.mixup_cutmix_alpha, cutmix_prob=cfg.cutmix_prob)
            logits = model(data)
            loss = mixed_ce(logits, target, target_a, lam, criterion)

        scaler.scale(loss).backward()

        if cfg.grad_clip is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        bs = data.size(0)
        loss_meter.update(loss.item(), bs)
        correct = mixup_cutmix_accuracy(logits.detach(), target, target_a, lam)
        acc_meter.update(correct, bs)

        curr_lr = optimizer.param_groups[0]["lr"]

        if rank == 0:
            pbar.set_description(
                f"Epoch {epoch} | loss={loss_meter.avg:.4f} acc={100.0*acc_meter.sum/acc_meter.count:.2f}% lr={curr_lr:.5f}"
            )

            # Log batch progress periodically
            if logger and batch_idx % cfg.log_freq == 0:
                logger.log_batch(
                    epoch, batch_idx, total_batches,
                    loss_meter.avg, 100.0 * acc_meter.sum / acc_meter.count, curr_lr
                )

    return loss_meter.avg, 100.0 * acc_meter.sum / acc_meter.count


@torch.no_grad()
def validate(model, device, val_loader, criterion, rank, world_size):
    model.eval()
    loss_meter = AverageMeter()
    correct = 0
    total = 0

    for data, target in val_loader:
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        logits = model(data)
        loss = criterion(logits, target)

        loss_meter.update(loss.item(), data.size(0))
        pred = logits.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += data.size(0)

    if world_size > 1:
        loss_tensor = torch.tensor(loss_meter.sum, device=device)
        correct_tensor = torch.tensor(correct, device=device)
        total_tensor = torch.tensor(total, device=device)

        loss_tensor = reduce_tensor(loss_tensor, world_size)
        correct_tensor = reduce_tensor(correct_tensor, world_size)
        total_tensor = reduce_tensor(total_tensor, world_size)

        avg_loss = loss_tensor.item() / total_tensor.item()
        avg_acc = 100.0 * correct_tensor.item() / total_tensor.item()
    else:
        avg_loss = loss_meter.avg
        avg_acc = 100.0 * correct / total

    return avg_loss, avg_acc


def main(rank, world_size):
    cfg = get_config()
    set_seed(cfg.seed + rank)

    # Initialize logger
    logger = Logger(log_dir=cfg.log_dir, experiment_name=cfg.experiment_name, rank=rank)

    if world_size > 1:
        setup_ddp(rank, world_size)

    device = torch.device(f"cuda:{rank}")

    # Log configuration
    logger.log_config(cfg)

    # Log system info
    if rank == 0:
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"Number of GPUs: {world_size}")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    train_dl, val_dl, train_sampler, val_sampler = get_dataloaders(cfg, rank, world_size)

    logger.info(f"Training samples: {len(train_dl.dataset):,}")
    logger.info(f"Validation samples: {len(val_dl.dataset):,}")
    logger.info(f"Batch size per GPU: {cfg.batch_size}")
    logger.info(f"Effective batch size: {cfg.batch_size * world_size}")
    logger.info(f"Batches per epoch: {len(train_dl):,}")

    model = resnet50(num_classes=cfg.num_classes).to(device)
    model = model.to(memory_format=torch.channels_last)

    # Log model info
    logger.log_model_info(model)

    if world_size > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[rank], find_unused_parameters=cfg.find_unused_parameters)

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg.max_lr / cfg.div_factor,
        momentum=cfg.momentum,
        nesterov=cfg.nesterov,
        weight_decay=cfg.weight_decay
    )

    steps_per_epoch = len(train_dl)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.max_lr,
        epochs=cfg.epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=cfg.pct_start,
        anneal_strategy="cos",
        div_factor=cfg.div_factor,
        final_div_factor=cfg.final_div_factor,
        base_momentum=cfg.base_momentum,
        max_momentum=cfg.max_momentum,
        cycle_momentum=True
    )

    scaler = GradScaler(enabled=cfg.use_amp)

    start_epoch = 0
    best_val_acc = 0.0
    best_epoch = 0

    if rank == 0 and os.path.exists(f"{cfg.checkpoint_dir}/latest.pth"):
        start_epoch = load_checkpoint(
            f"{cfg.checkpoint_dir}/latest.pth",
            model.module if world_size > 1 else model,
            optimizer,
            scheduler
        )
        logger.info(f"Resumed training from epoch {start_epoch}")

    if world_size > 1:
        dist.barrier()

    logger.info("\nStarting training...\n")

    for epoch in range(start_epoch + 1, cfg.epochs + 1):
        logger.start_epoch()

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_loss, train_acc = train_epoch(
            model, device, train_dl, optimizer, criterion, scheduler, scaler, cfg, epoch, rank, world_size, logger
        )
        val_loss, val_acc = validate(model, device, val_dl, criterion, rank, world_size)

        # Get current learning rate
        curr_lr = optimizer.param_groups[0]["lr"]

        # Log epoch results
        logger.log_epoch(epoch, train_loss, train_acc, val_loss, val_acc, curr_lr)

        # Track best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch

        if rank == 0:
            # Save checkpoints
            if epoch % cfg.save_freq == 0 or epoch == cfg.epochs:
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.module.state_dict() if world_size > 1 else model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "train_loss": train_loss,
                        "train_acc": train_acc,
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                        "best_val_acc": best_val_acc,
                    },
                    cfg.checkpoint_dir,
                    f"epoch_{epoch}.pth"
                )
                logger.info(f"Checkpoint saved: epoch_{epoch}.pth")

            # Always save latest checkpoint
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state_dict": model.module.state_dict() if world_size > 1 else model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_acc": best_val_acc,
                },
                cfg.checkpoint_dir,
                "latest.pth"
            )

            # Save best model
            if val_acc == best_val_acc:
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.module.state_dict() if world_size > 1 else model.state_dict(),
                        "val_acc": val_acc,
                    },
                    cfg.checkpoint_dir,
                    "best_model.pth"
                )
                logger.info(f"Best model saved with validation accuracy: {best_val_acc:.2f}%")

            # Log GPU stats periodically
            if epoch % 10 == 0:
                logger.log_gpu_stats()

    # Final summary
    logger.log_final_summary(best_epoch, best_val_acc)
    logger.close()

    if world_size > 1:
        cleanup_ddp()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for DDP")
    parser.add_argument("--world_size", type=int, default=1, help="Number of GPUs")
    args = parser.parse_args()

    if args.local_rank != -1:
        rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
        world_size = int(os.environ.get("WORLD_SIZE", args.world_size))
    else:
        rank = 0
        world_size = 1

    main(rank, world_size)
