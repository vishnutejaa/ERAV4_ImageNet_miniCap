#!/bin/bash

# Launch script for multi-GPU training on AWS EC2
# Usage: bash launch_multi_gpu.sh

NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)

echo "Detected $NUM_GPUS GPUs"
echo "Starting distributed training..."

python -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    train.py \
    --world_size=$NUM_GPUS

# Alternative using torchrun (PyTorch >= 1.10)
# torchrun --nproc_per_node=$NUM_GPUS --master_port=29500 train.py --world_size=$NUM_GPUS
