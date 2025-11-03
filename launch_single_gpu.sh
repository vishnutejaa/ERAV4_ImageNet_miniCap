#!/bin/bash

# Launch script for single-GPU training
# Usage: bash launch_single_gpu.sh

echo "Starting single-GPU training..."

python train.py --world_size=1
