#!/bin/bash

# Quick launch script for LR Finder
# Usage: bash run_lr_finder.sh

echo "=================================="
echo "ImageNet-1k LR Finder"
echo "=================================="
echo ""

# Default settings optimized for ImageNet-1k
MIN_LR=1e-7
MAX_LR=10
NUM_ITER=300
BATCH_SIZE=128

echo "Settings:"
echo "  Min LR: $MIN_LR"
echo "  Max LR: $MAX_LR"
echo "  Iterations: $NUM_ITER"
echo "  Batch Size: $BATCH_SIZE"
echo ""
echo "This will take approximately 10-15 minutes..."
echo ""

python lr_finder.py \
    --min_lr $MIN_LR \
    --max_lr $MAX_LR \
    --num_iter $NUM_ITER \
    --batch_size $BATCH_SIZE \
    --smooth_f 0.05 \
    --diverge_th 5.0

echo ""
echo "=================================="
echo "LR Finder Complete!"
echo "=================================="
echo ""
echo "Check the results:"
echo "  - Plot: lr_finder_result.png"
echo "  - Details: lr_finder_results.txt"
echo ""
