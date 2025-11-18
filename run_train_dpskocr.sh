#!/bin/bash

# Multi-GPU training script for train_dpskocr.py
# Usage: bash run_train_dpskocr.sh [num_gpus]
# Example: bash run_train_dpskocr.sh 4

NUM_GPUS=${1:-1}  # Default to 1 GPU if not specified

echo "Starting training on $NUM_GPUS GPU(s)..."

torchrun --standalone --nproc_per_node=$NUM_GPUS train_dpskocr.py






