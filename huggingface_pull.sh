#!/bin/bash

# Set your Hugging Face repository name
HF_REPO="mihirpd/quiet-grass-102-checkpoint-2000"

# Set the local directory where you want to download the checkpoint
LOCAL_DIR="./quiet-grass-102-checkpoint-2000"

# Make sure you're logged in to Hugging Face CLI
# Run: huggingface-cli login (if not already logged in)
echo "$CKPT_DIR/dpsk_ckpts/$LOCAL_DIR"
# Pull the checkpoint from Hugging Face Hub
huggingface-cli download $HF_REPO --local-dir $CKPT_DIR/dpsk_ckpts/$LOCAL_DIR

echo "Checkpoint downloaded from Hugging Face Hub: $HF_REPO to $LOCAL_DIR"
