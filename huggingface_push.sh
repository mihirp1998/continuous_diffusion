#!/bin/bash

# Set the checkpoint path
CHECKPOINT_PATH="/grogu/user/mprabhud/dpsk_ckpts/quiet-grass-102/checkpoint-2000"

# Set your Hugging Face repository name (replace with your actual repo)
HF_REPO="mihirpd/quiet-grass-102-checkpoint-2000"

# Make sure you're logged in to Hugging Face CLI
# Run: huggingface-cli login (if not already logged in)

# Push the checkpoint to Hugging Face Hub
huggingface-cli upload $HF_REPO $CHECKPOINT_PATH --repo-type model

echo "Checkpoint pushed to Hugging Face Hub: $HF_REPO"
