#!/bin/bash

# Multi-GPU training script for VAE using all 4 NVIDIA GPUs
# Uses DistributedDataParallel (DDP) for efficient parallel training

cd "$(dirname "$0")/.."

# Activate virtual environment
source .venv/bin/activate

# Set environment variables for better GPU utilization
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=4
export NCCL_DEBUG=INFO

# Default simulation category
SIM=${1:-fixate1}

# Shift to remove the first positional argument, combine remaining
shift 2>/dev/null
ARGS="${*}"

echo "**************************************************************************"
echo "Starting multi-GPU VAE training"
echo "Simulation: ${SIM}"
echo "GPUs: 4 x NVIDIA RTX 6000 Ada"
echo "Extra args: ${ARGS}"
echo "**************************************************************************"

# Run the DDP training script
python -m vae.train_vae_ddp "${SIM}" --n_gpus 4 ${ARGS}

echo "**************************************************************************"
printf "Done! (%s)\n" "$(date '+%m/%d/%Y %H:%M:%S')"
echo "**************************************************************************"
