#!/bin/bash
#SBATCH --job-name=tcd-jepa
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/%j.out
#SBATCH --error=logs/slurm/%j.err
#
# Single-node multi-GPU training via SLURM + torchrun
#
# Usage:
#   sbatch scripts/slurm_train.sh configs/dist_cifar10.yaml
#   sbatch scripts/slurm_train.sh configs/dist_cifar10.yaml --tcd
#
# Multi-node (2 nodes, 4 GPUs each):
#   sbatch --nodes=2 scripts/slurm_train.sh configs/dist_cifar10.yaml

set -euo pipefail

CONFIG=${1:?"Usage: sbatch scripts/slurm_train.sh <config.yaml> [extra args...]"}
shift
EXTRA_ARGS="$@"

# Create log directories
mkdir -p logs/slurm

# Detect multi-node setup
NNODES=${SLURM_NNODES:-1}
NPROC=${SLURM_GPUS_PER_NODE:-4}
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=${MASTER_PORT:-29500}

echo "============================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodes: ${NNODES}"
echo "GPUs per node: ${NPROC}"
echo "Master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "Config: ${CONFIG}"
echo "Extra args: ${EXTRA_ARGS}"
echo "============================================"

# Launch with torchrun (handles multi-node automatically via SLURM env vars)
srun torchrun \
    --nnodes="${NNODES}" \
    --nproc_per_node="${NPROC}" \
    --rdzv_id="${SLURM_JOB_ID}" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    train_distributed.py \
    --config "${CONFIG}" \
    --auto-resume \
    ${EXTRA_ARGS}

echo "Training complete. Job ID: ${SLURM_JOB_ID}"
