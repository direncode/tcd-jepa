#!/bin/bash
#SBATCH --job-name=tcd-jepa-multinode
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/slurm/%j.out
#SBATCH --error=logs/slurm/%j.err
#SBATCH --exclusive
#
# Multi-node distributed training for TCD-JEPA.
# Designed for 2+ nodes with 8 GPUs each (e.g., 8x H100/H200).
#
# Usage:
#   sbatch scripts/slurm_multinode.sh configs/default.yaml
#   sbatch --nodes=4 scripts/slurm_multinode.sh configs/default.yaml --tcd

set -euo pipefail

CONFIG=${1:?"Usage: sbatch scripts/slurm_multinode.sh <config.yaml> [extra args...]"}
shift
EXTRA_ARGS="$@"

mkdir -p logs/slurm

# Multi-node configuration
NNODES=${SLURM_NNODES}
NPROC=${SLURM_GPUS_PER_NODE:-8}
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=${MASTER_PORT:-29500}
WORLD_SIZE=$((NNODES * NPROC))

echo "============================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodes: ${NNODES}"
echo "GPUs per node: ${NPROC}"
echo "Total GPUs: ${WORLD_SIZE}"
echo "Master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "Config: ${CONFIG}"
echo "============================================"

# Set NCCL environment for multi-node
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=0

# Elastic launch for fault tolerance
srun torchrun \
    --nnodes="${NNODES}" \
    --nproc_per_node="${NPROC}" \
    --rdzv_id="${SLURM_JOB_ID}" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    --max_restarts=3 \
    train_distributed.py \
    --config "${CONFIG}" \
    --auto-resume \
    --max-retries 3 \
    ${EXTRA_ARGS}

echo "Multi-node training complete. Job ID: ${SLURM_JOB_ID}"
