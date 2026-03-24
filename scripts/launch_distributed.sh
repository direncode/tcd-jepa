#!/bin/bash
# Launch distributed TCD-JEPA benchmarks on multi-GPU nodes.
#
# Usage:
#   # 8xH100 — all benchmarks (~2h total)
#   bash scripts/launch_distributed.sh all
#
#   # Single dataset
#   bash scripts/launch_distributed.sh stl10
#
#   # Resume after crash
#   bash scripts/launch_distributed.sh stl10 --resume
#
#   # Quick sanity check (5 epochs)
#   bash scripts/launch_distributed.sh cifar10 --quick

set -euo pipefail

DATASET="${1:-all}"
shift || true  # Remaining args passed through

# Auto-detect GPU count
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "$NUM_GPUS" -eq 0 ]; then
    echo "ERROR: No GPUs detected"
    exit 1
fi
echo "=== TCD-JEPA Distributed Benchmark ==="
echo "GPUs: ${NUM_GPUS}"
echo "Dataset: ${DATASET}"
echo "Extra args: $@"
echo "======================================="

# Ensure data directory exists
mkdir -p ./data

# Set optimal NCCL settings for H100
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5
export NCCL_P2P_LEVEL=NVL

# Pin memory and workers
export OMP_NUM_THREADS=4

# Launch with torchrun
exec torchrun \
    --standalone \
    --nproc_per_node="${NUM_GPUS}" \
    -m benchmarks.run_distributed \
    --dataset "${DATASET}" \
    --resume \
    "$@" \
    2>&1 | tee "logs/distributed_${DATASET}_$(date +%Y%m%d_%H%M%S).log"
