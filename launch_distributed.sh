#!/bin/bash
# =============================================================================
# Launch all distributed trainings on 8x H100 GPUs
# Target: ~30 minutes total for CIFAR-10 + STL-10 + ImageNet
#
# Usage:
#   bash launch_distributed.sh                    # JEPA only
#   bash launch_distributed.sh --tcd              # TCD-JEPA (with recursive loop)
#   bash launch_distributed.sh --tcd --compile    # TCD-JEPA + torch.compile
#   bash launch_distributed.sh --tcd --probe      # + linear probe evaluation
# =============================================================================

set -euo pipefail

NGPUS=${NGPUS:-8}
TCD_FLAG=""
COMPILE_FLAG=""
PROBE_FLAG=""
EXTRA_ARGS=""

for arg in "$@"; do
    case $arg in
        --tcd)      TCD_FLAG="--tcd" ;;
        --compile)  COMPILE_FLAG="--compile" ;;
        --probe)    PROBE_FLAG="--probe" ;;
        *)          EXTRA_ARGS="$EXTRA_ARGS $arg" ;;
    esac
done

FLAGS="$TCD_FLAG $COMPILE_FLAG $PROBE_FLAG $EXTRA_ARGS"

echo "========================================"
echo "  TCD-JEPA Distributed Training"
echo "  GPUs: $NGPUS"
echo "  Flags: $FLAGS"
echo "========================================"

TOTAL_START=$SECONDS

# --- CIFAR-10 ---
echo ""
echo "========================================"
echo "  [1/3] CIFAR-10 (50k images, 32x32)"
echo "  Per-GPU batch: 512, Effective: $((512 * NGPUS))"
echo "  Epochs: 50"
echo "========================================"
START=$SECONDS

torchrun --nproc_per_node=$NGPUS \
    train_distributed.py \
    --config configs/dist_cifar10.yaml \
    $FLAGS

ELAPSED=$((SECONDS - START))
echo "  CIFAR-10 done in ${ELAPSED}s ($((ELAPSED / 60))m $((ELAPSED % 60))s)"

# --- STL-10 ---
echo ""
echo "========================================"
echo "  [2/3] STL-10 (105k images, 96x96)"
echo "  Per-GPU batch: 256, Effective: $((256 * NGPUS))"
echo "  Epochs: 50"
echo "========================================"
START=$SECONDS

torchrun --nproc_per_node=$NGPUS \
    train_distributed.py \
    --config configs/dist_stl10.yaml \
    $FLAGS

ELAPSED=$((SECONDS - START))
echo "  STL-10 done in ${ELAPSED}s ($((ELAPSED / 60))m $((ELAPSED % 60))s)"

# --- ImageNet ---
echo ""
echo "========================================"
echo "  [3/3] ImageNet (1.28M images, 224x224)"
echo "  Per-GPU batch: 256, Effective: $((256 * NGPUS))"
echo "  Epochs: 30"
echo "========================================"
START=$SECONDS

torchrun --nproc_per_node=$NGPUS \
    train_distributed.py \
    --config configs/dist_imagenet.yaml \
    $FLAGS

ELAPSED=$((SECONDS - START))
echo "  ImageNet done in ${ELAPSED}s ($((ELAPSED / 60))m $((ELAPSED % 60))s)"

# --- Summary ---
TOTAL_ELAPSED=$((SECONDS - TOTAL_START))
echo ""
echo "========================================"
echo "  ALL TRAININGS COMPLETE"
echo "  Total time: ${TOTAL_ELAPSED}s ($((TOTAL_ELAPSED / 60))m $((TOTAL_ELAPSED % 60))s)"
echo "========================================"
echo ""
echo "Checkpoints:"
echo "  CIFAR-10:  ./logs/dist_cifar10/checkpoints/"
echo "  STL-10:    ./logs/dist_stl10/checkpoints/"
echo "  ImageNet:  ./logs/dist_imagenet/checkpoints/"
