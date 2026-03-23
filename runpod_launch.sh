#!/bin/bash
# ============================================================================
# TCD-JEPA RunPod H100 Launch Script
# ============================================================================
# Deploy this on a RunPod H100 instance for the 12-hour benchmark run.
#
# Usage:
#   1. Create a RunPod pod with:
#      - GPU: 1x H100 80GB SXM5
#      - Template: RunPod PyTorch 2.3+ (CUDA 12.x)
#      - Disk: 50GB (enough for CIFAR-10 + checkpoints)
#   2. SSH into the pod
#   3. Run: bash runpod_launch.sh
#
# For a quick sanity check first:
#   bash runpod_launch.sh --quick-test
# ============================================================================

set -euo pipefail

echo "=============================================="
echo "TCD-JEPA H100 RunPod Benchmark"
echo "$(date)"
echo "=============================================="

# --- System Info ---
echo ""
echo "--- System Info ---"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "No GPU detected"
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')" 2>/dev/null || true
echo ""

# --- Setup ---
cd /workspace 2>/dev/null || cd ~

# Clone if needed
if [ ! -d "tcd-jepa" ]; then
    echo "Cloning repository..."
    git clone https://github.com/direncode/tcd-jepa.git
    cd tcd-jepa
else
    cd tcd-jepa
    echo "Repository exists, pulling latest..."
    git pull origin main 2>/dev/null || true
fi

# Install dependencies
echo "Installing dependencies..."
pip install -e ".[all]" --quiet 2>/dev/null || pip install -e . --quiet
pip install wandb --quiet 2>/dev/null || true

# Verify Flash Attention availability
python3 -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device: {torch.cuda.get_device_name()}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
    print(f'Flash SDP available: {torch.backends.cuda.flash_sdp_enabled() if hasattr(torch.backends.cuda, \"flash_sdp_enabled\") else \"unknown\"}')
    print(f'torch.compile available: {hasattr(torch, \"compile\")}')
    # Quick test
    q = torch.randn(2, 8, 64, 32, device='cuda', dtype=torch.bfloat16)
    k = torch.randn(2, 8, 64, 32, device='cuda', dtype=torch.bfloat16)
    v = torch.randn(2, 8, 64, 32, device='cuda', dtype=torch.bfloat16)
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    print(f'Flash Attention test: PASSED (output shape: {out.shape})')
"

# --- Configure wandb (optional) ---
if [ -n "${WANDB_API_KEY:-}" ]; then
    echo "Wandb configured"
    wandb login "$WANDB_API_KEY" 2>/dev/null || true
else
    echo "No WANDB_API_KEY set — logging to local files only"
    echo "(Set WANDB_API_KEY env var to enable wandb tracking)"
fi

# --- Run ---
echo ""
echo "=============================================="
echo "Starting benchmark..."
echo "=============================================="

# Pass through any arguments (e.g., --quick-test)
python3 run_h100_benchmark.py \
    --config configs/h100_runpod.yaml \
    --data-dir ./data \
    --time-limit-hours 11.5 \
    "$@"

echo ""
echo "=============================================="
echo "Benchmark complete!"
echo "Results in: results/h100_run/"
echo "=============================================="

# Show final report
if [ -f results/h100_run/final_report.txt ]; then
    cat results/h100_run/final_report.txt
fi
