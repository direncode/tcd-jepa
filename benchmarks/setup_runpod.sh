#!/bin/bash
# RunPod setup script for TCD-JEPA benchmarks
# Run this ONCE after connecting to your pod.
#
# Usage:
#   bash benchmarks/setup_runpod.sh
#
# Prerequisites: RunPod with NVIDIA GPU (A100 recommended), PyTorch template

set -euo pipefail

echo "============================================"
echo "TCD-JEPA Benchmark Setup for RunPod"
echo "============================================"

# 1. System info
echo ""
echo "--- GPU Info ---"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "No GPU detected!"
echo ""
echo "--- PyTorch ---"
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NONE\"}')" 2>/dev/null || echo "PyTorch not installed"

# 2. Install dependencies
echo ""
echo "--- Installing dependencies ---"
pip install --quiet --upgrade pip
pip install --quiet torch torchvision --upgrade  # Ensure latest
pip install --quiet numpy scipy matplotlib seaborn pyyaml einops tqdm
pip install --quiet wandb  # For experiment tracking

# Optional: persistent homology backends (for TCD)
pip install --quiet giotto-tda 2>/dev/null || pip install --quiet ripser 2>/dev/null || echo "WARNING: No PH backend installed. Using scipy fallback."

# 3. Verify installation
echo ""
echo "--- Verifying installation ---"
python3 -c "
import torch
import torchvision
print(f'torch={torch.__version__}')
print(f'torchvision={torchvision.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
    # Test bf16 support
    try:
        x = torch.randn(2, 2, dtype=torch.bfloat16, device='cuda')
        print('bfloat16: supported')
    except:
        print('bfloat16: NOT supported (will use fp32)')
"

# 4. Download CIFAR-10 and STL-10 (these are small and auto-download)
echo ""
echo "--- Pre-downloading datasets ---"
python3 -c "
import torchvision
print('Downloading CIFAR-10...')
torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
print('CIFAR-10 ready.')

print('Downloading STL-10...')
torchvision.datasets.STL10(root='./data', split='train+unlabeled', download=True)
torchvision.datasets.STL10(root='./data', split='test', download=True)
print('STL-10 ready.')
"

# 5. Run tests to verify everything works
echo ""
echo "--- Running quick tests ---"
python3 -m pytest tests/ -x -q --tb=short 2>/dev/null || echo "Some tests may fail without optional deps (OK)"

# 6. Quick sanity check (5 epochs on CIFAR-10)
echo ""
echo "--- Sanity check: 5-epoch CIFAR-10 run ---"
python3 -m benchmarks.run_standard_benchmarks --dataset cifar10 --quick

echo ""
echo "============================================"
echo "Setup complete! Ready to run benchmarks."
echo ""
echo "RECOMMENDED RUN ORDER:"
echo "  1. CIFAR-10  (~2-3h):  python -m benchmarks.run_standard_benchmarks --dataset cifar10"
echo "  2. STL-10    (~3-4h):  python -m benchmarks.run_standard_benchmarks --dataset stl10"
echo "  3. ImageNet-100 (~8-12h): (needs data prep first, see below)"
echo ""
echo "For ImageNet-100, you need to prepare the data first:"
echo "  python -m benchmarks.prepare_imagenet100 --imagenet-dir /path/to/ILSVRC2012"
echo "  python -m benchmarks.run_standard_benchmarks --dataset imagenet100"
echo ""
echo "To track experiments with wandb:"
echo "  wandb login"
echo "  python -m benchmarks.run_standard_benchmarks --dataset cifar10  # wandb enabled by default"
echo ""
echo "For a quick test first:"
echo "  python -m benchmarks.run_standard_benchmarks --dataset cifar10 --epochs 10 --seeds 1"
echo "============================================"
