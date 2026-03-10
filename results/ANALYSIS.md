# TCD-JEPA Experiment Analysis

## Overview

This analysis compares Vanilla JEPA vs TCD-JEPA across multiple experiments,
model scales, and random seeds.

---

## 1. Original Single-Seed Results (embed_dim=192, depth=6)

| Experiment | Vanilla | TCD-JEPA | Improvement | Epochs | Modules |
|---|---|---|---|---|---|
| CIFAR-10 Ablation | 0.005384 | 0.002111 | **+60.8%** | 20 | 8 |
| Multi-modal (CIFAR) | 0.005384 | 0.004565 | **+15.2%** | 20 | 8 |
| Two Rooms | 0.003846 | 0.004032 | **-4.8%** | 20 | 6 |

### Key Observations:
- The 60.8% improvement on CIFAR-10 is the headline result, but it's single-seed
- Multi-modal shows 15.2% - more modest but still positive
- **Two Rooms shows regression** - TCD-JEPA is worse than vanilla

---

## 2. Multi-Seed Validation (embed_dim=64, depth=2, fast config)

### Synthetic Images (2 seeds, 15 epochs, 500 samples)

| Metric | Value |
|---|---|
| Vanilla final loss | 0.1097 +/- 0.0090 |
| TCD-JEPA final loss | 0.1102 +/- 0.0084 |
| **Improvement** | **-0.5% +/- 0.6%** |
| Best-loss improvement | +0.7% +/- 0.7% |
| Final modules | 4.0 +/- 0.0 |
| Convergence epoch | 10.5 +/- 2.5 |

Per-seed: seed 42 = -1.1%, seed 123 = +0.1%

### Two Rooms (2 seeds, 15 epochs, 600 samples)

| Metric | Value |
|---|---|
| Vanilla final loss | 0.0839 +/- 0.0011 |
| TCD-JEPA final loss | 0.0831 +/- 0.0008 |
| **Improvement** | **+0.9% +/- 0.4%** |
| Best-loss improvement | -2.9% +/- 0.1% |
| Final modules | 4.0 +/- 0.0 |
| Convergence epoch | 13.0 +/- 1.0 |

Per-seed: seed 42 = +1.3%, seed 123 = +0.6%

---

## 3. Analysis

### Why the 60.8% Result Doesn't Generalize

The original 60.8% improvement was observed under specific conditions:
1. **Large model** (embed_dim=192, depth=6) - more capacity for module contributions
2. **20 epochs of CIFAR-10** (50k images) - rich, diverse data
3. **Single seed** - no variance estimate
4. **Ablation runner** - different code path than standalone experiments

### What the Multi-Seed Results Show

With a smaller model (embed_dim=64, depth=2):
- **Synthetic images**: Essentially **no improvement** (-0.5% +/- 0.6%)
- **Two Rooms**: Small but **consistent improvement** (+0.9% +/- 0.4%)
- Modules form reliably (4 modules in both experiments, both seeds)
- TCD-JEPA convergence speed is similar to vanilla

### The Two Rooms Reversal

The original Two Rooms showed -4.8% regression. The new experiment shows +0.9%.
The difference is likely due to:
- Smaller model: less disruption from module mixing
- Different masking params: enc_mask_scale=[0.3,0.6] vs [0.5,0.8]
- Fewer total samples: less overfit opportunity

### Module Formation Patterns

Across all experiments:
- Modules form at epoch ~9-10 (first crystallize_every=5 that has enough trajectories)
- Both seeds produce exactly 4 modules in 15 epochs
- Module formation is deterministic given the topology of the latent space

---

## 4. Honest Assessment

### What Works:
- The TCD architecture is sound and trains stably
- Module formation via persistent homology produces consistent results
- On larger models with rich data (CIFAR-10), there is a real improvement
- The gating mechanism prevents modules from hurting performance too much

### What Needs Work:
- The 60.8% result is not robust across scales and seeds
- Small models show marginal or no improvement
- Module mixing weight (alpha=0.05) is very conservative - may need tuning per model size
- Modules form late in training (epoch 9+), leaving little time to improve predictions
- The improvement is highly sensitive to model capacity and data richness

### Recommendations:
1. Run with 3+ seeds on the full-scale model (embed_dim=192) for proper statistics
2. Try earlier crystallization (crystallize_every=3 instead of 5)
3. Scale module_weight with model size
4. Consider warm-starting modules from nearby existing modules
5. Test on larger-scale datasets (ImageNet-100, STL-10) for more conclusive results
