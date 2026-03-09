# TCD-JEPA Architecture

## Overview

TCD-JEPA extends the Joint Embedding Predictive Architecture (JEPA) with three self-organizing systems that operate over JEPA's latent energy landscape. The key innovation is that predictive modules are not designed a priori but emerge from the system's own exploration dynamics.

## Base JEPA Architecture (Phase 1)

The foundation is a standard JEPA setup:

1. **Context Encoder** (`s_θ`): A Vision Transformer that encodes visible (context) patches into latent representations.
2. **Target Encoder** (`s_ξ`): An EMA copy of the context encoder that encodes target patches. No gradients flow through this encoder.
3. **Predictor** (`p_φ`): A smaller transformer that takes context representations and predicts target representations in latent space.
4. **Loss**: Smooth L1 loss between predicted and actual target representations.

### Training Loop

```
For each batch:
    1. Sample context masks (small blocks) and target masks (larger blocks)
    2. Context encoder: z = s_θ(images, context_masks)
    3. Predictor: z_pred = p_φ(z, context_masks, target_masks)
    4. Target encoder (no grad): h = sg(s_ξ(images, target_masks))
    5. Loss: L = smooth_l1(z_pred, h)
    6. Backprop through context encoder and predictor
    7. EMA update: ξ ← m·ξ + (1-m)·θ
```

## System 1 — Knowledge Corpus Releaser

Wraps the context encoder with instrumentation hooks that monitor:
- Per-layer embedding statistics (mean, std, norm)
- Representation diversity across the batch
- Information flow rates between layers

This data feeds into System 2's exploration decisions.

## System 2 — Energy Explorer (Phase 2)

Operates over the energy surface E(z) = ||p_φ(z) - target||². Instead of just minimizing the loss, System 2 actively explores:

1. **Blank Space Detection**: Identifies regions where the energy Hessian has small eigenvalues (flat landscape) or where the predictor has high output variance.
2. **Langevin Dynamics**: Samples from the energy surface biased toward blank space regions.
3. **Trajectory Tracking**: Records full paths through latent space during exploration.

## System 3 — Module Crystallizer (Phase 3)

Analyzes System 2's exploration trajectories using topological data analysis:

1. **Persistent Homology**: Computes Vietoris-Rips complexes on trajectory point clouds.
2. **Feature Identification**: Stable topological features (high persistence) become module candidates.
3. **Module Instantiation**: Converts topological features into lightweight predictor heads.

## Recursive Loop (Phase 4)

The three systems form a feedback loop:
- System 3's modules enrich System 1's representations
- Richer representations create a different energy landscape
- System 2 explores the new landscape, finding new patterns
- System 3 crystallizes new modules from new patterns
- Convergence is monitored via module formation rate and representation entropy
