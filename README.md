# TCD-JEPA: Tripartite Conditional Dynamics for Joint Embedding Predictive Architectures

> Self-organizing modular extensions to JEPA that enable recursive capability growth through energy landscape exploration and topological module crystallization.

## Abstract

Joint Embedding Predictive Architectures (JEPA) predict in latent space using a static predictor trained end-to-end. TCD-JEPA extends this paradigm with three dynamically interacting systems: (1) a **Stream Encoder** that instruments JEPA's context pipeline with information flow monitoring, (2) an **Energy Explorer** that uses Langevin dynamics to explore uncertain regions of the latent energy landscape, and (3) a **Module Crystallizer** that applies persistent homology to exploration trajectories, identifying and instantiating reusable predictor modules. These systems form a recursive loop where crystallized modules enrich representations, enabling qualitatively new predictive capabilities that static JEPA architectures cannot achieve.

## Key Insight

JEPA's predictor is *designed*, not *discovered*. TCD-JEPA proposes that predictive modules should **emerge** from the dynamics of the system's own exploration of what it doesn't yet know.

## Architecture

TCD-JEPA consists of three self-organizing systems operating over JEPA's latent space:

**System 1 — Knowledge Corpus Releaser (Stream Encoder)**
- Wraps JEPA's ViT encoder with instrumentation hooks
- Monitors representation diversity and information flow rates
- Provides the energy surface that System 2 explores

**System 2 — Recursive Manifold Knowledge Enveloper (Energy Explorer)**
- Explores the energy landscape via Langevin dynamics
- Targets high-entropy, low-confidence "blank space" regions
- Generates trajectories through latent space

**System 3 — Universal Module Former (Module Crystallizer)**
- Applies persistent homology to trajectory point clouds
- Identifies stable topological features (cycles, components, voids)
- Crystallizes features into lightweight predictor modules

**Recursive Loop:** System 3's modules feed back into System 1, enriching representations → System 2 explores the richer landscape → System 3 discovers new modules → repeat until convergence.

## Quick Start

### Installation

```bash
git clone https://github.com/direncode/tcd-jepa.git
cd tcd-jepa
pip install -e .
```

### Minimal Example

```python
from tcd_jepa.models import build_tcd_jepa

# Build a small-scale model
model = build_tcd_jepa(
    img_size=32,
    patch_size=4,
    embed_dim=192,
    depth=6,
    num_heads=3,
    predictor_embed_dim=96,
    predictor_depth=4,
)

# Forward pass with masks
import torch
images = torch.randn(4, 3, 32, 32)
masks_enc = [torch.randint(0, 64, (4, 10))]
masks_pred = [torch.randint(0, 64, (4, 16))]
result = model(images, masks_enc, masks_pred)
print(f"Loss: {result['loss'].item():.4f}")
```

## Mathematical Framework

**Energy Function:**
```
E_pred(x, y) = ||p_φ(s_θ(x)) - sg(s_ξ(y))||²
```
where `s_θ` is the context encoder, `s_ξ` is the EMA target encoder, `p_φ` is the predictor, and `sg` is stop-gradient.

**Langevin Exploration (System 2):**
```
z_{t+1} = z_t - η∇_z E(z_t) + √(2η/β) · ε_t
```

**Persistent Homology (System 3):**
Vietoris-Rips complexes computed at multiple scales on trajectory point clouds, with features thresholded by persistence for module instantiation.

**Convergence:**
```
C(t) = |M(t) - M(t-1)| / M(t) + KL(R(t) || R(t-1)) + |S(t) - S(t-1)|
```

## Project Structure

```
tcd-jepa/
├── tcd_jepa/
│   ├── core/           # Systems 1-3 and recursive loop
│   ├── modules/        # Predictor variants and module registry
│   ├── exploration/    # Langevin dynamics and trajectory tracking
│   ├── topology/       # Persistent homology and feature extraction
│   ├── models/         # ViT encoder, context/target encoders, full model
│   ├── training/       # Trainer, losses, schedulers, metrics
│   └── utils/          # Config, logging, masking, checkpointing
├── configs/            # YAML experiment configs
├── experiments/        # Experiment runners and analysis
├── tests/              # Unit and integration tests
└── docs/               # Architecture and math documentation
```

## Acknowledgments

TCD-JEPA builds on the JEPA paradigm introduced by Yann LeCun and implemented by Meta's FAIR team. We gratefully acknowledge the following open-source implementations:

- [I-JEPA](https://github.com/facebookresearch/ijepa) (Assran et al., 2023)
- [V-JEPA](https://github.com/facebookresearch/jepa) (Bardes et al., 2024)
- [V-JEPA 2](https://github.com/facebookresearch/vjepa2) (Bardes et al., 2025)
- [eb_jepa](https://github.com/facebookresearch/eb_jepa) (Meta FAIR)

## Citation

```bibtex
@article{diren2026tcdjepa,
  title={TCD-JEPA: Tripartite Conditional Dynamics for Self-Organizing Prediction in Latent Space},
  author={Diren},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

## License

MIT
