# TCD-JEPA / Crystara: Tripartite Conditional Dynamics for Joint Embedding Predictive Architectures

> Self-organizing modular extensions to JEPA that enable recursive capability growth through energy landscape exploration and topological module crystallization. Marketed as **Crystara**, the structure primitive in the horizontal-intelligence stack.

## Abstract

Joint Embedding Predictive Architectures (JEPA) learn representations by predicting target embeddings from context embeddings in latent space, using a static predictor trained end-to-end. While this paradigm avoids the pitfalls of pixel-level reconstruction, the predictor architecture remains fixed -- its structure is designed, not discovered. TCD-JEPA extends JEPA with three dynamically interacting systems: (1) a **Stream Encoder** that instruments JEPA's context pipeline with information flow monitoring, (2) an **Energy Explorer** that uses Langevin dynamics to probe uncertain regions of the latent energy landscape, and (3) a **Module Crystallizer** that applies persistent homology to exploration trajectories, identifying stable topological features and instantiating them as reusable predictor modules. These systems form a recursive feedback loop where crystallized modules enrich representations, enabling the discovery of predictive capabilities that static architectures cannot achieve. We demonstrate this self-organizing mechanism on CIFAR-10, a Two Rooms navigation environment, and three real-world heterogeneous graphs (semiconductor supply chain, GDELT global news, SEC EDGAR), showing module formation, convergence of the recursive loop, and quantitative gains over baseline JEPA and supervised GNNs.

## Key Insight

JEPA's predictor is *designed*, not *discovered*. TCD-JEPA proposes that predictive modules should **emerge** from the dynamics of the system's own exploration of what it doesn't yet know.

## Graph benchmarks (Crystara)

Self-supervised link prediction (AUC) on three real heterogeneous graphs, against
baseline JEPA and three supervised GNN baselines (GAT, GCN, GraphSAGE). Full
tables, JSON artifacts, and an honest assessment live in
[`results/GRAPH_BENCHMARKS.md`](results/GRAPH_BENCHMARKS.md).

| Benchmark                       | Entities             | Δ vs baseline JEPA |
|---------------------------------|---------------------:|-------------------:|
| CSET semiconductor supply chain | 519                  | **+36.6 AUC pts**  |
| GDELT global news events        | 380                  | **+22.1 AUC pts**  |
| SEC EDGAR filings               | 9,725 (~3.9M edges)  | **+20.0 AUC pts**  |

On CSET (519 entities), Crystara reaches **82.7% AUC**, beating GAT (DeepMind,
70.3%), GCN (Google Brain, 63.9%), GraphSAGE (33.8%) and baseline JEPA (46.1%).
On GDELT (380 entities), Crystara is +22.1 pts over JEPA but behind GAT/GCN —
the news-event graph has weaker physical-cluster structure than semiconductors.
On SEC EDGAR (9,725 entities, ~3.9M edges), Crystara is the only model that
finishes: GAT runs out of memory, GraphSAGE produces no usable output, GCN gets
90.8% on edges but ~7% on downstream classification.

On the semiconductor graph the pipeline crystallizes **16 interpretable modules**
that map 1-to-1 to real industry clusters (CMP pipeline, ASML lithography,
Singapore ATP corridor, China packaging cluster, EUV ↔ etch/clean flows, …) —
with no labels, no prompting. See
[`results/graph_benchmarks/semiconductor_modules.json`](results/graph_benchmarks/semiconductor_modules.json).

Reproduction surface:

- Data adapters: `scripts/data_adapters/eto_semiconductor.py`,
  `scripts/data_adapters/gdelt_events.py`, `scripts/data_adapters/sec_edgar.py`.
- GNN baselines: `scripts/gnn_baselines.py`.
- Manifold trainer: `train_manifold.py`, `train_manifold_distributed.py`.
- Module analysis: `scripts/analyze_modules.py`, `scripts/quick_module_analysis.py`.

## Architecture

```
                        RECURSIVE LOOP
    ┌─────────────────────────────────────────────────┐
    │                                                 │
    ▼                                                 │
┌──────────┐     ┌──────────────┐     ┌───────────┐  │
│ System 1 │────▶│   System 2   │────▶│ System 3  │──┘
│  Stream  │     │   Energy     │     │  Module   │
│ Encoder  │     │  Explorer    │     │Crystallizer│
└──────────┘     └──────────────┘     └───────────┘
     │                 │                    │
     │            Langevin             Persistent
  ViT +           Dynamics             Homology
  EMA              on E(z)            on trajectories
  JEPA                │                    │
                 Trajectories    ┌─────────┴─────────┐
                  {z_0,...,z_T}  │  H_0 → Attractor  │
                                │  H_1 → Cycle       │
                                │  H_2 → Boundary    │
                                └────────────────────┘
```

**System 1 -- Knowledge Corpus Releaser (Stream Encoder)**
- Wraps JEPA's ViT encoder with instrumentation hooks
- Monitors representation diversity and information flow rates per layer
- Provides the energy surface E(z) = ||p(s_theta(x)) - sg(s_xi(y))||^2

**System 2 -- Recursive Manifold Knowledge Enveloper (Energy Explorer)**
- Detects blank spaces via Hessian eigenvalue analysis and perturbation variance
- Explores the energy landscape via Langevin dynamics: z_{t+1} = z_t - eta * grad E + noise
- Temperature biased toward blank regions (lower beta = more exploration)
- Computes Fisher information metric for Riemannian geometry of the latent space
- Records exploration trajectories for System 3

**System 3 -- Universal Module Former (Module Crystallizer)**
- Computes Vietoris-Rips persistent homology on trajectory point clouds
- Analyzes persistence diagrams to identify stable topological features
- Converts features to predictor modules:
  - H_0 (connected components) -> **AttractorModule**: local predictor centered on cluster centroid
  - H_1 (loops) -> **CycleModule**: periodic predictor with learnable frequencies
  - H_2 (voids) -> **BoundaryModule**: interface predictor with boundary detection
- Manages module lifecycle: registration, performance tracking, pruning

**Recursive Loop:** System 3's modules feed back into System 1, enriching representations -> System 2 explores the richer landscape -> System 3 discovers new modules -> convergence monitored via C(t).

## Quick Start

### Installation

```bash
git clone https://github.com/direncode/tcd-jepa.git
cd tcd-jepa
pip install -e ".[dev]"

# For persistent homology (optional, falls back to scipy)
pip install giotto-tda  # or: pip install ripser persim
```

### Training

```bash
# Vanilla JEPA on CIFAR-10
python train.py --config configs/small_scale.yaml

# TCD-JEPA with recursive loop
python train.py --config configs/small_scale.yaml --tcd

# Two Rooms experiment (vanilla vs TCD-JEPA comparison)
python -m experiments.two_rooms.run --config configs/two_rooms.yaml

# CIFAR-10 comparison experiment
python -m experiments.image.run --config configs/small_scale.yaml
```

### Minimal Code Example

```python
import torch
from tcd_jepa.models.tcd_jepa_model import build_tcd_jepa
from tcd_jepa.core.recursive_loop import RecursiveLoop
from tcd_jepa.core.system1_encoder import StreamEncoder

# Build model
model = build_tcd_jepa(img_size=32, patch_size=4, embed_dim=192, depth=6,
                       num_heads=3, predictor_embed_dim=96, predictor_depth=4)

# Standard JEPA forward pass
images = torch.randn(4, 3, 32, 32)
masks_enc = [torch.randint(0, 64, (4, 10))]
masks_pred = [torch.randint(0, 64, (4, 16))]
result = model(images, masks_enc, masks_pred)
print(f"Loss: {result['loss'].item():.4f}")

# TCD recursive loop
loop = RecursiveLoop(embed_dim=192, explore_every=1, crystallize_every=2,
                     langevin_steps=20)
stream = StreamEncoder(model.context_encoder, model.target_encoder)

with torch.no_grad():
    z = model.context_encoder(images)
    t = model.target_encoder(images)
energy_fn = stream.make_energy_fn(t)
loop_result = loop.step(z, energy_fn, epoch=0)
print(f"Explored: {loop_result['explored']}, Modules: {loop.num_modules}")
```

## Experiments

### Two Rooms

The Two Rooms environment is a gridworld with two rooms connected by a doorway. The agent generates random trajectories producing RGB observations. TCD-JEPA is compared against vanilla JEPA to demonstrate:
- Module formation over training
- Convergence of the recursive loop
- Qualitative representation differences

```bash
python -m experiments.two_rooms.run --config configs/two_rooms.yaml training.epochs=20
```

### CIFAR-10

Standard CIFAR-10 pretraining with ViT-tiny. The contribution is not SOTA benchmarks but demonstrating that the self-organizing mechanism works.

```bash
python -m experiments.image.run --config configs/small_scale.yaml training.epochs=30
```

### Ablation Studies

Test each system individually:

```bash
# Full TCD-JEPA
python train.py --config configs/ablation.yaml --tcd

# Vanilla JEPA baseline
python train.py --config configs/ablation.yaml
```

## Mathematical Framework

**Energy Function:**

E(x, y) = ||s_theta(x) - s_xi(y)||^2

where s_theta is the context encoder and s_xi is the EMA target encoder. The predictor energy:

E_pred(x, y) = ||p_phi(s_theta(x)) - sg(s_xi(y))||^2

**Blank Space Detection (System 2):**

H(z) = nabla^2_z E(z) -- the Hessian of energy
blank_space(z) = True if lambda_min(H(z)) < tau

Also: regions where predictor output variance is high under perturbation.

**Langevin Exploration (System 2):**

z_{t+1} = z_t - eta * nabla_z E(z_t) + sqrt(2*eta/beta) * epsilon_t,  epsilon ~ N(0, I)

Temperature beta biased to explore blank space regions (lower beta = more exploration in uncertain areas).

**Fisher Information Metric (System 2):**

F_ij(z) = (1/sigma^2) * sum_k (dp_k/dz_i)(dp_k/dz_j)

Defines the Riemannian geometry of the latent space for geometrically-aware exploration.

**Persistent Homology (System 3):**

Given trajectory T = {z_0, z_1, ..., z_N}, compute the Vietoris-Rips complex at multiple scales:
- H_0: connected components (clusters) -> AttractorModule
- H_1: loops (cycles) -> CycleModule
- H_2: voids (cavities) -> BoundaryModule

Features with persistence > tau_module become module candidates.

**Convergence Metric:**

C(t) = |M(t) - M(t-1)| / M(t) + KL(R(t) || R(t-1)) + |S(t) - S(t-1)|

where M(t) = active modules, R(t) = representation distribution, S(t) = energy landscape smoothness. System converges when C(t) < epsilon for k consecutive iterations.

## Project Structure

```
tcd-jepa/
├── train.py                    # Main training entry point
├── configs/
│   ├── small_scale.yaml        # CIFAR-10 small-scale config
│   ├── two_rooms.yaml          # Two Rooms environment config
│   ├── ablation.yaml           # Ablation study config
│   └── default.yaml            # Full-scale default config
├── tcd_jepa/
│   ├── core/
│   │   ├── energy_landscape.py # Energy function E(z)
│   │   ├── system1_encoder.py  # StreamEncoder orchestrator
│   │   ├── system2_explorer.py # EnergyExplorer orchestrator
│   │   ├── system3_crystallizer.py  # ModuleCrystallizer orchestrator
│   │   └── recursive_loop.py  # RecursiveLoop + ConvergenceMonitor
│   ├── exploration/
│   │   ├── blank_space_detector.py  # Hessian + variance detection
│   │   ├── langevin.py         # Langevin dynamics sampler
│   │   ├── trajectory_tracker.py    # Trajectory storage
│   │   └── fisher_metric.py    # Fisher information metric
│   ├── topology/
│   │   ├── persistent_homology.py   # Vietoris-Rips PH (giotto/ripser/scipy)
│   │   ├── persistence_diagrams.py  # Diagram analysis + thresholding
│   │   └── feature_extraction.py    # TopologicalFeature -> module mapping
│   ├── modules/
│   │   ├── predictor.py        # Vanilla JEPA predictor (I-JEPA compatible)
│   │   ├── dynamic_predictor.py     # Predictor + crystallized modules
│   │   ├── module_factory.py   # Attractor/Cycle/Boundary module creation
│   │   └── module_registry.py  # Module lifecycle management
│   ├── models/
│   │   ├── vision_transformer.py    # ViT encoder (I-JEPA compatible)
│   │   ├── context_encoder.py  # Instrumented encoder (System 1)
│   │   ├── target_encoder.py   # EMA target encoder
│   │   └── tcd_jepa_model.py   # Full JEPA model assembly
│   ├── training/
│   │   ├── trainer.py          # Training loop
│   │   ├── losses.py           # JEPA loss functions
│   │   ├── schedulers.py       # LR and WD schedules
│   │   └── metrics.py          # Metric accumulation
│   └── utils/
│       ├── config.py           # YAML config + CLI overrides
│       ├── logging.py          # CSV + wandb logging
│       ├── masking.py          # Block mask generation (I-JEPA)
│       ├── visualization.py    # Energy heatmaps, trajectories, PD plots
│       ├── tensors.py          # Tensor utilities
│       └── checkpointing.py    # Model checkpoint save/load
├── experiments/
│   ├── two_rooms/              # Two Rooms gridworld experiments
│   │   ├── environment.py      # TwoRoomsEnv + TwoRoomsDataset
│   │   ├── run.py              # Vanilla vs TCD-JEPA comparison
│   │   └── analysis.py         # Result analysis
│   └── image/                  # CIFAR-10 experiments
│       ├── run.py              # Training + comparison
│       └── analysis.py         # Result analysis
├── tests/                      # 65 tests across all systems
│   ├── test_system1.py         # Encoder, context, target, utils
│   ├── test_system2.py         # Blank space, Langevin, Fisher, explorer
│   ├── test_system3.py         # PH, persistence, factory, registry, crystallizer
│   ├── test_recursive_loop.py  # Convergence, loop orchestration
│   └── test_integration.py     # Full model, losses, schedulers
├── paper/
│   └── tcd_jepa.tex            # ArXiv preprint (LaTeX)
└── pyproject.toml              # Package metadata + dependencies
```

## Dependencies

```
torch >= 2.0
torchvision
numpy
scipy
matplotlib
seaborn
pyyaml
einops
tqdm
giotto-tda          # persistent homology (optional, falls back to scipy)
wandb               # experiment tracking (optional)
```

## Acknowledgments

TCD-JEPA builds on the JEPA paradigm introduced by Yann LeCun and implemented by Meta's FAIR team. We gratefully acknowledge:

- [I-JEPA](https://github.com/facebookresearch/ijepa) (Assran et al., 2023) -- core architecture reference
- [V-JEPA](https://github.com/facebookresearch/jepa) (Bardes et al., 2024) -- video extension
- [V-JEPA 2](https://github.com/facebookresearch/vjepa2) (Bardes et al., 2025) -- scaled video JEPA
- [eb_jepa](https://github.com/facebookresearch/eb_jepa) (Meta FAIR) -- energy-based JEPA

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
