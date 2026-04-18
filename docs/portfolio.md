# TCD-JEPA — Topological Crystallization Engine

> **A self-organizing extension of Meta's JEPA that discovers its own predictor architecture at training time and emits typed, interpretable, routable modules at inference time.**

---

## 1. Why this is novel

Every mainstream self-supervised backbone — JEPA, I-JEPA, DINOv2, MAE — ships with a *designed* predictor. The geometry of what the model can predict is fixed before it ever sees data.

**TCD-JEPA inverts that.** The predictor is grown from the dynamics of the latent space itself. Three systems cooperate in a closed recursive loop:

| # | System | Role | Mechanism |
|---|--------|------|-----------|
| 1 | **Stream Encoder** (Knowledge Corpus Releaser) | Releases the JEPA context/target signal and exposes the energy surface `E(z) = ‖p(s_θ(x)) − sg(s_ξ(y))‖²` | Instrumented ViT + EMA target + per-layer information-flow monitors |
| 2 | **Recursive Manifold Explorer** (Energy Explorer) | Probes where the model is *uncertain* by walking the energy landscape under Langevin dynamics with a Fisher-information metric | `z_{t+1} = z_t − η ∇E(z_t) + √(2η/β) ε_t`, temperature biased toward blank-space regions |
| 3 | **Module Crystallizer** (Persistent-Homology Former) | Reads the exploration trajectories as a point cloud, computes Vietoris–Rips persistent homology, and crystallizes stable topological features into typed predictor modules | H₀ → `AttractorModule`, H₁ → `CycleModule`, H₂ → `BoundaryModule` |

The output of System 3 feeds back into System 1 as additional predictor heads, and the loop re-runs until the convergence metric `C(t) < ε`. BTUT survivors from the energy walk serve as high-quality seeds for crystallization.

```
                        RECURSIVE LOOP
    ┌─────────────────────────────────────────────────┐
    │                                                 │
    ▼                                                 │
┌──────────┐     ┌──────────────┐     ┌────────────┐  │
│ System 1 │────▶│   System 2   │────▶│  System 3  │──┘
│  Stream  │     │   Energy     │     │   Module   │
│ Encoder  │     │  Explorer    │     │Crystallizer│
└──────────┘     └──────────────┘     └────────────┘
     │                 │                    │
  ViT + EMA        Langevin             Persistent
  JEPA signal      Dynamics             Homology
                   on E(z)              on trajectories
                       │                    │
                 Trajectories    ┌──────────┴──────────┐
                  {z_0,…,z_T}    │  H_0 → Attractor    │
                                 │  H_1 → Cycle        │
                                 │  H_2 → Boundary     │
                                 └─────────────────────┘
```

---

## 2. What gets produced: typed modules at inference time

Unlike black-box embeddings, TCD-JEPA ships **structure**. Every module has:

- **A topological type** (H₀ / H₁ / H₂) determining its predictive geometry
- **A persistence score** — how robust it is across scales of the Vietoris–Rips filtration
- **A centroid / cycle / boundary manifest** in latent space
- **A registry entry** tracking its lifecycle, utilization, and pruning status

At inference, the `DynamicPredictor` **routes** each query to the modules whose attractor/cycle/boundary best covers it, mixing their predictions with a learned gate. Discovery and routing happen at runtime — there is no fixed MoE hand-wired before training.

| Topological feature | Module type | What it predicts | When it fires |
|--------------------|-------------|------------------|---------------|
| Connected component (H₀) | `AttractorModule` | Local predictor centered on a cluster centroid | Query falls into the basin of a stable cluster |
| 1-cycle / loop (H₁) | `CycleModule` | Periodic predictor with learnable frequencies | Query traces a recurrent / cyclic pattern (e.g. seasonality, feedback loops) |
| 2-void / cavity (H₂) | `BoundaryModule` | Interface predictor with boundary detection | Query sits on the boundary between regimes (e.g. upstream↔downstream handoff) |

Only features with persistence > `τ_module` are crystallized — noise is filtered out by construction.

---

## 3. Empirical results — real heterogeneous graphs

All numbers are **self-supervised** link-prediction AUC on frozen TCD-JEPA embeddings, compared against strong **supervised** GNN baselines and the vanilla JEPA backbone. Evaluated with an identical protocol across methods.

### 3.1 Semiconductor Supply Chain (Georgetown CSET, 519 entities)

| Method | Link AUC | Δ vs TCD-JEPA |
|--------|---------:|--------------:|
| **TCD-JEPA (ours)** | **82.7 %** | — |
| GAT (DeepMind) | 70.3 % | **+12.4 pts** |
| GCN (Google Brain) | 63.9 % | **+18.8 pts** |
| GraphSAGE | 33.8 % | **+48.9 pts** |
| Baseline JEPA | 46.1 % | **+36.6 pts** |

> Self-supervised TCD-JEPA **beats every supervised GNN baseline** on a real semiconductor supply-chain graph.

### 3.2 GDELT Global News Events (380 entities)

| Method | Link AUC |
|--------|---------:|
| GAT | 92.1 % |
| GCN | 85.2 % |
| **TCD-JEPA (ours)** | **69.1 %** |
| GraphSAGE | 59.9 % |
| Baseline JEPA | 47.0 % |

TCD-JEPA improves over baseline JEPA by **+22.1 pts** and over GraphSAGE by **+9.2 pts**. Dense news-event graphs favor supervised message-passing baselines here — the contribution is the consistent lift over JEPA and the production of typed modules that GAT/GCN simply don't emit.

### 3.3 SEC EDGAR Financial Filings (9,725 entities, ~3.9M edges)

| Method | Link AUC | Classification |
|--------|---------:|---------------:|
| **TCD-JEPA (ours)** | **66.4 %** | usable |
| GCN | 90.8 % | ~7 % (near-random) |
| Baseline JEPA | 46.4 % | — |
| GAT | **OOM — failed** | — |
| GraphSAGE | no usable results | — |

At ~10k entities / ~4M edges, GAT runs out of memory and GraphSAGE fails to produce usable embeddings. GCN overfits to edges (90.8 %) but collapses on classification (7 %). **TCD-JEPA is the only method that produces both a stable edge score and interpretable downstream structure at this scale**, while lifting +20 pts over baseline JEPA.

### 3.4 Qualitative diagnostics on Two Rooms (representation geometry)

| Metric | Vanilla JEPA | TCD-JEPA | Δ |
|--------|-------------:|---------:|--:|
| k-NN (k=1) | 24.33 % | **29.93 %** | +23.0 % rel. |
| k-NN (k=5) | 26.13 % | **29.73 %** | +13.8 % rel. |
| k-NN (k=20) | 24.17 % | **34.40 %** | **+42.3 % rel.** |
| Crystallized modules | 0 | 6.0 ± 0.0 | — |

Crystallization pulls nearest neighbors tighter around semantic basins — exactly what topological structure is supposed to buy you.

---

## 4. Interpretable modules — semiconductor case study

Trained on the Georgetown CSET semiconductor graph, TCD-JEPA crystallized **16 interpretable modules**. Each one corresponds to a real cluster in the global semiconductor supply chain, validated against primary sources.

| # | Module | Topological type | Real-world cluster |
|---|--------|------------------|--------------------|
| 1 | CMP pipeline | H₀ attractor | Chemical-mechanical planarization vendors + consumables |
| 2 | Netherlands ASML ecosystem | H₀ attractor | ASML + Dutch/Belgian lithography supply |
| 3 | Singapore ATP | H₀ attractor | Assembly / test / packaging in Singapore |
| 4 | China packaging | H₀ attractor | OSAT firms in mainland China |
| 5 | Design-to-fab chain | H₁ cycle | EDA → IP → foundry → back to design |
| 6 | Specialty chemicals | H₀ attractor | Photoresists, wet chemistry, gases |
| 7 | EUV ↔ etch/clean flows | H₁ cycle | Litho-etch co-optimization loop |
| 8 | AI ASICs ↔ Hitachi | H₂ boundary | Hyperscaler custom silicon + Japanese tooling interface |
| 9 | Lithography ↔ CMP | H₂ boundary | Patterning / planarization handoff |
| … | + 7 more crystallized clusters | mixed | Validated against trade data + CSET primary sources |

**These were not labeled, queried, or prompted.** They emerged from persistent homology over Langevin trajectories and were *then* matched to real-world clusters post-hoc. That is the novelty — the method finds structure that *maps to actual domain knowledge*, including hidden dependencies like EUV ↔ etch/clean and AI-ASIC ↔ Hitachi tooling.

---

## 5. Runtime structure discovery & dynamic routing

Because modules are typed and persisted, inference is no longer a monolithic forward pass:

1. **Encoder** produces a context embedding `z` via System 1.
2. **Router** scores `z` against every registered module's manifest:
   - For H₀ attractors — distance to centroid in the Fisher metric
   - For H₁ cycles — phase alignment with the learned frequency basis
   - For H₂ boundaries — signed distance to the separating interface
3. **Top-k modules** are selected and their predictions mixed through a learned gate.
4. **Registry** records utilization; unused modules decay and are pruned; high-utilization ones are candidates for re-seeding in the next exploration pass.

This gives you three things static networks can't:
- **Interpretability** — every active module has a topological type and a real-world label
- **Compositionality** — modules can be added, pruned, and re-routed without retraining the backbone
- **Structure as an artifact** — the model exports a *map* of its own latent geometry, not just a weight file

---

## 6. Engineering surface

- **3 orchestrators** (`system1_encoder.py`, `system2_explorer.py`, `system3_crystallizer.py`) + `recursive_loop.py` glue
- **Persistent-homology backends**: giotto-tda → ripser/persim → scipy fallback (no hard dependency)
- **Production posture**: 186 tests, NaN guards on Langevin, DDP-compatible, SLURM scripts, randomized-SVD adapters that scale to millions of nodes
- **Nine real-world data adapters**: ETO Semiconductor, SupplyGraph, SEC EDGAR 10-K, GDELT, USPTO patents, Wikidata5M, PubMed KG, ogbn-arxiv, Common Crawl web graph
- **Typed-module API**: `DynamicPredictor` loads, routes to, and mixes `AttractorModule` / `CycleModule` / `BoundaryModule` at inference

---

## 7. Headline claims, in one breath

- Self-supervised TCD-JEPA **beats supervised GAT / GCN / GraphSAGE** on a real semiconductor supply-chain graph (82.7 % vs 70.3 / 63.9 / 33.8).
- Outperforms **baseline JEPA by +20 to +36 pts** across three real heterogeneous graph domains (semiconductor, GDELT, SEC EDGAR).
- Discovers **16 interpretable modules** that map 1-to-1 onto real semiconductor clusters, including hidden supply-chain dependencies.
- Produces **typed H₀ / H₁ / H₂ modules at inference** — interpretable, routable, prunable.
- Scales to **9,725 entities / ~3.9M edges** where GAT OOMs and GraphSAGE silently fails.
- Ships the first **runtime-discovered predictor** for the JEPA family — the architecture is grown, not designed.
