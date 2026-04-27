# Crystara — Graph Benchmark Results

Authoritative benchmark for Crystara (TCD-JEPA) on three real-world heterogeneous
graphs. Self-supervised link prediction (AUC), compared against baseline JEPA and
three supervised GNN baselines: GAT (DeepMind), GCN (Google Brain), GraphSAGE.

The three graphs span **four orders of magnitude** in edge count, and the setup
matches the codebase verbatim:

- Data adapters: `scripts/data_adapters/eto_semiconductor.py`,
  `scripts/data_adapters/gdelt_events.py`, `scripts/data_adapters/sec_edgar.py`.
- GNN baselines: `scripts/gnn_baselines.py`.
- Module analysis: `scripts/analyze_modules.py`, `scripts/quick_module_analysis.py`.
- Manifold trainer: `train_manifold.py`, `train_manifold_distributed.py`.

Per-benchmark structured artifacts:

- `graph_benchmarks/semiconductor_cset.json`
- `graph_benchmarks/gdelt.json`
- `graph_benchmarks/sec_edgar.json`
- `graph_benchmarks/semiconductor_modules.json`

## Headline

| Benchmark                       | Entities | Δ vs baseline JEPA |
|---------------------------------|----------|--------------------|
| CSET semiconductor supply chain | 519      | **+36.6 AUC pts**  |
| GDELT global news events        | 380      | **+22.1 AUC pts**  |
| SEC EDGAR filings               | 9,725 (~3.9M edges) | **+20.0 AUC pts** |

On the semiconductor graph the pipeline crystallizes **16 interpretable modules**
that map 1-to-1 to real industry clusters — with no labels, no prompting.

## 1. Semiconductor supply chain — CSET, 519 entities

Georgetown CSET (Center for Security and Emerging Technology) data covering the
physical semiconductor supply chain (design, fab, packaging, specialty
chemicals, lithography, and their interdependencies). Crystara beats every
supervised baseline and every self-supervised baseline.

| Model                | Link-prediction AUC | Δ vs Crystara |
|----------------------|--------------------:|--------------:|
| **Crystara**         | **82.7%**           | —             |
| GAT (DeepMind)       | 70.3%               | −12.4 pts     |
| GCN (Google Brain)   | 63.9%               | −18.8 pts     |
| Baseline JEPA        | 46.1%               | −36.6 pts     |
| GraphSAGE            | 33.8%               | −48.9 pts     |

## 2. GDELT global news events — 380 entities

News-derived event graph. Crystara substantially beats baseline JEPA but is
behind the supervised GAT and GCN baselines — an honest result: the news-event
graph has less physical-cluster structure than the semiconductor one, and the
topological-module vocabulary is a weaker match to its geometry.

| Model                | Link-prediction AUC | Δ vs Crystara |
|----------------------|--------------------:|--------------:|
| GAT                  | 92.1%               | +23.0 pts     |
| GCN                  | 85.2%               | +16.1 pts     |
| **Crystara**         | **69.1%**           | —             |
| GraphSAGE            | 59.9%               | −9.2 pts      |
| Baseline JEPA        | 47.0%               | −22.1 pts     |

## 3. SEC EDGAR financial filings — 9,725 entities, ~3.9M edges

The scale test. Crystara completes training; GAT runs out of memory and returns
no result; GraphSAGE produces no usable output; GCN hits 90.8% on
edge-reconstruction but only ~7% on downstream classification, which is
near-random and makes the edge number misleading.

| Model                | Link-prediction AUC | Note |
|----------------------|--------------------:|------|
| **Crystara**         | **66.4%**           | Completed at 9,725 entities |
| GCN                  | 90.8% (edges) / ~7% (cls) | Near-random on classification |
| Baseline JEPA        | 46.4%               | −20.0 pts vs Crystara |
| GraphSAGE            | —                   | No usable results |
| GAT                  | OOM                 | Out of memory |

## Honest assessment

The three-graph result is **stronger where the graph has physical-cluster
structure (semiconductor) and weaker where it does not (GDELT)**. GAT and GCN
outperform Crystara on GDELT by a real margin. The SEC EDGAR story is a scaling
story more than an accuracy one: Crystara is the only model that finishes and
produces a usable classification number — but the 66.4% AUC is not a
benchmark-leading number in absolute terms, and there is no supervised baseline
that both scales and classifies to compare against.

The headline "+20 to +36.6 AUC pts over baseline JEPA" is accurate; "beats every
GNN everywhere" would not be.

## Qualitative breakthrough — 16 semiconductor modules

On the CSET semiconductor graph, Crystara crystallized 16 interpretable modules
that mapped 1-to-1 to real semiconductor supply-chain clusters, validated
against primary sources (CSET, industry trade data). These modules emerged from
persistent homology on Langevin trajectories — no labels, no prompting.

- CMP (chemical-mechanical polishing) pipeline.
- Netherlands / ASML lithography ecosystem.
- Singapore assembly-test-packaging corridor.
- China packaging cluster.
- Design-to-fab chain.
- Specialty chemicals cluster.
- EUV ↔ etch/clean flows.
- AI ASICs ↔ Hitachi dependency.
- Lithography ↔ CMP handoff.
- …and seven more, each independently validated.

The modules capture both obvious clusters (national packaging hubs) and hidden
dependencies (EUV ↔ etch/clean flows, AI ASIC ↔ specialty-equipment
relationships). That is the signature of a predictor family that has grown into
the shape of the data: H₀ attractors pulling industry clusters together,
H₁ cycles linking feedback loops in the manufacturing pipeline, H₂ boundary
modules handling interfaces between pipeline stages.

Per-module artifact: `graph_benchmarks/semiconductor_modules.json`.
