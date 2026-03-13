"""Deep signal harvesting from trained TCD-JEPA model.

Extracts every deep architectural signal after training:
- Energy landscape (blank scores, Hessian traces)
- Fisher information (metric tensor traces)
- Attention structure (aggregated attention graph, hubs, bridges)
- Module crystallization (types, performance, attractor positions)
- Convergence dynamics (history, stability)
- Representation quality (effective rank, uniformity, neighborhood preservation)
- Uncertainty field (epistemic uncertainty, coverage)

All signals are aggregated into a DeepSignalProfile that flows into the
intelligence engines.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("tcd_jepa.deep_signals")


@dataclass
class DeepSignalProfile:
    """Unified container for all deep architectural signals."""

    # Energy landscape
    energy_map: Optional[torch.Tensor] = None        # [N] per-chunk energy
    blank_scores: Optional[torch.Tensor] = None      # [N] blank space scores
    hessian_traces: Optional[torch.Tensor] = None     # [N] curvature per chunk
    fisher_traces: Optional[torch.Tensor] = None      # [N] Fisher info trace

    # Topology
    betti_numbers: dict = field(default_factory=dict)  # {0: int, 1: int, 2: int}
    persistence_diagram: dict = field(default_factory=dict)  # {dim: [(birth, death, persistence)]}
    total_persistence: float = 0.0
    topological_features: list = field(default_factory=list)

    # Attention structure
    attention_graph: Optional[torch.Tensor] = None    # [N, N] aggregated attention
    attention_hubs: list = field(default_factory=list)  # chunk indices
    attention_bridges: list = field(default_factory=list)  # (i, j, weight) tuples

    # Module crystallization
    num_modules: int = 0
    module_types: dict = field(default_factory=dict)   # {attractor: n, cycle: n, boundary: n}
    module_performance: dict = field(default_factory=dict)  # {module_id: avg_loss}
    attractor_positions: list = field(default_factory=list)
    cycle_frequencies: list = field(default_factory=list)

    # Convergence
    convergence_score: float = 0.0
    convergence_history: list = field(default_factory=list)
    representation_stability: float = 0.0

    # Representation quality
    effective_rank: float = 0.0
    uniformity: float = 0.0
    neighborhood_preservation: float = 0.0
    link_prediction_auc: float = 0.0

    # Uncertainty field
    epistemic_uncertainty: Optional[torch.Tensor] = None  # [N]
    coverage_ratio: float = 0.0


class DeepSignalHarvester:
    """Extracts all deep signals from a trained TCD-JEPA model."""

    def __init__(
        self,
        blank_detection_samples: int = 200,
        fisher_samples: int = 100,
        attention_extraction: bool = True,
    ):
        self.blank_detection_samples = blank_detection_samples
        self.fisher_samples = fisher_samples
        self.attention_extraction = attention_extraction

    def harvest(
        self,
        model: nn.Module,
        dataset,
        corpus,
        recursive_loop=None,
        device: torch.device = torch.device("cpu"),
    ) -> DeepSignalProfile:
        """Run all signal extraction and return unified profile."""
        profile = DeepSignalProfile()
        N = len(corpus.chunks)

        logger.info(f"Harvesting deep signals from {N} chunks...")

        # 1. Extract per-chunk energy + blank detection
        self._harvest_energy_landscape(profile, model, dataset, N, device)

        # 2. Compute Fisher traces
        self._harvest_fisher_information(profile, model, dataset, N, device)

        # 3. Extract attention patterns
        if self.attention_extraction:
            self._harvest_attention_structure(profile, model, dataset, N, device)

        # 4. Get topology and module info from recursive loop
        if recursive_loop is not None:
            self._harvest_topology(profile, recursive_loop)
            self._harvest_convergence(profile, recursive_loop)

        # 5. Compute representation quality
        self._harvest_representation_quality(profile, model, dataset, corpus, device)

        # 6. Compute uncertainty field
        self._harvest_uncertainty(profile, N)

        logger.info(f"Deep signal harvest complete. Coverage: {profile.coverage_ratio:.1%}")
        return profile

    def _harvest_energy_landscape(
        self, profile: DeepSignalProfile, model: nn.Module,
        dataset, N: int, device: torch.device,
    ) -> None:
        """Extract energy landscape: blank scores and Hessian traces."""
        from tcd_jepa.exploration.blank_space_detector import BlankSpaceDetector

        model.eval()
        detector = BlankSpaceDetector()
        all_blank_scores = []
        all_hessian_traces = []
        all_energies = []

        num_samples = min(self.blank_detection_samples, len(dataset))

        with torch.no_grad():
            for i in range(num_samples):
                sample = dataset[i]
                fp = sample["fingerprints"].unsqueeze(0).to(device)
                coords = sample.get("coords")
                if coords is not None:
                    coords = coords.unsqueeze(0).to(device)
                adj = sample.get("adjacency")
                if adj is not None:
                    adj = adj.unsqueeze(0).to(device)
                vel = sample.get("velocity")
                if vel is not None:
                    vel = vel.unsqueeze(0).to(device)

                z = model.context_encoder(fp, coords=coords, adjacency=adj, velocity=vel)
                z_flat = z.reshape(-1, z.shape[-1])

                # Energy: prediction error as energy proxy
                t = model.target_encoder(fp, coords=coords, adjacency=adj, velocity=vel)
                t_flat = t.reshape(-1, t.shape[-1])
                energy = (z_flat - t_flat).pow(2).sum(dim=-1)
                all_energies.append(energy.cpu())

        # Run blank detection on actual encoder representations
        if all_energies:
            energies_cat = torch.cat(all_energies)

            # Collect real encoder outputs for blank detection
            all_z = []
            with torch.no_grad():
                for i in range(min(4, num_samples)):
                    sample = dataset[i]
                    fp = sample["fingerprints"].unsqueeze(0).to(device)
                    c_ = sample.get("coords")
                    if c_ is not None:
                        c_ = c_.unsqueeze(0).to(device)
                    a_ = sample.get("adjacency")
                    if a_ is not None:
                        a_ = a_.unsqueeze(0).to(device)
                    v_ = sample.get("velocity")
                    if v_ is not None:
                        v_ = v_.unsqueeze(0).to(device)
                    z_enc = model.context_encoder(fp, coords=c_, adjacency=a_, velocity=v_)
                    all_z.append(z_enc.reshape(-1, z_enc.shape[-1]).cpu())

            sample_z = torch.cat(all_z, dim=0)[:min(64, N)]

            # Energy function based on prediction error
            def energy_fn(z_in):
                return z_in.pow(2).sum(dim=-1)

            try:
                blank_result = detector.detect(sample_z, energy_fn)
                combined = blank_result["combined_score"]
                # Normalize to [0, 1]
                if combined.max() > combined.min():
                    combined = (combined - combined.min()) / (combined.max() - combined.min())
                # Pad/trim to N
                blank_scores = torch.zeros(N)
                blank_scores[:min(len(combined), N)] = combined[:N]
                profile.blank_scores = blank_scores

                hessian = blank_result["hessian_info"]["trace"]
                hessian_padded = torch.zeros(N)
                hessian_padded[:min(len(hessian), N)] = hessian[:N]
                profile.hessian_traces = hessian_padded
            except Exception as e:
                logger.warning(f"Blank detection failed: {e}")
                profile.blank_scores = torch.zeros(N)
                profile.hessian_traces = torch.zeros(N)

            # Aggregate energy to per-chunk
            energy_map = torch.zeros(N)
            chunk_counts = torch.zeros(N)
            offset = 0
            for i in range(len(all_energies)):
                e = all_energies[i]
                sample = dataset[i]
                indices = sample.get("indices", torch.arange(e.shape[0]))
                for j, idx in enumerate(indices):
                    idx_val = idx.item() if isinstance(idx, torch.Tensor) else idx
                    if idx_val < N and j < len(e):
                        energy_map[idx_val] += e[j].item()
                        chunk_counts[idx_val] += 1
            valid = chunk_counts > 0
            energy_map[valid] /= chunk_counts[valid]
            profile.energy_map = energy_map

    def _harvest_fisher_information(
        self, profile: DeepSignalProfile, model: nn.Module,
        dataset, N: int, device: torch.device,
    ) -> None:
        """Compute Fisher information traces."""
        from tcd_jepa.exploration.fisher_metric import FisherMetric

        model.eval()
        fisher = FisherMetric(num_jacobian_samples=8)
        fisher_traces = torch.zeros(N)
        fisher_counts = torch.zeros(N)

        num_samples = min(self.fisher_samples, len(dataset))

        try:
            with torch.no_grad():
                for i in range(num_samples):
                    sample = dataset[i]
                    fp = sample["fingerprints"].unsqueeze(0).to(device)
                    coords = sample.get("coords")
                    if coords is not None:
                        coords = coords.unsqueeze(0).to(device)
                    adj = sample.get("adjacency")
                    if adj is not None:
                        adj = adj.unsqueeze(0).to(device)
                    vel = sample.get("velocity")
                    if vel is not None:
                        vel = vel.unsqueeze(0).to(device)

                    z = model.context_encoder(fp, coords=coords, adjacency=adj, velocity=vel)
                    z_flat = z.reshape(-1, z.shape[-1])

                    # Use target encoder as predictor proxy for Fisher computation.
                    # The target encoder maps nearby inputs to nearby outputs, so its
                    # Jacobian captures how sensitively the model maps this region.
                    target_mean = model.target_encoder(fp, coords=coords, adjacency=adj, velocity=vel)
                    target_flat = target_mean.reshape(-1, target_mean.shape[-1])

                    def pred_fn(z_in, _tf=target_flat[:16]):
                        # Prediction-error-based output: how different is z from target?
                        # This makes Fisher trace high where prediction is sensitive
                        # and low where it's flat (blank spaces).
                        return z_in - _tf[:len(z_in)]

                    traces = fisher.compute_metric_tensor_trace(z_flat[:16], pred_fn)

                    indices = sample.get("indices", torch.arange(z_flat.shape[0]))
                    for j in range(min(len(traces), len(indices))):
                        idx = indices[j].item() if isinstance(indices[j], torch.Tensor) else indices[j]
                        if idx < N:
                            fisher_traces[idx] += traces[j].item()
                            fisher_counts[idx] += 1

            valid = fisher_counts > 0
            fisher_traces[valid] /= fisher_counts[valid]
            profile.fisher_traces = fisher_traces
        except Exception as e:
            logger.warning(f"Fisher computation failed: {e}")
            profile.fisher_traces = torch.zeros(N)

    def _harvest_attention_structure(
        self, profile: DeepSignalProfile, model: nn.Module,
        dataset, N: int, device: torch.device,
    ) -> None:
        """Extract aggregated attention patterns via forward hooks."""
        model.eval()

        # Guard against OOM for large corpora
        if N > 5000:
            logger.warning(f"Corpus has {N} chunks — attention matrix would be {N}x{N}. "
                           f"Capping at 5000 for memory safety.")
            N = 5000

        attention_accumulator = torch.zeros(N, N)
        attention_counts = torch.zeros(N, N)

        # Hook to capture attention weights
        captured_attentions = []

        def attention_hook(module, input, output):
            if isinstance(output, tuple) and len(output) > 1:
                # Some attention modules return (output, attention_weights)
                captured_attentions.append(output[1].detach().cpu())

        # Register hooks on attention layers
        hooks = []
        for name, module in model.context_encoder.named_modules():
            if "attn" in name.lower() and hasattr(module, "forward"):
                try:
                    h = module.register_forward_hook(attention_hook)
                    hooks.append(h)
                except Exception:
                    pass

        num_samples = min(100, len(dataset))

        try:
            with torch.no_grad():
                for i in range(num_samples):
                    captured_attentions.clear()
                    sample = dataset[i]
                    fp = sample["fingerprints"].unsqueeze(0).to(device)
                    coords = sample.get("coords")
                    if coords is not None:
                        coords = coords.unsqueeze(0).to(device)
                    adj = sample.get("adjacency")
                    if adj is not None:
                        adj = adj.unsqueeze(0).to(device)
                    vel = sample.get("velocity")
                    if vel is not None:
                        vel = vel.unsqueeze(0).to(device)

                    model.context_encoder(fp, coords=coords, adjacency=adj, velocity=vel)

                    indices = sample.get("indices", torch.arange(fp.shape[1]))

                    # Aggregate captured attention weights
                    for attn_w in captured_attentions:
                        if attn_w.dim() >= 3:
                            # Average over heads: [B, H, T, T] -> [T, T]
                            avg_attn = attn_w.mean(dim=0)
                            if avg_attn.dim() == 3:
                                avg_attn = avg_attn.mean(dim=0)
                            T = min(avg_attn.shape[0], len(indices))
                            for si in range(T):
                                for sj in range(T):
                                    gi = indices[si].item() if isinstance(indices[si], torch.Tensor) else indices[si]
                                    gj = indices[sj].item() if isinstance(indices[sj], torch.Tensor) else indices[sj]
                                    if gi < N and gj < N:
                                        attention_accumulator[gi, gj] += avg_attn[si, sj].item()
                                        attention_counts[gi, gj] += 1
        finally:
            for h in hooks:
                h.remove()

        # Normalize
        valid = attention_counts > 0
        attention_accumulator[valid] /= attention_counts[valid]
        profile.attention_graph = attention_accumulator

        # Find attention hubs (high out-degree)
        out_strength = attention_accumulator.sum(dim=1)
        if out_strength.max() > 0:
            threshold = out_strength.mean() + out_strength.std()
            hub_indices = torch.where(out_strength > threshold)[0].tolist()
            profile.attention_hubs = hub_indices[:20]

        # Find attention bridges (strong cross-cluster edges)
        # Will be populated when labels are available in later steps
        top_k = min(50, N * N)
        flat = attention_accumulator.flatten()
        if flat.max() > 0:
            vals, indices_flat = flat.topk(min(top_k, len(flat)))
            bridges = []
            for v, idx in zip(vals, indices_flat):
                if v.item() > 0:
                    i_idx = idx.item() // N
                    j_idx = idx.item() % N
                    if i_idx != j_idx:
                        bridges.append((i_idx, j_idx, v.item()))
            profile.attention_bridges = bridges[:30]

    def _harvest_topology(self, profile: DeepSignalProfile, recursive_loop) -> None:
        """Extract topology from the recursive loop's crystallizer."""
        crystallizer = recursive_loop.crystallizer
        registry = crystallizer.registry

        # Module info
        stats = registry.get_statistics()
        profile.num_modules = stats["num_modules"]
        profile.module_types = stats.get("module_types", {})

        # Per-module performance
        module_perf = {}
        for mid, record in registry._modules.items():
            module_perf[mid] = record.avg_loss
        profile.module_performance = module_perf

        # Attractor positions from attractor modules
        attractor_positions = []
        cycle_frequencies = []
        for mid, record in registry._modules.items():
            feat = record.feature
            if feat.module_type == "attractor" and feat.centroid is not None:
                attractor_positions.append(feat.centroid.tolist() if hasattr(feat.centroid, 'tolist') else list(feat.centroid))
            elif feat.module_type == "cycle":
                cycle_frequencies.append(feat.persistence)
        profile.attractor_positions = attractor_positions
        profile.cycle_frequencies = cycle_frequencies

        # Topology from crystallization history
        history = crystallizer.crystallization_history
        if history:
            last = history[-1]
            # Betti numbers
            profile.betti_numbers = {
                0: profile.module_types.get("attractor", 0),
                1: profile.module_types.get("cycle", 0),
                2: profile.module_types.get("boundary", 0),
            }
            profile.total_persistence = last.get("total_persistence", 0.0)

            # Build persistence diagram from topological features
            topo_features = last.get("topological_features", [])
            if isinstance(topo_features, list):
                profile.topological_features = topo_features
                diag = {}
                for tf in topo_features:
                    dim = tf.dim if hasattr(tf, 'dim') else 0
                    entry = (
                        tf.birth if hasattr(tf, 'birth') else 0.0,
                        tf.death if hasattr(tf, 'death') else 0.0,
                        tf.persistence if hasattr(tf, 'persistence') else 0.0,
                    )
                    diag.setdefault(dim, []).append(entry)
                profile.persistence_diagram = diag

    def _harvest_convergence(self, profile: DeepSignalProfile, recursive_loop) -> None:
        """Extract convergence dynamics."""
        convergence = recursive_loop.convergence
        profile.convergence_history = list(convergence.history)

        if convergence.history:
            latest = convergence.history[-1]
            profile.convergence_score = latest.get("convergence_score", 0.0)

            # Representation stability: inverse of recent repr_divergence
            recent = convergence.history[-5:]
            avg_div = sum(h.get("repr_divergence", 1.0) for h in recent) / len(recent)
            profile.representation_stability = max(0.0, 1.0 - avg_div)

    def _harvest_representation_quality(
        self, profile: DeepSignalProfile, model: nn.Module,
        dataset, corpus, device: torch.device,
    ) -> None:
        """Compute representation quality metrics."""
        from tcd_jepa.manifold.evaluation import (
            extract_manifold_features,
            compute_representation_metrics,
            causal_link_prediction,
            manifold_neighborhood_preservation,
        )

        try:
            features, labels, indices = extract_manifold_features(
                model, dataset, device, max_samples=min(300, len(dataset)),
            )

            rep_metrics = compute_representation_metrics(features)
            profile.effective_rank = rep_metrics.get("effective_rank", 0.0)
            profile.uniformity = rep_metrics.get("uniformity", 0.0)

            # Neighborhood preservation
            if hasattr(dataset, "coords"):
                all_coords = []
                for i in range(min(300, len(dataset))):
                    c = dataset[i].get("coords")
                    if c is not None:
                        all_coords.append(c)
                if all_coords:
                    coords_flat = torch.cat(all_coords, dim=0)[:len(features)]
                    if len(coords_flat) == len(features):
                        mnp = manifold_neighborhood_preservation(features, coords_flat)
                        profile.neighborhood_preservation = mnp.get("neighborhood_preservation", 0.0)

            # Link prediction
            if hasattr(dataset, "adjacency") and dataset.adjacency is not None:
                clp = causal_link_prediction(
                    features, dataset.adjacency, indices, dataset.num_entities,
                )
                profile.link_prediction_auc = clp.get("link_auc", 0.0)

        except Exception as e:
            logger.warning(f"Representation quality harvest failed: {e}")

    def _harvest_uncertainty(self, profile: DeepSignalProfile, N: int) -> None:
        """Compute uncertainty field from blank scores and Fisher traces."""
        epistemic = torch.zeros(N)

        if profile.blank_scores is not None and profile.fisher_traces is not None:
            # High blank score + low Fisher trace = high uncertainty
            blank_norm = profile.blank_scores
            fisher_norm = profile.fisher_traces
            if fisher_norm.max() > 0:
                fisher_norm = fisher_norm / fisher_norm.max()

            # Uncertainty = blank_score * (1 - normalized_fisher)
            epistemic = blank_norm * (1.0 - fisher_norm)
        elif profile.blank_scores is not None:
            epistemic = profile.blank_scores

        profile.epistemic_uncertainty = epistemic

        # Coverage: fraction of chunks with low uncertainty
        if epistemic.numel() > 0:
            well_modeled = (epistemic < 0.5).float().mean().item()
            profile.coverage_ratio = well_modeled
        else:
            profile.coverage_ratio = 0.0
