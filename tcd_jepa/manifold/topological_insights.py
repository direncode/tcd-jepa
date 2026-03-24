"""Topological intelligence — insights from persistent homology.

Generates insights that are impossible with standard NLP:
- Knowledge gaps (H2 voids)
- Feedback loops (H1 cycles)
- Hierarchical topic structure (multi-scale H0)
- Topological stability (persistence-based robustness)
"""

import logging

import torch
import torch.nn.functional as F

from tcd_jepa.manifold.insight_engine import Insight
from tcd_jepa.manifold.lineage import make_lineage_id

logger = logging.getLogger("tcd_jepa.topological_insights")


class TopologicalInsightGenerator:
    """Generates insights from persistent homology and topological features."""

    def generate(self, profile, chunks, features, labels) -> list["Insight"]:
        """Generate all topological insights.

        Args:
            profile: DeepSignalProfile with topology data.
            chunks: List of DocumentChunk objects.
            features: [N, D] learned representations.
            labels: [N] cluster labels.
        """
        insights = []
        try:
            insights.extend(self._knowledge_gap_insights(profile, chunks, features, labels))
            insights.extend(self._feedback_loop_insights(profile, chunks, features, labels))
            insights.extend(self._hierarchical_cluster_insights(profile, chunks, features, labels))
            insights.extend(self._topological_stability_insights(profile, chunks, features, labels))
        except Exception as e:
            logger.warning(f"Topological insight generation failed: {e}")
        return insights

    def _knowledge_gap_insights(self, profile, chunks, features, labels) -> list[Insight]:
        """H2 voids → knowledge gaps between topic clusters."""
        insights = []
        num_clusters = int(labels.max().item()) + 1 if len(labels) > 0 else 0

        # H2 features (boundaries/voids) indicate missing knowledge
        h2_count = profile.betti_numbers.get(2, 0)
        h2_features = [f for f in profile.topological_features
                       if hasattr(f, 'dim') and f.dim == 2]

        if h2_count > 0 or h2_features:
            # Find cluster pairs with weakest connections
            feat_norm = F.normalize(features, dim=1)
            cluster_centroids = []
            cluster_names = []
            for c in range(num_clusters):
                mask = labels == c
                if mask.sum() > 0:
                    centroid = feat_norm[mask].mean(0)
                    cluster_centroids.append(centroid)
                    # Get representative text for cluster name
                    c_indices = torch.where(mask)[0]
                    rep_idx = c_indices[0].item()
                    name = chunks[rep_idx].text[:50] if rep_idx < len(chunks) else f"Cluster {c}"
                    cluster_names.append(name)

            if len(cluster_centroids) >= 2:
                centroids_t = torch.stack(cluster_centroids)
                inter_sim = centroids_t @ centroids_t.t()

                # Find least-connected cluster pairs (excluding self)
                inter_sim.fill_diagonal_(float('inf'))
                for _ in range(min(3, h2_count + 1)):
                    min_val = inter_sim.min()
                    if min_val == float('inf'):
                        break
                    min_idx = (inter_sim == min_val).nonzero(as_tuple=False)
                    if len(min_idx) == 0:
                        break
                    ci, cj = min_idx[0][0].item(), min_idx[0][1].item()
                    inter_sim[ci, cj] = float('inf')
                    inter_sim[cj, ci] = float('inf')

                    # Get boundary chunks
                    source_ids = []
                    for idx in torch.where(labels == ci)[0][:3].tolist():
                        if idx < len(chunks) and hasattr(chunks[idx], 'chunk_id'):
                            source_ids.append(chunks[idx].chunk_id)

                    persistence = h2_features[0].persistence if h2_features else 0.0

                    insights.append(Insight(
                        category="knowledge_gap",
                        severity="high" if min_val.item() < 0.3 else "medium",
                        title=f"Knowledge void between topic clusters {ci} and {cj}",
                        description=(
                            f"There is a void in the knowledge space between "
                            f"cluster {ci} ('{cluster_names[ci][:40]}...') and "
                            f"cluster {cj} ('{cluster_names[cj][:40]}...'). "
                            f"No documents adequately bridge these topics. "
                            f"Inter-cluster similarity: {min_val.item():.3f}. "
                            f"H2 persistence: {persistence:.3f}. "
                            f"This represents a blind spot in the corpus."
                        ),
                        evidence=[chunks[torch.where(labels == ci)[0][0].item()].text[:200]
                                  if torch.where(labels == ci)[0][0].item() < len(chunks) else ""],
                        entities=[cluster_names[ci][:50], cluster_names[cj][:50]],
                        confidence=0.7 + 0.2 * (1.0 - min_val.item()),
                        insight_id=make_lineage_id("insight"),
                        source_chunk_ids=source_ids,
                        derivation_steps=["persistent_homology", "H2_void_detection", "cluster_gap_analysis"],
                    ))

        return insights

    def _feedback_loop_insights(self, profile, chunks, features, labels) -> list[Insight]:
        """H1 cycles → circular dependencies and feedback loops."""
        insights = []
        h1_count = profile.betti_numbers.get(1, 0)
        h1_features = [f for f in profile.topological_features
                       if hasattr(f, 'dim') and f.dim == 1]

        if not h1_features and h1_count == 0:
            return insights

        # Use attention graph to find actual cycles
        attn = profile.attention_graph
        if attn is not None and attn.max() > 0:
            N = attn.shape[0]
            # Find strong 3-hop cycles: i->j->k->i
            attn_thresh = attn.clone()
            mean_attn = attn[attn > 0].mean() if (attn > 0).any() else 0.1
            attn_thresh[attn_thresh < mean_attn] = 0

            cycles_found = []
            checked = set()
            for i in range(min(N, 50)):
                neighbors_i = torch.where(attn_thresh[i] > 0)[0]
                for j in neighbors_i[:10]:
                    j = j.item()
                    if j == i:
                        continue
                    neighbors_j = torch.where(attn_thresh[j] > 0)[0]
                    for k in neighbors_j[:10]:
                        k = k.item()
                        if k == i or k == j:
                            continue
                        if attn_thresh[k, i] > 0:
                            cycle_key = tuple(sorted([i, j, k]))
                            if cycle_key not in checked:
                                checked.add(cycle_key)
                                strength = (attn_thresh[i, j] * attn_thresh[j, k] * attn_thresh[k, i]).item()
                                cycles_found.append((i, j, k, strength))

            cycles_found.sort(key=lambda x: x[3], reverse=True)

            for idx, (i, j, k, strength) in enumerate(cycles_found[:3]):
                names = []
                source_ids = []
                for node_idx in [i, j, k]:
                    if node_idx < len(chunks):
                        names.append(chunks[node_idx].text[:40])
                        if hasattr(chunks[node_idx], 'chunk_id'):
                            source_ids.append(chunks[node_idx].chunk_id)

                h1_persistence = h1_features[idx].persistence if idx < len(h1_features) else 0.0

                insights.append(Insight(
                    category="feedback_loop",
                    severity="medium",
                    title=f"Cyclic dependency detected: {len(names)}-node feedback loop",
                    description=(
                        f"A cyclic dependency exists: chunk {i} → chunk {j} → chunk {k} → chunk {i}. "
                        f"Cycle strength: {strength:.4f}. "
                        f"H1 persistence: {h1_persistence:.3f}. "
                        f"This indicates a stable feedback loop in the knowledge structure."
                    ),
                    evidence=[n[:200] for n in names[:2]],
                    entities=[f"chunk_{i}", f"chunk_{j}", f"chunk_{k}"],
                    confidence=min(0.9, 0.5 + strength * 2),
                    insight_id=make_lineage_id("insight"),
                    source_chunk_ids=source_ids,
                    derivation_steps=["persistent_homology", "H1_cycle_detection", "attention_cycle_validation"],
                ))

        return insights

    def _hierarchical_cluster_insights(self, profile, chunks, features, labels) -> list[Insight]:
        """Multi-scale H0 analysis → topic hierarchy discovery."""
        insights = []
        num_clusters = int(labels.max().item()) + 1 if len(labels) > 0 else 0

        if num_clusters < 3:
            return insights

        feat_norm = F.normalize(features, dim=1)

        # Multi-scale clustering via threshold sweep
        cluster_centroids = []
        for c in range(num_clusters):
            mask = labels == c
            if mask.sum() > 0:
                cluster_centroids.append(feat_norm[mask].mean(0))

        if len(cluster_centroids) < 3:
            return insights

        centroids_t = torch.stack(cluster_centroids)
        sim_matrix = centroids_t @ centroids_t.t()

        # Sweep scales to find hierarchy
        scales = [0.3, 0.5, 0.7, 0.9]
        hierarchy_levels = []
        for eps in scales:
            # Count connected components at this scale
            connected = sim_matrix > eps
            visited = set()
            components = 0
            for i in range(len(centroids_t)):
                if i not in visited:
                    components += 1
                    # BFS
                    queue = [i]
                    while queue:
                        node = queue.pop(0)
                        if node in visited:
                            continue
                        visited.add(node)
                        for j in range(len(centroids_t)):
                            if j not in visited and connected[node, j]:
                                queue.append(j)
            hierarchy_levels.append((eps, components))

        # Report if there's genuine hierarchy (different component counts at different scales)
        component_counts = [h[1] for h in hierarchy_levels]
        if len(set(component_counts)) >= 2:
            desc_parts = []
            for eps, count in hierarchy_levels:
                desc_parts.append(f"ε={eps}: {count} clusters")

            insights.append(Insight(
                category="hierarchy",
                severity="medium",
                title=f"Multi-level topic hierarchy discovered ({num_clusters} base clusters)",
                description=(
                    f"The corpus exhibits a hierarchical topic structure. "
                    f"{'. '.join(desc_parts)}. "
                    f"This reveals a {len(set(component_counts))}-level topic hierarchy "
                    f"from fine-grained subtopics to macro-themes."
                ),
                evidence=[],
                entities=[f"cluster_{c}" for c in range(min(num_clusters, 10))],
                confidence=0.75,
                insight_id=make_lineage_id("insight"),
                derivation_steps=["H0_multi_scale", "threshold_sweep", "component_counting"],
            ))

        return insights

    def _topological_stability_insights(self, profile, chunks, features, labels) -> list[Insight]:
        """Persistence-based robustness of topological features."""
        insights = []

        if not profile.topological_features:
            return insights

        # Find most and least persistent features
        sorted_features = sorted(
            [f for f in profile.topological_features if hasattr(f, 'persistence')],
            key=lambda f: f.persistence,
            reverse=True,
        )

        if not sorted_features:
            return insights

        most_robust = sorted_features[0]
        least_robust = sorted_features[-1] if len(sorted_features) > 1 else None

        type_names = {0: "component/cluster", 1: "cycle/loop", 2: "void/boundary"}

        insights.append(Insight(
            category="topological_stability",
            severity="low",
            title=f"Most robust structure: H{most_robust.dim} with persistence {most_robust.persistence:.3f}",
            description=(
                f"The most robust topological feature is an H{most_robust.dim} "
                f"({type_names.get(most_robust.dim, 'feature')}) with persistence "
                f"{most_robust.persistence:.3f}. This structure is highly stable and "
                f"resistant to perturbation. Total persistence across all features: "
                f"{profile.total_persistence:.3f}."
            ),
            evidence=[],
            entities=[],
            confidence=0.8,
            insight_id=make_lineage_id("insight"),
            derivation_steps=["persistence_analysis", "stability_ranking"],
        ))

        if least_robust and least_robust.persistence < most_robust.persistence * 0.3:
            insights.append(Insight(
                category="topological_stability",
                severity="medium",
                title=f"Fragile structure: H{least_robust.dim} with persistence {least_robust.persistence:.3f}",
                description=(
                    f"An H{least_robust.dim} ({type_names.get(least_robust.dim, 'feature')}) "
                    f"has low persistence ({least_robust.persistence:.3f}), only "
                    f"{least_robust.persistence/most_robust.persistence:.0%} of the most robust feature. "
                    f"This structure is fragile and could dissolve with new data or slight "
                    f"parameter changes."
                ),
                evidence=[],
                entities=[],
                confidence=0.65,
                insight_id=make_lineage_id("insight"),
                derivation_steps=["persistence_analysis", "fragility_detection"],
            ))

        return insights
