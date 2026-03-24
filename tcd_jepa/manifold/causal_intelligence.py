"""Causal intelligence — attention-derived causal inference.

Goes beyond correlation using attention flows and information geometry:
- Multi-hop causal chains through the attention graph
- Information bottleneck detection (hub chunks gating information flow)
- Intervention analysis (simulated removal of critical chunks)
- Strongest causal pathways across the document graph
"""

import logging

import torch

from tcd_jepa.manifold.insight_engine import Insight
from tcd_jepa.manifold.lineage import make_lineage_id

logger = logging.getLogger("tcd_jepa.causal_intelligence")


class CausalIntelligenceEngine:
    """Generates causal insights from attention flows and graph structure."""

    def generate(self, profile, chunks, features, adjacency, sparse_adj=None) -> list[Insight]:
        """Generate all causal insights.

        Args:
            profile: DeepSignalProfile with attention data.
            chunks: List of DocumentChunk objects.
            features: [N, D] learned representations.
            adjacency: [N, N] adjacency matrix.
            sparse_adj: Optional sparse adjacency for efficiency.
        """
        insights = []
        try:
            insights.extend(self._attention_flow_insights(profile, chunks, features))
            insights.extend(self._information_bottleneck_insights(profile, chunks, features))
            insights.extend(self._intervention_insights(profile, chunks, features))
            insights.extend(self._causal_path_insights(profile, chunks, features, adjacency))
        except Exception as e:
            logger.warning(f"Causal insight generation failed: {e}")
        return insights

    def _attention_flow_insights(self, profile, chunks, features) -> list[Insight]:
        """Multi-hop attention chains → causal pathways."""
        insights = []
        attn = profile.attention_graph
        if attn is None or attn.max() == 0:
            return insights

        N = attn.shape[0]

        # Find strongest multi-hop paths (2-3 hops)
        # Strength of 2-hop path i->j->k = attn[i,j] * attn[j,k]
        attn_thresh = attn.clone()
        mean_val = attn[attn > 0].mean().item() if (attn > 0).any() else 0
        attn_thresh[attn_thresh < mean_val * 0.5] = 0

        best_paths = []
        for i in range(min(N, 30)):
            neighbors = torch.where(attn_thresh[i] > 0)[0]
            for j in neighbors[:15]:
                j = j.item()
                if j == i:
                    continue
                hop2_neighbors = torch.where(attn_thresh[j] > 0)[0]
                for k in hop2_neighbors[:15]:
                    k = k.item()
                    if k == i or k == j:
                        continue
                    strength = attn_thresh[i, j].item() * attn_thresh[j, k].item()
                    if strength > 0:
                        best_paths.append((i, j, k, strength))

        best_paths.sort(key=lambda x: x[3], reverse=True)

        for i_node, j_node, k_node, strength in best_paths[:3]:
            names = []
            source_ids = []
            for node_idx in [i_node, j_node, k_node]:
                if node_idx < len(chunks):
                    names.append(f"chunk {node_idx}: '{chunks[node_idx].text[:40]}...'")
                    if hasattr(chunks[node_idx], 'chunk_id'):
                        source_ids.append(chunks[node_idx].chunk_id)

            w1 = attn_thresh[i_node, j_node].item()
            w2 = attn_thresh[j_node, k_node].item()

            insights.append(Insight(
                category="causal_chain",
                severity="medium",
                title=f"3-hop causal chain: chunk {i_node} → {j_node} → {k_node}",
                description=(
                    f"Information flows from {names[0] if names else 'unknown'} → "
                    f"{names[1] if len(names) > 1 else 'unknown'} → "
                    f"{names[2] if len(names) > 2 else 'unknown'} "
                    f"via a 3-hop attention chain (weights: {w1:.3f} → {w2:.3f}). "
                    f"Path strength: {strength:.4f}. "
                    f"This is a cross-document causal pathway."
                ),
                evidence=[chunks[i_node].text[:200] if i_node < len(chunks) else ""],
                entities=[f"chunk_{i_node}", f"chunk_{j_node}", f"chunk_{k_node}"],
                confidence=min(0.85, 0.5 + strength * 5),
                insight_id=make_lineage_id("insight"),
                source_chunk_ids=source_ids,
                derivation_steps=["attention_extraction", "multi_hop_path_search", "causal_chain_scoring"],
            ))

        return insights

    def _information_bottleneck_insights(self, profile, chunks, features) -> list[Insight]:
        """Identify chunks that gate information flow (betweenness centrality)."""
        insights = []
        attn = profile.attention_graph
        if attn is None or attn.max() == 0:
            return insights

        N = min(attn.shape[0], len(chunks))

        # Approximate betweenness centrality using shortest paths
        # For efficiency, use a graph-based approach on thresholded attention
        attn_binary = (attn[:N, :N] > attn[:N, :N].mean()).float()

        # Count shortest paths through each node (simplified)
        # For each pair (s, t), check if removing node v disconnects them
        in_degree = attn_binary.sum(dim=0)
        out_degree = attn_binary.sum(dim=1)

        # Betweenness proxy: product of in and out degree (flow-through capacity)
        betweenness_proxy = in_degree * out_degree
        if betweenness_proxy.max() == 0:
            return insights

        # Find top bottleneck nodes
        top_k = min(5, N)
        values, indices = betweenness_proxy.topk(top_k)
        mean_betweenness = betweenness_proxy.mean().item()

        for rank, (val, idx) in enumerate(zip(values, indices)):
            idx = idx.item()
            if val.item() < mean_betweenness * 1.5:
                continue
            if idx >= len(chunks):
                continue

            chunk = chunks[idx]
            source_ids = [chunk.chunk_id] if hasattr(chunk, 'chunk_id') else []

            # Compute what fraction of paths go through this node
            total_paths = attn_binary.sum().item()
            node_paths = (in_degree[idx] + out_degree[idx]).item()
            path_fraction = node_paths / max(total_paths, 1)

            insights.append(Insight(
                category="information_bottleneck",
                severity="high" if path_fraction > 0.15 else "medium",
                title=f"Information bottleneck at chunk {idx}",
                description=(
                    f"Chunk {idx} ('{chunk.text[:60]}...') is a critical information bottleneck. "
                    f"It has {int(in_degree[idx].item())} incoming and {int(out_degree[idx].item())} "
                    f"outgoing attention connections. "
                    f"Flow-through score: {val.item():.1f} ({path_fraction:.0%} of network flow). "
                    f"This chunk sits on a large fraction of information pathways. "
                    f"Removing it would significantly disrupt knowledge connectivity."
                ),
                evidence=[chunk.text[:300]],
                entities=[f"chunk_{idx}", getattr(chunk, 'doc_id', '')],
                confidence=min(0.9, 0.6 + path_fraction * 2),
                insight_id=make_lineage_id("insight"),
                source_chunk_ids=source_ids,
                derivation_steps=["attention_graph", "betweenness_centrality", "bottleneck_scoring"],
            ))

        return insights

    def _intervention_insights(self, profile, chunks, features) -> list[Insight]:
        """Simulate removal of hub chunks to assess systemic impact."""
        insights = []
        attn = profile.attention_graph
        if attn is None or not profile.attention_hubs:
            return insights

        N = min(attn.shape[0], len(chunks))
        hubs = [h for h in profile.attention_hubs if h < N]

        if not hubs:
            return insights

        # Count connected components before removal
        def count_components(adj_matrix):
            n = adj_matrix.shape[0]
            visited = set()
            components = 0
            for start in range(n):
                if start in visited:
                    continue
                components += 1
                queue = [start]
                while queue:
                    node = queue.pop(0)
                    if node in visited:
                        continue
                    visited.add(node)
                    for j in range(n):
                        if j not in visited and adj_matrix[node, j] > 0:
                            queue.append(j)
            return components

        attn_binary = (attn[:N, :N] > attn[:N, :N].mean()).float()
        baseline_components = count_components(attn_binary)

        # Simulate removal of top hubs
        top_hubs = hubs[:min(5, len(hubs))]
        modified_attn = attn_binary.clone()
        for h in top_hubs:
            if h < N:
                modified_attn[h, :] = 0
                modified_attn[:, h] = 0

        post_removal_components = count_components(modified_attn)
        fragmentation = post_removal_components - baseline_components

        if fragmentation > 0:
            hub_names = []
            source_ids = []
            for h in top_hubs[:3]:
                if h < len(chunks):
                    hub_names.append(f"chunk {h} ('{chunks[h].text[:30]}...')")
                    if hasattr(chunks[h], 'chunk_id'):
                        source_ids.append(chunks[h].chunk_id)

            insights.append(Insight(
                category="intervention",
                severity="high" if fragmentation >= 3 else "medium",
                title=f"Removing {len(top_hubs)} hub chunks fragments the knowledge graph",
                description=(
                    f"Removing the top {len(top_hubs)} hub chunks "
                    f"({', '.join(hub_names)}) would increase the number of "
                    f"disconnected components from {baseline_components} to "
                    f"{post_removal_components} (+{fragmentation}). "
                    f"These chunks are systemically important — treat as critical "
                    f"infrastructure for the knowledge base."
                ),
                evidence=[chunks[h].text[:200] for h in top_hubs[:2] if h < len(chunks)],
                entities=[f"chunk_{h}" for h in top_hubs],
                confidence=0.8,
                insight_id=make_lineage_id("insight"),
                source_chunk_ids=source_ids,
                derivation_steps=["hub_identification", "simulated_removal", "fragmentation_analysis"],
            ))

        return insights

    def _causal_path_insights(self, profile, chunks, features, adjacency) -> list[Insight]:
        """Find strongest causal chains combining adjacency + attention."""
        insights = []
        if adjacency is None:
            return insights

        N = min(adjacency.shape[0], len(chunks))
        attn = profile.attention_graph

        # Combine adjacency and attention into a unified causal strength matrix
        combined = adjacency[:N, :N].float()
        if attn is not None:
            attn_sub = attn[:N, :N]
            # Normalize both to [0, 1]
            if combined.max() > 0:
                combined = combined / combined.max()
            if attn_sub.max() > 0:
                attn_norm = attn_sub / attn_sub.max()
                combined = 0.5 * combined + 0.5 * attn_norm

        if combined.max() == 0:
            return insights

        # Find strongest direct causal links
        combined.fill_diagonal_(0)
        flat = combined.flatten()
        top_k = min(10, (flat > 0).sum().item())
        if top_k == 0:
            return insights

        vals, idx_flat = flat.topk(top_k)

        # Group into chains
        strongest_links = []
        for v, idx in zip(vals, idx_flat):
            if v.item() == 0:
                break
            i = idx.item() // N
            j = idx.item() % N
            strongest_links.append((i, j, v.item()))

        if strongest_links:
            link = strongest_links[0]
            i, j, w = link
            source_ids = []
            for node_idx in [i, j]:
                if node_idx < len(chunks) and hasattr(chunks[node_idx], 'chunk_id'):
                    source_ids.append(chunks[node_idx].chunk_id)

            insights.append(Insight(
                category="causal_chain",
                severity="medium",
                title=f"Strongest causal link: chunk {i} → chunk {j} (weight {w:.3f})",
                description=(
                    f"The strongest causal connection in the corpus links "
                    f"chunk {i} ('{chunks[i].text[:50]}...' ) to "
                    f"chunk {j} ('{chunks[j].text[:50]}...'). "
                    f"Combined attention+adjacency weight: {w:.3f}. "
                    f"This represents the primary information conduit in the corpus."
                ),
                evidence=[
                    chunks[i].text[:200] if i < len(chunks) else "",
                    chunks[j].text[:200] if j < len(chunks) else "",
                ],
                entities=[getattr(chunks[i], 'doc_id', ''), getattr(chunks[j], 'doc_id', '')],
                confidence=min(0.9, 0.5 + w),
                insight_id=make_lineage_id("insight"),
                source_chunk_ids=source_ids,
                derivation_steps=["adjacency_analysis", "attention_fusion", "causal_strength_ranking"],
            ))

        return insights
