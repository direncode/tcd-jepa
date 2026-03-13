"""Insight Engine: TCD-JEPA signals → structured commercial intelligence → natural language.

Takes the outputs of TCD-JEPA training/inference (learned representations, topological
features, prediction errors, KPIs) and produces human-readable commercial intelligence
insights about the document corpus.

Insight categories:
1. **Cluster Insights** (H0 attractors) — Topic discovery, theme identification
2. **Relationship Insights** (causal links) — Cross-document connections, dependencies
3. **Trend Insights** (velocity/drift) — Evolving topics, emerging patterns
4. **Opportunity Insights** (H1 cycles) — Cross-domain feedback loops, hidden connections
5. **Risk Insights** (H2 boundaries) — Unstable entities, boundary conditions
6. **Anomaly Insights** (prediction error) — Novel content, surprising connections
7. **Strategic Summary** — Latent Ocean KPIs mapped to business intelligence
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F

from tcd_jepa.manifold.lineage import (
    InsightNode, ClusterNode, LineageGraph, make_lineage_id,
)

logger = logging.getLogger("tcd_jepa.insight_engine")


@dataclass
class EvidenceItem:
    """Structured evidence with source reference."""
    chunk_id: str = ""         # Lineage ID of source chunk
    doc_id: str = ""           # Parent document
    doc_title: str = ""        # Human-readable source
    text_excerpt: str = ""     # The relevant text
    char_start: int = 0        # Offset in source document
    char_end: int = 0          # End offset
    relevance: float = 1.0     # 0-1 how relevant


@dataclass
class Insight:
    """A single commercial intelligence insight with full lineage."""
    category: str          # cluster, relationship, trend, opportunity, risk, anomaly, strategic
    severity: str          # high, medium, low
    title: str             # One-line summary
    description: str       # Detailed explanation
    evidence: list[str]    # Supporting text chunks (backwards compat)
    entities: list[str]    # Involved entities
    confidence: float      # 0-1 confidence score
    metadata: dict = field(default_factory=dict)
    # --- Oracle-grade additions ---
    insight_id: str = field(default_factory=lambda: make_lineage_id("insight"))
    structured_evidence: list[EvidenceItem] = field(default_factory=list)
    confidence_breakdown: dict = field(default_factory=dict)
    source_chunk_ids: list[str] = field(default_factory=list)
    source_link_ids: list[str] = field(default_factory=list)
    source_cluster_ids: list[str] = field(default_factory=list)
    derivation_steps: list[str] = field(default_factory=list)
    parent_insight_ids: list[str] = field(default_factory=list)
    contradicts: list[str] = field(default_factory=list)
    supports: list[str] = field(default_factory=list)
    sensitivity: float = 0.0   # 0-1, stability across parameter variations

    def to_nl(self) -> str:
        """Render as natural language paragraph with source citations."""
        severity_prefix = {
            "high": "CRITICAL",
            "medium": "NOTABLE",
            "low": "OBSERVATION",
        }
        prefix = severity_prefix.get(self.severity, "NOTE")
        lines = [f"[{prefix}] {self.title} (ID: {self.insight_id})"]
        lines.append(self.description)
        if self.entities:
            lines.append(f"Entities involved: {', '.join(self.entities[:10])}")
        # Prefer structured evidence with source citations
        if self.structured_evidence:
            lines.append("Supporting evidence:")
            for ev in self.structured_evidence[:5]:
                truncated = ev.text_excerpt[:200] + "..." if len(ev.text_excerpt) > 200 else ev.text_excerpt
                source_ref = f"[Source: {ev.doc_title or ev.doc_id}, chunk {ev.chunk_id}]"
                lines.append(f"  - \"{truncated}\" {source_ref}")
        elif self.evidence:
            lines.append("Supporting evidence:")
            for ev in self.evidence[:3]:
                truncated = ev[:200] + "..." if len(ev) > 200 else ev
                lines.append(f"  - \"{truncated}\"")
        if self.contradicts:
            lines.append(f"Contradicts: {', '.join(self.contradicts[:3])}")
        if self.supports:
            lines.append(f"Corroborated by: {', '.join(self.supports[:3])}")
        if self.derivation_steps:
            lines.append(f"Derivation: {' -> '.join(self.derivation_steps[:5])}")
        conf_str = f"Confidence: {self.confidence:.0%}"
        if self.sensitivity > 0:
            conf_str += f" | Sensitivity: {self.sensitivity:.0%}"
        lines.append(conf_str)
        if self.source_chunk_ids:
            lines.append(f"Lineage: {len(self.source_chunk_ids)} source chunks")
        return "\n".join(lines)


@dataclass
class InsightReport:
    """Complete intelligence report from document analysis."""
    insights: list[Insight]
    summary: str
    kpis: dict
    num_documents: int
    num_chunks: int
    num_clusters: int
    num_links: int
    lineage_graph: Optional[LineageGraph] = None

    def export_lineage(self, insight_id: str) -> dict:
        """Export full lineage chain for a specific insight."""
        if self.lineage_graph is None:
            return {"insight_id": insight_id, "error": "no lineage graph"}
        return self.lineage_graph.export_lineage(insight_id)

    def get_contradictions(self) -> list[tuple[str, str]]:
        """Return pairs of contradicting insight IDs."""
        pairs = []
        seen = set()
        for ins in self.insights:
            for cid in ins.contradicts:
                pair = tuple(sorted([ins.insight_id, cid]))
                if pair not in seen:
                    seen.add(pair)
                    pairs.append(pair)
        return pairs

    def to_nl(self) -> str:
        """Render full report as natural language."""
        lines = []
        lines.append("=" * 70)
        lines.append("COMMERCIAL INTELLIGENCE REPORT")
        lines.append("=" * 70)
        lines.append("")
        lines.append(f"Corpus: {self.num_documents} documents, {self.num_chunks} semantic chunks")
        lines.append(f"Structure: {self.num_clusters} topic clusters, {self.num_links} causal links")
        lines.append("")

        # Strategic KPIs
        lines.append("--- STRATEGIC INDICATORS ---")
        if "clarity_score" in self.kpis:
            cs = self.kpis["clarity_score"]
            clarity_desc = "excellent" if cs > 70 else "good" if cs > 40 else "moderate" if cs > 20 else "low"
            benchmark = 70
            direction = "ABOVE" if cs >= benchmark else "BELOW"
            lines.append(f"  Clarity Score: {cs:.1f}/100 ({clarity_desc}) — {direction} the {benchmark} enterprise benchmark")
            lines.append(f"    → How well-separated are the topic clusters in this corpus")
        if "drift_velocity" in self.kpis:
            dv = self.kpis["drift_velocity"]
            drift_desc = "highly dynamic" if dv > 0.8 else "active" if dv > 0.4 else "stable" if dv > 0.1 else "static"
            lines.append(f"  Drift Velocity: {dv:.3f} ({drift_desc})")
            lines.append(f"    → Rate of semantic evolution across the corpus")
        if "opportunity_surface" in self.kpis:
            os_val = self.kpis["opportunity_surface"]
            lines.append(f"  Opportunity Surface: {os_val}")
            lines.append(f"    → Cross-topic high-similarity connections (potential synergies)")
        if "risk_horizon" in self.kpis:
            rh = self.kpis["risk_horizon"]
            lines.append(f"  Risk Horizon: {rh}")
            lines.append(f"    → Chunks at topic boundaries (ambiguous classification)")
        lines.append("")

        # TCD topology
        if "tcd_num_modules" in self.kpis:
            lines.append("--- TOPOLOGICAL STRUCTURE ---")
            lines.append(f"  Crystallized modules: {self.kpis.get('tcd_num_modules', 0)}")
            lines.append(f"  Attractors (H0): {self.kpis.get('tcd_attractors_h0', 0)} — stable topic centers")
            lines.append(f"  Cycles (H1): {self.kpis.get('tcd_cycles_h1', 0)} — feedback loops between topics")
            lines.append(f"  Boundaries (H2): {self.kpis.get('tcd_boundaries_h2', 0)} — topic transition zones")
            lines.append(f"  Converged: {self.kpis.get('tcd_converged', False)}")
            lines.append("")

        # Executive summary
        lines.append("--- EXECUTIVE SUMMARY ---")
        lines.append(self.summary)
        lines.append("")

        # Insights by category
        categories = [
            "strategic", "cluster", "opportunity", "risk", "trend", "relationship", "anomaly",
            "knowledge_gap", "feedback_loop", "hierarchy", "topological_stability",
            "causal_chain", "information_bottleneck", "intervention",
            "blind_spot", "confidence_field", "coverage_gap",
            "topic_drift", "phase_transition", "exploration_dynamics",
            "model_quality", "insight_reliability", "corpus_diagnosis",
        ]
        category_titles = {
            "strategic": "STRATEGIC INSIGHTS",
            "cluster": "TOPIC DISCOVERY",
            "opportunity": "OPPORTUNITIES",
            "risk": "RISKS & VULNERABILITIES",
            "trend": "TRENDS & DYNAMICS",
            "relationship": "KEY RELATIONSHIPS",
            "anomaly": "ANOMALIES & NOVELTY",
            "knowledge_gap": "KNOWLEDGE GAPS (Topological Voids)",
            "feedback_loop": "FEEDBACK LOOPS (Topological Cycles)",
            "hierarchy": "TOPIC HIERARCHY (Multi-Scale Structure)",
            "topological_stability": "TOPOLOGICAL STABILITY",
            "causal_chain": "CAUSAL CHAINS (Attention Flow)",
            "information_bottleneck": "INFORMATION BOTTLENECKS",
            "intervention": "INTERVENTION ANALYSIS",
            "blind_spot": "BLIND SPOTS (Uncertainty Field)",
            "confidence_field": "CONFIDENCE MAPPING (Fisher Metric)",
            "coverage_gap": "COVERAGE ANALYSIS",
            "topic_drift": "TOPIC DRIFT (Temporal Dynamics)",
            "phase_transition": "PHASE TRANSITIONS",
            "exploration_dynamics": "EXPLORATION DYNAMICS",
            "model_quality": "MODEL QUALITY (Meta-Intelligence)",
            "insight_reliability": "INSIGHT RELIABILITY",
            "corpus_diagnosis": "CORPUS DIAGNOSIS",
        }

        for cat in categories:
            cat_insights = [i for i in self.insights if i.category == cat]
            if not cat_insights:
                continue
            lines.append(f"--- {category_titles.get(cat, cat.upper())} ---")
            for insight in cat_insights:
                lines.append("")
                lines.append(insight.to_nl())
            lines.append("")

        # Contradiction summary
        contradictions = self.get_contradictions()
        if contradictions:
            lines.append("--- CONTRADICTIONS ---")
            for a, b in contradictions[:5]:
                lines.append(f"  {a} <-> {b}")
            lines.append("")

        # Lineage summary
        if self.lineage_graph:
            lines.append("--- LINEAGE ---")
            lines.append(f"  Total provenance nodes: {self.lineage_graph.num_nodes}")
            lines.append(f"  Chunk nodes: {len(self.lineage_graph.nodes_by_kind('chunk'))}")
            lines.append(f"  Link nodes: {len(self.lineage_graph.nodes_by_kind('link'))}")
            lines.append(f"  Insight nodes: {len(self.lineage_graph.nodes_by_kind('insight'))}")
            lines.append("  Every insight is traceable to source documents via lineage IDs.")
            lines.append("")

        lines.append("=" * 70)
        return "\n".join(lines)

    def get_high_priority(self) -> list[Insight]:
        """Get only high-severity insights."""
        return [i for i in self.insights if i.severity == "high"]

    def get_by_category(self, category: str) -> list[Insight]:
        """Get insights by category."""
        return [i for i in self.insights if i.category == category]


class InsightEngine:
    """Generates commercial intelligence insights from TCD-JEPA outputs.

    Takes:
    - Learned representations (encoder outputs)
    - Topological features (H0/H1/H2 from crystallization)
    - Prediction errors (JEPA reconstruction quality)
    - KPIs (Clarity, Drift, Opportunity, Risk)
    - Document metadata (chunk texts, entities, structure)

    Produces:
    - Structured Insight objects
    - Natural language InsightReport
    """

    def __init__(
        self,
        top_k_per_category: int = 5,
        anomaly_threshold: float = 2.0,
        opportunity_sim_threshold: float = 0.7,
        risk_boundary_ratio: float = 0.7,
        enable_divergent: bool = True,
    ):
        self.top_k = top_k_per_category
        self.anomaly_threshold = anomaly_threshold
        self.opportunity_sim_threshold = opportunity_sim_threshold
        self.risk_boundary_ratio = risk_boundary_ratio
        self.enable_divergent = enable_divergent

    def generate_report(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        chunk_texts: list[str],
        chunks: list,
        adjacency: torch.Tensor,
        kpis: dict,
        prediction_errors: Optional[torch.Tensor] = None,
        recursive_loop=None,
        lineage_graph: Optional[LineageGraph] = None,
        deep_signal_profile=None,
        coords: Optional[torch.Tensor] = None,
        velocity: Optional[torch.Tensor] = None,
    ) -> InsightReport:
        """Generate full intelligence report.

        Args:
            features: [N, D] learned representations from encoder.
            labels: [N] cluster assignments.
            chunk_texts: List of original text strings.
            chunks: List of DocumentChunk objects.
            adjacency: [N, N] causal link weights.
            kpis: Dict of computed KPIs.
            prediction_errors: Optional [N] per-chunk prediction errors.
            recursive_loop: Optional TCD recursive loop for topology info.
            lineage_graph: Optional lineage graph for provenance tracking.
            deep_signal_profile: Optional DeepSignalProfile from signal harvesting.
            coords: Optional [N, 3] manifold coordinates.
            velocity: Optional [N, D] velocity field.
        """
        insights = []

        # Generate insights by category
        insights.extend(self._cluster_insights(features, labels, chunk_texts, chunks, lineage_graph))
        insights.extend(self._relationship_insights(features, labels, chunk_texts, chunks, adjacency, lineage_graph))
        insights.extend(self._opportunity_insights(features, labels, chunk_texts, chunks, lineage_graph))
        insights.extend(self._risk_insights(features, labels, chunk_texts, chunks, lineage_graph))

        if prediction_errors is not None:
            insights.extend(self._anomaly_insights(prediction_errors, chunk_texts, chunks, lineage_graph))
            insights.extend(self._trend_insights(features, labels, chunk_texts, chunks, prediction_errors, lineage_graph))

        # Strategic insights from KPIs
        insights.extend(self._strategic_insights(kpis, features, labels, chunk_texts))

        # Deep signal intelligence engines
        if deep_signal_profile is not None:
            from tcd_jepa.manifold.topological_insights import TopologicalInsightGenerator
            from tcd_jepa.manifold.causal_intelligence import CausalIntelligenceEngine
            from tcd_jepa.manifold.uncertainty_intelligence import UncertaintyIntelligenceEngine
            from tcd_jepa.manifold.temporal_intelligence import TemporalIntelligenceEngine
            from tcd_jepa.manifold.meta_intelligence import MetaIntelligenceEngine

            logger.info("Running deep signal intelligence engines...")

            insights.extend(
                TopologicalInsightGenerator().generate(deep_signal_profile, chunks, features, labels)
            )
            insights.extend(
                CausalIntelligenceEngine().generate(deep_signal_profile, chunks, features, adjacency)
            )
            insights.extend(
                UncertaintyIntelligenceEngine().generate(deep_signal_profile, chunks, features, labels)
            )
            insights.extend(
                TemporalIntelligenceEngine().generate(
                    deep_signal_profile, chunks, features, labels,
                    coords=coords, velocity=velocity,
                )
            )
            # Meta runs last — it analyzes all other insights
            insights.extend(
                MetaIntelligenceEngine().generate(insights, deep_signal_profile, kpis)
            )

            logger.info(f"Deep signal engines produced {sum(1 for i in insights if i.category in ('knowledge_gap', 'feedback_loop', 'hierarchy', 'topological_stability', 'causal_chain', 'information_bottleneck', 'intervention', 'blind_spot', 'confidence_field', 'coverage_gap', 'topic_drift', 'phase_transition', 'exploration_dynamics', 'model_quality', 'insight_reliability', 'corpus_diagnosis'))} deep insights")

        # Divergent analysis (sensitivity, contradictions, counterfactuals)
        if self.enable_divergent and len(insights) > 1:
            self._sensitivity_analysis(insights, features, labels, chunk_texts, chunks)
            self._contradiction_detection(insights)
            self._counterfactual_analysis(insights, features, labels, chunk_texts, chunks)
            self._consensus_scoring(insights)

        # Register insight nodes in lineage graph
        if lineage_graph is not None:
            for ins in insights:
                node = InsightNode(
                    lineage_id=ins.insight_id,
                    kind="insight",
                    source_ids=ins.source_chunk_ids[:],
                    source_chunk_ids=ins.source_chunk_ids[:],
                    source_link_ids=ins.source_link_ids[:],
                    source_cluster_ids=ins.source_cluster_ids[:],
                    derivation_steps=ins.derivation_steps[:],
                    confidence_breakdown=dict(ins.confidence_breakdown),
                    contradicts=ins.contradicts[:],
                    supports=ins.supports[:],
                )
                lineage_graph.add(node)

        # Sort by confidence (descending)
        insights.sort(key=lambda x: (-{"high": 3, "medium": 2, "low": 1}[x.severity], -x.confidence))

        # Generate executive summary
        summary = self._generate_summary(insights, kpis, labels, chunk_texts, chunks)

        num_docs = len(set(c.doc_id for c in chunks)) if chunks else 0
        num_links = (adjacency > 0).sum().item() if adjacency is not None else 0

        return InsightReport(
            insights=insights,
            summary=summary,
            kpis=kpis,
            num_documents=num_docs,
            num_chunks=len(chunk_texts),
            num_clusters=int(labels.max().item()) + 1 if len(labels) > 0 else 0,
            num_links=num_links,
            lineage_graph=lineage_graph,
        )

    def _cluster_insights(
        self, features: torch.Tensor, labels: torch.Tensor,
        texts: list[str], chunks: list,
        lineage_graph: Optional[LineageGraph] = None,
    ) -> list[Insight]:
        """Discover topic clusters and their characteristics."""
        insights = []
        num_classes = int(labels.max().item()) + 1

        feat_norm = F.normalize(features, dim=1)

        for c in range(num_classes):
            mask = labels == c
            count = mask.sum().item()
            if count < 2:
                continue

            # Find representative chunks (closest to centroid)
            c_feats = feat_norm[mask]
            centroid = c_feats.mean(0)
            dists = (c_feats - centroid).pow(2).sum(dim=-1)
            _, top_idx = dists.topk(min(3, count), largest=False)

            # Map back to global indices (bounded by text count)
            global_indices = torch.where(mask)[0]
            representative_texts = [
                texts[global_indices[i].item()] for i in top_idx
                if global_indices[i].item() < len(texts)
            ]

            # Build structured evidence
            structured_ev = []
            source_chunk_ids = []
            for i in top_idx:
                gi = global_indices[i].item()
                if gi < len(chunks):
                    ch = chunks[gi]
                    structured_ev.append(EvidenceItem(
                        chunk_id=getattr(ch, "chunk_id", ""),
                        doc_id=ch.doc_id,
                        doc_title=getattr(ch, "doc_title", ""),
                        text_excerpt=ch.text[:300],
                        char_start=getattr(ch, "char_start", 0),
                        char_end=getattr(ch, "char_end", 0),
                    ))
                    if hasattr(ch, "chunk_id"):
                        source_chunk_ids.append(ch.chunk_id)
            # Collect all chunk IDs in cluster
            for idx in global_indices.tolist():
                if idx < len(chunks) and hasattr(chunks[idx], "chunk_id"):
                    cid = chunks[idx].chunk_id
                    if cid not in source_chunk_ids:
                        source_chunk_ids.append(cid)

            # Cluster coherence
            intra_sim = (c_feats @ c_feats.t()).fill_diagonal_(0)
            coherence = intra_sim.sum() / max(count * (count - 1), 1)

            # Extract common entities
            cluster_entities = []
            for idx in global_indices.tolist():
                if idx < len(chunks):
                    cluster_entities.extend(chunks[idx].entities)
            entity_counts = {}
            for e in cluster_entities:
                entity_counts[e] = entity_counts.get(e, 0) + 1
            top_entities = sorted(entity_counts.items(), key=lambda x: -x[1])[:5]

            topic_keywords = [e[0] for e in top_entities] if top_entities else ["(unnamed)"]

            severity = "high" if count > len(texts) * 0.15 else "medium" if count > 5 else "low"

            # Register cluster node in lineage
            cluster_id = make_lineage_id("cluster")
            if lineage_graph is not None:
                centroid_idx = global_indices[top_idx[0]].item() if len(top_idx) > 0 else 0
                centroid_cid = chunks[centroid_idx].chunk_id if centroid_idx < len(chunks) and hasattr(chunks[centroid_idx], "chunk_id") else ""
                cn = ClusterNode(
                    lineage_id=cluster_id,
                    kind="cluster",
                    source_ids=source_chunk_ids[:],
                    member_chunk_ids=source_chunk_ids[:],
                    centroid_chunk_id=centroid_cid,
                    coherence=coherence.item(),
                    method_params={"cluster_label": c},
                )
                lineage_graph.add(cn)

            insights.append(Insight(
                category="cluster",
                severity=severity,
                title=f"Topic cluster: {', '.join(topic_keywords[:3])} ({count} chunks, {coherence:.2f} coherence)",
                description=(
                    f"Identified a coherent topic cluster containing {count} semantic chunks "
                    f"({count/len(texts)*100:.0f}% of corpus). "
                    f"Internal coherence: {coherence:.2f}. "
                    f"Key entities: {', '.join(topic_keywords)}. "
                    f"RECOMMENDATION: {'Cluster is well-formed, no action needed.' if coherence > 0.5 else 'Consider subdividing — low coherence suggests mixed themes.'}"
                ),
                evidence=representative_texts,
                entities=[e[0] for e in top_entities],
                confidence=min(1.0, coherence.item() * 1.5),
                metadata={"cluster_id": c, "size": count, "coherence": coherence.item()},
                structured_evidence=structured_ev,
                source_chunk_ids=source_chunk_ids,
                source_cluster_ids=[cluster_id],
                derivation_steps=[
                    "Compute feature centroids per label",
                    f"Identified cluster {c} with {count} members",
                    f"Computed coherence={coherence:.3f}",
                    "Ranked by distance to centroid",
                ],
                confidence_breakdown={
                    "coherence_signal": min(1.0, coherence.item() * 1.5),
                    "size_signal": min(1.0, count / len(texts)),
                },
            ))

        return insights[:self.top_k]

    def _relationship_insights(
        self, features: torch.Tensor, labels: torch.Tensor,
        texts: list[str], chunks: list, adjacency: torch.Tensor,
        lineage_graph: Optional[LineageGraph] = None,
    ) -> list[Insight]:
        """Identify key relationships between document chunks."""
        insights = []
        N = len(texts)

        # Find strongest cross-cluster links
        cross_links = []
        for i in range(N):
            for j in range(N):
                if adjacency[i, j] > 0.5 and labels[i] != labels[j]:
                    cross_links.append((i, j, adjacency[i, j].item()))

        cross_links.sort(key=lambda x: -x[2])

        for i, j, strength in cross_links[:self.top_k]:
            entities = list(set(
                (chunks[i].entities if i < len(chunks) else []) +
                (chunks[j].entities if j < len(chunks) else [])
            ))
            src_ids = []
            s_ev = []
            for idx in [i, j]:
                if idx < len(chunks):
                    ch = chunks[idx]
                    if hasattr(ch, "chunk_id"):
                        src_ids.append(ch.chunk_id)
                    s_ev.append(EvidenceItem(
                        chunk_id=getattr(ch, "chunk_id", ""),
                        doc_id=ch.doc_id,
                        doc_title=getattr(ch, "doc_title", ""),
                        text_excerpt=ch.text[:300],
                        char_start=getattr(ch, "char_start", 0),
                        char_end=getattr(ch, "char_end", 0),
                        relevance=strength,
                    ))

            insights.append(Insight(
                category="relationship",
                severity="high" if strength > 0.8 else "medium",
                title=f"Cross-topic link (strength {strength:.2f}) between clusters {labels[i].item()} and {labels[j].item()}",
                description=(
                    f"Strong connection ({strength:.2f}) between chunks from different topic clusters. "
                    f"This suggests a causal or semantic bridge between otherwise separate themes. "
                    f"RECOMMENDATION: Investigate shared concepts for cross-domain synergy."
                ),
                evidence=[texts[i][:300], texts[j][:300]],
                entities=entities[:5],
                confidence=strength,
                metadata={"source": i, "target": j, "strength": strength},
                structured_evidence=s_ev,
                source_chunk_ids=src_ids,
                derivation_steps=[
                    "Scanned adjacency for cross-cluster edges",
                    f"Found link {i}->{j} with strength {strength:.2f}",
                    f"Clusters {labels[i].item()} and {labels[j].item()} bridged",
                ],
            ))

        # Find hub nodes (highly connected)
        degree = (adjacency > 0).float().sum(dim=1)
        top_hubs = degree.topk(min(3, N)).indices

        for hub_idx in top_hubs:
            hub_idx = hub_idx.item()
            hub_degree = degree[hub_idx].item()
            if hub_degree < 3:
                continue

            hub_entities = chunks[hub_idx].entities if hub_idx < len(chunks) else []

            insights.append(Insight(
                category="relationship",
                severity="medium",
                title=f"Hub chunk (degree {hub_degree:.0f}) connecting multiple topics",
                description=(
                    f"This chunk connects to {hub_degree:.0f} other chunks, acting as a "
                    f"central node in the document graph. It likely contains key concepts "
                    f"that bridge multiple themes."
                ),
                evidence=[texts[hub_idx][:300]],
                entities=hub_entities[:5],
                confidence=min(1.0, hub_degree / 10),
                metadata={"chunk_idx": hub_idx, "degree": hub_degree},
            ))

        return insights

    def _opportunity_insights(
        self, features: torch.Tensor, labels: torch.Tensor,
        texts: list[str], chunks: list,
        lineage_graph: Optional[LineageGraph] = None,
    ) -> list[Insight]:
        """Find cross-domain opportunities (H1 cycles in topic space)."""
        insights = []
        feat_norm = F.normalize(features, dim=1)
        sim = feat_norm @ feat_norm.t()

        # Cross-type high-similarity pairs
        num_classes = int(labels.max().item()) + 1
        opportunities = []

        for c1 in range(num_classes):
            for c2 in range(c1 + 1, num_classes):
                mask1 = labels == c1
                mask2 = labels == c2
                if mask1.sum() < 2 or mask2.sum() < 2:
                    continue

                cross_sim = sim[mask1][:, mask2]
                high_sim = (cross_sim > self.opportunity_sim_threshold)

                if high_sim.any():
                    count = high_sim.sum().item()
                    max_sim = cross_sim.max().item()
                    avg_sim = cross_sim[high_sim].mean().item()

                    # Find best example pair
                    flat_idx = cross_sim.argmax()
                    i_local = flat_idx // cross_sim.shape[1]
                    j_local = flat_idx % cross_sim.shape[1]
                    i_global = torch.where(mask1)[0][i_local].item()
                    j_global = torch.where(mask2)[0][j_local].item()

                    opportunities.append({
                        "c1": c1, "c2": c2, "count": count,
                        "max_sim": max_sim, "avg_sim": avg_sim,
                        "example_i": i_global, "example_j": j_global,
                    })

        opportunities.sort(key=lambda x: -x["max_sim"])

        for opp in opportunities[:self.top_k]:
            i, j = opp["example_i"], opp["example_j"]
            entities = list(set(
                (chunks[i].entities if i < len(chunks) else []) +
                (chunks[j].entities if j < len(chunks) else [])
            ))

            insights.append(Insight(
                category="opportunity",
                severity="high" if opp["count"] > 5 else "medium",
                title=f"Cross-topic synergy between clusters {opp['c1']} and {opp['c2']} ({opp['count']} connections)",
                description=(
                    f"Found {opp['count']} high-similarity connections between topic clusters "
                    f"{opp['c1']} and {opp['c2']} (max similarity: {opp['max_sim']:.2f}). "
                    f"These cross-domain connections suggest unexploited synergies or "
                    f"emerging convergence between otherwise separate themes."
                ),
                evidence=[texts[i][:300], texts[j][:300]],
                entities=entities[:5],
                confidence=opp["max_sim"],
                metadata=opp,
            ))

        return insights

    def _risk_insights(
        self, features: torch.Tensor, labels: torch.Tensor,
        texts: list[str], chunks: list,
        lineage_graph: Optional[LineageGraph] = None,
    ) -> list[Insight]:
        """Identify at-risk entities near cluster boundaries."""
        insights = []
        num_classes = int(labels.max().item()) + 1

        # Compute centroids
        centroids = torch.zeros(num_classes, features.shape[1])
        for c in range(num_classes):
            mask = labels == c
            if mask.sum() > 0:
                centroids[c] = features[mask].mean(0)

        # Find boundary chunks
        boundary_chunks = []
        for i in range(len(texts)):
            own_class = labels[i].item()
            own_dist = (features[i] - centroids[own_class]).norm().item()

            other_dists = []
            for c in range(num_classes):
                if c != own_class:
                    other_dists.append((c, (features[i] - centroids[c]).norm().item()))

            if other_dists:
                nearest_other = min(other_dists, key=lambda x: x[1])
                ratio = own_dist / (nearest_other[1] + 1e-8)

                if ratio > self.risk_boundary_ratio:
                    boundary_chunks.append({
                        "idx": i, "own_class": own_class,
                        "nearest_other": nearest_other[0],
                        "ratio": ratio, "own_dist": own_dist,
                        "other_dist": nearest_other[1],
                    })

        boundary_chunks.sort(key=lambda x: -x["ratio"])

        for bc in boundary_chunks[:self.top_k]:
            idx = bc["idx"]
            entities = chunks[idx].entities if idx < len(chunks) else []

            insights.append(Insight(
                category="risk",
                severity="high" if bc["ratio"] > 1.0 else "medium",
                title=f"Boundary chunk between clusters {bc['own_class']} and {bc['nearest_other']}",
                description=(
                    f"This chunk sits at the boundary between topic clusters "
                    f"{bc['own_class']} and {bc['nearest_other']} "
                    f"(distance ratio: {bc['ratio']:.2f}). Its classification is unstable — "
                    f"it could belong to either cluster. This ambiguity may indicate "
                    f"evolving themes, contested territory, or content requiring clarification."
                ),
                evidence=[texts[idx][:300]],
                entities=entities[:5],
                confidence=min(1.0, bc["ratio"]),
                metadata=bc,
            ))

        return insights

    def _anomaly_insights(
        self, prediction_errors: torch.Tensor,
        texts: list[str], chunks: list,
        lineage_graph: Optional[LineageGraph] = None,
    ) -> list[Insight]:
        """Identify anomalous content via prediction error."""
        insights = []

        mean_err = prediction_errors.mean()
        std_err = prediction_errors.std()

        # Z-score anomalies
        z_scores = (prediction_errors - mean_err) / (std_err + 1e-8)
        anomaly_mask = z_scores > self.anomaly_threshold

        anomaly_indices = torch.where(anomaly_mask)[0]

        # Sort by z-score
        sorted_anomalies = sorted(
            anomaly_indices.tolist(),
            key=lambda i: -z_scores[i].item(),
        )

        for idx in sorted_anomalies[:self.top_k]:
            z = z_scores[idx].item()
            err = prediction_errors[idx].item()
            entities = chunks[idx].entities if idx < len(chunks) else []

            insights.append(Insight(
                category="anomaly",
                severity="high" if z > 3.0 else "medium",
                title=f"Anomalous content (z-score: {z:.1f})",
                description=(
                    f"This chunk has a prediction error {z:.1f}σ above the mean "
                    f"(error: {err:.3f}, corpus mean: {mean_err:.3f}). "
                    f"The model finds this content surprising — it doesn't fit the "
                    f"learned patterns. This may indicate genuinely novel information, "
                    f"an outlier topic, or content that contradicts corpus consensus."
                ),
                evidence=[texts[idx][:300]],
                entities=entities[:5],
                confidence=min(1.0, z / 5.0),
                metadata={"z_score": z, "error": err, "chunk_idx": idx},
            ))

        return insights

    def _trend_insights(
        self, features: torch.Tensor, labels: torch.Tensor,
        texts: list[str], chunks: list,
        prediction_errors: torch.Tensor,
        lineage_graph: Optional[LineageGraph] = None,
    ) -> list[Insight]:
        """Identify evolving topics via prediction error patterns."""
        insights = []
        num_classes = int(labels.max().item()) + 1

        # Per-cluster prediction error (high = active dynamics)
        cluster_dynamics = []
        for c in range(num_classes):
            mask = labels == c
            if mask.sum() < 2:
                continue
            cluster_err = prediction_errors[mask].mean().item()
            cluster_std = prediction_errors[mask].std().item()
            cluster_dynamics.append({
                "cluster": c, "mean_error": cluster_err,
                "std_error": cluster_std, "size": mask.sum().item(),
            })

        cluster_dynamics.sort(key=lambda x: -x["mean_error"])
        global_mean = prediction_errors.mean().item()

        for cd in cluster_dynamics[:self.top_k]:
            c = cd["cluster"]
            mask = labels == c
            global_indices = torch.where(mask)[0]

            # Representative chunks from this cluster
            evidence_texts = [texts[global_indices[i].item()]
                              for i in range(min(2, len(global_indices)))]

            cluster_entities = []
            for idx in global_indices[:10].tolist():
                if idx < len(chunks):
                    cluster_entities.extend(chunks[idx].entities)
            entity_counts = {}
            for e in cluster_entities:
                entity_counts[e] = entity_counts.get(e, 0) + 1
            top_entities = sorted(entity_counts.items(), key=lambda x: -x[1])[:5]

            is_dynamic = cd["mean_error"] > global_mean * 1.2
            severity = "high" if is_dynamic and cd["size"] > 5 else "medium" if is_dynamic else "low"

            insights.append(Insight(
                category="trend",
                severity=severity,
                title=f"{'Rapidly evolving' if is_dynamic else 'Stable'} topic cluster {c} (error: {cd['mean_error']:.3f})",
                description=(
                    f"Cluster {c} ({cd['size']} chunks) has "
                    f"{'above-average' if is_dynamic else 'below-average'} prediction error "
                    f"({cd['mean_error']:.3f} vs corpus mean {global_mean:.3f}). "
                    f"{'This topic is actively evolving — the model finds it harder to predict, suggesting dynamic content.' if is_dynamic else 'This topic is well-understood by the model — stable, predictable content.'}"
                ),
                evidence=evidence_texts,
                entities=[e[0] for e in top_entities],
                confidence=min(1.0, abs(cd["mean_error"] - global_mean) / (global_mean + 1e-8)),
                metadata=cd,
            ))

        return insights

    def _strategic_insights(
        self, kpis: dict, features: torch.Tensor,
        labels: torch.Tensor, texts: list[str],
    ) -> list[Insight]:
        """Generate strategic insights from KPI analysis."""
        insights = []

        # Clarity assessment
        clarity = kpis.get("clarity_score", 0)
        if clarity < 20:
            insights.append(Insight(
                category="strategic",
                severity="high",
                title="Low corpus clarity — topics are poorly separated",
                description=(
                    f"Clarity Score: {clarity:.1f}/100. The document corpus lacks clear "
                    f"topic separation. This may indicate overlapping themes, inconsistent "
                    f"categorization, or a need for better document organization. "
                    f"Consider: more specific topic labeling, content deduplication, "
                    f"or splitting multi-topic documents."
                ),
                evidence=[], entities=[],
                confidence=0.8,
                metadata={"clarity_score": clarity},
            ))
        elif clarity > 70:
            insights.append(Insight(
                category="strategic",
                severity="low",
                title="High corpus clarity — well-organized topic structure",
                description=(
                    f"Clarity Score: {clarity:.1f}/100. Topics are well-separated in the "
                    f"learned representation space. The corpus has clear thematic structure."
                ),
                evidence=[], entities=[],
                confidence=0.9,
                metadata={"clarity_score": clarity},
            ))

        # Opportunity assessment
        opp = kpis.get("opportunity_surface", 0)
        if opp > 100:
            insights.append(Insight(
                category="strategic",
                severity="high",
                title=f"Large opportunity surface ({opp} cross-topic connections)",
                description=(
                    f"Found {opp} cross-topic connections where semantically similar "
                    f"content spans different topic clusters. This large opportunity surface "
                    f"suggests significant potential for cross-pollination, knowledge transfer, "
                    f"or discovering hidden connections between business domains."
                ),
                evidence=[], entities=[],
                confidence=0.7,
                metadata={"opportunity_surface": opp},
            ))

        # Risk assessment
        risk = kpis.get("risk_horizon", 0)
        total = len(texts) if texts else 1
        risk_pct = risk / total * 100
        if risk_pct > 30:
            insights.append(Insight(
                category="strategic",
                severity="high",
                title=f"High risk horizon ({risk} boundary chunks, {risk_pct:.0f}%)",
                description=(
                    f"{risk} chunks ({risk_pct:.0f}% of corpus) sit at topic boundaries "
                    f"with ambiguous classification. High boundary density indicates "
                    f"rapidly evolving themes, contested categorizations, or areas "
                    f"where the current topic structure doesn't fit the data well."
                ),
                evidence=[], entities=[],
                confidence=0.75,
                metadata={"risk_horizon": risk, "risk_pct": risk_pct},
            ))

        return insights

    def _generate_summary(
        self, insights: list[Insight], kpis: dict,
        labels: torch.Tensor, texts: list[str], chunks: list,
    ) -> str:
        """Generate executive summary from all insights."""
        num_clusters = int(labels.max().item()) + 1 if len(labels) > 0 else 0
        num_docs = len(set(c.doc_id for c in chunks)) if chunks else 0

        high_priority = [i for i in insights if i.severity == "high"]
        opportunities = [i for i in insights if i.category == "opportunity"]
        risks = [i for i in insights if i.category == "risk"]
        anomalies = [i for i in insights if i.category == "anomaly"]

        lines = []
        lines.append(
            f"Analysis of {num_docs} documents ({len(texts)} semantic chunks) reveals "
            f"{num_clusters} distinct topic clusters."
        )

        if high_priority:
            lines.append(
                f"\n{len(high_priority)} high-priority findings require attention."
            )

        clarity = kpis.get("clarity_score", 0)
        if clarity > 50:
            lines.append(
                f"The corpus has strong thematic structure (clarity: {clarity:.0f}/100)."
            )
        else:
            lines.append(
                f"Topic separation is moderate (clarity: {clarity:.0f}/100) — "
                f"some themes overlap significantly."
            )

        if opportunities:
            lines.append(
                f"\n{len(opportunities)} cross-topic opportunities identified — "
                f"connections between otherwise separate themes that may indicate "
                f"untapped synergies."
            )

        if risks:
            lines.append(
                f"\n{len(risks)} chunks identified at topic boundaries — "
                f"ambiguous content that may need reclassification or represents "
                f"emerging themes."
            )

        if anomalies:
            lines.append(
                f"\n{len(anomalies)} anomalous chunks detected — content that "
                f"doesn't fit established patterns and warrants investigation."
            )

        return " ".join(lines)

    # ------------------------------------------------------------------
    # Divergent analysis methods
    # ------------------------------------------------------------------

    def _sensitivity_analysis(
        self,
        insights: list[Insight],
        features: torch.Tensor,
        labels: torch.Tensor,
        texts: list[str],
        chunks: list,
    ) -> None:
        """Run insight generators at multiple thresholds to measure stability.

        Modifies insights in-place: sets ``sensitivity`` score (0-1).
        Insights that appear at all threshold levels get high sensitivity.
        """
        # Vary the opportunity threshold at 3 levels
        levels = [0.6, 0.7, 0.85]
        original_thresh = self.opportunity_sim_threshold

        # Collect insight titles at each level
        per_level: list[set[str]] = []
        for thresh in levels:
            self.opportunity_sim_threshold = thresh
            level_insights = []
            level_insights.extend(self._cluster_insights(features, labels, texts, chunks))
            level_insights.extend(self._opportunity_insights(features, labels, texts, chunks))
            level_insights.extend(self._risk_insights(features, labels, texts, chunks))
            per_level.append({i.title for i in level_insights})

        self.opportunity_sim_threshold = original_thresh

        # Score each insight by how many levels it appeared in
        for ins in insights:
            appearances = sum(1 for level_titles in per_level if ins.title in level_titles)
            ins.sensitivity = appearances / len(levels)
            ins.derivation_steps.append(f"Sensitivity: appeared in {appearances}/{len(levels)} parameter sweeps")

    def _contradiction_detection(self, insights: list[Insight]) -> None:
        """Cross-check insights for contradictions. Modifies in-place."""
        cluster_insights = [i for i in insights if i.category == "cluster"]
        risk_insights = [i for i in insights if i.category == "risk"]
        opp_insights = [i for i in insights if i.category == "opportunity"]
        rel_insights = [i for i in insights if i.category == "relationship"]

        # Cluster "cohesive" vs Risk "boundary" contradiction
        for ci in cluster_insights:
            ci_cluster_id = ci.metadata.get("cluster_id")
            if ci_cluster_id is None:
                continue
            for ri in risk_insights:
                ri_own = ri.metadata.get("own_class")
                ri_near = ri.metadata.get("nearest_other")
                if ri_own == ci_cluster_id or ri_near == ci_cluster_id:
                    coherence = ci.metadata.get("coherence", 0)
                    if coherence > 0.5:
                        ci.contradicts.append(ri.insight_id)
                        ri.contradicts.append(ci.insight_id)
                        ci.derivation_steps.append(
                            f"Contradiction: cluster {ci_cluster_id} is cohesive "
                            f"({coherence:.2f}) but chunk at boundary"
                        )

        # Opportunity "connected" vs Relationship "weak" contradiction
        for oi in opp_insights:
            c1 = oi.metadata.get("c1")
            c2 = oi.metadata.get("c2")
            for ri in rel_insights:
                ri_src = ri.metadata.get("source")
                ri_tgt = ri.metadata.get("target")
                if ri_src is not None and ri_tgt is not None:
                    strength = ri.metadata.get("strength", 1.0)
                    if strength < 0.5 and c1 is not None and c2 is not None:
                        oi.contradicts.append(ri.insight_id)
                        ri.contradicts.append(oi.insight_id)

    def _counterfactual_analysis(
        self,
        insights: list[Insight],
        features: torch.Tensor,
        labels: torch.Tensor,
        texts: list[str],
        chunks: list,
        remove_pct: float = 0.05,
    ) -> None:
        """For cluster insights: would they survive if we remove top anomalous chunks?

        Measures robustness. Modifies insights in-place.
        """
        N = features.shape[0]
        if N < 10:
            return

        # Use feature norm as anomaly proxy
        norms = features.norm(dim=-1)
        k_remove = max(1, int(N * remove_pct))
        _, anomaly_idx = norms.topk(k_remove, largest=True)
        keep_mask = torch.ones(N, dtype=torch.bool)
        keep_mask[anomaly_idx] = False

        kept_features = features[keep_mask]
        kept_labels = labels[keep_mask]

        for ins in insights:
            if ins.category != "cluster":
                continue
            c = ins.metadata.get("cluster_id")
            if c is None:
                continue
            # Check if cluster survives
            orig_count = ins.metadata.get("size", 0)
            new_count = (kept_labels == c).sum().item()
            survival_ratio = new_count / max(orig_count, 1)
            ins.metadata["counterfactual_survival"] = survival_ratio
            ins.derivation_steps.append(
                f"Counterfactual: removing top {remove_pct*100:.0f}% anomalous chunks "
                f"retains {survival_ratio*100:.0f}% of cluster"
            )

    def _consensus_scoring(self, insights: list[Insight]) -> None:
        """Boost confidence for robust insights, downgrade fragile ones."""
        for ins in insights:
            adjustments = []
            # Sensitivity boost/penalty
            if ins.sensitivity >= 0.8:
                adjustments.append(0.05)
            elif ins.sensitivity > 0 and ins.sensitivity < 0.4:
                adjustments.append(-0.1)

            # Contradiction penalty
            if ins.contradicts:
                adjustments.append(-0.05 * len(ins.contradicts))

            # Counterfactual survival boost
            survival = ins.metadata.get("counterfactual_survival")
            if survival is not None and survival > 0.9:
                adjustments.append(0.05)

            if adjustments:
                total_adj = sum(adjustments)
                ins.confidence = max(0.0, min(1.0, ins.confidence + total_adj))
                ins.derivation_steps.append(
                    f"Consensus scoring: confidence adjusted by {total_adj:+.2f}"
                )
