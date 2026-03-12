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

logger = logging.getLogger("tcd_jepa.insight_engine")


@dataclass
class Insight:
    """A single commercial intelligence insight."""
    category: str          # cluster, relationship, trend, opportunity, risk, anomaly, strategic
    severity: str          # high, medium, low
    title: str             # One-line summary
    description: str       # Detailed explanation
    evidence: list[str]    # Supporting text chunks
    entities: list[str]    # Involved entities
    confidence: float      # 0-1 confidence score
    metadata: dict = field(default_factory=dict)

    def to_nl(self) -> str:
        """Render as natural language paragraph."""
        severity_prefix = {
            "high": "CRITICAL",
            "medium": "NOTABLE",
            "low": "OBSERVATION",
        }
        prefix = severity_prefix.get(self.severity, "NOTE")
        lines = [f"[{prefix}] {self.title}"]
        lines.append(self.description)
        if self.entities:
            lines.append(f"Entities involved: {', '.join(self.entities[:10])}")
        if self.evidence:
            lines.append("Supporting evidence:")
            for ev in self.evidence[:3]:
                truncated = ev[:200] + "..." if len(ev) > 200 else ev
                lines.append(f"  - \"{truncated}\"")
        lines.append(f"Confidence: {self.confidence:.0%}")
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
            lines.append(f"  Clarity Score: {cs:.1f}/100 ({clarity_desc})")
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
        categories = ["strategic", "cluster", "opportunity", "risk", "trend", "relationship", "anomaly"]
        category_titles = {
            "strategic": "STRATEGIC INSIGHTS",
            "cluster": "TOPIC DISCOVERY",
            "opportunity": "OPPORTUNITIES",
            "risk": "RISKS & VULNERABILITIES",
            "trend": "TRENDS & DYNAMICS",
            "relationship": "KEY RELATIONSHIPS",
            "anomaly": "ANOMALIES & NOVELTY",
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
    ):
        self.top_k = top_k_per_category
        self.anomaly_threshold = anomaly_threshold
        self.opportunity_sim_threshold = opportunity_sim_threshold
        self.risk_boundary_ratio = risk_boundary_ratio

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
        """
        insights = []

        # Generate insights by category
        insights.extend(self._cluster_insights(features, labels, chunk_texts, chunks))
        insights.extend(self._relationship_insights(features, labels, chunk_texts, chunks, adjacency))
        insights.extend(self._opportunity_insights(features, labels, chunk_texts, chunks))
        insights.extend(self._risk_insights(features, labels, chunk_texts, chunks))

        if prediction_errors is not None:
            insights.extend(self._anomaly_insights(prediction_errors, chunk_texts, chunks))
            insights.extend(self._trend_insights(features, labels, chunk_texts, chunks, prediction_errors))

        # Strategic insights from KPIs
        insights.extend(self._strategic_insights(kpis, features, labels, chunk_texts))

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
        )

    def _cluster_insights(
        self, features: torch.Tensor, labels: torch.Tensor,
        texts: list[str], chunks: list,
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

            # Topic summary from representative chunks
            topic_keywords = [e[0] for e in top_entities] if top_entities else ["(unnamed)"]

            severity = "high" if count > len(texts) * 0.15 else "medium" if count > 5 else "low"

            insights.append(Insight(
                category="cluster",
                severity=severity,
                title=f"Topic cluster: {', '.join(topic_keywords[:3])} ({count} chunks, {coherence:.2f} coherence)",
                description=(
                    f"Identified a coherent topic cluster containing {count} semantic chunks "
                    f"({count/len(texts)*100:.0f}% of corpus). "
                    f"Internal coherence: {coherence:.2f}. "
                    f"Key entities: {', '.join(topic_keywords)}."
                ),
                evidence=representative_texts,
                entities=[e[0] for e in top_entities],
                confidence=min(1.0, coherence.item() * 1.5),
                metadata={"cluster_id": c, "size": count, "coherence": coherence.item()},
            ))

        return insights[:self.top_k]

    def _relationship_insights(
        self, features: torch.Tensor, labels: torch.Tensor,
        texts: list[str], chunks: list, adjacency: torch.Tensor,
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

            insights.append(Insight(
                category="relationship",
                severity="high" if strength > 0.8 else "medium",
                title=f"Cross-topic link (strength {strength:.2f}) between clusters {labels[i].item()} and {labels[j].item()}",
                description=(
                    f"Strong connection ({strength:.2f}) between chunks from different topic clusters. "
                    f"This suggests a causal or semantic bridge between otherwise separate themes."
                ),
                evidence=[texts[i][:300], texts[j][:300]],
                entities=entities[:5],
                confidence=strength,
                metadata={"source": i, "target": j, "strength": strength},
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
