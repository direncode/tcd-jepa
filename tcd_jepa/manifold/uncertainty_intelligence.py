"""Uncertainty intelligence — blank spaces and Fisher information as insights.

Turns model uncertainty into actionable intelligence:
- Blind spots (high blank score regions)
- Confidence calibration (Fisher-based confidence mapping)
- Coverage analysis (what fraction of topic space is well-modeled)
- Epistemic uncertainty heatmap summary
"""

import logging

import torch

from tcd_jepa.manifold.insight_engine import Insight
from tcd_jepa.manifold.lineage import make_lineage_id

logger = logging.getLogger("tcd_jepa.uncertainty_intelligence")


class UncertaintyIntelligenceEngine:
    """Generates insights from uncertainty quantification."""

    def generate(self, profile, chunks, features, labels) -> list[Insight]:
        """Generate all uncertainty insights.

        Args:
            profile: DeepSignalProfile with uncertainty data.
            chunks: List of DocumentChunk objects.
            features: [N, D] learned representations.
            labels: [N] cluster labels.
        """
        insights = []
        try:
            insights.extend(self._blank_space_insights(profile, chunks, features, labels))
            insights.extend(self._confidence_calibration_insights(profile, chunks, features, labels))
            insights.extend(self._coverage_analysis_insights(profile, chunks, features, labels))
            insights.extend(self._epistemic_map_insights(profile, chunks, features, labels))
        except Exception as e:
            logger.warning(f"Uncertainty insight generation failed: {e}")
        return insights

    def _blank_space_insights(self, profile, chunks, features, labels) -> list[Insight]:
        """High blank scores → knowledge blind spots."""
        insights = []
        blank = profile.blank_scores
        fisher = profile.fisher_traces
        if blank is None:
            return insights

        N = min(len(blank), len(chunks))
        if N == 0:
            return insights

        # Find chunks with highest uncertainty
        uncertainty = profile.epistemic_uncertainty[:N] if profile.epistemic_uncertainty is not None else blank[:N]
        mean_unc = uncertainty.mean().item()
        std_unc = uncertainty.std().item()

        # High uncertainty chunks (> mean + 1 std)
        threshold = mean_unc + std_unc
        high_unc_mask = uncertainty > threshold
        high_unc_indices = torch.where(high_unc_mask)[0]

        if len(high_unc_indices) == 0:
            return insights

        # Group high-uncertainty chunks by cluster
        cluster_uncertainty = {}
        num_clusters = int(labels[:N].max().item()) + 1 if len(labels[:N]) > 0 else 0
        for c in range(num_clusters):
            c_mask = labels[:N] == c
            if c_mask.sum() > 0:
                c_unc = uncertainty[c_mask].mean().item()
                cluster_uncertainty[c] = c_unc

        # Report clusters with highest uncertainty
        sorted_clusters = sorted(cluster_uncertainty.items(), key=lambda x: x[1], reverse=True)

        for cluster_id, avg_unc in sorted_clusters[:2]:
            if avg_unc < threshold:
                continue

            c_mask = labels[:N] == cluster_id
            c_indices = torch.where(c_mask)[0]
            source_ids = []
            evidence = []
            for idx in c_indices[:3]:
                idx_val = idx.item()
                if idx_val < len(chunks):
                    evidence.append(chunks[idx_val].text[:200])
                    if hasattr(chunks[idx_val], 'chunk_id'):
                        source_ids.append(chunks[idx_val].chunk_id)

            fisher_mean = fisher[c_mask].mean().item() if fisher is not None else 0.0
            corpus_fisher_mean = fisher.mean().item() if fisher is not None else 0.0

            insights.append(Insight(
                category="blind_spot",
                severity="high" if avg_unc > mean_unc + 2 * std_unc else "medium",
                title=f"Knowledge blind spot in cluster {cluster_id}",
                description=(
                    f"The model has HIGH uncertainty (mean blank score: {avg_unc:.3f}) "
                    f"in cluster {cluster_id} ({c_mask.sum().item()} chunks). "
                    f"This is a knowledge blind spot — the model struggles to "
                    f"confidently represent these documents. "
                    f"Fisher trace: {fisher_mean:.3f} (vs corpus mean {corpus_fisher_mean:.3f})."
                ),
                evidence=evidence,
                entities=[f"cluster_{cluster_id}"],
                confidence=min(0.85, 0.5 + avg_unc * 0.5),
                insight_id=make_lineage_id("insight"),
                source_chunk_ids=source_ids,
                derivation_steps=["blank_space_detection", "cluster_aggregation", "blind_spot_ranking"],
            ))

        return insights

    def _confidence_calibration_insights(self, profile, chunks, features, labels) -> list[Insight]:
        """Fisher-based confidence mapping across the corpus."""
        insights = []
        fisher = profile.fisher_traces
        if fisher is None:
            return insights

        N = min(len(fisher), len(chunks))
        if N == 0:
            return insights

        fisher_sub = fisher[:N]
        well_modeled_threshold = fisher_sub.mean().item() * 0.5
        well_modeled = (fisher_sub > well_modeled_threshold).float()
        well_modeled_pct = well_modeled.mean().item()
        uncertain_pct = 1.0 - well_modeled_pct

        # Find the most uncertain chunks
        worst_chunks = torch.argsort(fisher_sub)[:5]
        worst_texts = []
        for idx in worst_chunks:
            idx_val = idx.item()
            if idx_val < len(chunks):
                worst_texts.append(f"'{chunks[idx_val].text[:40]}...'")

        insights.append(Insight(
            category="confidence_field",
            severity="medium" if uncertain_pct > 0.3 else "low",
            title=f"{well_modeled_pct:.0%} of corpus is well-modeled by Fisher metric",
            description=(
                f"{well_modeled_pct:.0%} of the corpus has confident representations "
                f"(Fisher trace > {well_modeled_threshold:.3f}). "
                f"The remaining {uncertain_pct:.0%} has uncertain representations. "
                f"Highest uncertainty chunks: {', '.join(worst_texts[:3])}."
            ),
            evidence=[],
            entities=[],
            confidence=0.75,
            insight_id=make_lineage_id("insight"),
            derivation_steps=["fisher_metric_computation", "confidence_thresholding", "coverage_assessment"],
        ))

        return insights

    def _coverage_analysis_insights(self, profile, chunks, features, labels) -> list[Insight]:
        """Per-cluster coverage analysis."""
        insights = []
        epistemic = profile.epistemic_uncertainty
        if epistemic is None:
            return insights

        N = min(len(epistemic), len(chunks))
        num_clusters = int(labels[:N].max().item()) + 1 if len(labels[:N]) > 0 else 0

        if num_clusters < 2:
            return insights

        # Per-cluster coverage
        cluster_coverage = {}
        poor_clusters = []
        excellent_clusters = []
        for c in range(num_clusters):
            c_mask = labels[:N] == c
            if c_mask.sum() == 0:
                continue
            c_unc = epistemic[c_mask]
            coverage = (c_unc < 0.5).float().mean().item()
            cluster_coverage[c] = coverage
            if coverage < 0.6:
                poor_clusters.append((c, coverage))
            elif coverage > 0.85:
                excellent_clusters.append((c, coverage))

        if poor_clusters or len(cluster_coverage) >= 3:
            poor_desc = ", ".join([f"cluster {c} ({cov:.0%})" for c, cov in poor_clusters[:3]])
            excellent_desc = ", ".join([f"cluster {c} ({cov:.0%})" for c, cov in excellent_clusters[:3]])

            insights.append(Insight(
                category="coverage_gap",
                severity="medium" if poor_clusters else "low",
                title=f"Coverage analysis: {len(excellent_clusters)}/{num_clusters} clusters well-modeled",
                description=(
                    f"Of {num_clusters} topic clusters, {len(excellent_clusters)} have excellent "
                    f"coverage (>85% of chunks well-modeled)"
                    f"{': ' + excellent_desc if excellent_desc else ''}. "
                    f"{len(poor_clusters)} clusters have poor coverage (<60%)"
                    f"{': ' + poor_desc if poor_desc else ''}. "
                    f"Overall corpus coverage: {profile.coverage_ratio:.0%}."
                ),
                evidence=[],
                entities=[f"cluster_{c}" for c, _ in poor_clusters[:5]],
                confidence=0.7,
                insight_id=make_lineage_id("insight"),
                derivation_steps=["epistemic_uncertainty", "per_cluster_aggregation", "coverage_assessment"],
            ))

        return insights

    def _epistemic_map_insights(self, profile, chunks, features, labels) -> list[Insight]:
        """Uncertainty field summary — where the model knows and doesn't know."""
        insights = []
        epistemic = profile.epistemic_uncertainty
        if epistemic is None:
            return insights

        N = min(len(epistemic), len(chunks))
        if N == 0:
            return insights

        # Spatial distribution of uncertainty
        unc_mean = epistemic[:N].mean().item()
        unc_std = epistemic[:N].std().item()
        unc_max = epistemic[:N].max().item()
        unc_min = epistemic[:N].min().item()

        # Only report if there's meaningful variation
        if unc_std < 0.01:
            return insights

        # Find the boundary between known and unknown
        high_unc = (epistemic[:N] > unc_mean + unc_std)
        low_unc = (epistemic[:N] < unc_mean - unc_std)

        insights.append(Insight(
            category="blind_spot",
            severity="low",
            title=f"Epistemic uncertainty field: mean={unc_mean:.3f}, spread={unc_std:.3f}",
            description=(
                f"Uncertainty field summary: mean={unc_mean:.3f}, std={unc_std:.3f}, "
                f"range=[{unc_min:.3f}, {unc_max:.3f}]. "
                f"{high_unc.sum().item()} chunks ({high_unc.float().mean():.0%}) have "
                f"high uncertainty. {low_unc.sum().item()} chunks ({low_unc.float().mean():.0%}) "
                f"have low uncertainty. The uncertainty gradient suggests "
                f"{'a concentrated knowledge gap' if unc_std > 0.2 else 'uniformly distributed uncertainty'}."
            ),
            evidence=[],
            entities=[],
            confidence=0.65,
            insight_id=make_lineage_id("insight"),
            derivation_steps=["blank_detection", "fisher_metric", "uncertainty_field_analysis"],
        ))

        return insights
