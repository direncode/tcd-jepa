"""Temporal intelligence — velocity fields, convergence dynamics, phase transitions.

Uses convergence history and representation dynamics to detect:
- Topic drift (velocity field analysis)
- Phase transitions (sudden structural changes)
- Exploration dynamics (coverage of representation space)
"""

import logging

import torch
import torch.nn.functional as F

from tcd_jepa.manifold.insight_engine import Insight
from tcd_jepa.manifold.lineage import make_lineage_id

logger = logging.getLogger("tcd_jepa.temporal_intelligence")


class TemporalIntelligenceEngine:
    """Generates temporal insights from convergence dynamics and velocity fields."""

    def generate(self, profile, chunks, features, labels, coords=None, velocity=None) -> list[Insight]:
        """Generate all temporal insights.

        Args:
            profile: DeepSignalProfile with convergence data.
            chunks: List of DocumentChunk objects.
            features: [N, D] learned representations.
            labels: [N] cluster labels.
            coords: Optional [N, 3] manifold coordinates.
            velocity: Optional [N, D] velocity field.
        """
        insights = []
        try:
            insights.extend(self._velocity_field_insights(profile, chunks, features, labels, velocity))
            insights.extend(self._phase_transition_insights(profile, chunks))
            insights.extend(self._convergence_insights(profile, chunks, features, labels))
            insights.extend(self._exploration_dynamics_insights(profile, chunks, features))
        except Exception as e:
            logger.warning(f"Temporal insight generation failed: {e}")
        return insights

    def _velocity_field_insights(self, profile, chunks, features, labels, velocity=None) -> list[Insight]:
        """Detect topics in motion via velocity analysis."""
        insights = []
        N = min(len(features), len(chunks))
        num_clusters = int(labels[:N].max().item()) + 1 if len(labels[:N]) > 0 else 0

        if num_clusters < 2:
            return insights

        # Use energy gradient as velocity proxy if no explicit velocity
        if velocity is not None and len(velocity) >= N:
            vel = velocity[:N]
        elif profile.energy_map is not None:
            # Approximate velocity from energy gradient differences
            energy = profile.energy_map[:N]
            feat_norm = F.normalize(features[:N], dim=1)
            # Velocity proxy: energy differences between neighbors
            sim = feat_norm @ feat_norm.t()
            vel_magnitude = torch.zeros(N)
            for i in range(N):
                neighbors = torch.where(sim[i] > sim[i].mean())[0]
                if len(neighbors) > 1:
                    energy_diffs = (energy[neighbors] - energy[i]).abs()
                    vel_magnitude[i] = energy_diffs.mean()
            vel = vel_magnitude
        else:
            return insights

        # Per-cluster velocity
        cluster_vel = {}
        vel_mag = vel if vel.dim() == 1 else vel.norm(dim=-1)
        corpus_mean_vel = vel_mag.mean().item()

        for c in range(num_clusters):
            c_mask = labels[:N] == c
            if c_mask.sum() > 0:
                c_vel = vel_mag[c_mask].mean().item()
                cluster_vel[c] = c_vel

        # Find fast-moving clusters
        sorted_clusters = sorted(cluster_vel.items(), key=lambda x: x[1], reverse=True)

        for cluster_id, avg_vel in sorted_clusters[:2]:
            if corpus_mean_vel > 0 and avg_vel > corpus_mean_vel * 1.5:
                ratio = avg_vel / corpus_mean_vel

                # Find direction of movement (toward which other cluster?)
                c_mask = labels[:N] == cluster_id
                c_features = F.normalize(features[:N][c_mask], dim=1)
                c_centroid = c_features.mean(0)

                closest_cluster = -1
                closest_sim = -1
                for other_c in range(num_clusters):
                    if other_c == cluster_id:
                        continue
                    o_mask = labels[:N] == other_c
                    if o_mask.sum() > 0:
                        o_centroid = F.normalize(features[:N][o_mask], dim=1).mean(0)
                        sim = F.cosine_similarity(c_centroid.unsqueeze(0), o_centroid.unsqueeze(0)).item()
                        if sim > closest_sim:
                            closest_sim = sim
                            closest_cluster = other_c

                direction_str = ""
                if closest_cluster >= 0:
                    direction_str = (
                        f" The drift direction is toward cluster {closest_cluster} "
                        f"(similarity: {closest_sim:.3f})."
                    )

                insights.append(Insight(
                    category="topic_drift",
                    severity="medium",
                    title=f"Cluster {cluster_id} is drifting ({ratio:.1f}x corpus average velocity)",
                    description=(
                        f"Cluster {cluster_id} has mean velocity magnitude {avg_vel:.4f} "
                        f"({ratio:.1f}x corpus average {corpus_mean_vel:.4f}).{direction_str} "
                        f"This indicates rapid semantic evolution in this topic area."
                    ),
                    evidence=[],
                    entities=[f"cluster_{cluster_id}"],
                    confidence=min(0.8, 0.5 + ratio * 0.1),
                    insight_id=make_lineage_id("insight"),
                    derivation_steps=["velocity_field_extraction", "cluster_velocity_aggregation", "drift_detection"],
                ))

        return insights

    def _phase_transition_insights(self, profile, chunks) -> list[Insight]:
        """Detect sudden structural changes in convergence history."""
        insights = []
        history = profile.convergence_history
        if len(history) < 5:
            return insights

        # Look for sudden drops or spikes in convergence score
        scores = [h.get("convergence_score", 0) for h in history]

        max_drop = 0
        drop_epoch = -1
        for i in range(1, len(scores)):
            drop = scores[i - 1] - scores[i]
            if drop > max_drop:
                max_drop = drop
                drop_epoch = i

        max_spike = 0
        for i in range(1, len(scores)):
            spike = scores[i] - scores[i - 1]
            if spike > max_spike:
                max_spike = spike
                _spike_epoch = i

        # Report phase transitions (sudden drops = crystallization events)
        if max_drop > 0.05 and drop_epoch > 0:
            before = history[drop_epoch - 1]
            after = history[drop_epoch]

            module_before = before.get("module_change", 0)
            module_after = after.get("module_change", 0)

            insights.append(Insight(
                category="phase_transition",
                severity="medium",
                title=f"Phase transition detected at iteration {drop_epoch}",
                description=(
                    f"Between iteration {drop_epoch - 1} and {drop_epoch}, the convergence "
                    f"score dropped from {scores[drop_epoch - 1]:.4f} to {scores[drop_epoch]:.4f} "
                    f"(Δ = {max_drop:.4f}). The representation underwent a phase transition — "
                    f"the system discovered new structure. "
                    f"Module change rate: {module_before:.4f} → {module_after:.4f}."
                ),
                evidence=[],
                entities=[],
                confidence=min(0.85, 0.5 + max_drop * 5),
                insight_id=make_lineage_id("insight"),
                derivation_steps=["convergence_monitoring", "change_point_detection", "phase_transition_analysis"],
            ))

        return insights

    def _convergence_insights(self, profile, chunks, features, labels) -> list[Insight]:
        """Report on convergence status and what it means."""
        insights = []
        history = profile.convergence_history
        if not history:
            return insights

        latest = history[-1]
        score = latest.get("convergence_score", 0)
        is_converged = latest.get("is_converged", False)
        consecutive = latest.get("consecutive_converged", 0)

        status = "converged" if is_converged else "still evolving"
        stability_desc = f"({consecutive} consecutive stable iterations)" if consecutive > 0 else ""

        insights.append(Insight(
            category="topic_drift",
            severity="low",
            title=f"Model state: {status} {stability_desc}",
            description=(
                f"The model has {'converged' if is_converged else 'not yet converged'}. "
                f"Latest convergence score: {score:.6f}. "
                f"Representation stability: {profile.representation_stability:.3f}. "
                f"{'The learned structure is stable and insights are reliable.' if is_converged else 'The model is still evolving — insights may shift with additional training.'}"
            ),
            evidence=[],
            entities=[],
            confidence=0.9 if is_converged else 0.6,
            insight_id=make_lineage_id("insight"),
            derivation_steps=["convergence_monitoring", "stability_assessment"],
        ))

        return insights

    def _exploration_dynamics_insights(self, profile, chunks, features) -> list[Insight]:
        """Report on how much of the representation space was explored."""
        insights = []

        if profile.coverage_ratio == 0:
            return insights

        coverage = profile.coverage_ratio
        effective_rank = profile.effective_rank
        total_dim = features.shape[1] if len(features) > 0 else 0
        utilization = effective_rank / total_dim if total_dim > 0 else 0

        insights.append(Insight(
            category="exploration_dynamics",
            severity="low" if coverage > 0.7 else "medium",
            title=f"Exploration covered {coverage:.0%} of representation space",
            description=(
                f"The model's exploration covered {coverage:.0%} of the representation volume. "
                f"Effective rank: {effective_rank:.1f}/{total_dim} ({utilization:.0%} utilization). "
                f"{'The model has explored most of the space — good coverage.' if coverage > 0.7 else 'Significant unexplored regions remain — consider longer training or more diverse data.'}"
            ),
            evidence=[],
            entities=[],
            confidence=0.7,
            insight_id=make_lineage_id("insight"),
            derivation_steps=["coverage_analysis", "effective_rank_computation", "exploration_assessment"],
        ))

        return insights
