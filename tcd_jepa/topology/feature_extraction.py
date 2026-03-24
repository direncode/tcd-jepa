"""Maps persistent homology features to predictor module candidates.

- H_0 (connected components) -> point attractor modules (local predictors)
- H_1 (loops/cycles) -> cycle modules (periodic/oscillatory predictors)
- H_2 (voids) -> boundary modules (surface/interface predictors)
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass
class TopologicalFeature:
    """A persistent topological feature extracted from exploration trajectories."""
    dim: int                    # Homology dimension (0, 1, or 2)
    birth: float                # Birth time (scale)
    death: float                # Death time (scale)
    persistence: float          # death - birth
    centroid: Optional[np.ndarray] = None   # Representative point in latent space
    points: Optional[np.ndarray] = None     # Points belonging to this feature
    module_type: str = ""       # Assigned module type


class TopologicalFeatureExtractor:
    """Extracts module-forming features from persistence analysis results.

    Assigns each significant topological feature a module type based
    on its homology dimension and characteristics.
    """

    def __init__(
        self,
        persistence_threshold: float = 0.3,
        min_cluster_size: int = 5,
    ) -> None:
        self.persistence_threshold = persistence_threshold
        self.min_cluster_size = min_cluster_size

    def extract_features(
        self,
        ph_analysis: dict[str, dict],
        point_cloud: Optional[torch.Tensor] = None,
    ) -> list[TopologicalFeature]:
        """Extract TopologicalFeature objects from persistence analysis.

        Args:
            ph_analysis: Output from PersistenceDiagramAnalyzer.analyze().
            point_cloud: Optional [N, D] point cloud for centroid computation.

        Returns:
            List of TopologicalFeature candidates for module formation.
        """
        features = []

        for dim, dim_key in enumerate(["H0", "H1", "H2"]):
            info = ph_analysis.get(dim_key, {})
            dgm = info.get("features", np.empty((0, 2)))

            if len(dgm) == 0:
                continue

            for i in range(len(dgm)):
                birth, death = dgm[i]
                persistence = death - birth

                # Skip low-persistence features
                if persistence < self.persistence_threshold:
                    continue

                module_type = {0: "attractor", 1: "cycle", 2: "boundary"}[dim]

                feat = TopologicalFeature(
                    dim=dim,
                    birth=float(birth),
                    death=float(death),
                    persistence=float(persistence),
                    module_type=module_type,
                )

                # Compute centroid if point cloud available
                if point_cloud is not None and dim == 0:
                    feat.centroid = self._compute_cluster_centroid(
                        point_cloud, birth, death
                    )

                features.append(feat)

        # Sort by persistence (most significant first)
        features.sort(key=lambda f: f.persistence, reverse=True)
        return features

    def _compute_cluster_centroid(
        self,
        point_cloud: torch.Tensor,
        birth_scale: float,
        death_scale: float,
    ) -> np.ndarray:
        """Estimate the centroid of a cluster corresponding to an H_0 feature.

        Uses the midpoint scale to identify which points belong to the cluster.
        """
        points = point_cloud.numpy() if isinstance(point_cloud, torch.Tensor) else point_cloud
        # Use a random subset as centroid proxy (exact cluster assignment
        # would require the full filtration which is expensive)
        mid_scale = (birth_scale + death_scale) / 2.0

        # Rough approach: find dense region at the characteristic scale
        from scipy.spatial.distance import cdist
        N = len(points)
        if N > 200:
            idx = np.random.choice(N, 200, replace=False)
            subset = points[idx]
        else:
            subset = points

        dists = cdist(subset, subset)
        # Count neighbors within mid_scale
        neighbor_counts = (dists < mid_scale).sum(axis=1)
        densest = np.argmax(neighbor_counts)
        nearby = subset[dists[densest] < mid_scale]

        return nearby.mean(axis=0)

    def classify_feature(self, feature: TopologicalFeature) -> str:
        """Classify what kind of predictor module a feature should become.

        Args:
            feature: Topological feature to classify.

        Returns:
            Module type string.
        """
        return feature.module_type
