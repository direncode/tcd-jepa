"""Persistence diagram analysis and thresholding.

Identifies topological features with high persistence (long-lived)
that become candidates for module crystallization by System 3.
"""

from typing import Optional

import numpy as np


class PersistenceDiagramAnalyzer:
    """Analyzes persistence diagrams to identify stable topological features.

    Features with persistence > threshold are candidates for module formation.
    """

    def __init__(
        self,
        persistence_threshold: float = 0.3,
        relative_threshold: bool = True,
        min_features: int = 0,
        max_features: int = 5,
    ) -> None:
        self.persistence_threshold = persistence_threshold
        self.relative_threshold = relative_threshold
        self.min_features = min_features
        self.max_features = max_features

    def compute_persistence(self, diagram: np.ndarray) -> np.ndarray:
        """Compute persistence (death - birth) for each feature.

        Args:
            diagram: Persistence diagram [K, 2] with (birth, death) pairs.

        Returns:
            Persistence values [K].
        """
        if len(diagram) == 0:
            return np.array([])
        return diagram[:, 1] - diagram[:, 0]

    def filter_by_persistence(
        self,
        diagram: np.ndarray,
        threshold: Optional[float] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Filter features by persistence threshold.

        Args:
            diagram: [K, 2] persistence diagram.
            threshold: Override default threshold.

        Returns:
            (filtered_diagram, indices) — surviving features and their original indices.
        """
        if len(diagram) == 0:
            return diagram, np.array([], dtype=int)

        persistence = self.compute_persistence(diagram)
        thresh = threshold or self.persistence_threshold

        if self.relative_threshold:
            # Threshold relative to max persistence
            max_pers = persistence.max() if len(persistence) > 0 else 1.0
            thresh = thresh * max_pers

        mask = persistence >= thresh
        indices = np.where(mask)[0]

        # Enforce min/max feature counts
        if len(indices) < self.min_features and len(persistence) > 0:
            # Take top-k by persistence
            top_k = min(self.min_features, len(persistence))
            indices = np.argsort(persistence)[-top_k:]

        if len(indices) > self.max_features:
            # Keep only top-k
            pers_at_indices = persistence[indices]
            top_k_local = np.argsort(pers_at_indices)[-self.max_features:]
            indices = indices[top_k_local]

        return diagram[indices], indices

    def analyze(
        self,
        ph_result: dict[str, np.ndarray],
    ) -> dict[str, dict]:
        """Full analysis of a persistence homology result.

        Args:
            ph_result: Output from PersistentHomologyComputer.compute().

        Returns:
            Dict per dimension with filtered features and statistics:
            {
                'H0': {'features': [...], 'persistence': [...], 'count': int},
                'H1': {...},
                'H2': {...},
            }
        """
        result = {}
        for dim_key in ["H0", "H1", "H2"]:
            dgm = ph_result.get(dim_key, np.empty((0, 2)))
            if len(dgm) == 0:
                result[dim_key] = {
                    "features": np.empty((0, 2)),
                    "persistence": np.array([]),
                    "count": 0,
                    "mean_persistence": 0.0,
                    "max_persistence": 0.0,
                }
                continue

            filtered, indices = self.filter_by_persistence(dgm)
            persistence = self.compute_persistence(filtered)

            result[dim_key] = {
                "features": filtered,
                "persistence": persistence,
                "count": len(filtered),
                "mean_persistence": float(persistence.mean()) if len(persistence) > 0 else 0.0,
                "max_persistence": float(persistence.max()) if len(persistence) > 0 else 0.0,
            }

        return result

    def total_persistence(self, ph_result: dict[str, np.ndarray]) -> float:
        """Compute total persistence across all dimensions (landscape summary statistic)."""
        total = 0.0
        for dim_key in ["H0", "H1", "H2"]:
            dgm = ph_result.get(dim_key, np.empty((0, 2)))
            if len(dgm) > 0:
                total += float(self.compute_persistence(dgm).sum())
        return total
