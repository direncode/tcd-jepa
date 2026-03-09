"""Persistent homology computation on trajectory point clouds.

Computes Vietoris-Rips complexes at multiple scales, tracking:
- H_0: connected components (clusters)
- H_1: loops (cycles)
- H_2: voids (cavities)

Uses giotto-tda as primary backend, falls back to ripser.
"""

from typing import Optional

import numpy as np
import torch


def _get_backend():
    """Detect available persistent homology backend."""
    try:
        from gtda.homology import VietorisRipsPersistence
        return "giotto"
    except ImportError:
        pass
    try:
        from ripser import ripser
        return "ripser"
    except ImportError:
        pass
    return None


class PersistentHomologyComputer:
    """Computes persistent homology on point clouds from exploration trajectories.

    Wraps giotto-tda or ripser to compute Vietoris-Rips persistence diagrams.
    """

    def __init__(
        self,
        max_homology_dim: int = 2,
        max_edge_length: float = float("inf"),
        n_jobs: int = 1,
    ) -> None:
        self.max_homology_dim = max_homology_dim
        self.max_edge_length = max_edge_length
        self.n_jobs = n_jobs
        self._backend = _get_backend()

    @property
    def backend(self) -> Optional[str]:
        return self._backend

    def compute(
        self,
        point_cloud: torch.Tensor,
        max_points: int = 500,
    ) -> dict[str, np.ndarray]:
        """Compute persistent homology of a point cloud.

        Args:
            point_cloud: Points [N, D].
            max_points: Subsample if point cloud is larger (PH is O(N^3)).

        Returns:
            Dict with:
                'diagram': persistence diagram array [K, 3] (birth, death, dim)
                'H0', 'H1', 'H2': filtered diagrams per dimension
        """
        points = point_cloud.detach().cpu().numpy()

        # Subsample for computational feasibility
        if points.shape[0] > max_points:
            indices = np.random.choice(points.shape[0], max_points, replace=False)
            points = points[indices]

        if self._backend == "giotto":
            return self._compute_giotto(points)
        elif self._backend == "ripser":
            return self._compute_ripser(points)
        else:
            return self._compute_fallback(points)

    def _compute_giotto(self, points: np.ndarray) -> dict[str, np.ndarray]:
        """Compute using giotto-tda."""
        from gtda.homology import VietorisRipsPersistence

        vr = VietorisRipsPersistence(
            homology_dimensions=list(range(self.max_homology_dim + 1)),
            max_edge_length=self.max_edge_length,
            n_jobs=self.n_jobs,
        )
        # giotto expects [n_samples, n_points, n_dims]
        diagrams = vr.fit_transform(points[np.newaxis])[0]

        result = {"diagram": diagrams}
        for dim in range(self.max_homology_dim + 1):
            mask = diagrams[:, 2] == dim
            result[f"H{dim}"] = diagrams[mask][:, :2]
        return result

    def _compute_ripser(self, points: np.ndarray) -> dict[str, np.ndarray]:
        """Compute using ripser."""
        from ripser import ripser

        result_ripser = ripser(
            points,
            maxdim=self.max_homology_dim,
            thresh=self.max_edge_length if self.max_edge_length != float("inf") else np.inf,
        )

        # Convert to unified format
        all_features = []
        result = {}
        for dim, dgm in enumerate(result_ripser["dgms"]):
            # Filter infinite deaths
            finite_mask = np.isfinite(dgm[:, 1])
            dgm_finite = dgm[finite_mask]
            result[f"H{dim}"] = dgm_finite

            dims_col = np.full((dgm_finite.shape[0], 1), dim)
            all_features.append(np.hstack([dgm_finite, dims_col]))

        if all_features:
            result["diagram"] = np.vstack(all_features)
        else:
            result["diagram"] = np.empty((0, 3))
        return result

    def _compute_fallback(self, points: np.ndarray) -> dict[str, np.ndarray]:
        """Approximate PH using pairwise distances when no backend is available.

        Computes H_0 via single-linkage clustering (connected components).
        """
        from scipy.spatial.distance import pdist, squareform
        from scipy.cluster.hierarchy import linkage

        dists = pdist(points)
        Z = linkage(dists, method="single")

        # H_0: connected components merge at distances given by linkage
        births = np.zeros(len(Z))
        deaths = Z[:, 2]
        h0 = np.column_stack([births, deaths])

        diagram = np.column_stack([births, deaths, np.zeros(len(births))])

        return {
            "diagram": diagram,
            "H0": h0,
            "H1": np.empty((0, 2)),
            "H2": np.empty((0, 2)),
        }
