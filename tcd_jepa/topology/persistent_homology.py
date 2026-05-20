"""Persistent homology computation on trajectory point clouds.

Computes Vietoris-Rips complexes at multiple scales, tracking:
- H_0: connected components (clusters)
- H_1: loops (cycles)
- H_2: voids (cavities)

Backend selection (M7 of the SPU custom PH kernel R&D track):
- OCEAN_PH_BACKEND env var = "ocean" -> route to the OCEAN-built PH kernel
  (scripts/spu/ocean_ph/run_ocean_ph.py::ocean_ph_diagram). The OCEAN
  kernel auto-routes to its own CUDA binary when present
  (OCEAN_PH_KERNEL_BACKEND=auto) or to the pure-Python production path
  (M1 H0 + M3 H1 + M5 H2) otherwise.
- OCEAN_PH_BACKEND env var = "giotto" or unset (default) -> existing
  giotto-tda / ripser fallback chain. Preserves operator behaviour for
  any caller that has not opted in.
- The plan-of-plans says: keep the default at the existing third-party
  backend until the pod run validates OCEAN; the user flips the default
  after that gate fires green.
"""

import logging
import os
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger("tcd_jepa")

OCEAN_PH_BACKEND_ENV = "OCEAN_PH_BACKEND"


def _get_backend():
    """Detect available persistent homology backend.

    Honors OCEAN_PH_BACKEND env var: 'ocean' (route to OCEAN PH kernel),
    'giotto' (force giotto-tda), 'ripser' (force ripser), or unset
    (default chain: giotto -> ripser -> scipy fallback).
    """
    env_choice = os.environ.get(OCEAN_PH_BACKEND_ENV, "").lower()

    if env_choice == "ocean":
        # The OCEAN kernel is dep-free at the wrapper level; it owns its
        # own backend selection (auto -> CUDA ph_bench when built, else
        # pure-Python production path). Returning "ocean" here delegates.
        return "ocean"

    if env_choice == "giotto":
        try:
            from gtda.homology import VietorisRipsPersistence  # noqa: F401
            return "giotto"
        except ImportError:
            logger.warning(
                "OCEAN_PH_BACKEND=giotto but giotto-tda not installed; "
                "falling through to the default chain."
            )

    if env_choice == "ripser":
        try:
            from ripser import ripser  # noqa: F401
            return "ripser"
        except ImportError:
            logger.warning(
                "OCEAN_PH_BACKEND=ripser but ripser not installed; "
                "falling through to the default chain."
            )

    try:
        from gtda.homology import VietorisRipsPersistence  # noqa: F401
        return "giotto"
    except ImportError:
        pass
    try:
        from ripser import ripser  # noqa: F401
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

        # Subsample for computational feasibility (seeded for reproducibility)
        if points.shape[0] > max_points:
            rng = np.random.RandomState(42)
            indices = rng.choice(points.shape[0], max_points, replace=False)
            points = points[indices]

        if self._backend == "ocean":
            return self._compute_ocean(points)
        if self._backend == "giotto":
            return self._compute_giotto(points)
        elif self._backend == "ripser":
            return self._compute_ripser(points)
        else:
            logger.warning(
                "No PH backend (giotto-tda/ripser) installed — using scipy fallback "
                "(only H0 via single-linkage). Install giotto-tda or ripser for full PH."
            )
            return self._compute_fallback(points)

    def _compute_ocean(self, points: np.ndarray) -> dict[str, np.ndarray]:
        """Compute via the OCEAN PH kernel.

        The OCEAN wrapper (scripts/spu/ocean_ph/run_ocean_ph.py) returns a
        unified [K, 3] diagram of (birth, death, dim) with finite deaths
        only and diagonal pairs already stripped. We split it back into
        per-dimension arrays for the operator's existing API contract.
        """
        # Lazy import: the operator does not depend on the OCEAN package
        # unless OCEAN_PH_BACKEND=ocean is set.
        from scripts.spu.ocean_ph import ocean_ph_diagram

        diagram = ocean_ph_diagram(points, max_dim=self.max_homology_dim)
        result = {"diagram": diagram}
        for dim in range(self.max_homology_dim + 1):
            mask = diagram[:, 2] == dim
            result[f"H{dim}"] = diagram[mask][:, :2]
        return result

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
        from scipy.cluster.hierarchy import linkage
        from scipy.spatial.distance import pdist

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
