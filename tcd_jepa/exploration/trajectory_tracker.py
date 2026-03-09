"""Trajectory tracking for System 2 exploration.

Stores sequences of latent-space positions during Langevin exploration
for subsequent topological analysis by System 3.
"""

from typing import Optional

import torch


class TrajectoryTracker:
    """Records and manages exploration trajectories through latent space.

    Maintains a buffer of trajectory point clouds that System 3
    uses for persistent homology computation.
    """

    def __init__(
        self,
        max_trajectories: int = 100,
        max_points_per_trajectory: int = 1000,
    ) -> None:
        self.max_trajectories = max_trajectories
        self.max_points_per_trajectory = max_points_per_trajectory
        self._trajectories: list[torch.Tensor] = []
        self._energies: list[torch.Tensor] = []
        self._metadata: list[dict] = []

    def add_trajectory(
        self,
        z_trajectory: torch.Tensor,
        energies: Optional[torch.Tensor] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """Store a new exploration trajectory.

        Args:
            z_trajectory: Trajectory points [T, B, D] or [T, D].
            energies: Optional energy values at each point [T, B] or [T].
            metadata: Optional dict with extra info (epoch, blank_scores, etc.).
        """
        # Flatten batch dimension if present: [T, B, D] -> [T*B, D]
        if z_trajectory.dim() == 3:
            T, B, D = z_trajectory.shape
            z_flat = z_trajectory.reshape(T * B, D)
        else:
            z_flat = z_trajectory

        # Subsample if too many points
        if z_flat.shape[0] > self.max_points_per_trajectory:
            indices = torch.linspace(
                0, z_flat.shape[0] - 1, self.max_points_per_trajectory
            ).long()
            z_flat = z_flat[indices]

        self._trajectories.append(z_flat.detach().cpu())

        if energies is not None:
            self._energies.append(energies.detach().cpu().flatten())
        else:
            self._energies.append(None)

        self._metadata.append(metadata or {})

        # Evict oldest if over capacity
        if len(self._trajectories) > self.max_trajectories:
            self._trajectories.pop(0)
            self._energies.pop(0)
            self._metadata.pop(0)

    def get_point_cloud(
        self,
        last_n: Optional[int] = None,
    ) -> torch.Tensor:
        """Get combined point cloud from recent trajectories.

        Args:
            last_n: Use only the last N trajectories. None = all.

        Returns:
            Point cloud [N_total, D].
        """
        if not self._trajectories:
            raise ValueError("No trajectories recorded yet")

        trajectories = self._trajectories
        if last_n is not None:
            trajectories = trajectories[-last_n:]

        return torch.cat(trajectories, dim=0)

    def get_trajectory(self, index: int) -> torch.Tensor:
        """Get a specific trajectory by index."""
        return self._trajectories[index]

    def get_energies(self, index: int) -> Optional[torch.Tensor]:
        """Get energy values for a specific trajectory."""
        return self._energies[index]

    @property
    def num_trajectories(self) -> int:
        return len(self._trajectories)

    @property
    def total_points(self) -> int:
        return sum(t.shape[0] for t in self._trajectories)

    def clear(self) -> None:
        """Remove all stored trajectories."""
        self._trajectories.clear()
        self._energies.clear()
        self._metadata.clear()
