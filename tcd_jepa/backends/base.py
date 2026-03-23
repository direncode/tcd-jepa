"""Physics backend protocol for System 2 exploration."""

from typing import Dict, Optional, Protocol, runtime_checkable
import torch


@runtime_checkable
class PhysicsBackend(Protocol):
    """Abstract interface for energy landscape computation.

    System 2's explorer delegates physics to a backend that
    computes energy, performs exploration steps, and reports metrics.
    """

    def compute_energy(self, z: torch.Tensor) -> torch.Tensor:
        """Compute energy at latent positions.

        Args:
            z: [B, D] latent positions
        Returns:
            [B] per-sample energy values
        """
        ...

    def explore_step(
        self, z: torch.Tensor, temperature: float, gradient: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Perform one exploration step from current positions.

        Args:
            z: [B, D] current positions
            temperature: Exploration temperature (lower = more focused)
            gradient: Optional pre-computed energy gradient [B, D]
        Returns:
            [B, D] updated positions
        """
        ...

    def get_landscape_metrics(self) -> Dict[str, float]:
        """Return current landscape statistics."""
        ...
