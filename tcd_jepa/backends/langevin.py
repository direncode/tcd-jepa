"""Default Langevin dynamics backend — wraps existing exploration code."""

from typing import Dict, Optional
import torch
from tcd_jepa.exploration.langevin import LangevinSampler


class LangevinBackend:
    """PhysicsBackend wrapping TCD-JEPA's built-in Langevin explorer."""

    def __init__(
        self,
        embed_dim: int,
        step_size: float = 0.01,
        temperature: float = 1.0,
        grad_clip: float = 1.0,
    ):
        self.embed_dim = embed_dim
        self.sampler = LangevinSampler(
            step_size=step_size,
            temperature=temperature,
            grad_clip=grad_clip,
        )
        self._step_count = 0
        self._last_energy = None
        self._energy_fn = None

    def set_energy_fn(self, energy_fn):
        """Set the energy function (typically JEPA prediction error)."""
        self._energy_fn = energy_fn

    def compute_energy(self, z: torch.Tensor) -> torch.Tensor:
        if self._energy_fn is None:
            return torch.norm(z, dim=-1)
        return self._energy_fn(z)

    def explore_step(
        self, z: torch.Tensor, temperature: float, gradient: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        self._step_count += 1
        if self._energy_fn is not None:
            z_new = self.sampler.step(z, self._energy_fn)
        else:
            noise = torch.randn_like(z) * (temperature * self.sampler.step_size) ** 0.5
            z_new = z + noise
        with torch.no_grad():
            self._last_energy = self.compute_energy(z_new).mean().item()
        return z_new

    def get_landscape_metrics(self) -> Dict[str, float]:
        return {
            "step_count": self._step_count,
            "mean_energy": self._last_energy or 0.0,
            "temperature": self.sampler.temperature,
            "step_size": self.sampler.step_size,
        }
