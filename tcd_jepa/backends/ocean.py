"""Ocean physics backend — delegates to latent-ocean-core's spectral evolution."""

from typing import Dict, List, Optional
import torch
import numpy as np

from ocean_core import OceanConfig, get_config
from ocean_core.math.shcg import SHCGEngine
from ocean_core.math.drift import EnhancedDriftKernel
from ocean_core.evolution.btut import BTUTEngine
from ocean_core.irdb.adapter import IRDBAdapter


class OceanBackend:
    """PhysicsBackend powered by latent-ocean-core's spectral evolution."""

    def __init__(self, embed_dim: int, config: Optional[OceanConfig] = None):
        self.embed_dim = embed_dim
        self.config = config or get_config()
        self.shcg = SHCGEngine(dimension=self.config.shcg_base_dimension)
        self.irdb = IRDBAdapter(max_entities=self.config.irdb_max_entities)
        self.drift = EnhancedDriftKernel(self.shcg)
        self.btut: Optional[BTUTEngine] = None
        self._step_count = 0

    def ingest(self, records: List[Dict]) -> Dict:
        """Ingest records into the ocean intelligence pipeline."""
        result = self.irdb.ingest(records)
        if self.irdb.count >= 10 and self.btut is None:
            self.btut = BTUTEngine(l_max=self.config.btut_l_max)
            self.btut.initialize_from_arrays(
                self.irdb.get_positions(),
                self.irdb.get_embeddings(),
                ["entity"] * self.irdb.count,
            )
        return result

    def compute_energy(self, z: torch.Tensor) -> torch.Tensor:
        """Compute spectral energy at latent positions."""
        z_np = z.detach().cpu().numpy()
        d = self.shcg.manifold.d
        if z_np.shape[1] > d:
            z_np = z_np[:, :d]
        elif z_np.shape[1] < d:
            z_np = np.pad(z_np, ((0, 0), (0, d - z_np.shape[1])))
        norms = np.linalg.norm(z_np, axis=1, keepdims=True)
        z_np = z_np / np.maximum(norms, 1e-8)
        velocities = np.zeros_like(z_np)
        mf_velocity = self.shcg.drift_kernel.compute_mean_field_velocity(z_np, velocities)
        energy = np.sum(mf_velocity ** 2, axis=1)
        return torch.from_numpy(energy).float().to(z.device)

    def explore_step(
        self, z: torch.Tensor, temperature: float, gradient: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Exploration step via SHCG drift + BTUT spectral evolution."""
        self._step_count += 1
        z_np = z.detach().cpu().numpy()
        d = self.shcg.manifold.d
        if z_np.shape[1] > d:
            z_np = z_np[:, :d]
        elif z_np.shape[1] < d:
            z_np = np.pad(z_np, ((0, 0), (0, d - z_np.shape[1])))
        norms = np.linalg.norm(z_np, axis=1, keepdims=True)
        z_np = z_np / np.maximum(norms, 1e-8)

        velocities = np.zeros_like(z_np)
        mf_velocity = self.shcg.drift_kernel.compute_mean_field_velocity(z_np, velocities)
        sigma = self.shcg.drift_kernel.sigma * temperature
        noise = np.random.randn(*z_np.shape) * sigma
        z_new = z_np + 0.001 * mf_velocity + noise
        z_new = z_new / np.maximum(np.linalg.norm(z_new, axis=1, keepdims=True), 1e-8)

        if self.btut is not None:
            self.btut.step(n_steps=1)

        if z_new.shape[1] < self.embed_dim:
            z_new = np.pad(z_new, ((0, 0), (0, self.embed_dim - z_new.shape[1])))
        elif z_new.shape[1] > self.embed_dim:
            z_new = z_new[:, :self.embed_dim]

        return torch.from_numpy(z_new).float().to(z.device)

    def get_landscape_metrics(self) -> Dict[str, float]:
        state = self.shcg.get_state()
        metrics = {
            "step_count": self._step_count,
            "n_entities": self.irdb.count or state["n_entities"],
            "dimension": state["dimension"],
            "sigma": state["sigma"],
        }
        if self.btut is not None:
            btut_state = self.btut.get_state()
            metrics["btut_step"] = btut_state.get("step", 0)
        return metrics
