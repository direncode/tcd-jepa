"""Recursive loop orchestrating the tripartite dynamics.

Systems 1 -> 2 -> 3 -> 1 feedback loop with convergence monitoring.

The loop:
1. System 1 (Encoder) produces representations
2. System 2 (Explorer) explores the energy landscape, produces trajectories
3. System 3 (Crystallizer) analyzes trajectories, creates modules
4. New modules enrich System 1's predictor -> back to step 1

Convergence:
    C(t) = |M(t) - M(t-1)| / M(t) + KL(R(t) || R(t-1)) + |S(t) - S(t-1)|
    System converges when C(t) < epsilon for k consecutive iterations.
"""

from typing import Optional
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

from tcd_jepa.core.system2_explorer import EnergyExplorer
from tcd_jepa.core.system3_crystallizer import ModuleCrystallizer


class ConvergenceMonitor:
    """Tracks convergence of the recursive loop.

    C(t) = module_change + representation_divergence + smoothness_change

    Converged when C(t) < epsilon for k consecutive iterations.
    """

    def __init__(
        self,
        epsilon: float = 0.01,
        patience: int = 5,
        history_size: int = 50,
    ) -> None:
        self.epsilon = epsilon
        self.patience = patience
        self._history: list[dict] = []
        self._convergence_scores: list[float] = []
        self._consecutive_converged = 0
        self._prev_num_modules = 0
        self._prev_repr_histogram: Optional[torch.Tensor] = None
        self._prev_smoothness = 0.0

    def update(
        self,
        num_modules: int,
        representations: torch.Tensor,
        energy_landscape_smoothness: float,
    ) -> dict:
        """Record one iteration and compute convergence metrics.

        Args:
            num_modules: Current active module count M(t).
            representations: Current encoder outputs [B, D] for distribution tracking.
            energy_landscape_smoothness: Energy std as smoothness proxy.

        Returns:
            Dict with convergence metrics and is_converged flag.
        """
        # Module formation rate: |M(t) - M(t-1)| / max(M(t), 1)
        module_change = abs(num_modules - self._prev_num_modules) / max(num_modules, 1)

        # Representation distribution shift (histogram-based KL approximation)
        repr_histogram = self._compute_repr_histogram(representations)
        if self._prev_repr_histogram is not None:
            repr_divergence = self._kl_divergence(repr_histogram, self._prev_repr_histogram)
        else:
            repr_divergence = 1.0

        # Energy landscape smoothness change
        smoothness_change = abs(energy_landscape_smoothness - self._prev_smoothness)

        # Combined convergence score
        convergence_score = module_change + repr_divergence + smoothness_change

        # Track consecutive convergence
        if convergence_score < self.epsilon:
            self._consecutive_converged += 1
        else:
            self._consecutive_converged = 0

        is_converged = self._consecutive_converged >= self.patience

        # Update state
        self._prev_num_modules = num_modules
        self._prev_repr_histogram = repr_histogram
        self._prev_smoothness = energy_landscape_smoothness

        result = {
            "convergence_score": convergence_score,
            "module_change": module_change,
            "repr_divergence": repr_divergence,
            "smoothness_change": smoothness_change,
            "is_converged": is_converged,
            "consecutive_converged": self._consecutive_converged,
        }
        self._convergence_scores.append(convergence_score)
        self._history.append(result)
        return result

    @staticmethod
    def _compute_repr_histogram(
        representations: torch.Tensor,
        num_bins: int = 50,
    ) -> torch.Tensor:
        """Compute a normalized histogram of representation norms."""
        norms = representations.detach().norm(dim=-1).cpu()
        hist = torch.histc(norms, bins=num_bins, min=0.0, max=float(norms.max() + 1e-6))
        return hist / (hist.sum() + 1e-8)

    @staticmethod
    def _kl_divergence(p: torch.Tensor, q: torch.Tensor) -> float:
        """Compute KL(p || q) with smoothing."""
        p = p + 1e-8
        q = q + 1e-8
        p = p / p.sum()
        q = q / q.sum()
        return float(F.kl_div(q.log(), p, reduction="sum").item())

    @property
    def history(self) -> list[dict]:
        return self._history

    @property
    def is_converged(self) -> bool:
        return self._consecutive_converged >= self.patience


class RecursiveLoop:
    """Orchestrates the Systems 1-2-3 feedback loop.

    Each iteration:
    1. System 1 produces latent representations (handled by caller)
    2. System 2 explores energy landscape around those representations
    3. System 3 crystallizes modules from exploration trajectories
    4. New modules are fed back into the predictor

    Stability controls:
    - Dampening: module contributions are scaled by a learnable factor
    - Maximum module count: enforced by registry
    - Pruning: underperforming modules are removed
    """

    def __init__(
        self,
        embed_dim: int,
        explore_every: int = 5,
        crystallize_every: int = 10,
        min_trajectories_for_crystallization: int = 3,
        langevin_steps: int = 50,
        langevin_step_size: float = 0.01,
        langevin_temperature: float = 1.0,
        persistence_threshold: float = 0.1,
        max_modules: int = 20,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.embed_dim = embed_dim
        self.explore_every = explore_every
        self.crystallize_every = crystallize_every
        self.min_trajectories = min_trajectories_for_crystallization
        self.device = device

        self.explorer = EnergyExplorer(
            embed_dim=embed_dim,
            langevin_step_size=langevin_step_size,
            langevin_temperature=langevin_temperature,
            langevin_steps=langevin_steps,
        )

        self.crystallizer = ModuleCrystallizer(
            embed_dim=embed_dim,
            persistence_threshold=persistence_threshold,
            max_modules=max_modules,
            device=device,
        )

        self.convergence = ConvergenceMonitor()
        self._iteration = 0

    def step(
        self,
        z_context: torch.Tensor,
        energy_fn: callable,
        predictor_fn: Optional[callable] = None,
        epoch: int = 0,
    ) -> dict:
        """Execute one iteration of the recursive loop.

        Args:
            z_context: Current encoder output [B, N, D] or [B, D].
            energy_fn: Energy function mapping latent points to scalars.
            predictor_fn: Optional predictor for variance-based detection.
            epoch: Current training epoch.

        Returns:
            Dict with exploration results, crystallization results, convergence.
        """
        self._iteration += 1
        result = {
            "iteration": self._iteration,
            "explored": False,
            "crystallized": False,
        }

        # System 2: Explore (periodic)
        if self._iteration % self.explore_every == 0:
            explore_result = self.explorer.explore(
                z_context, energy_fn, predictor_fn, epoch=epoch
            )
            result["exploration"] = {
                "num_blank": int(explore_result["blank_info"]["is_blank"].sum()),
                "mean_fisher_trace": float(explore_result["fisher_trace"].mean()),
                "trajectory_length": explore_result["trajectory"].shape[0],
            }
            result["explored"] = True

        # System 3: Crystallize (less frequent, needs enough trajectories)
        if (self._iteration % self.crystallize_every == 0
                and self.explorer.num_trajectories >= self.min_trajectories):
            point_cloud = self.explorer.get_point_cloud(last_n=5)
            crystal_result = self.crystallizer.crystallize(point_cloud, epoch)
            result["crystallization"] = {
                "new_modules": crystal_result["new_modules"],
                "pruned_modules": crystal_result["pruned_modules"],
                "total_persistence": crystal_result["total_persistence"],
                "num_active_modules": self.crystallizer.num_modules,
            }
            result["crystallized"] = True

        # Convergence monitoring
        if z_context.dim() == 3:
            z_flat = z_context.reshape(-1, z_context.shape[-1])
        else:
            z_flat = z_context

        smoothness = float(energy_fn(z_flat[:min(64, len(z_flat))]).std())

        conv = self.convergence.update(
            num_modules=self.crystallizer.num_modules,
            representations=z_flat[:min(256, len(z_flat))],
            energy_landscape_smoothness=smoothness,
        )
        result["convergence"] = conv

        return result

    def get_crystallized_modules(self) -> nn.ModuleList:
        """Get all active crystallized modules."""
        return self.crystallizer.get_active_modules()

    @property
    def is_converged(self) -> bool:
        return self.convergence.is_converged

    @property
    def num_modules(self) -> int:
        return self.crystallizer.num_modules
