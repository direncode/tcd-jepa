"""System 2 — Energy Explorer.

Orchestrates blank space detection, Langevin exploration, and trajectory
recording over JEPA's latent energy surface.
"""

from typing import Optional

import torch

from tcd_jepa.exploration.blank_space_detector import BlankSpaceDetector
from tcd_jepa.exploration.fisher_metric import FisherMetric
from tcd_jepa.exploration.langevin import LangevinSampler
from tcd_jepa.exploration.trajectory_tracker import TrajectoryTracker


class EnergyExplorer:
    """System 2: explores JEPA's energy landscape to find blank spaces.

    Pipeline per iteration:
    1. Sample latent points from encoder output
    2. Detect blank spaces (Hessian + variance)
    3. Run Langevin dynamics biased toward blank regions
    4. Record trajectories for System 3
    """

    def __init__(
        self,
        embed_dim: int,
        langevin_step_size: float = 0.01,
        langevin_temperature: float = 1.0,
        langevin_steps: int = 50,
        flatness_threshold: float = 0.1,
        variance_threshold: float = 1.0,
        perturbation_std: float = 0.01,
        max_trajectories: int = 100,
    ) -> None:
        self.embed_dim = embed_dim

        self.blank_detector = BlankSpaceDetector(
            flatness_threshold=flatness_threshold,
            variance_threshold=variance_threshold,
            perturbation_std=perturbation_std,
        )

        self.langevin = LangevinSampler(
            step_size=langevin_step_size,
            temperature=langevin_temperature,
            max_steps=langevin_steps,
        )

        self.fisher = FisherMetric()

        self.trajectory_tracker = TrajectoryTracker(
            max_trajectories=max_trajectories,
        )

    def explore(
        self,
        z_context: torch.Tensor,
        energy_fn: callable,
        predictor_fn: Optional[callable] = None,
        num_langevin_steps: Optional[int] = None,
        epoch: Optional[int] = None,
    ) -> dict[str, torch.Tensor]:
        """Run one exploration cycle.

        Args:
            z_context: Encoded latent representations [B, N, D] or [B, D].
            energy_fn: Maps [B, D] -> [B] energy.
            predictor_fn: Optional predictor for variance detection.
            num_langevin_steps: Override default step count.
            epoch: Current epoch for metadata.

        Returns:
            Dict with exploration results:
                'trajectory': [T, B_flat, D]
                'blank_info': blank space detection results
                'fisher_trace': Fisher metric trace at starting points
        """
        # Flatten spatial dims if needed: [B, N, D] -> [B*N, D]
        if z_context.dim() == 3:
            B, N, D = z_context.shape
            z_flat = z_context.reshape(B * N, D)
        else:
            z_flat = z_context
            D = z_flat.shape[-1]

        # Subsample for efficiency (explore a subset of points)
        max_explore = min(z_flat.shape[0], 256)
        if z_flat.shape[0] > max_explore:
            indices = torch.randperm(z_flat.shape[0], device=z_flat.device)[:max_explore]
            z_explore = z_flat[indices]
        else:
            z_explore = z_flat

        # Step 1: Detect blank spaces
        blank_info = self.blank_detector.detect(
            z_explore, energy_fn, predictor_fn
        )

        # Step 2: Run Langevin dynamics biased toward blank regions
        trajectory = self.langevin.sample_trajectory(
            z_explore,
            energy_fn,
            num_steps=num_langevin_steps,
            blank_score=blank_info["combined_score"],
        )

        # Step 3: Compute Fisher metric at starting points
        if predictor_fn is not None:
            fisher_trace = self.fisher.compute_metric_tensor_trace(
                z_explore, predictor_fn
            )
        else:
            fisher_trace = torch.zeros(z_explore.shape[0], device=z_explore.device)

        # Step 4: Record trajectory
        self.trajectory_tracker.add_trajectory(
            trajectory,
            metadata={"epoch": epoch},
        )

        return {
            "trajectory": trajectory,
            "blank_info": blank_info,
            "fisher_trace": fisher_trace,
            "z_explored": z_explore,
        }

    def get_point_cloud(self, last_n: Optional[int] = None) -> torch.Tensor:
        """Get accumulated point cloud for System 3."""
        return self.trajectory_tracker.get_point_cloud(last_n=last_n)

    @property
    def num_trajectories(self) -> int:
        return self.trajectory_tracker.num_trajectories
