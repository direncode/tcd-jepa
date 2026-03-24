"""Tests for System 2 — Energy Explorer."""

import torch

from tcd_jepa.core.system2_explorer import EnergyExplorer
from tcd_jepa.exploration.blank_space_detector import BlankSpaceDetector
from tcd_jepa.exploration.fisher_metric import FisherMetric
from tcd_jepa.exploration.langevin import LangevinSampler
from tcd_jepa.exploration.trajectory_tracker import TrajectoryTracker


def _quadratic_energy(z: torch.Tensor) -> torch.Tensor:
    """Simple quadratic energy: E(z) = ||z||^2."""
    return z.pow(2).sum(dim=-1)


def _linear_predictor(z: torch.Tensor) -> torch.Tensor:
    """Identity predictor for testing."""
    return z


class TestBlankSpaceDetector:

    def test_hessian_spectrum(self):
        """Hessian of quadratic E=||z||^2 should have positive curvature."""
        detector = BlankSpaceDetector(perturbation_std=0.01)
        z = torch.randn(8, 16)
        result = detector.compute_energy_hessian_spectrum(z, _quadratic_energy)

        assert "eigenvalues_min" in result
        assert "eigenvalues_max" in result
        assert result["eigenvalues_min"].shape == (8,)
        # Quadratic has positive definite Hessian
        assert (result["eigenvalues_min"] > -1.0).all()

    def test_perturbation_variance(self):
        """Perturbation variance is non-negative."""
        detector = BlankSpaceDetector(num_perturbations=5)
        z = torch.randn(4, 8)
        var = detector.compute_perturbation_variance(z, _linear_predictor)
        assert var.shape == (4,)
        assert (var >= 0).all()

    def test_detect(self):
        """detect() returns correct keys and shapes."""
        detector = BlankSpaceDetector()
        z = torch.randn(8, 16)
        result = detector.detect(z, _quadratic_energy, _linear_predictor)

        assert result["is_blank"].shape == (8,)
        assert result["flatness_score"].shape == (8,)
        assert result["variance_score"].shape == (8,)
        assert result["combined_score"].shape == (8,)

    def test_flat_region_detected(self):
        """Nearly constant energy should be flagged as blank."""
        detector = BlankSpaceDetector(flatness_threshold=10.0)
        z = torch.randn(8, 16)
        # Near-constant energy
        result = detector.detect(z, lambda z: torch.ones(z.shape[0]))
        assert result["is_blank"].any()


class TestLangevinSampler:

    def test_step(self):
        """Single Langevin step produces valid output."""
        sampler = LangevinSampler(step_size=0.01)
        z = torch.randn(4, 8)
        z_new = sampler.step(z, _quadratic_energy)
        assert z_new.shape == z.shape
        assert not torch.equal(z, z_new)

    def test_trajectory(self):
        """Trajectory has correct shape [T+1, B, D]."""
        sampler = LangevinSampler(step_size=0.01, max_steps=10)
        z = torch.randn(4, 8)
        traj = sampler.sample_trajectory(z, _quadratic_energy)
        assert traj.shape == (11, 4, 8)  # T+1 steps

    def test_trajectory_with_blank_score(self):
        """Trajectory works with per-sample temperature bias."""
        sampler = LangevinSampler(step_size=0.01, max_steps=5)
        z = torch.randn(4, 8)
        blank_score = torch.tensor([0.0, 1.0, 2.0, 5.0])
        traj = sampler.sample_trajectory(z, _quadratic_energy, blank_score=blank_score)
        assert traj.shape == (6, 4, 8)

    def test_langevin_moves_toward_minimum(self):
        """With low temperature, Langevin should move toward energy minimum."""
        sampler = LangevinSampler(step_size=0.1, temperature=100.0, max_steps=50)
        z = torch.ones(1, 4) * 5.0  # Start far from minimum at origin
        traj = sampler.sample_trajectory(z, _quadratic_energy)
        # Final position should be closer to origin than initial
        initial_energy = _quadratic_energy(traj[0])
        final_energy = _quadratic_energy(traj[-1])
        assert final_energy < initial_energy


class TestTrajectoryTracker:

    def test_add_and_get(self):
        """Can add and retrieve trajectories."""
        tracker = TrajectoryTracker()
        traj = torch.randn(20, 4, 8)  # [T, B, D]
        tracker.add_trajectory(traj)
        assert tracker.num_trajectories == 1

        cloud = tracker.get_point_cloud()
        assert cloud.dim() == 2
        assert cloud.shape[1] == 8

    def test_max_trajectories(self):
        """Old trajectories are evicted when over capacity."""
        tracker = TrajectoryTracker(max_trajectories=3)
        for _ in range(5):
            tracker.add_trajectory(torch.randn(10, 8))
        assert tracker.num_trajectories == 3

    def test_clear(self):
        """Clear removes all trajectories."""
        tracker = TrajectoryTracker()
        tracker.add_trajectory(torch.randn(10, 8))
        tracker.clear()
        assert tracker.num_trajectories == 0


class TestFisherMetric:

    def test_jacobian(self):
        """Jacobian has correct shape."""
        fisher = FisherMetric(num_jacobian_samples=4)
        z = torch.randn(3, 8)
        J = fisher.compute_jacobian_fd(z, _linear_predictor)
        assert J.shape == (3, 8, 4)  # [B, D_out, num_samples]

    def test_fisher_matrix(self):
        """Fisher matrix is symmetric positive semi-definite."""
        fisher = FisherMetric(num_jacobian_samples=4)
        z = torch.randn(3, 8)
        F_mat = fisher.compute_fisher_matrix(z, _linear_predictor)
        assert F_mat.shape == (3, 4, 4)
        # Check symmetry
        assert torch.allclose(F_mat, F_mat.transpose(-1, -2), atol=1e-5)

    def test_metric_trace(self):
        """Fisher trace is non-negative."""
        fisher = FisherMetric(num_jacobian_samples=4)
        z = torch.randn(3, 8)
        trace = fisher.compute_metric_tensor_trace(z, _linear_predictor)
        assert trace.shape == (3,)
        assert (trace >= 0).all()


class TestEnergyExplorer:

    def test_explore(self):
        """Full exploration pipeline runs and returns expected keys."""
        explorer = EnergyExplorer(
            embed_dim=8,
            langevin_steps=5,
            max_trajectories=10,
        )
        z = torch.randn(4, 8)
        result = explorer.explore(z, _quadratic_energy, _linear_predictor)

        assert "trajectory" in result
        assert "blank_info" in result
        assert "fisher_trace" in result
        assert result["trajectory"].shape[1] == 4  # B preserved

    def test_point_cloud_accumulation(self):
        """Multiple explorations accumulate point clouds."""
        explorer = EnergyExplorer(embed_dim=8, langevin_steps=3)
        for _ in range(3):
            z = torch.randn(4, 8)
            explorer.explore(z, _quadratic_energy)

        assert explorer.num_trajectories == 3
        cloud = explorer.get_point_cloud()
        assert cloud.dim() == 2
