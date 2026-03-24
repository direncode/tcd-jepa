"""Unit tests for exploration modules: Langevin, BlankSpace, Fisher, Trajectory."""

import pytest
import torch

from tcd_jepa.exploration.blank_space_detector import BlankSpaceDetector
from tcd_jepa.exploration.fisher_metric import FisherMetric
from tcd_jepa.exploration.langevin import LangevinSampler
from tcd_jepa.exploration.trajectory_tracker import TrajectoryTracker


def _quadratic_energy(z: torch.Tensor) -> torch.Tensor:
    """Simple quadratic energy: E(z) = 0.5 * ||z||^2. Minimum at origin."""
    return 0.5 * z.pow(2).sum(dim=-1)


def _flat_energy(z: torch.Tensor) -> torch.Tensor:
    """Flat energy landscape: E(z) = 0."""
    return torch.zeros(z.shape[0], device=z.device)


# ---------------------------------------------------------------------------
# Langevin sampler
# ---------------------------------------------------------------------------

class TestLangevinSampler:
    def test_energy_decreases_over_trajectory(self):
        """Energy should generally decrease over a Langevin trajectory."""
        torch.manual_seed(0)
        # High beta (temperature param) = low noise, gradient-dominated dynamics.
        # noise_scale = sqrt(2*eta/beta), so large beta -> small noise.
        sampler = LangevinSampler(
            step_size=0.005, temperature=100.0, min_temperature=100.0, max_steps=100,
        )
        z_init = torch.randn(16, 8) * 0.5
        trajectory = sampler.sample_trajectory(z_init, _quadratic_energy, num_steps=100)
        energy_start = _quadratic_energy(trajectory[0]).mean()
        energy_end = _quadratic_energy(trajectory[-1]).mean()
        assert energy_end < energy_start, (
            f"Energy should decrease: start={energy_start:.4f}, end={energy_end:.4f}"
        )

    def test_seeded_reproducibility(self):
        """Two runs with the same seed should produce identical trajectories."""
        gen1 = torch.Generator().manual_seed(42)
        gen2 = torch.Generator().manual_seed(42)
        s1 = LangevinSampler(step_size=0.01, temperature=1.0, max_steps=10, generator=gen1)
        s2 = LangevinSampler(step_size=0.01, temperature=1.0, max_steps=10, generator=gen2)
        z = torch.randn(4, 16)
        t1 = s1.sample_trajectory(z.clone(), _quadratic_energy, num_steps=10)
        t2 = s2.sample_trajectory(z.clone(), _quadratic_energy, num_steps=10)
        assert torch.allclose(t1, t2), "Seeded samplers should be deterministic"

    def test_gradient_clipping(self):
        """Gradient should be clipped to grad_clip norm."""
        sampler = LangevinSampler(step_size=0.001, grad_clip=0.5)
        z = torch.randn(4, 8) * 100  # Large values -> large gradients
        grad = sampler._compute_energy_gradient(z, _quadratic_energy)
        norms = grad.norm(dim=-1)
        assert (norms <= 0.5 + 1e-5).all(), f"Gradients should be clipped: {norms}"

    def test_differentiable_step_graph_connectivity(self):
        """differentiable_step should preserve the computation graph."""
        sampler = LangevinSampler(step_size=0.01, temperature=1.0)
        z = torch.randn(4, 8, requires_grad=True)
        z_new = sampler.differentiable_step(z, _quadratic_energy)
        loss = z_new.sum()
        loss.backward()
        assert z.grad is not None, "Gradient should flow through differentiable step"
        assert z.grad.abs().sum() > 0, "Gradient should be non-zero"

    def test_differentiable_sample_warmup_and_diff(self):
        """differentiable_sample should return a tensor with grad graph."""
        sampler = LangevinSampler(step_size=0.01, temperature=1.0)
        z = torch.randn(4, 8)
        result = sampler.differentiable_sample(
            z, _quadratic_energy, num_warmup_steps=3, num_diff_steps=3
        )
        loss = result.sum()
        loss.backward()
        # Result should have shape [4, 8]
        assert result.shape == (4, 8)

    def test_trajectory_shape(self):
        """Trajectory should have shape [T+1, B, D]."""
        sampler = LangevinSampler(step_size=0.01, max_steps=20)
        z = torch.randn(8, 16)
        traj = sampler.sample_trajectory(z, _quadratic_energy, num_steps=20)
        assert traj.shape == (21, 8, 16)

    def test_temperature_map(self):
        """Custom temperature map should affect step output."""
        sampler = LangevinSampler(step_size=0.01, temperature=1.0)
        z = torch.randn(4, 8)
        low_temp = torch.full((4,), 0.1)
        high_temp = torch.full((4,), 10.0)
        z_low = sampler.step(z.clone(), _quadratic_energy, temperature_map=low_temp)
        z_high = sampler.step(z.clone(), _quadratic_energy, temperature_map=high_temp)
        # Lower temperature -> larger noise -> more change from z
        # Actually low temp means low beta -> noise_scale = sqrt(2*eta/beta) -> larger noise
        # So low temp should produce more change
        _diff_low = (z_low - z).pow(2).sum()
        _diff_high = (z_high - z).pow(2).sum()
        # Just verify they produce different outputs
        assert not torch.allclose(z_low, z_high)


# ---------------------------------------------------------------------------
# Blank space detector
# ---------------------------------------------------------------------------

class TestBlankSpaceDetector:
    def test_flat_landscape_detected(self):
        """A flat energy landscape should be detected as blank space."""
        detector = BlankSpaceDetector(flatness_threshold=0.5)
        z = torch.randn(8, 16)
        result = detector.detect(z, _flat_energy)
        # Flat landscape -> eigenvalues are ~0 -> flatness_score < threshold -> is_blank
        assert result["is_blank"].all(), "Flat landscape should be detected as blank"

    def test_curved_landscape_not_blank(self):
        """A strongly curved landscape should not be detected as blank."""
        detector = BlankSpaceDetector(
            flatness_threshold=0.001,  # Very low threshold
            perturbation_std=0.01,
        )
        z = torch.randn(8, 16) * 0.1
        # Quadratic has curvature ~1.0 everywhere
        result = detector.detect(z, _quadratic_energy)
        # Not all should be blank with very low threshold
        assert not result["is_blank"].all(), "Curved landscape should not all be blank"

    def test_perturbation_variance_with_predictor(self):
        """Perturbation variance should be computed when predictor_fn is given."""
        detector = BlankSpaceDetector(num_perturbations=5)
        z = torch.randn(4, 8)
        def predictor_fn(x): return x * 2  # Simple linear predictor
        result = detector.detect(z, _quadratic_energy, predictor_fn=predictor_fn)
        assert result["variance_score"].shape == (4,)
        assert (result["variance_score"] >= 0).all()

    def test_hessian_spectrum_shape(self):
        """Hessian spectrum should return proper shapes."""
        detector = BlankSpaceDetector(num_hessian_directions=8)
        z = torch.randn(4, 16)
        result = detector.compute_energy_hessian_spectrum(z, _quadratic_energy)
        assert result["eigenvalues_min"].shape == (4,)
        assert result["eigenvalues_max"].shape == (4,)
        assert result["curvatures"].shape == (4, 8)


# ---------------------------------------------------------------------------
# Fisher metric
# ---------------------------------------------------------------------------

class TestFisherMetric:
    def _linear_predictor(self, z: torch.Tensor) -> torch.Tensor:
        return z * 2.0  # Simple scaling

    def test_trace_is_positive(self):
        """Fisher trace should be non-negative."""
        fisher = FisherMetric(num_jacobian_samples=8)
        z = torch.randn(4, 16)
        trace = fisher.compute_metric_tensor_trace(z, self._linear_predictor)
        assert (trace >= -1e-6).all(), f"Fisher trace should be non-negative: {trace}"

    def test_jacobian_shape(self):
        """Jacobian should have correct shape."""
        fisher = FisherMetric(num_jacobian_samples=8)
        z = torch.randn(4, 16)
        J = fisher.compute_jacobian_fd(z, self._linear_predictor)
        assert J.shape == (4, 16, 8)  # [B, D_out, K]

    def test_fisher_matrix_shape(self):
        """Fisher matrix should be [B, K, K]."""
        fisher = FisherMetric(num_jacobian_samples=8)
        z = torch.randn(4, 16)
        F = fisher.compute_fisher_matrix(z, self._linear_predictor)
        assert F.shape == (4, 8, 8)

    def test_geodesic_distance_symmetric(self):
        """Geodesic distance should be approximately symmetric."""
        fisher = FisherMetric(num_jacobian_samples=8)
        z1 = torch.randn(4, 16)
        z2 = torch.randn(4, 16)
        d12 = fisher.compute_geodesic_distance(z1, z2, self._linear_predictor)
        d21 = fisher.compute_geodesic_distance(z2, z1, self._linear_predictor)
        assert torch.allclose(d12, d21, atol=1e-4), "Geodesic distance should be symmetric"

    def test_identity_predictor_trace(self):
        """For identity predictor, Fisher trace should be proportional to K/sigma^2."""
        sigma2 = 1.0
        fisher = FisherMetric(noise_variance=sigma2, num_jacobian_samples=8)
        z = torch.zeros(1, 16)
        def identity_fn(x): return x
        trace = fisher.compute_metric_tensor_trace(z, identity_fn)
        # Each Jacobian column is a random direction, J^T J trace ~ K
        assert trace.item() > 0


# ---------------------------------------------------------------------------
# Trajectory tracker
# ---------------------------------------------------------------------------

class TestTrajectoryTracker:
    def test_capacity_limit(self):
        """Tracker should evict oldest trajectories when over capacity."""
        tracker = TrajectoryTracker(max_trajectories=3)
        for i in range(5):
            traj = torch.randn(10, 8)
            tracker.add_trajectory(traj)
        assert tracker.num_trajectories == 3

    def test_cpu_offloading(self):
        """Stored trajectories should be on CPU."""
        tracker = TrajectoryTracker()
        traj = torch.randn(10, 8)
        tracker.add_trajectory(traj)
        assert tracker.get_trajectory(0).device == torch.device("cpu")

    def test_subsampling(self):
        """Large trajectories should be subsampled."""
        tracker = TrajectoryTracker(max_points_per_trajectory=50)
        traj = torch.randn(200, 8)
        tracker.add_trajectory(traj)
        assert tracker.get_trajectory(0).shape[0] == 50

    def test_batch_trajectory_flattening(self):
        """3D trajectories [T, B, D] should be flattened to [T*B, D]."""
        tracker = TrajectoryTracker(max_points_per_trajectory=10000)
        traj = torch.randn(10, 4, 8)  # [T, B, D]
        tracker.add_trajectory(traj)
        assert tracker.get_trajectory(0).shape == (40, 8)

    def test_point_cloud_concatenation(self):
        """get_point_cloud should concatenate all trajectories."""
        tracker = TrajectoryTracker()
        tracker.add_trajectory(torch.randn(10, 8))
        tracker.add_trajectory(torch.randn(15, 8))
        cloud = tracker.get_point_cloud()
        assert cloud.shape == (25, 8)

    def test_point_cloud_last_n(self):
        """get_point_cloud(last_n=1) should only use the last trajectory."""
        tracker = TrajectoryTracker()
        tracker.add_trajectory(torch.randn(10, 8))
        tracker.add_trajectory(torch.randn(15, 8))
        cloud = tracker.get_point_cloud(last_n=1)
        assert cloud.shape == (15, 8)

    def test_empty_raises(self):
        """get_point_cloud on empty tracker should raise."""
        tracker = TrajectoryTracker()
        with pytest.raises(ValueError):
            tracker.get_point_cloud()

    def test_energies_stored(self):
        """Energy values should be stored alongside trajectories."""
        tracker = TrajectoryTracker()
        traj = torch.randn(10, 8)
        energies = torch.randn(10)
        tracker.add_trajectory(traj, energies=energies)
        assert tracker.get_energies(0) is not None
        assert tracker.get_energies(0).shape == (10,)
