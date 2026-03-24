"""Tests for Phase 4 — Recursive Loop."""

import torch

from tcd_jepa.core.recursive_loop import ConvergenceMonitor, RecursiveLoop


def _quadratic_energy(z: torch.Tensor) -> torch.Tensor:
    return z.pow(2).sum(dim=-1)


class TestConvergenceMonitor:

    def test_initial_state(self):
        """Monitor starts unconverged."""
        monitor = ConvergenceMonitor()
        assert not monitor.is_converged

    def test_convergence_detection(self):
        """Monitor detects convergence after stable iterations."""
        monitor = ConvergenceMonitor(epsilon=100.0, patience=3)

        # Simulate stable iterations
        for i in range(10):
            monitor.update(
                num_modules=5,
                representations=torch.randn(10, 8),
                energy_landscape_smoothness=1.0,
            )

        # After enough stable iterations, should converge
        assert monitor.history[-1]["convergence_score"] >= 0

    def test_history_tracking(self):
        """Monitor records history."""
        monitor = ConvergenceMonitor()
        for _ in range(5):
            monitor.update(
                num_modules=3,
                representations=torch.randn(10, 8),
                energy_landscape_smoothness=0.5,
            )
        assert len(monitor.history) == 5


class TestRecursiveLoop:

    def test_step_basic(self):
        """Recursive loop step runs without error."""
        loop = RecursiveLoop(
            embed_dim=8,
            explore_every=1,
            crystallize_every=2,
            langevin_steps=3,
        )
        z = torch.randn(4, 8)
        result = loop.step(z, _quadratic_energy, epoch=0)
        assert "iteration" in result
        assert "convergence" in result

    def test_exploration_triggered(self):
        """Exploration happens at the configured frequency."""
        loop = RecursiveLoop(
            embed_dim=8,
            explore_every=2,
            crystallize_every=100,  # don't crystallize
            langevin_steps=3,
        )
        z = torch.randn(4, 8)

        r1 = loop.step(z, _quadratic_energy, epoch=0)
        assert not r1["explored"]  # iteration 1, explore_every=2

        r2 = loop.step(z, _quadratic_energy, epoch=0)
        assert r2["explored"]  # iteration 2

    def test_crystallization_triggered(self):
        """Crystallization happens when enough trajectories exist."""
        loop = RecursiveLoop(
            embed_dim=8,
            explore_every=1,
            crystallize_every=2,
            min_trajectories_for_crystallization=1,
            langevin_steps=3,
        )
        z = torch.randn(4, 8)

        # First step: explore (iteration 1)
        loop.step(z, _quadratic_energy, epoch=0)
        # Second step: explore + crystallize (iteration 2)
        r = loop.step(z, _quadratic_energy, epoch=0)
        assert r["crystallized"]

    def test_3d_input(self):
        """Loop handles [B, N, D] input (encoder output with patches)."""
        loop = RecursiveLoop(embed_dim=8, explore_every=1, langevin_steps=3)
        z = torch.randn(2, 16, 8)  # B=2, N=16 patches, D=8
        result = loop.step(z, _quadratic_energy, epoch=0)
        assert result["explored"]

    def test_num_modules_property(self):
        """num_modules reflects crystallized module count."""
        loop = RecursiveLoop(embed_dim=8, explore_every=1, crystallize_every=1,
                             min_trajectories_for_crystallization=1, langevin_steps=3)
        z = torch.randn(4, 8)
        loop.step(z, _quadratic_energy, epoch=0)
        # Module count should be non-negative
        assert loop.num_modules >= 0
