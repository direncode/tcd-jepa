import torch
import pytest
from tcd_jepa.backends.base import PhysicsBackend
from tcd_jepa.backends.langevin import LangevinBackend


def test_satisfies_protocol():
    backend = LangevinBackend(embed_dim=192)
    assert isinstance(backend, PhysicsBackend)


def test_compute_energy():
    backend = LangevinBackend(embed_dim=192)
    z = torch.randn(8, 192)
    energy = backend.compute_energy(z)
    assert energy.shape == (8,)


def test_explore_step():
    backend = LangevinBackend(embed_dim=192)
    z = torch.randn(8, 192)
    z_new = backend.explore_step(z, temperature=1.0)
    assert z_new.shape == z.shape
    assert not torch.equal(z_new, z)


def test_get_metrics():
    backend = LangevinBackend(embed_dim=192)
    metrics = backend.get_landscape_metrics()
    assert isinstance(metrics, dict)
    assert "step_count" in metrics
