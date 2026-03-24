import torch
import pytest
from tcd_jepa.core.system2_explorer import EnergyExplorer
from tcd_jepa.backends.langevin import LangevinBackend


def test_explorer_accepts_backend():
    backend = LangevinBackend(embed_dim=192)
    explorer = EnergyExplorer(embed_dim=192, backend=backend)
    assert explorer.backend is backend


def test_explorer_default_backend():
    explorer = EnergyExplorer(embed_dim=192)
    assert explorer.backend is None


def test_explorer_with_backend_explore():
    backend = LangevinBackend(embed_dim=192)
    explorer = EnergyExplorer(embed_dim=192, backend=backend)
    z = torch.randn(4, 8, 192)
    energy_fn = lambda z: torch.norm(z, dim=-1)
    result = explorer.explore(z, energy_fn)
    assert "trajectory" in result
