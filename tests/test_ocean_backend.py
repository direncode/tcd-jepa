import torch
import numpy as np
import pytest

try:
    from ocean_core import OceanConfig, set_config
    from tcd_jepa.backends.ocean import OceanBackend
    HAS_OCEAN = True
except ImportError:
    HAS_OCEAN = False


@pytest.mark.skipif(not HAS_OCEAN, reason="latent-ocean-core not installed")
class TestOceanBackend:
    def setup_method(self):
        set_config(OceanConfig(
            irdb_max_entities=1000,
            btut_l_max=8,
            shcg_base_dimension=32,
        ))
        self.backend = OceanBackend(embed_dim=192)

    def test_satisfies_protocol(self):
        from tcd_jepa.backends.base import PhysicsBackend
        assert isinstance(self.backend, PhysicsBackend)

    def test_compute_energy(self):
        z = torch.randn(8, 192)
        energy = self.backend.compute_energy(z)
        assert energy.shape == (8,)

    def test_explore_step(self):
        z = torch.randn(8, 192)
        z_new = self.backend.explore_step(z, temperature=1.0)
        assert z_new.shape == z.shape

    def test_ingest_entities(self):
        records = [{"id": i, "val": i * 10} for i in range(20)]
        self.backend.ingest(records)
        metrics = self.backend.get_landscape_metrics()
        assert metrics["n_entities"] == 20

    def test_get_metrics(self):
        metrics = self.backend.get_landscape_metrics()
        assert "sigma" in metrics
        assert "n_entities" in metrics
