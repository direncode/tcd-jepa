import torch
import pytest
from tcd_jepa.backends.base import PhysicsBackend


def test_protocol_signature():
    assert hasattr(PhysicsBackend, "compute_energy")
    assert hasattr(PhysicsBackend, "explore_step")
    assert hasattr(PhysicsBackend, "get_landscape_metrics")


def test_protocol_enforcement():
    class Incomplete:
        pass

    assert not isinstance(Incomplete(), PhysicsBackend)
