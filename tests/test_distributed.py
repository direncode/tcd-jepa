"""Tests for distributed training components (CPU-only, no GPU required).

Uses torch.multiprocessing.spawn to simulate multi-rank environments.
"""

import os
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import pytest

from tcd_jepa.core.recursive_loop import ConvergenceMonitor


def _setup_process_group(rank: int, world_size: int, backend: str = "gloo"):
    """Initialize a process group for testing."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def _cleanup():
    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Convergence monitor distributed sync
# ---------------------------------------------------------------------------

def _convergence_sync_worker(rank: int, world_size: int, results_dict):
    """Worker that tests convergence monitor synchronization."""
    _setup_process_group(rank, world_size)

    monitor = ConvergenceMonitor(epsilon=0.01, patience=3, distributed=True)

    # Each rank has different representations
    torch.manual_seed(42 + rank)
    representations = torch.randn(32, 16)

    result = monitor.update(
        num_modules=5,
        representations=representations,
        energy_landscape_smoothness=0.1,
    )

    # Store convergence score per rank
    results_dict[rank] = result["convergence_score"]

    _cleanup()


class TestDistributedConvergence:
    @pytest.mark.skipif(
        not hasattr(dist, "init_process_group"),
        reason="torch.distributed not available",
    )
    def test_convergence_scores_synchronized(self):
        """After all_reduce, all ranks should have the same convergence score."""
        world_size = 2
        manager = mp.Manager()
        results = manager.dict()

        mp.spawn(
            _convergence_sync_worker,
            args=(world_size, results),
            nprocs=world_size,
            join=True,
        )

        scores = [results[r] for r in range(world_size)]
        # All ranks should have the same score after all_reduce
        assert abs(scores[0] - scores[1]) < 1e-6, (
            f"Ranks should agree on convergence score: {scores}"
        )


# ---------------------------------------------------------------------------
# DDP gradient sync
# ---------------------------------------------------------------------------

def _ddp_gradient_worker(rank: int, world_size: int, results_dict):
    """Worker that checks DDP gradient synchronization."""
    _setup_process_group(rank, world_size)

    model = torch.nn.Linear(16, 16)
    ddp_model = torch.nn.parallel.DistributedDataParallel(model)

    # Each rank has different input
    torch.manual_seed(rank)
    x = torch.randn(4, 16)
    out = ddp_model(x)
    loss = out.sum()
    loss.backward()

    # After backward, gradients should be averaged across ranks
    grad = model.weight.grad.clone()
    results_dict[rank] = grad

    _cleanup()


class TestDDPGradientSync:
    @pytest.mark.skipif(
        not hasattr(dist, "init_process_group"),
        reason="torch.distributed not available",
    )
    def test_gradients_synchronized(self):
        """DDP should synchronize gradients across ranks."""
        world_size = 2
        manager = mp.Manager()
        results = manager.dict()

        mp.spawn(
            _ddp_gradient_worker,
            args=(world_size, results),
            nprocs=world_size,
            join=True,
        )

        grad_0 = results[0]
        grad_1 = results[1]
        assert torch.allclose(grad_0, grad_1, atol=1e-5), (
            "DDP gradients should be synchronized"
        )


# ---------------------------------------------------------------------------
# Checkpoint save/load in distributed setting
# ---------------------------------------------------------------------------

def _checkpoint_worker(rank: int, world_size: int, tmpdir: str, results_dict):
    """Worker that tests checkpoint save/load."""
    _setup_process_group(rank, world_size)

    model = torch.nn.Linear(16, 16)
    # Ensure all ranks start with same weights
    torch.manual_seed(42)
    torch.nn.init.normal_(model.weight)

    # Only rank 0 saves
    path = str(Path(tmpdir) / "test_ckpt.pt")
    if rank == 0:
        torch.save({"model": model.state_dict(), "epoch": 5}, path)

    dist.barrier()

    # All ranks load
    checkpoint = torch.load(path, map_location="cpu")
    model2 = torch.nn.Linear(16, 16)
    model2.load_state_dict(checkpoint["model"])

    results_dict[rank] = model2.weight.detach().clone()

    _cleanup()


class TestDistributedCheckpoint:
    @pytest.mark.skipif(
        not hasattr(dist, "init_process_group"),
        reason="torch.distributed not available",
    )
    def test_checkpoint_consistent_across_ranks(self):
        """All ranks should load identical model state from checkpoint."""
        world_size = 2
        manager = mp.Manager()
        results = manager.dict()

        with tempfile.TemporaryDirectory() as tmpdir:
            mp.spawn(
                _checkpoint_worker,
                args=(world_size, tmpdir, results),
                nprocs=world_size,
                join=True,
            )

        w0 = results[0]
        w1 = results[1]
        assert torch.allclose(w0, w1), "All ranks should have identical weights"
