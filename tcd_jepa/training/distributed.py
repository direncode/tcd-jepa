"""Distributed training utilities using PyTorch DDP.

Supports multi-GPU training on a single node (e.g. 8xH100).
Wraps the existing Trainer with DDP, DistributedSampler, and
checkpoint resume for crash resilience.

Usage:
    torchrun --nproc_per_node=8 train.py --config configs/benchmark_stl10.yaml --tcd
"""

import logging
import os
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

logger = logging.getLogger("tcd_jepa.distributed")


def setup_distributed() -> tuple[int, int, int]:
    """Initialize DDP process group. Returns (rank, local_rank, world_size)."""
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    logger.info(f"DDP rank={rank}, local_rank={local_rank}, world_size={world_size}")
    return rank, local_rank, world_size


def cleanup_distributed():
    """Destroy the process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """True if this is rank 0 or not distributed."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def wrap_model_ddp(model: nn.Module, local_rank: int,
                   find_unused: bool = True) -> DDP:
    """Wrap model in DDP."""
    model = model.to(local_rank)
    return DDP(model, device_ids=[local_rank], find_unused_parameters=find_unused)


def make_distributed_loader(dataset, batch_size: int, num_workers: int = 4,
                            collate_fn=None, drop_last: bool = True,
                            seed: int = 42) -> tuple[DataLoader, DistributedSampler]:
    """Create a DataLoader with DistributedSampler."""
    sampler = DistributedSampler(dataset, shuffle=True, seed=seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=drop_last,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    return loader, sampler


def save_distributed_checkpoint(path: str, epoch: int, model: nn.Module,
                                optimizer: torch.optim.Optimizer,
                                scaler=None, extra: dict = None):
    """Save checkpoint only on rank 0."""
    if not is_main_process():
        return
    # Unwrap DDP
    raw_model = model.module if isinstance(model, DDP) else model
    state = {
        "epoch": epoch,
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if scaler is not None:
        state["scaler_state_dict"] = scaler.state_dict()
    if extra:
        state.update(extra)
    torch.save(state, path)
    logger.info(f"Checkpoint saved: {path} (epoch {epoch})")


def load_distributed_checkpoint(path: str, model: nn.Module,
                                optimizer: torch.optim.Optimizer = None,
                                scaler=None, device="cuda") -> dict:
    """Load checkpoint, returns metadata dict with 'epoch' etc."""
    if not os.path.exists(path):
        return {}
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    raw_model = model.module if isinstance(model, DDP) else model
    raw_model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    logger.info(f"Resumed from checkpoint: {path} (epoch {checkpoint.get('epoch', '?')})")
    return checkpoint


def reduce_metric(value: float) -> float:
    """Average a scalar metric across all DDP ranks."""
    if not dist.is_initialized():
        return value
    t = torch.tensor(value, device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.AVG)
    return t.item()
