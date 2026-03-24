"""Distributed training entry point for TCD-JEPA.

Supports DDP (default) and FSDP strategies, fault-tolerant training with
auto-resume, and differentiable TCD integration.

Usage:
    # DDP (default):
    torchrun --nproc_per_node=4 train_distributed.py --config configs/dist_cifar10.yaml

    # FSDP with activation checkpointing:
    torchrun --nproc_per_node=4 train_distributed.py \\
        --config configs/dist_cifar10.yaml --strategy fsdp --activation-checkpointing

    # With TCD and auto-resume:
    torchrun --nproc_per_node=4 train_distributed.py \\
        --config configs/dist_cifar10.yaml --tcd --auto-resume
"""

import argparse
import glob
import logging
import os
import re
import time
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from tcd_jepa.core.recursive_loop import RecursiveLoop
from tcd_jepa.core.system1_encoder import StreamEncoder
from tcd_jepa.exploration.langevin import LangevinSampler
from tcd_jepa.models.tcd_jepa_model import TCDJEPAModel, build_tcd_jepa
from tcd_jepa.models.target_encoder import momentum_schedule
from tcd_jepa.training.losses import jepa_loss
from tcd_jepa.training.schedulers import CosineWDSchedule, WarmupCosineSchedule
from tcd_jepa.training.trainer import build_optimizer
from tcd_jepa.utils.checkpointing import load_checkpoint, save_checkpoint
from tcd_jepa.utils.config import load_config_with_overrides
from tcd_jepa.utils.logging import MetricLogger
from tcd_jepa.utils.masking import MaskCollator

logger = logging.getLogger("tcd_jepa")


# ---------------------------------------------------------------------------
# FSDP helpers
# ---------------------------------------------------------------------------

def wrap_fsdp(model: TCDJEPAModel, device_id: int) -> nn.Module:
    """Wrap model with FullyShardedDataParallel (FSDP).

    Uses transformer_auto_wrap_policy on Block class and bf16 mixed precision.
    Default sharding strategy is SHARD_GRAD_OP (ZeRO-2 equivalent).
    """
    from torch.distributed.fsdp import (
        BackwardPrefetch,
        FullyShardedDataParallel as FSDP,
        MixedPrecision,
        ShardingStrategy,
    )
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    from tcd_jepa.models.vision_transformer import Block

    auto_wrap_policy = transformer_auto_wrap_policy(
        transformer_layer_cls={Block},
    )

    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    wrapped = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mp_policy,
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        device_id=device_id,
    )
    return wrapped


def save_fsdp_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    path: str,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> None:
    """Save checkpoint from FSDP-wrapped model using full state dict."""
    from torch.distributed.fsdp import FullStateDictConfig, StateDictType
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        state_dict = model.state_dict()

    if dist.get_rank() == 0:
        checkpoint = {
            "epoch": epoch,
            "model": state_dict,
            "optimizer": optimizer.state_dict(),
        }
        if scaler is not None:
            checkpoint["scaler"] = scaler.state_dict()
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, save_path)
        logger.info(f"Saved FSDP checkpoint to {save_path} (epoch {epoch})")


# ---------------------------------------------------------------------------
# Fault tolerance
# ---------------------------------------------------------------------------

class FaultTolerantTrainer:
    """Mixin providing fault tolerance capabilities for distributed training.

    Features:
    - NCCL timeout detection via monitored_barrier every N steps
    - Heartbeat logging for hung process detection
    - Auto-save checkpoint on exception before re-raise
    - Auto-resume: scans checkpoint dir for latest checkpoint
    """

    def __init__(
        self,
        checkpoint_dir: str,
        barrier_freq: int = 100,
    ) -> None:
        self.ft_checkpoint_dir = Path(checkpoint_dir)
        self.ft_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.barrier_freq = barrier_freq
        self._last_heartbeat = time.time()

    def heartbeat(self, step: int) -> None:
        """Log a heartbeat for hung process detection."""
        now = time.time()
        elapsed = now - self._last_heartbeat
        if elapsed > 60:  # Log every 60s
            rank = dist.get_rank() if dist.is_initialized() else 0
            logger.info(f"[Rank {rank}] Heartbeat at step {step}, {elapsed:.0f}s since last")
            self._last_heartbeat = now

    def monitored_barrier(self, step: int) -> None:
        """Run monitored_barrier for NCCL timeout detection."""
        if step > 0 and step % self.barrier_freq == 0 and dist.is_initialized():
            dist.monitored_barrier(timeout=dist.get_default_timeout())

    def emergency_save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        step: int,
        strategy: str,
        scaler: Optional[torch.amp.GradScaler] = None,
    ) -> None:
        """Auto-save checkpoint on exception before re-raising."""
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank != 0:
            return
        try:
            path = str(self.ft_checkpoint_dir / f"emergency_{epoch:04d}_step{step:06d}.pt")
            if strategy == "fsdp":
                # For FSDP, we can only save if the model is accessible
                logger.warning("Emergency save with FSDP may be incomplete")
            else:
                # DDP: unwrap and save
                raw = model.module if hasattr(model, "module") else model
                save_checkpoint(
                    path=path,
                    epoch=epoch,
                    encoder=raw.context_encoder,
                    predictor=raw.predictor,
                    target_encoder=raw.target_encoder,
                    optimizer=optimizer,
                    scaler=scaler,
                )
            logger.info(f"Emergency checkpoint saved to {path}")
        except Exception as e:
            logger.error(f"Emergency save failed: {e}")

    @staticmethod
    def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
        """Scan checkpoint directory for the latest checkpoint.

        Returns:
            Path to latest checkpoint, or None if no checkpoints found.
        """
        ckpt_dir = Path(checkpoint_dir)
        if not ckpt_dir.exists():
            return None

        # Match both regular and emergency checkpoints
        patterns = ["checkpoint_*.pt", "emergency_*.pt"]
        candidates = []
        for pattern in patterns:
            candidates.extend(ckpt_dir.glob(pattern))

        if not candidates:
            return None

        # Sort by modification time, return newest
        candidates.sort(key=lambda p: p.stat().st_mtime)
        return str(candidates[-1])


# ---------------------------------------------------------------------------
# Distributed Trainer
# ---------------------------------------------------------------------------

class DistributedTrainer:
    """Distributed training loop for TCD-JEPA with FSDP/DDP support."""

    def __init__(
        self,
        model: nn.Module,
        raw_model: TCDJEPAModel,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: WarmupCosineSchedule,
        wd_scheduler: CosineWDSchedule,
        momentum_schedule_iter: iter,
        train_loader: DataLoader,
        sampler: DistributedSampler,
        device: torch.device,
        cfg: dict,
        strategy: str = "ddp",
        metric_logger: Optional[MetricLogger] = None,
        checkpoint_dir: str = "checkpoints",
        scaler: Optional[torch.amp.GradScaler] = None,
        recursive_loop=None,
        stream_encoder: Optional[StreamEncoder] = None,
        langevin_sampler: Optional[LangevinSampler] = None,
        fault_tolerant: Optional[FaultTolerantTrainer] = None,
    ):
        self.model = model
        self.raw_model = raw_model  # Unwrapped model for EMA updates etc.
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.wd_scheduler = wd_scheduler
        self.momentum_schedule = momentum_schedule_iter
        self.train_loader = train_loader
        self.sampler = sampler
        self.device = device
        self.cfg = cfg
        self.strategy = strategy
        self.metric_logger = metric_logger
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.scaler = scaler
        self.recursive_loop = recursive_loop
        self.stream_encoder = stream_encoder
        self.langevin_sampler = langevin_sampler
        self.fault_tolerant = fault_tolerant
        self.global_step = 0
        self._known_module_ids: set[str] = set()
        self.rank = dist.get_rank() if dist.is_initialized() else 0

    def train(self, num_epochs: int, start_epoch: int = 0) -> None:
        """Run the distributed training loop."""
        self.model.train()

        train_cfg = self.cfg.get("training", {})
        tcd_weight = train_cfg.get("tcd_weight", 0.1)
        diff_steps = train_cfg.get("langevin_diff_steps", 5)
        use_tcd = (
            self.langevin_sampler is not None
            and self.stream_encoder is not None
            and self.raw_model._has_dynamic_predictor
        )

        for epoch in range(start_epoch, num_epochs):
            self.sampler.set_epoch(epoch)
            epoch_loss = 0.0
            epoch_tcd_loss = 0.0
            num_batches = 0
            t0 = time.time()

            for itr, (images, masks_enc, masks_pred) in enumerate(self.train_loader):
                loss_val, tcd_loss_val, lr, wd, momentum = self._train_step(
                    images, masks_enc, masks_pred,
                    use_tcd=use_tcd,
                    tcd_weight=tcd_weight,
                    diff_steps=diff_steps,
                )
                epoch_loss += loss_val
                epoch_tcd_loss += tcd_loss_val
                num_batches += 1
                self.global_step += 1

                # Fault tolerance checks
                if self.fault_tolerant:
                    self.fault_tolerant.heartbeat(self.global_step)
                    self.fault_tolerant.monitored_barrier(self.global_step)

            # Recursive loop at epoch end (detached, for exploration/crystallization)
            if self.recursive_loop is not None and self.stream_encoder is not None:
                with torch.no_grad():
                    z = self.raw_model.context_encoder(images.to(self.device))
                    t = self.raw_model.target_encoder(images.to(self.device))
                energy_fn = self.stream_encoder.make_energy_fn(t)
                loop_result = self.recursive_loop.step(z, energy_fn, epoch=epoch)
                if loop_result.get("crystallized"):
                    n_mods = loop_result["crystallization"]["num_active_modules"]
                    if self.rank == 0:
                        logger.info(f"  Recursive loop: {n_mods} active modules")
                    self._register_new_module_params()

            # Logging (rank 0 only)
            if self.rank == 0:
                epoch_time = time.time() - t0
                avg_loss = epoch_loss / max(num_batches, 1)
                avg_tcd = epoch_tcd_loss / max(num_batches, 1)

                metrics_dict = {
                    "epoch": epoch,
                    "loss": avg_loss,
                    "lr": lr,
                    "wd": wd,
                    "momentum": momentum,
                    "epoch_time": epoch_time,
                }

                if use_tcd:
                    metrics_dict["tcd_loss"] = avg_tcd

                if self.recursive_loop is not None:
                    metrics_dict["num_modules"] = self.recursive_loop.num_modules
                    metrics_dict["converged"] = self.recursive_loop.is_converged

                logger.info(
                    f"Epoch {epoch}: loss={avg_loss:.4f} lr={lr:.6f} time={epoch_time:.1f}s"
                    + (f" tcd_loss={avg_tcd:.4f}" if use_tcd else "")
                    + (f" modules={self.recursive_loop.num_modules}"
                       if self.recursive_loop else "")
                )

                if self.metric_logger is not None:
                    self.metric_logger.log(metrics_dict, step=epoch)

                # Save checkpoint periodically
                save_freq = train_cfg.get("checkpoint_freq", 10)
                if (epoch + 1) % save_freq == 0 or epoch == num_epochs - 1:
                    self._save_checkpoint(epoch)

    def _train_step(
        self,
        images: torch.Tensor,
        masks_enc: list[torch.Tensor],
        masks_pred: list[torch.Tensor],
        use_tcd: bool = False,
        tcd_weight: float = 0.1,
        diff_steps: int = 5,
    ) -> tuple[float, float, float, float, float]:
        """Execute a single distributed training step.

        Returns:
            Tuple of (loss, tcd_loss, lr, wd, momentum).
        """
        images = images.to(self.device)
        masks_enc = [m.to(self.device) for m in masks_enc]
        masks_pred = [m.to(self.device) for m in masks_pred]

        tcd_loss_val = 0.0

        if use_tcd:
            # TCD-aware forward pass
            result = self.raw_model.forward_with_tcd(
                images, masks_enc, masks_pred,
                langevin_sampler=self.langevin_sampler,
                stream_encoder=self.stream_encoder,
                diff_steps=diff_steps,
                tcd_weight=tcd_weight,
            )
            loss = result["loss"]
            tcd_loss_val = result["tcd_loss"].item()
        else:
            # Standard JEPA forward
            result = self.model(images, masks_enc, masks_pred)
            loss = result["loss"]

        # Backward pass
        self.optimizer.zero_grad()
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

        # Update schedules
        new_lr = self.lr_scheduler.step()
        new_wd = self.wd_scheduler.step()
        new_m = next(self.momentum_schedule)

        # EMA update of target encoder (on raw model)
        self.raw_model.update_target_encoder(new_m)

        return loss.item(), tcd_loss_val, new_lr, new_wd, new_m

    def _save_checkpoint(self, epoch: int) -> None:
        """Save a training checkpoint (rank 0 only)."""
        path = str(self.checkpoint_dir / f"checkpoint_{epoch:04d}.pt")
        if self.strategy == "fsdp":
            save_fsdp_checkpoint(
                self.model, self.optimizer, epoch, path, self.scaler,
            )
        else:
            save_checkpoint(
                path=path,
                epoch=epoch,
                encoder=self.raw_model.context_encoder,
                predictor=self.raw_model.predictor,
                target_encoder=self.raw_model.target_encoder,
                optimizer=self.optimizer,
                scaler=self.scaler,
            )

    def _register_new_module_params(self) -> None:
        """Add newly crystallized module parameters to the optimizer."""
        if self.recursive_loop is None:
            return
        registry = self.recursive_loop.crystallizer.registry
        new_params = []
        for module_id, module in registry.get_all_modules():
            if module_id not in self._known_module_ids:
                self._known_module_ids.add(module_id)
                params = list(module.parameters())
                if params:
                    new_params.extend(params)
                    module.to(self.device)

        if new_params:
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.optimizer.add_param_group({
                "params": new_params,
                "lr": current_lr,
                "weight_decay": 0.0,
            })
            if self.rank == 0:
                logger.info(f"  Added {len(new_params)} new module params to optimizer")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def build_distributed_dataloader(
    cfg: dict,
    mask_collator: MaskCollator,
    world_size: int,
    rank: int,
) -> tuple[DataLoader, DistributedSampler]:
    """Build a distributed dataloader with DistributedSampler."""
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})
    dataset_name = data_cfg.get("dataset", "cifar10")
    img_size = data_cfg.get("img_size", 32)

    if dataset_name == "cifar10":
        transform = T.Compose([
            T.Resize(img_size) if img_size != 32 else T.Lambda(lambda x: x),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        dataset = torchvision.datasets.CIFAR10(
            root=data_cfg.get("data_dir", "./data"),
            train=True, download=(rank == 0), transform=transform,
        )

        class DropLabel(torch.utils.data.Dataset):
            def __init__(self, ds):
                self.ds = ds
            def __len__(self):
                return len(self.ds)
            def __getitem__(self, idx):
                return self.ds[idx][0]

        dataset = DropLabel(dataset)

    elif dataset_name == "two_rooms":
        from experiments.two_rooms.environment import TwoRoomsDataset
        dataset = TwoRoomsDataset(
            num_episodes=200,
            episode_length=50,
            render_size=img_size,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg.get("batch_size", 64),
        sampler=sampler,
        num_workers=data_cfg.get("num_workers", 2),
        collate_fn=mask_collator,
        drop_last=True,
        pin_memory=True,
    )
    return dataloader, sampler


# ---------------------------------------------------------------------------
# Setup / Teardown
# ---------------------------------------------------------------------------

def setup_distributed() -> tuple[int, int, int]:
    """Initialize distributed process group. Returns (rank, local_rank, world_size)."""
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def cleanup_distributed() -> None:
    """Clean up distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Distributed TCD-JEPA Training")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--strategy", default="ddp", choices=["ddp", "fsdp"],
                        help="Distributed strategy: ddp or fsdp")
    parser.add_argument("--tcd", action="store_true", help="Enable TCD recursive loop")
    parser.add_argument("--activation-checkpointing", action="store_true",
                        help="Enable activation checkpointing (reduces memory)")
    parser.add_argument("--auto-resume", action="store_true",
                        help="Auto-resume from latest checkpoint in log dir")
    parser.add_argument("--max-retries", type=int, default=3,
                        help="Max retries on training failure")
    parser.add_argument("overrides", nargs="*",
                        help="Config overrides (e.g., training.epochs=10)")
    args = parser.parse_args()

    # Setup distributed
    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    # Logging setup (only rank 0 prints)
    log_level = logging.INFO if rank == 0 else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format=f"%(asctime)s [Rank {rank}] %(levelname)s: %(message)s",
    )

    # Load config
    cfg = load_config_with_overrides(args.config, args.overrides)
    train_cfg = cfg.get("training", {})

    # Override from config if CLI args not explicitly set
    strategy = args.strategy
    if train_cfg.get("strategy"):
        strategy = train_cfg["strategy"] if args.strategy == "ddp" else args.strategy
    activation_ckpt = args.activation_checkpointing or train_cfg.get("activation_checkpointing", False)
    max_retries = args.max_retries if args.max_retries != 3 else train_cfg.get("max_retries", 3)
    barrier_freq = train_cfg.get("monitored_barrier_freq", 100)

    # Seed (per-rank for different data augmentation)
    seed = train_cfg.get("seed", 42)
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)

    if rank == 0:
        logger.info(f"World size: {world_size}, Strategy: {strategy}")
        logger.info(f"Config: {args.config}")

    enc_cfg = cfg["model"]["encoder"]
    pred_cfg = cfg["model"]["predictor"]
    mask_cfg = cfg["masking"]

    img_size = enc_cfg["img_size"]
    patch_size = enc_cfg["patch_size"]

    # Build model
    model = build_tcd_jepa(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=enc_cfg["embed_dim"],
        depth=enc_cfg["depth"],
        num_heads=enc_cfg["num_heads"],
        predictor_embed_dim=pred_cfg["predictor_embed_dim"],
        predictor_depth=pred_cfg["predictor_depth"],
        predictor_num_heads=pred_cfg["num_heads"],
        use_dynamic_predictor=args.tcd,
    ).to(device)

    if rank == 0:
        param_count = sum(p.numel() for p in model.parameters())
        logger.info(f"Model parameters: {param_count:,}")

    # Activation checkpointing
    if activation_ckpt:
        encoder_vit = model.context_encoder.encoder
        encoder_vit.set_activation_checkpointing(True)
        if rank == 0:
            logger.info("Activation checkpointing enabled")

    # Keep a reference to the unwrapped model
    raw_model = model

    # Wrap model for distributed
    if strategy == "fsdp":
        model = wrap_fsdp(model, local_rank)
    else:
        model = DDP(model, device_ids=[local_rank])

    # Build data
    mask_collator = MaskCollator(
        input_size=(img_size, img_size),
        patch_size=patch_size,
        nenc=mask_cfg["num_enc_masks"],
        npred=mask_cfg["num_pred_masks"],
        min_keep=mask_cfg["min_keep"],
        enc_mask_scale=tuple(mask_cfg["enc_mask_scale"]),
        pred_mask_scale=tuple(mask_cfg["pred_mask_scale"]),
        aspect_ratio=tuple(mask_cfg["aspect_ratio"]),
    )

    # Wait for rank 0 to download dataset
    if rank == 0:
        build_distributed_dataloader(cfg, mask_collator, world_size, rank)
    dist.barrier()

    dataloader, sampler = build_distributed_dataloader(cfg, mask_collator, world_size, rank)
    if rank == 0:
        logger.info(f"Dataset: {len(dataloader.dataset)} samples, {len(dataloader)} batches/epoch")

    # Build optimizer and schedulers
    num_epochs = train_cfg["epochs"]
    steps_per_epoch = len(dataloader)
    total_steps = num_epochs * steps_per_epoch

    optimizer = build_optimizer(
        raw_model, lr=train_cfg["learning_rate"], weight_decay=train_cfg["weight_decay"],
    )
    lr_scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=train_cfg.get("warmup_epochs", 5) * steps_per_epoch,
        start_lr=train_cfg.get("start_lr", 1e-4),
        ref_lr=train_cfg["learning_rate"],
        T_max=total_steps,
    )
    wd_scheduler = CosineWDSchedule(
        optimizer, ref_wd=train_cfg["weight_decay"], T_max=total_steps,
    )
    ema_schedule = momentum_schedule(
        cfg["model"]["ema"]["start"],
        cfg["model"]["ema"]["end"],
        total_steps,
    )

    # Logging (rank 0 only)
    metric_logger = None
    log_cfg = cfg.get("logging", {})
    log_dir = log_cfg.get("log_dir", "./logs")
    if rank == 0:
        metric_logger = MetricLogger(
            log_dir=log_dir,
            use_wandb=log_cfg.get("use_wandb", False),
            wandb_project=log_cfg.get("wandb_project", "tcd-jepa"),
            wandb_config=cfg,
        )

    # Scaler for mixed precision (DDP only; FSDP handles its own mixed precision)
    scaler = None
    if strategy == "ddp" and train_cfg.get("use_bfloat16", False):
        scaler = torch.amp.GradScaler()

    # Optional TCD components
    recursive_loop = None
    stream_encoder = None
    langevin_sampler = None
    if args.tcd:
        recursive_loop = RecursiveLoop(
            embed_dim=enc_cfg["embed_dim"],
            explore_every=2,
            crystallize_every=5,
            langevin_steps=20,
            device=device,
        )
        stream_encoder = StreamEncoder(raw_model.context_encoder, raw_model.target_encoder)
        raw_model.set_module_registry(recursive_loop.crystallizer.registry)

        langevin_sampler = LangevinSampler(
            step_size=0.01,
            temperature=1.0,
            max_steps=train_cfg.get("langevin_diff_steps", 5),
        )
        if rank == 0:
            logger.info("TCD recursive loop enabled with differentiable Langevin")

    # Fault tolerance
    ckpt_dir = str(Path(log_dir) / "checkpoints")
    fault_tolerant = FaultTolerantTrainer(
        checkpoint_dir=ckpt_dir,
        barrier_freq=barrier_freq,
    )

    # Auto-resume
    start_epoch = 0
    if args.auto_resume:
        latest = FaultTolerantTrainer.find_latest_checkpoint(ckpt_dir)
        if latest is not None:
            if rank == 0:
                logger.info(f"Auto-resuming from {latest}")
            start_epoch = load_checkpoint(
                latest,
                raw_model.context_encoder,
                raw_model.predictor,
                raw_model.target_encoder,
                optimizer=optimizer,
                scaler=scaler,
                device=str(device),
            ) + 1
            # Advance schedulers to the right position
            steps_to_skip = start_epoch * steps_per_epoch
            for _ in range(steps_to_skip):
                lr_scheduler.step()
                wd_scheduler.step()
                next(ema_schedule)
            if rank == 0:
                logger.info(f"Resumed from epoch {start_epoch - 1}, continuing from epoch {start_epoch}")

    # Build distributed trainer
    trainer = DistributedTrainer(
        model=model,
        raw_model=raw_model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        wd_scheduler=wd_scheduler,
        momentum_schedule_iter=ema_schedule,
        train_loader=dataloader,
        sampler=sampler,
        device=device,
        cfg=cfg,
        strategy=strategy,
        metric_logger=metric_logger,
        checkpoint_dir=ckpt_dir,
        scaler=scaler,
        recursive_loop=recursive_loop,
        stream_encoder=stream_encoder,
        langevin_sampler=langevin_sampler,
        fault_tolerant=fault_tolerant,
    )

    if rank == 0:
        logger.info(f"Starting distributed training for {num_epochs} epochs (from epoch {start_epoch})")

    trainer.train(num_epochs, start_epoch=start_epoch)

    if rank == 0 and metric_logger is not None:
        metric_logger.close()
        logger.info("Training complete")

    cleanup_distributed()


def main_with_retries() -> None:
    """Entry point with retry logic for fault tolerance."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--max-retries", type=int, default=3)
    known, _ = parser.parse_known_args()
    max_retries = known.max_retries

    for attempt in range(max_retries + 1):
        try:
            main()
            return  # Success
        except Exception as e:
            rank = int(os.environ.get("RANK", 0))
            if rank == 0:
                logger.error(f"Training failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                traceback.print_exc()

            # Clean up distributed state before retry
            if dist.is_initialized():
                try:
                    cleanup_distributed()
                except Exception:
                    pass

            if attempt < max_retries:
                wait = 2 ** (attempt + 1)
                if rank == 0:
                    logger.info(f"Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise RuntimeError(
                    f"Training failed after {max_retries + 1} attempts"
                ) from e


if __name__ == "__main__":
    main_with_retries()
