"""Distributed training for Latent Ocean TCD-JEPA.

Multi-GPU manifold training with DDP, fault tolerance, and TCD integration.

Usage:
    # 3 GPUs, semiconductor data
    torchrun --nproc_per_node=3 train_manifold_distributed.py \
        --config configs/manifold_large.yaml --tcd --eval \
        data.dataset=latent_ocean data.data_dir=./data/arxiv

    # 8 GPUs, with auto-resume
    torchrun --nproc_per_node=8 train_manifold_distributed.py \
        --config configs/manifold_large.yaml --tcd --eval --auto-resume
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from tcd_jepa.core.recursive_loop import RecursiveLoop
from tcd_jepa.manifold.dataset import LatentOceanDataset, SyntheticManifoldDataset
from tcd_jepa.manifold.evaluation import run_full_manifold_evaluation
from tcd_jepa.manifold.masking import ManifoldMaskCollator
from tcd_jepa.manifold.model import ManifoldJEPAModel, build_manifold_jepa
from tcd_jepa.models.target_encoder import momentum_schedule
from tcd_jepa.training.guards import (
    LossTracker,
    check_loss_health,
    compute_gradient_stats,
)
from tcd_jepa.training.schedulers import CosineWDSchedule, WarmupCosineSchedule
from tcd_jepa.training.trainer import build_optimizer
from tcd_jepa.utils.checkpointing import load_checkpoint, save_checkpoint
from tcd_jepa.utils.config import load_config_with_overrides

logger = logging.getLogger("tcd_jepa.manifold")


def setup_distributed() -> tuple[int, int, int]:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def build_dataset(cfg: dict):
    data_cfg = cfg.get("data", {})
    name = data_cfg.get("dataset", "synthetic_manifold")

    if name in ("synthetic_manifold", "synthetic"):
        return SyntheticManifoldDataset(
            num_entities=data_cfg.get("num_entities", 256),
            fingerprint_dim=data_cfg.get("fingerprint_dim", 384),
            num_clusters=data_cfg.get("num_clusters", 6),
            num_tokens=data_cfg.get("num_tokens", 64),
            num_samples=data_cfg.get("num_samples", 500),
            causal_link_prob=data_cfg.get("causal_link_prob", 0.15),
            cross_cluster_prob=data_cfg.get("cross_cluster_prob", 0.03),
            sphere_radius=data_cfg.get("sphere_radius", 4.5),
            seed=cfg.get("training", {}).get("seed", 42),
        )
    elif name in ("latent_ocean", "spherical_db"):
        return LatentOceanDataset(
            data_dir=data_cfg.get("data_dir", "./data/latent_ocean"),
            num_tokens=data_cfg.get("num_tokens", 64),
            num_samples=data_cfg.get("num_samples", 1000),
        )
    else:
        raise ValueError(f"Unknown dataset: {name}")


class DistributedManifoldTrainer:
    """Distributed manifold training with DDP and TCD support."""

    def __init__(
        self,
        model: nn.Module,
        raw_model: ManifoldJEPAModel,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: WarmupCosineSchedule,
        wd_scheduler: CosineWDSchedule,
        momentum_schedule_iter: iter,
        train_loader: DataLoader,
        sampler: DistributedSampler,
        device: torch.device,
        cfg: dict,
        checkpoint_dir: str = "checkpoints",
        recursive_loop=None,
    ):
        self.model = model
        self.raw_model = raw_model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.wd_scheduler = wd_scheduler
        self.momentum_schedule = momentum_schedule_iter
        self.train_loader = train_loader
        self.sampler = sampler
        self.device = device
        self.cfg = cfg
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.recursive_loop = recursive_loop
        self.global_step = 0
        self._known_module_ids: set[str] = set()
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.grad_clip_norm = cfg.get("training", {}).get("grad_clip_norm", 1.0)
        self._loss_tracker = LossTracker()

    def train(self, num_epochs: int, start_epoch: int = 0) -> None:
        self.model.train()

        for epoch in range(start_epoch, num_epochs):
            self.sampler.set_epoch(epoch)
            epoch_loss = 0.0
            num_batches = 0
            epoch_grad_norm_sum = 0.0
            t0 = time.time()

            for itr, (batch_data, masks_enc, masks_pred) in enumerate(self.train_loader):
                step_result = self._train_step(batch_data, masks_enc, masks_pred)
                epoch_loss += step_result["loss"]
                epoch_grad_norm_sum += step_result["grad_norm"]
                num_batches += 1
                self.global_step += 1

            # TCD crystallization at epoch end (rank 0 explores, then broadcast)
            if self.recursive_loop is not None:
                self._run_recursive_loop(batch_data, epoch)

            # Logging (rank 0 only)
            if self.rank == 0:
                epoch_time = time.time() - t0
                avg_loss = epoch_loss / max(num_batches, 1)
                avg_grad_norm = epoch_grad_norm_sum / max(num_batches, 1)

                logger.info(
                    f"Epoch {epoch}: loss={avg_loss:.4f} "
                    f"lr={step_result['lr']:.6f} time={epoch_time:.1f}s "
                    f"grad_norm={avg_grad_norm:.4f}"
                    + (f" modules={self.recursive_loop.num_modules}"
                       if self.recursive_loop else "")
                )

                # Save checkpoint
                save_freq = self.cfg.get("training", {}).get("checkpoint_freq", 10)
                if (epoch + 1) % save_freq == 0 or epoch == num_epochs - 1:
                    save_checkpoint(
                        path=str(self.checkpoint_dir / f"manifold_checkpoint_{epoch:04d}.pt"),
                        epoch=epoch,
                        encoder=self.raw_model.context_encoder,
                        predictor=self.raw_model.predictor,
                        target_encoder=self.raw_model.target_encoder,
                        optimizer=self.optimizer,
                    )

    def _train_step(self, batch_data: dict, masks_enc: list, masks_pred: list) -> dict:
        batch_data = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch_data.items()
        }
        masks_enc = [m.to(self.device) for m in masks_enc]
        masks_pred = [m.to(self.device) for m in masks_pred]

        result = self.model(batch_data, masks_enc, masks_pred)
        loss = result["loss"]

        loss_healthy = check_loss_health(loss)
        if not loss_healthy:
            new_lr = self.lr_scheduler.step()
            new_wd = self.wd_scheduler.step()
            new_m = next(self.momentum_schedule)
            return {
                "loss": 0.0, "lr": new_lr, "wd": new_wd, "momentum": new_m,
                "grad_norm": 0.0, "skipped": True,
            }

        self.optimizer.zero_grad()
        loss.backward()
        grad_stats = compute_gradient_stats(self.model)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
        self.optimizer.step()

        new_lr = self.lr_scheduler.step()
        new_wd = self.wd_scheduler.step()
        new_m = next(self.momentum_schedule)

        loss_val = loss.item()
        is_spike = self._loss_tracker.update(loss_val)
        skip_ema = is_spike or not grad_stats.is_healthy

        if not skip_ema:
            self.raw_model.update_target_encoder(new_m)
        elif self.rank == 0:
            logger.warning("Skipping EMA update due to unhealthy step")

        return {
            "loss": loss_val, "lr": new_lr, "wd": new_wd, "momentum": new_m,
            "grad_norm": grad_stats.grad_norm, "skipped": False,
        }

    def _run_recursive_loop(self, batch_data: dict, epoch: int) -> None:
        with torch.no_grad():
            bd = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                  for k, v in batch_data.items()}

            z = self.raw_model.context_encoder(
                bd["fingerprints"], coords=bd.get("coords"),
                adjacency=bd.get("adjacency"), velocity=bd.get("velocity"),
            )
            t = self.raw_model.target_encoder(
                bd["fingerprints"], coords=bd.get("coords"),
                adjacency=bd.get("adjacency"), velocity=bd.get("velocity"),
            )

            t_mean = t.mean(dim=1) if t.dim() == 3 else t
            t_det = t_mean.detach()

            def energy_fn(z_in):
                B_z, B_t = z_in.shape[0], t_det.shape[0]
                tt = t_det.repeat((B_z + B_t - 1) // B_t, 1)[:B_z] if B_z != B_t else t_det
                return (z_in - tt).pow(2).sum(dim=-1)

            loop_result = self.recursive_loop.step(z, energy_fn, epoch=epoch)
            if loop_result.get("crystallized"):
                n = loop_result["crystallization"]["num_active_modules"]
                if self.rank == 0:
                    logger.info(f"  TCD crystallization: {n} active modules")
                self._sync_crystallized_modules()
                dist.barrier()
                self._register_new_module_params()

    def _sync_crystallized_modules(self) -> None:
        if not dist.is_initialized() or self.recursive_loop is None:
            return
        dist.barrier()
        registry = self.recursive_loop.crystallizer.registry
        for _, module in registry.get_all_modules():
            params = list(module.parameters())
            if not params:
                continue
            flat = torch.cat([p.data.reshape(-1) for p in params])
            dist.broadcast(flat, src=0)
            offset = 0
            for p in params:
                numel = p.numel()
                p.data.copy_(flat[offset:offset + numel].reshape(p.shape))
                offset += numel

    def _register_new_module_params(self) -> None:
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
                "params": new_params, "lr": current_lr, "weight_decay": 0.0,
            })
            if self.rank == 0:
                logger.info(f"  Added {len(new_params)} new module params to optimizer")


def main():
    parser = argparse.ArgumentParser(description="Distributed Latent Ocean TCD-JEPA")
    parser.add_argument("--config", default="configs/manifold_large.yaml")
    parser.add_argument("--tcd", action="store_true", help="Enable TCD crystallization")
    parser.add_argument("--eval", action="store_true", help="Run evaluation after training")
    parser.add_argument("--auto-resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    log_level = logging.INFO if rank == 0 else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format=f"%(asctime)s [Rank {rank}] %(levelname)s: %(message)s",
    )

    cfg = load_config_with_overrides(args.config, args.overrides)
    train_cfg = cfg.get("training", {})
    enc_cfg = cfg["model"]["encoder"]
    pred_cfg = cfg["model"]["predictor"]
    mask_cfg = cfg["masking"]
    tcd_cfg = cfg.get("tcd", {})

    seed = train_cfg.get("seed", 42)
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)

    use_tcd = args.tcd or tcd_cfg.get("use_dynamic_predictor", False)

    # Build model
    model = build_manifold_jepa(
        fingerprint_dim=enc_cfg.get("fingerprint_dim", 384),
        num_tokens=enc_cfg.get("num_tokens", 64),
        coord_dim=enc_cfg.get("coord_dim", 3),
        embed_dim=enc_cfg["embed_dim"],
        depth=enc_cfg["depth"],
        num_heads=enc_cfg["num_heads"],
        mlp_ratio=enc_cfg.get("mlp_ratio", 4.0),
        predictor_embed_dim=pred_cfg["predictor_embed_dim"],
        predictor_depth=pred_cfg["predictor_depth"],
        predictor_num_heads=pred_cfg["num_heads"],
        use_dynamic_predictor=use_tcd,
        use_causal_encoding=enc_cfg.get("use_causal_encoding", True),
        use_velocity_encoding=enc_cfg.get("use_velocity_encoding", True),
        sphere_radius=enc_cfg.get("sphere_radius", 4.5),
    ).to(device)

    if rank == 0:
        param_count = sum(p.numel() for p in model.parameters())
        logger.info(f"Model parameters: {param_count:,}")

    raw_model = model
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=use_tcd)

    # Build dataset
    dataset = build_dataset(cfg)
    if rank == 0:
        logger.info(f"Dataset: {len(dataset)} samples, {dataset.num_entities} entities, "
                     f"{dataset.num_clusters} clusters")

    mask_collator = ManifoldMaskCollator(
        num_tokens=enc_cfg.get("num_tokens", 64),
        context_ratio=tuple(mask_cfg.get("context_ratio", [0.3, 0.5])),
        target_ratio=tuple(mask_cfg.get("target_ratio", [0.15, 0.3])),
        num_context_masks=mask_cfg.get("num_context_masks", 4),
        num_target_masks=mask_cfg.get("num_target_masks", 1),
        min_keep=mask_cfg.get("min_keep", 4),
        use_geodesic=mask_cfg.get("use_geodesic", True),
        sphere_radius=mask_cfg.get("sphere_radius", 4.5),
    )

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(
        dataset, batch_size=train_cfg.get("batch_size", 32),
        sampler=sampler, num_workers=0, collate_fn=mask_collator, drop_last=True,
        pin_memory=True,
    )

    # Optimizer and schedulers
    num_epochs = train_cfg["epochs"]
    steps_per_epoch = len(dataloader)
    total_steps = num_epochs * steps_per_epoch

    optimizer = build_optimizer(raw_model, lr=train_cfg["learning_rate"], weight_decay=train_cfg["weight_decay"])
    lr_scheduler = WarmupCosineSchedule(
        optimizer, warmup_steps=train_cfg.get("warmup_epochs", 5) * steps_per_epoch,
        start_lr=train_cfg.get("start_lr", 1e-4), ref_lr=train_cfg["learning_rate"], T_max=total_steps,
    )
    wd_scheduler = CosineWDSchedule(optimizer, ref_wd=train_cfg["weight_decay"], T_max=total_steps)
    ema_schedule = momentum_schedule(
        cfg["model"]["ema"]["start"], cfg["model"]["ema"]["end"], total_steps,
    )

    # TCD
    recursive_loop = None
    if use_tcd:
        recursive_loop = RecursiveLoop(
            embed_dim=enc_cfg["embed_dim"],
            explore_every=tcd_cfg.get("explore_every", 3),
            crystallize_every=tcd_cfg.get("crystallize_every", 6),
            langevin_steps=tcd_cfg.get("langevin_steps", 20),
            langevin_step_size=tcd_cfg.get("langevin_step_size", 0.01),
            persistence_threshold=tcd_cfg.get("persistence_threshold", 0.2),
            max_modules=tcd_cfg.get("max_modules", 16),
            device=device,
        )
        raw_model.set_module_registry(recursive_loop.crystallizer.registry)
        if rank == 0:
            logger.info("TCD recursive loop enabled")

    # Auto-resume
    log_dir = cfg.get("logging", {}).get("log_dir", "./logs/manifold_large")
    ckpt_dir = str(Path(log_dir) / "checkpoints")
    start_epoch = 0
    if args.auto_resume:
        from tcd_jepa.utils.checkpointing import validate_checkpoint
        ckpt_path = Path(ckpt_dir)
        if ckpt_path.exists():
            candidates = sorted(ckpt_path.glob("manifold_checkpoint_*.pt"), key=lambda p: p.stat().st_mtime)
            if candidates:
                latest = str(candidates[-1])
                valid, msg = validate_checkpoint(latest)
                if valid:
                    start_epoch = load_checkpoint(
                        latest, raw_model.context_encoder, raw_model.predictor,
                        raw_model.target_encoder, optimizer=optimizer, device=str(device),
                    ) + 1
                    for _ in range(start_epoch * steps_per_epoch):
                        lr_scheduler.step()
                        wd_scheduler.step()
                        next(ema_schedule)
                    if rank == 0:
                        logger.info(f"Resumed from {latest} (epoch {start_epoch})")

    # Build trainer
    trainer = DistributedManifoldTrainer(
        model=model, raw_model=raw_model, optimizer=optimizer,
        lr_scheduler=lr_scheduler, wd_scheduler=wd_scheduler,
        momentum_schedule_iter=ema_schedule, train_loader=dataloader,
        sampler=sampler, device=device, cfg=cfg,
        checkpoint_dir=ckpt_dir, recursive_loop=recursive_loop,
    )

    if rank == 0:
        logger.info(f"Starting distributed manifold training: {num_epochs} epochs, "
                     f"{world_size} GPUs")

    trainer.train(num_epochs, start_epoch=start_epoch)

    if rank == 0:
        logger.info("Training complete")

    # Evaluation (rank 0 only, using unwrapped model)
    if args.eval and rank == 0:
        logger.info("=" * 70)
        logger.info("LATENT OCEAN TCD-JEPA EVALUATION")
        logger.info("=" * 70)

        results = run_full_manifold_evaluation(
            raw_model, dataset, device, num_classes=dataset.num_clusters,
            max_samples=min(500, len(dataset)), recursive_loop=recursive_loop,
        )

        logger.info("\n--- Standard SSL Metrics ---")
        for key in ["linear_probe_train", "linear_probe_test", "knn_k1", "knn_k5", "knn_k20"]:
            if key in results:
                logger.info(f"  {key:35s}: {results[key]:.4f}")

        logger.info("\n--- Causal Link Prediction ---")
        for key in ["link_auc", "link_pos_sim", "link_neg_sim", "link_sim_gap"]:
            if key in results:
                logger.info(f"  {key:35s}: {results[key]:.4f}")

        logger.info("\n--- Manifold Geometry ---")
        for key in ["neighborhood_preservation", "effective_rank", "feature_std", "uniformity"]:
            if key in results:
                logger.info(f"  {key:35s}: {results[key]:.4f}")

        logger.info("\n--- Latent Ocean KPIs ---")
        for key in ["clarity_score", "drift_velocity", "opportunity_surface", "risk_horizon"]:
            if key in results:
                logger.info(f"  {key:35s}: {results[key]:.4f}" if isinstance(results[key], float)
                             else f"  {key:35s}: {results[key]}")

        logger.info("\n--- TCD Topology ---")
        for key in ["tcd_num_modules", "tcd_converged", "tcd_attractors_h0", "tcd_cycles_h1",
                     "tcd_boundaries_h2", "tcd_total_persistence"]:
            if key in results:
                logger.info(f"  {key:35s}: {results[key]}")

        results_dir = Path(log_dir) / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        with open(results_dir / "latent_ocean_eval.json", "w") as f:
            serializable = {k: (bool(v) if isinstance(v, (np.bool_,)) else v) for k, v in results.items()}
            json.dump(serializable, f, indent=2, default=str)
        logger.info(f"\nResults saved to {results_dir / 'latent_ocean_eval.json'}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
