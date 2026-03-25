"""Training loop for Latent Ocean Manifold TCD-JEPA.

Handles manifold batch data (dicts with fingerprints, coords, velocity,
adjacency) instead of image tensors. Integrates the TCD recursive loop
for topological crystallization on manifold representations.
"""

import logging
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from tcd_jepa.manifold.model import ManifoldJEPAModel
from tcd_jepa.training.guards import (
    LossTracker,
    check_loss_health,
    compute_gradient_stats,
)
from tcd_jepa.training.metrics import TrainingMetrics
from tcd_jepa.training.schedulers import CosineWDSchedule, WarmupCosineSchedule
from tcd_jepa.utils.checkpointing import save_checkpoint

logger = logging.getLogger("tcd_jepa.manifold")


class ManifoldTrainer:
    """Training loop for Manifold TCD-JEPA."""

    def __init__(
        self,
        model: ManifoldJEPAModel,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: WarmupCosineSchedule,
        wd_scheduler: CosineWDSchedule,
        momentum_schedule: iter,
        train_loader: DataLoader,
        device: torch.device,
        cfg: dict,
        checkpoint_dir: str = "checkpoints",
        recursive_loop=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.wd_scheduler = wd_scheduler
        self.momentum_schedule = momentum_schedule
        self.train_loader = train_loader
        self.device = device
        self.cfg = cfg
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.recursive_loop = recursive_loop
        self.global_step = 0
        self._known_module_ids: set[str] = set()
        self.grad_clip_norm = cfg.get("training", {}).get("grad_clip_norm", 1.0)
        self._loss_tracker = LossTracker()

    def train(self, num_epochs: int, start_epoch: int = 0) -> None:
        self.model.train()

        for epoch in range(start_epoch, num_epochs):
            epoch_metrics = TrainingMetrics()
            t0 = time.time()

            for itr, (batch_data, masks_enc, masks_pred) in enumerate(
                tqdm(self.train_loader, desc=f"Epoch {epoch}", leave=False)
            ):
                step_result = self._train_step(batch_data, masks_enc, masks_pred)
                epoch_metrics.update(
                    loss=step_result["loss"],
                    lr=step_result["lr"],
                    wd=step_result["wd"],
                    momentum=step_result["momentum"],
                    grad_norm=step_result["grad_norm"],
                    param_norm=step_result["param_norm"],
                    nan_count=step_result["nan_count"],
                    inf_count=step_result["inf_count"],
                    is_spike=step_result["is_spike"],
                    skipped=step_result["skipped"],
                )
                self.global_step += 1

            # TCD crystallization at end of epoch
            if self.recursive_loop is not None:
                self._run_recursive_loop(batch_data, epoch)

            epoch_time = time.time() - t0
            metrics_dict = epoch_metrics.to_dict()
            metrics_dict["epoch"] = epoch
            metrics_dict["epoch_time"] = epoch_time

            if self.recursive_loop is not None:
                metrics_dict["num_modules"] = self.recursive_loop.num_modules
                metrics_dict["converged"] = self.recursive_loop.is_converged

            logger.info(
                f"Epoch {epoch}: loss={metrics_dict['loss']:.4f} "
                f"lr={metrics_dict['lr']:.6f} time={epoch_time:.1f}s "
                f"grad_norm={metrics_dict['grad_norm_avg']:.4f}"
                + (f" modules={self.recursive_loop.num_modules}" if self.recursive_loop else "")
                + (f" nan={metrics_dict['nan_count']}" if metrics_dict["nan_count"] > 0 else "")
            )

            save_freq = self.cfg.get("training", {}).get("checkpoint_freq", 10)
            if (epoch + 1) % save_freq == 0 or epoch == num_epochs - 1:
                save_checkpoint(
                    path=str(self.checkpoint_dir / f"manifold_checkpoint_{epoch:04d}.pt"),
                    epoch=epoch,
                    encoder=self.model.context_encoder,
                    predictor=self.model.predictor,
                    target_encoder=self.model.target_encoder,
                    optimizer=self.optimizer,
                    scaler=None,
                )

    def _run_recursive_loop(self, batch_data: dict, epoch: int) -> None:
        """Run TCD Systems 2->3 on the last batch."""
        with torch.no_grad():
            bd = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                  for k, v in batch_data.items()}

            z = self.model.context_encoder(
                bd["fingerprints"], coords=bd.get("coords"),
                adjacency=bd.get("adjacency"), velocity=bd.get("velocity"),
            )
            t = self.model.target_encoder(
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
                logger.info(f"  TCD crystallization: {n} active modules")
                self._register_new_module_params()

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
            logger.info(f"  Added {len(new_params)} new module params to optimizer")

    def _train_step(
        self, batch_data: dict, masks_enc: list[torch.Tensor], masks_pred: list[torch.Tensor],
    ) -> dict:
        """Execute a single training step with NaN guards and gradient clipping."""
        batch_data = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch_data.items()
        }
        masks_enc = [m.to(self.device) for m in masks_enc]
        masks_pred = [m.to(self.device) for m in masks_pred]

        result = self.model(batch_data, masks_enc, masks_pred)
        loss = result["loss"]

        # Check loss health before backward
        loss_healthy = check_loss_health(loss)
        if not loss_healthy:
            new_lr = self.lr_scheduler.step()
            new_wd = self.wd_scheduler.step()
            new_m = next(self.momentum_schedule)
            return {
                "loss": 0.0, "lr": new_lr, "wd": new_wd, "momentum": new_m,
                "grad_norm": 0.0, "param_norm": 0.0,
                "nan_count": 1, "inf_count": 0, "is_spike": False, "skipped": True,
            }

        self.optimizer.zero_grad()
        loss.backward()
        grad_stats = compute_gradient_stats(self.model)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
        self.optimizer.step()

        new_lr = self.lr_scheduler.step()
        new_wd = self.wd_scheduler.step()
        new_m = next(self.momentum_schedule)

        # Check for spike/gradient health before EMA update
        loss_val = loss.item()
        is_spike = self._loss_tracker.update(loss_val)
        skip_ema = is_spike or not grad_stats.is_healthy

        if not skip_ema:
            self.model.update_target_encoder(new_m)
        else:
            logger.warning("Skipping EMA update due to unhealthy step")

        return {
            "loss": loss_val,
            "lr": new_lr,
            "wd": new_wd,
            "momentum": new_m,
            "grad_norm": grad_stats.grad_norm,
            "param_norm": grad_stats.param_norm,
            "nan_count": grad_stats.nan_count,
            "inf_count": grad_stats.inf_count,
            "is_spike": is_spike,
            "skipped": False,
        }
