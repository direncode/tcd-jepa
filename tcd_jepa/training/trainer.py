"""Main training loop for TCD-JEPA."""

import logging
import time
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from tcd_jepa.models.tcd_jepa_model import TCDJEPAModel
from tcd_jepa.training.guards import (
    LossTracker,
    check_loss_health,
    compute_gradient_stats,
)
from tcd_jepa.training.metrics import TrainingMetrics
from tcd_jepa.training.schedulers import CosineWDSchedule, WarmupCosineSchedule
from tcd_jepa.utils.checkpointing import save_checkpoint
from tcd_jepa.utils.logging import MetricLogger

logger = logging.getLogger("tcd_jepa")


class Trainer:
    """Training loop for TCD-JEPA.

    Handles the standard JEPA training: forward pass through context encoder,
    predict targets, compute loss against EMA target encoder, update weights.
    """

    def __init__(
        self,
        model: TCDJEPAModel,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: WarmupCosineSchedule,
        wd_scheduler: CosineWDSchedule,
        momentum_schedule: iter,
        train_loader: DataLoader,
        device: torch.device,
        cfg: dict,
        metric_logger: Optional[MetricLogger] = None,
        checkpoint_dir: str = "checkpoints",
        scaler: Optional[torch.amp.GradScaler] = None,
        recursive_loop=None,
        stream_encoder=None,
        eval_runner=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.wd_scheduler = wd_scheduler
        self.momentum_schedule = momentum_schedule
        self.train_loader = train_loader
        self.device = device
        self.cfg = cfg
        self.metric_logger = metric_logger
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.scaler = scaler
        self.use_amp = scaler is not None
        self.grad_clip_norm = cfg.get("training", {}).get("grad_clip_norm", 1.0)
        self.recursive_loop = recursive_loop
        self.stream_encoder = stream_encoder
        self.eval_runner = eval_runner
        self.global_step = 0
        self._known_module_ids: set[str] = set()
        self._loss_tracker = LossTracker()

    def train(self, num_epochs: int, start_epoch: int = 0) -> None:
        """Run the training loop."""
        self.model.train()

        eval_freq = self.cfg.get("evaluation", {}).get("eval_freq", 10)

        for epoch in range(start_epoch, num_epochs):
            epoch_metrics = TrainingMetrics()
            t0 = time.time()

            for itr, (images, masks_enc, masks_pred) in enumerate(
                tqdm(self.train_loader, desc=f"Epoch {epoch}", leave=False)
            ):
                step_result = self._train_step(images, masks_enc, masks_pred)

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

            # Run recursive loop at end of epoch if enabled
            if self.recursive_loop is not None and self.stream_encoder is not None:
                with torch.no_grad():
                    z = self.model.context_encoder(images.to(self.device))
                    t = self.model.target_encoder(images.to(self.device))
                energy_fn = self.stream_encoder.make_energy_fn(t)
                loop_result = self.recursive_loop.step(z, energy_fn, epoch=epoch)
                if loop_result.get("crystallized"):
                    n_mods = loop_result["crystallization"]["num_active_modules"]
                    logger.info(f"  Recursive loop: {n_mods} active modules")
                    # Add new module parameters to optimizer
                    self._register_new_module_params()

            epoch_time = time.time() - t0
            metrics_dict = epoch_metrics.to_dict()
            metrics_dict["epoch"] = epoch
            metrics_dict["epoch_time"] = epoch_time

            # Add recursive loop metrics
            if self.recursive_loop is not None:
                metrics_dict["num_modules"] = self.recursive_loop.num_modules
                metrics_dict["converged"] = self.recursive_loop.is_converged

            # GPU memory stats
            if torch.cuda.is_available():
                metrics_dict["gpu_memory_allocated_mb"] = torch.cuda.memory_allocated() / 1e6
                metrics_dict["gpu_memory_peak_mb"] = torch.cuda.max_memory_allocated() / 1e6

            logger.info(
                f"Epoch {epoch}: loss={metrics_dict['loss']:.4f} "
                f"lr={metrics_dict['lr']:.6f} time={epoch_time:.1f}s "
                f"grad_norm={metrics_dict['grad_norm_avg']:.4f}"
                + (f" modules={self.recursive_loop.num_modules}"
                   if self.recursive_loop else "")
                + (f" nan={metrics_dict['nan_count']}" if metrics_dict["nan_count"] > 0 else "")
                + (f" spikes={metrics_dict['loss_spikes']}" if metrics_dict["loss_spikes"] > 0 else "")
            )

            if self.metric_logger is not None:
                self.metric_logger.log(metrics_dict, step=epoch)

            # Save checkpoint periodically
            save_freq = self.cfg.get("training", {}).get("checkpoint_freq", 10)
            if (epoch + 1) % save_freq == 0 or epoch == num_epochs - 1:
                save_checkpoint(
                    path=str(self.checkpoint_dir / f"checkpoint_{epoch:04d}.pt"),
                    epoch=epoch,
                    encoder=self.model.context_encoder,
                    predictor=self.model.predictor,
                    target_encoder=self.model.target_encoder,
                    optimizer=self.optimizer,
                    scaler=self.scaler,
                )

            # Run evaluation if configured
            if self.eval_runner is not None and (epoch + 1) % eval_freq == 0:
                eval_metrics = self.eval_runner.run_evaluation(epoch)
                if self.metric_logger is not None:
                    self.metric_logger.log(
                        {"epoch": epoch, **{f"eval/{k}": v for k, v in eval_metrics.items()}},
                        step=epoch,
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
            # Add as a new param group with reduced learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.optimizer.add_param_group({
                "params": new_params,
                "lr": current_lr,
                "weight_decay": 0.0,
            })
            logger.info(f"  Added {len(new_params)} new module params to optimizer")

    def _train_step(
        self,
        images: torch.Tensor,
        masks_enc: list[torch.Tensor],
        masks_pred: list[torch.Tensor],
    ) -> dict:
        """Execute a single training step.

        Returns:
            Dict with loss, lr, wd, momentum, grad stats, and skip indicators.
        """
        # Move data to device
        images = images.to(self.device)
        masks_enc = [m.to(self.device) for m in masks_enc]
        masks_pred = [m.to(self.device) for m in masks_pred]

        # Forward pass with optional mixed-precision autocast
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_amp):
            result = self.model(images, masks_enc, masks_pred)
            loss = result["loss"]

        # Check loss health before backward
        loss_healthy = check_loss_health(loss)
        if not loss_healthy:
            # Skip this step entirely — advance schedules but don't update weights/EMA
            new_lr = self.lr_scheduler.step()
            new_wd = self.wd_scheduler.step()
            new_m = next(self.momentum_schedule)
            return {
                "loss": 0.0, "lr": new_lr, "wd": new_wd, "momentum": new_m,
                "grad_norm": 0.0, "param_norm": 0.0,
                "nan_count": 1, "inf_count": 0, "is_spike": False, "skipped": True,
            }

        # Backward pass with gradient clipping
        self.optimizer.zero_grad()
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            grad_stats = compute_gradient_stats(self.model)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            grad_stats = compute_gradient_stats(self.model)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.optimizer.step()

        # Update schedules
        new_lr = self.lr_scheduler.step()
        new_wd = self.wd_scheduler.step()
        new_m = next(self.momentum_schedule)

        # Check for loss spike and gradient health before EMA update
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


def build_optimizer(
    model: TCDJEPAModel,
    lr: float = 1e-3,
    weight_decay: float = 0.05,
) -> torch.optim.Optimizer:
    """Build AdamW optimizer with separate param groups for weight decay."""
    # Separate weight decay for biases and layer norms
    decay_params = []
    no_decay_params = []

    for name, param in list(model.context_encoder.named_parameters()) + list(
        model.predictor.named_parameters()
    ):
        if "bias" in name or len(param.shape) == 1:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0, "WD_exclude": True},
    ]

    return torch.optim.AdamW(param_groups, lr=lr)
