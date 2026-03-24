"""Main training loop for TCD-JEPA."""

import logging
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from tcd_jepa.models.tcd_jepa_model import TCDJEPAModel
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
        self.global_step = 0
        self._known_module_ids: set[str] = set()

    def train(self, num_epochs: int, start_epoch: int = 0) -> None:
        """Run the training loop."""
        self.model.train()

        for epoch in range(start_epoch, num_epochs):
            epoch_metrics = TrainingMetrics()
            t0 = time.time()

            for itr, (images, masks_enc, masks_pred) in enumerate(
                tqdm(self.train_loader, desc=f"Epoch {epoch}", leave=False)
            ):
                loss, new_lr, new_wd, new_m = self._train_step(images, masks_enc, masks_pred)

                epoch_metrics.update(
                    loss=loss,
                    lr=new_lr,
                    wd=new_wd,
                    momentum=new_m,
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

            logger.info(
                f"Epoch {epoch}: loss={metrics_dict['loss']:.4f} "
                f"lr={metrics_dict['lr']:.6f} time={epoch_time:.1f}s"
                + (f" modules={self.recursive_loop.num_modules}"
                   if self.recursive_loop else "")
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
    ) -> tuple[float, float, float, float]:
        """Execute a single training step.

        Returns:
            Tuple of (loss_value, learning_rate, weight_decay, momentum).
        """
        # Move data to device
        images = images.to(self.device)
        masks_enc = [m.to(self.device) for m in masks_enc]
        masks_pred = [m.to(self.device) for m in masks_pred]

        # Forward pass with optional mixed-precision autocast
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_amp):
            result = self.model(images, masks_enc, masks_pred)
            loss = result["loss"]

        # Backward pass with gradient clipping
        self.optimizer.zero_grad()
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.optimizer.step()

        # Update schedules
        new_lr = self.lr_scheduler.step()
        new_wd = self.wd_scheduler.step()
        new_m = next(self.momentum_schedule)

        # EMA update of target encoder
        self.model.update_target_encoder(new_m)

        return loss.item(), new_lr, new_wd, new_m


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
