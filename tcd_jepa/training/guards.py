"""Training safety guards: NaN/Inf detection, gradient health monitoring, loss tracking.

Provides utilities to detect and handle numerical instabilities during training,
preventing silent corruption of model weights and EMA targets.
"""

import logging
from collections import deque
from dataclasses import dataclass

import torch
import torch.nn as nn

logger = logging.getLogger("tcd_jepa")


@dataclass
class GradientStats:
    """Gradient health statistics computed after backward pass."""

    grad_norm: float = 0.0
    grad_max: float = 0.0
    grad_min: float = float("inf")
    param_norm: float = 0.0
    nan_count: int = 0
    inf_count: int = 0

    @property
    def is_healthy(self) -> bool:
        return self.nan_count == 0 and self.inf_count == 0


def check_loss_health(loss: torch.Tensor, threshold: float = 1e6) -> bool:
    """Check if loss is finite and within reasonable bounds.

    Returns:
        True if loss is healthy, False if NaN/Inf/exceeds threshold.
    """
    if torch.isnan(loss):
        logger.warning("NaN loss detected — skipping step")
        return False
    if torch.isinf(loss):
        logger.warning("Inf loss detected — skipping step")
        return False
    if loss.item() > threshold:
        logger.warning(f"Loss exceeds threshold ({loss.item():.2e} > {threshold:.2e}) — skipping step")
        return False
    return True


def compute_gradient_stats(model: nn.Module) -> GradientStats:
    """Compute gradient health statistics for all model parameters.

    Call after backward() and before optimizer.step() to monitor gradient health.
    """
    stats = GradientStats()
    grad_norms_sq = 0.0
    param_norms_sq = 0.0
    total_params = 0

    for p in model.parameters():
        if p.grad is not None:
            g = p.grad.data
            grad_norms_sq += g.norm(2).item() ** 2
            g_abs_max = g.abs().max().item()
            g_abs_min = g.abs().min().item()
            stats.grad_max = max(stats.grad_max, g_abs_max)
            stats.grad_min = min(stats.grad_min, g_abs_min)
            stats.nan_count += torch.isnan(g).sum().item()
            stats.inf_count += torch.isinf(g).sum().item()
            total_params += 1

        param_norms_sq += p.data.norm(2).item() ** 2

    stats.grad_norm = grad_norms_sq**0.5
    stats.param_norm = param_norms_sq**0.5

    if total_params == 0:
        stats.grad_min = 0.0

    if stats.nan_count > 0:
        logger.warning(f"NaN gradients detected: {stats.nan_count} elements")
    if stats.inf_count > 0:
        logger.warning(f"Inf gradients detected: {stats.inf_count} elements")

    return stats


class LossTracker:
    """Tracks loss values over a rolling window to detect spikes.

    A spike is detected when the current loss exceeds `spike_factor` times
    the rolling average. When a spike is detected, EMA updates should be skipped
    to prevent corrupting the target encoder.
    """

    def __init__(self, window_size: int = 100, spike_factor: float = 10.0) -> None:
        self.window: deque[float] = deque(maxlen=window_size)
        self.spike_factor = spike_factor
        self.total_spikes = 0

    @property
    def rolling_avg(self) -> float:
        if not self.window:
            return 0.0
        return sum(self.window) / len(self.window)

    def update(self, loss: float) -> bool:
        """Record a loss value and check for spikes.

        Returns:
            True if loss is a spike (should skip EMA update).
        """
        avg = self.rolling_avg
        is_spike = False

        if len(self.window) >= 10 and avg > 0:
            if loss > self.spike_factor * avg:
                logger.warning(
                    f"Loss spike detected: {loss:.4f} > {self.spike_factor}x rolling avg {avg:.4f}"
                )
                is_spike = True
                self.total_spikes += 1

        self.window.append(loss)
        return is_spike
