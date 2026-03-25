"""Loss functions for JEPA training."""

import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger("tcd_jepa")


def jepa_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Standard JEPA loss: smooth L1 between predicted and target embeddings.

    Matches I-JEPA's loss: F.smooth_l1_loss(z, h) where z is the prediction
    and h is the stop-gradient target.

    Args:
        predictions: Predicted target embeddings from predictor.
        targets: Actual target embeddings from target encoder (detached).

    Returns:
        Scalar loss value.
    """
    if __debug__:
        if torch.isnan(predictions).any():
            logger.warning("NaN detected in predictions input to jepa_loss")
        if torch.isnan(targets).any():
            logger.warning("NaN detected in targets input to jepa_loss")
    return F.smooth_l1_loss(predictions, targets)


def l2_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """L2 loss between predictions and targets."""
    return F.mse_loss(predictions, targets)


def tcd_auxiliary_loss(
    energy_pre: torch.Tensor,
    energy_post: torch.Tensor,
    module_weights: torch.Tensor,
    margin: float = 1.0,
    energy_shaping_weight: float = 0.1,
) -> torch.Tensor:
    """TCD auxiliary loss for differentiable module improvement.

    Combines two objectives:
    1. Margin-based module improvement loss: encourages Langevin exploration
       to reduce energy (post < pre - margin).
    2. Energy shaping loss: regularizes module weights to prevent collapse.

    Args:
        energy_pre: Energy before differentiable Langevin steps [B].
        energy_post: Energy after differentiable Langevin steps [B].
        module_weights: Per-module routing weights from the predictor [M].
        margin: Minimum energy improvement margin.
        energy_shaping_weight: Weight for the energy shaping regularizer.

    Returns:
        Scalar TCD auxiliary loss.
    """
    # Guard against NaN energy values
    if torch.isnan(energy_pre).any() or torch.isnan(energy_post).any():
        logger.warning("NaN energy values in tcd_auxiliary_loss — returning zero loss")
        return torch.tensor(0.0, device=energy_pre.device, requires_grad=True)

    # Margin-based improvement: want energy_post < energy_pre - margin
    improvement_loss = F.relu(energy_post - energy_pre + margin).mean()

    # Energy shaping: encourage diverse module usage (entropy regularizer)
    if module_weights.numel() > 1:
        w = F.softmax(module_weights, dim=-1)
        entropy = -(w * (w + 1e-8).log()).sum()
        max_entropy = torch.tensor(w.shape[-1], dtype=w.dtype, device=w.device).float().log()
        shaping_loss = max_entropy - entropy
    else:
        shaping_loss = torch.tensor(0.0, device=energy_pre.device)

    return improvement_loss + energy_shaping_weight * shaping_loss
