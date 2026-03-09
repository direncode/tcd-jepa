"""Loss functions for JEPA training."""

import torch
import torch.nn.functional as F


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
    return F.smooth_l1_loss(predictions, targets)


def l2_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """L2 loss between predictions and targets."""
    return F.mse_loss(predictions, targets)
