"""Energy landscape computation for JEPA.

The energy function measures the distance between predicted and actual
target representations in latent space. System 2 will explore over this
surface in later phases.
"""

from typing import Optional

import torch
import torch.nn.functional as F


def compute_energy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute the energy (L2 distance) between predictions and targets in latent space.

    E(pred, target) = ||pred - target||^2

    Args:
        predictions: Predicted target embeddings [B, N, D].
        targets: Actual target embeddings [B, N, D].
        reduction: 'mean', 'sum', or 'none'.

    Returns:
        Energy value(s).
    """
    energy = (predictions - targets).pow(2).sum(dim=-1)
    if reduction == "mean":
        return energy.mean()
    elif reduction == "sum":
        return energy.sum()
    return energy


def compute_energy_statistics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, float]:
    """Compute energy landscape statistics over a batch.

    Args:
        predictions: Predicted target embeddings [B, N, D].
        targets: Actual target embeddings [B, N, D].

    Returns:
        Dictionary with mean, std, min, max energy values.
    """
    per_sample_energy = compute_energy(predictions, targets, reduction="none")
    per_sample_mean = per_sample_energy.mean(dim=-1)

    return {
        "energy_mean": per_sample_mean.mean().item(),
        "energy_std": per_sample_mean.std().item(),
        "energy_min": per_sample_mean.min().item(),
        "energy_max": per_sample_mean.max().item(),
    }


def compute_smooth_l1_energy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Compute smooth L1 energy, matching I-JEPA's loss function."""
    return F.smooth_l1_loss(predictions, targets)
