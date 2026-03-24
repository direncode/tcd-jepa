"""Loss functions for JEPA training.

Implements the I-JEPA loss with proper target normalization and
representation collapse prevention (variance-covariance regularization).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def jepa_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    loss_type: str = "smooth_l1",
) -> torch.Tensor:
    """Standard JEPA loss between predicted and target embeddings.

    Targets are layer-normalized per-patch before loss computation,
    matching the I-JEPA paper (Section 3.2).

    Args:
        predictions: Predicted target embeddings [B, N, D].
        targets: Target encoder outputs (detached) [B, N, D].
        loss_type: "smooth_l1" (default, matches I-JEPA) or "mse".

    Returns:
        Scalar loss value.
    """
    # Normalize targets per-patch (critical for stability)
    targets = F.layer_norm(targets, (targets.shape[-1],))

    if loss_type == "smooth_l1":
        return F.smooth_l1_loss(predictions, targets)
    elif loss_type == "mse":
        return F.mse_loss(predictions, targets)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")


def variance_covariance_loss(
    z: torch.Tensor,
    gamma_var: float = 1.0,
    gamma_cov: float = 0.04,
    eps: float = 1e-4,
) -> dict[str, torch.Tensor]:
    """VICReg-style variance-covariance regularization to prevent collapse.

    Prevents two common failure modes:
    1. Variance collapse: all representations converge to the same point
    2. Dimensional collapse: representations span a low-dimensional subspace

    Args:
        z: Representations [B, D] (mean-pooled over patches).
        gamma_var: Weight for variance term.
        gamma_cov: Weight for covariance term.
        eps: Minimum variance threshold.

    Returns:
        Dict with 'var_loss', 'cov_loss', and 'total' collapse prevention loss.
    """
    B, D = z.shape

    # Variance loss: penalize dimensions with variance below eps
    z_centered = z - z.mean(dim=0)
    std = torch.sqrt(z.var(dim=0) + 1e-6)
    var_loss = torch.mean(F.relu(eps - std))

    # Covariance loss: penalize off-diagonal correlations
    cov = (z_centered.T @ z_centered) / max(B - 1, 1)
    # Zero the diagonal (we only penalize off-diagonal)
    cov_off_diag = cov - torch.diag(cov.diag())
    cov_loss = (cov_off_diag ** 2).sum() / D

    total = gamma_var * var_loss + gamma_cov * cov_loss

    return {
        "var_loss": var_loss,
        "cov_loss": cov_loss,
        "total": total,
    }


def compute_collapse_metrics(z: torch.Tensor) -> dict[str, float]:
    """Compute metrics to detect representation collapse.

    Args:
        z: Representations [B, D] (detached, for monitoring).

    Returns:
        Dict with collapse indicators:
        - 'effective_rank': Effective dimensionality (higher is better, max=D)
        - 'std_mean': Mean per-dimension std (higher is better, near 0 = collapsed)
        - 'std_min': Minimum per-dimension std (0 = at least one dim collapsed)
        - 'uniformity': Log uniformity (more negative = more uniform = better)
    """
    with torch.no_grad():
        B, D = z.shape

        # Per-dimension statistics
        std = z.std(dim=0)
        std_mean = std.mean().item()
        std_min = std.min().item()

        # Effective rank via SVD
        centered = z - z.mean(dim=0)
        try:
            S = torch.linalg.svdvals(centered)
            S_norm = S / (S.sum() + 1e-8)
            entropy = -(S_norm * (S_norm + 1e-8).log()).sum()
            effective_rank = entropy.exp().item()
        except Exception:
            effective_rank = 0.0

        # Uniformity (Wang & Isola, 2020)
        z_norm = F.normalize(z, dim=1)
        n = min(B, 512)
        sq_dists = torch.cdist(z_norm[:n], z_norm[:n]).pow(2)
        uniformity = (-2.0 * sq_dists).exp().mean().log().item()

        return {
            "effective_rank": effective_rank,
            "std_mean": std_mean,
            "std_min": std_min,
            "uniformity": uniformity,
        }
