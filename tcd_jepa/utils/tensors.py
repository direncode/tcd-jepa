"""Tensor utilities for TCD-JEPA. Adapted from I-JEPA (Meta Platforms, Inc.)."""

import math

import torch


def trunc_normal_(tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0, a: float = -2.0, b: float = 2.0) -> torch.Tensor:
    """Truncated normal initialization. Adapted from I-JEPA."""

    def norm_cdf(x: float) -> float:
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    with torch.no_grad():
        lo = norm_cdf((a - mean) / std)
        hi = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * lo - 1, 2 * hi - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def apply_masks(x: torch.Tensor, masks: list[torch.Tensor]) -> torch.Tensor:
    """Apply index masks to select patches from a tensor.

    Args:
        x: Tensor of shape [B, N, D] (batch, num_patches, feature_dim).
        masks: List of tensors containing patch indices to keep.

    Returns:
        Concatenated masked tensors along the batch dimension.
    """
    all_x = []
    for m in masks:
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
        all_x.append(torch.gather(x, dim=1, index=mask_keep))
    return torch.cat(all_x, dim=0)


def repeat_interleave_batch(x: torch.Tensor, B: int, repeat: int) -> torch.Tensor:
    """Repeat-interleave tensor blocks along batch dimension."""
    N = len(x) // B
    x = torch.cat(
        [torch.cat([x[i * B : (i + 1) * B] for _ in range(repeat)], dim=0) for i in range(N)],
        dim=0,
    )
    return x
