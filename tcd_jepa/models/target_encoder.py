"""Target encoder with exponential moving average (EMA) updates.

The target encoder is a momentum-updated copy of the context encoder,
following the I-JEPA paradigm. Uses cosine momentum schedule matching
the FAIR implementation.
"""

import copy
import math
from typing import Iterator, Optional

import torch
import torch.nn as nn

from tcd_jepa.models.vision_transformer import VisionTransformer


class TargetEncoder(nn.Module):
    """EMA target encoder.

    Maintains a copy of the context encoder whose parameters are updated
    via exponential moving average. No gradients flow through this encoder.
    """

    def __init__(self, encoder: VisionTransformer):
        super().__init__()
        self.encoder = copy.deepcopy(encoder)
        # Freeze all parameters
        for p in self.encoder.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update_ema(self, context_encoder: nn.Module, momentum: float) -> None:
        """Update target encoder parameters via EMA.

        target = momentum * target + (1 - momentum) * context
        """
        # Handle ContextEncoder wrapper
        if hasattr(context_encoder, "encoder"):
            context_encoder = context_encoder.encoder

        # Handle torch.compile wrapper
        src = context_encoder
        if hasattr(src, "_orig_mod"):
            src = src._orig_mod

        for param_q, param_k in zip(src.parameters(), self.encoder.parameters()):
            param_k.data.mul_(momentum).add_((1.0 - momentum) * param_q.detach().data)

    def forward(
        self, x: torch.Tensor, masks: Optional[list[torch.Tensor]] = None
    ) -> torch.Tensor:
        """Encode images without gradient tracking."""
        with torch.no_grad():
            return self.encoder(x, masks=masks)

    @property
    def embed_dim(self) -> int:
        return self.encoder.embed_dim

    @property
    def patch_embed(self) -> nn.Module:
        return self.encoder.patch_embed


def cosine_momentum_schedule(
    base_value: float,
    final_value: float,
    num_steps: int,
) -> Iterator[float]:
    """Cosine momentum schedule matching I-JEPA/DINO.

    momentum(t) = final - (final - base) * (cos(pi * t / T) + 1) / 2

    This provides a slow start, fast middle, slow end ramp from
    base_value to final_value — much better than linear for EMA.

    Yields indefinitely (clamps at final_value after num_steps).
    """
    for i in range(num_steps):
        progress = i / max(num_steps - 1, 1)
        value = final_value - (final_value - base_value) * (
            math.cos(math.pi * progress) + 1.0
        ) / 2.0
        yield value
    while True:
        yield final_value


# Keep backward compat alias
def momentum_schedule(
    base_value: float, final_value: float, num_steps: int
) -> Iterator[float]:
    """Linear momentum schedule (deprecated, use cosine_momentum_schedule)."""
    return cosine_momentum_schedule(base_value, final_value, num_steps)
