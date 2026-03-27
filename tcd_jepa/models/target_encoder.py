"""Target encoder with exponential moving average (EMA) updates.

The target encoder is a momentum-updated copy of the context encoder,
following the standard JEPA paradigm from I-JEPA.
"""

import copy
import logging
from typing import Iterator, Optional

import torch
import torch.nn as nn

from tcd_jepa.models.vision_transformer import VisionTransformer

logger = logging.getLogger("tcd_jepa")


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

        Args:
            context_encoder: The context encoder (or its underlying encoder).
            momentum: EMA momentum in [0, 1]. Higher = slower update.
                target = momentum * target + (1 - momentum) * context
        """
        # Handle ContextEncoder wrapper
        if hasattr(context_encoder, "encoder"):
            context_encoder = context_encoder.encoder

        for param_q, param_k in zip(context_encoder.parameters(), self.encoder.parameters()):
            param_k.data.mul_(momentum).add_((1.0 - momentum) * param_q.detach().data)

        # Emergency NaN recovery: if EMA produced NaN, restore from context encoder
        for param_q, param_k in zip(context_encoder.parameters(), self.encoder.parameters()):
            if torch.isnan(param_k.data).any():
                logger.error("NaN detected in target encoder after EMA update — restoring from context encoder")
                param_k.data.copy_(param_q.detach().data)
                break

    def forward(
        self, x: torch.Tensor, masks: Optional[list[torch.Tensor]] = None
    ) -> torch.Tensor:
        """Encode images without gradient tracking.

        Args:
            x: Input images [B, C, H, W].
            masks: Optional list of index tensors for patch selection.

        Returns:
            Encoded patch representations.
        """
        with torch.no_grad():
            return self.encoder(x, masks=masks)

    @property
    def embed_dim(self) -> int:
        return self.encoder.embed_dim

    @property
    def patch_embed(self) -> nn.Module:
        return self.encoder.patch_embed


def momentum_schedule(base_value: float, final_value: float, num_steps: int) -> Iterator[float]:
    """Generate a linear momentum schedule from base_value to final_value.

    Args:
        base_value: Starting momentum (e.g., 0.996).
        final_value: Final momentum (e.g., 1.0).
        num_steps: Total number of training steps.

    Yields:
        Momentum value for each step.
    """
    for i in range(num_steps):
        yield base_value + i * (final_value - base_value) / max(num_steps - 1, 1)
