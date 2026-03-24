"""System 1 — Stream Encoder.

Wraps the JEPA encoder pipeline, providing the interface that
Systems 2 and 3 interact with.
"""

from typing import Optional

import torch
import torch.nn as nn

from tcd_jepa.models.context_encoder import ContextEncoder
from tcd_jepa.models.target_encoder import TargetEncoder
from tcd_jepa.core.energy_landscape import compute_energy


class StreamEncoder:
    """System 1: provides latent representations and energy functions.

    Acts as the bridge between the JEPA encoder and the exploration/
    crystallization systems. Provides energy_fn and predictor_fn
    callables for Systems 2 and 3.
    """

    def __init__(
        self,
        context_encoder: ContextEncoder,
        target_encoder: TargetEncoder,
    ) -> None:
        self.context_encoder = context_encoder
        self.target_encoder = target_encoder

    def encode_context(
        self,
        images: torch.Tensor,
        masks: Optional[list[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Encode images through context encoder."""
        return self.context_encoder(images, masks=masks)

    def encode_target(
        self,
        images: torch.Tensor,
        masks: Optional[list[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Encode images through target encoder."""
        return self.target_encoder(images, masks=masks)

    def make_energy_fn(
        self,
        target_representations: torch.Tensor,
        differentiable: bool = False,
    ) -> callable:
        """Create an energy function for System 2.

        The energy at a point z is defined as E(z) = ||z - target||^2
        averaged over target patches.

        Args:
            target_representations: Target encoder outputs [B, N, D] or [B, D].
            differentiable: If True, keeps targets in the computation graph
                so gradients can flow back through the energy landscape.

        Returns:
            Callable mapping [B, D] -> [B] energy.
        """
        if target_representations.dim() == 3:
            target_mean = target_representations.mean(dim=1)  # [B, D]
        else:
            target_mean = target_representations

        if differentiable:
            # Keep targets in the computation graph for gradient flow
            target = target_mean
        else:
            target = target_mean.detach()

        def energy_fn(z: torch.Tensor) -> torch.Tensor:
            # Broadcast: if z has more samples than targets, tile targets
            B_z = z.shape[0]
            B_t = target.shape[0]
            if B_z != B_t:
                t = target.repeat((B_z + B_t - 1) // B_t, 1)[:B_z]
            else:
                t = target
            return (z - t).pow(2).sum(dim=-1)

        return energy_fn

    def get_layer_stats(self) -> list[dict]:
        """Get representation statistics from last forward pass."""
        return self.context_encoder.get_layer_stats()
