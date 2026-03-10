"""Dynamic predictor that incorporates crystallized modules from System 3.

Extends the vanilla JEPA predictor with dynamically created module heads.
"""

from typing import Optional

import torch
import torch.nn as nn

from tcd_jepa.modules.module_registry import ModuleRegistry


class DynamicPredictor(nn.Module):
    """Predictor that combines base JEPA predictions with crystallized modules.

    The base predictor handles the standard JEPA prediction task.
    Crystallized modules contribute specialized predictions for regions
    of latent space they were designed for.
    """

    def __init__(
        self,
        base_predictor: nn.Module,
        embed_dim: int,
        module_weight: float = 0.1,
        registry: Optional[ModuleRegistry] = None,
    ) -> None:
        super().__init__()
        self.base_predictor = base_predictor
        self.embed_dim = embed_dim
        self.module_weight = module_weight
        self.registry = registry or ModuleRegistry()

        # Gate that learns to weight module contributions
        self.module_gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
        masks_x: list[torch.Tensor],
        masks: list[torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass combining base predictor and crystallized modules.

        Args:
            x: Encoded context patches [B*nenc, N_ctx, D].
            masks_x: Context mask indices.
            masks: Target mask indices.

        Returns:
            Combined predictions [B*npred*nenc, N_pred, D].
        """
        # Base prediction
        base_pred = self.base_predictor(x, masks_x, masks)

        # If no modules registered, return base prediction
        modules = self.registry.get_all_modules()
        if not modules:
            return base_pred

        # Aggregate module contributions
        # Modules operate on flattened representations
        B, N, D = base_pred.shape
        z_flat = base_pred.reshape(B * N, D)

        module_sum = torch.zeros_like(z_flat)
        num_active = 0
        for module_id, module in modules:
            try:
                contribution = module(z_flat)
                module_sum = module_sum + contribution
                num_active += 1
            except Exception:
                continue

        if num_active > 0:
            module_avg = module_sum / num_active
            module_avg = module_avg.reshape(B, N, D)

            # Learned gating
            gate = self.module_gate(base_pred.detach())
            combined = base_pred + self.module_weight * gate * module_avg
            return combined

        return base_pred

    def get_module_parameters(self) -> list[nn.Parameter]:
        """Get parameters from all registered modules (for optimizer)."""
        params = list(self.module_gate.parameters())
        for _, module in self.registry.get_all_modules():
            params.extend(module.parameters())
        return params
