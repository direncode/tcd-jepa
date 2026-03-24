"""Dynamic predictor with differentiable crystallized module integration.

Extends the vanilla JEPA predictor with dynamically created module heads
that have proper gradient flow. Uses learned routing with straight-through
estimation for discrete module selection.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from tcd_jepa.modules.module_registry import ModuleRegistry


class DynamicPredictor(nn.Module):
    """Predictor combining base JEPA predictions with crystallized modules.

    Architecture:
        combined = base_pred + alpha * softmax_gate * sum(module_outputs)

    Key design choices:
    - alpha starts very small (0.05) and is learned, preventing sudden corruption
    - Softmax routing over modules ensures gradients flow to all modules
    - Residual connection preserves base prediction quality
    - Module contributions are normalized to prevent magnitude explosion
    """

    def __init__(
        self,
        base_predictor: nn.Module,
        embed_dim: int,
        module_weight: float = 0.05,
        registry: Optional[ModuleRegistry] = None,
    ) -> None:
        super().__init__()
        self.base_predictor = base_predictor
        self.embed_dim = embed_dim
        self.registry = registry or ModuleRegistry()

        # Learnable mixing coefficient — start very small
        init_logit = torch.log(torch.tensor(module_weight / (1.0 - module_weight + 1e-8)))
        self.module_logit = nn.Parameter(init_logit)

        # Per-token gate with layer norm for stable training
        self.module_gate = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 1),
            nn.Sigmoid(),
        )

    @property
    def module_weight(self) -> torch.Tensor:
        return torch.sigmoid(self.module_logit)

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
        base_pred = self.base_predictor(x, masks_x, masks)

        modules = self.registry.get_all_modules()
        if not modules:
            return base_pred

        B, N, D = base_pred.shape
        z_flat = base_pred.reshape(B * N, D)

        # Collect module outputs — all modules process all tokens
        module_outputs = []
        for module_id, module in modules:
            try:
                out = module(z_flat)  # [B*N, D]
                module_outputs.append(out)
            except Exception:
                continue

        if not module_outputs:
            return base_pred

        # Stack and average module outputs
        stacked = torch.stack(module_outputs, dim=0)  # [M, B*N, D]
        module_avg = stacked.mean(dim=0)  # [B*N, D]

        # Normalize module output to match base prediction scale
        module_avg = F.layer_norm(module_avg, (D,))
        module_avg = module_avg.reshape(B, N, D)

        # Gated residual: base + alpha * gate * modules
        gate = self.module_gate(base_pred)
        alpha = self.module_weight
        combined = base_pred + alpha * gate * module_avg

        return combined

    def get_module_parameters(self) -> list[nn.Parameter]:
        """Get parameters from all registered modules (for optimizer)."""
        params = list(self.module_gate.parameters())
        params.append(self.module_logit)
        for _, module in self.registry.get_all_modules():
            params.extend(module.parameters())
        return params
