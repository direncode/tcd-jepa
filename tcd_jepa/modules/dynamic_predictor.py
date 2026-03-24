"""Dynamic predictor that incorporates crystallized modules from System 3.

Extends the vanilla JEPA predictor with dynamically created module heads.
Uses a learned soft router (ModuleRouter) with per-module weights via softmax,
per-token gating for spatial selectivity, and output normalization.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from tcd_jepa.modules.module_registry import ModuleRegistry

logger = logging.getLogger("tcd_jepa")


class ModuleRouter(nn.Module):
    """Learned soft router that assigns per-module weights via softmax.

    Routes input representations to crystallized modules with learned
    importance weights, enabling gradient-based module selection.
    """

    def __init__(self, embed_dim: int, max_modules: int = 64) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.max_modules = max_modules
        self.router_proj = nn.Linear(embed_dim, max_modules)
        nn.init.zeros_(self.router_proj.weight)
        nn.init.zeros_(self.router_proj.bias)

    def forward(self, x: torch.Tensor, num_active: int) -> torch.Tensor:
        """Compute soft routing weights over active modules.

        Args:
            x: Input representations [B, D].
            num_active: Number of currently active modules.

        Returns:
            Routing weights [B, num_active] summing to 1 per sample.
        """
        logits = self.router_proj(x)[:, :num_active]  # [B, num_active]
        return F.softmax(logits, dim=-1)


class DynamicPredictor(nn.Module):
    """Predictor that combines base JEPA predictions with crystallized modules.

    The base predictor handles the standard JEPA prediction task.
    Crystallized modules contribute specialized predictions for regions
    of latent space they were designed for.

    Architecture:
        combined = base_pred + alpha * token_gate(base_pred) * module_norm(routed_modules)
    where alpha is a learned scalar, token_gate provides spatial selectivity,
    and module_norm normalizes module outputs.
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

        # Learnable mixing coefficient — start very small so new modules
        # don't immediately corrupt predictions
        init_logit = torch.log(torch.tensor(module_weight / (1.0 - module_weight + 1e-8)))
        self.module_logit = nn.Parameter(init_logit)

        # Learned soft router for per-module weighting
        self.module_router = ModuleRouter(embed_dim)

        # Per-token gate for spatial selectivity
        self.token_gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 1),
            nn.Sigmoid(),
        )

        # Output normalization for module contributions
        self.module_norm = nn.LayerNorm(embed_dim)

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
        # Base prediction — no exception swallowing, let errors propagate
        base_pred = self.base_predictor(x, masks_x, masks)

        # If no modules registered, return base prediction
        modules = self.registry.get_all_modules()
        if not modules:
            return base_pred

        # Aggregate module contributions with learned routing
        B, N, D = base_pred.shape
        z_flat = base_pred.reshape(B * N, D)

        # Compute per-token routing weights (each token gets distinct routing)
        routing_weights = self.module_router(z_flat, len(modules))  # [B*N, M]

        # Compute weighted module contributions
        module_outputs = []
        for module_id, module in modules:
            contribution = module(z_flat)  # [B*N, D]
            module_outputs.append(contribution)

        if module_outputs:
            # Stack and apply routing: [B*N, M, D] * [B*N, M, 1] -> [B*N, D]
            stacked = torch.stack(module_outputs, dim=1)  # [B*N, M, D]
            weights = routing_weights.unsqueeze(-1)  # [B*N, M, 1]
            module_combined = (stacked * weights).sum(dim=1)  # [B*N, D]

            # Normalize module output
            module_combined = self.module_norm(module_combined)
            module_combined = module_combined.reshape(B, N, D)

            # Learned per-token gating for spatial selectivity
            gate = self.token_gate(base_pred)  # [B, N, 1]
            alpha = self.module_weight
            combined = base_pred + alpha * gate * module_combined
            return combined

        return base_pred

    def get_module_parameters(self) -> list[nn.Parameter]:
        """Get parameters from all registered modules (for optimizer)."""
        params = list(self.token_gate.parameters())
        params.extend(self.module_router.parameters())
        params.extend(self.module_norm.parameters())
        params.append(self.module_logit)
        for _, module in self.registry.get_all_modules():
            params.extend(module.parameters())
        return params
