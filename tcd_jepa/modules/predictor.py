"""Vanilla JEPA predictor. Compatible with I-JEPA's VisionTransformerPredictor.

This is the base predictor that System 3's dynamic predictor will extend.
Architecture adapted from I-JEPA (Meta Platforms, Inc.).
"""

import math
from functools import partial
from typing import Optional

import torch
import torch.nn as nn

from tcd_jepa.models.vision_transformer import Block, get_2d_sincos_pos_embed
from tcd_jepa.utils.tensors import apply_masks, repeat_interleave_batch, trunc_normal_


class VisionTransformerPredictor(nn.Module):
    """Transformer-based predictor that maps context embeddings to predicted target embeddings.

    Takes encoded context patches and mask token positions, predicts target patch
    representations in the encoder's embedding space.

    The forward() signature matches I-JEPA: forward(x, masks_x, masks).
    """

    def __init__(
        self,
        num_patches: int,
        embed_dim: int = 768,
        predictor_embed_dim: int = 384,
        depth: int = 6,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: type = nn.LayerNorm,
        init_std: float = 0.02,
        **kwargs,
    ):
        super().__init__()
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # Positional embeddings for predictor
        # Use 2D sincos for perfect-square token counts (vision), learnable otherwise (manifold)
        grid_size = int(num_patches**0.5)
        if grid_size * grid_size == num_patches:
            self.predictor_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, predictor_embed_dim), requires_grad=False
            )
            predictor_pos_embed = get_2d_sincos_pos_embed(
                predictor_embed_dim, grid_size, cls_token=False
            )
            self.predictor_pos_embed.data.copy_(
                torch.from_numpy(predictor_pos_embed).float().unsqueeze(0)
            )
        else:
            # Learnable positional embeddings for non-grid token layouts (manifold data)
            self.predictor_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, predictor_embed_dim)
            )
            nn.init.trunc_normal_(self.predictor_pos_embed, std=0.02)

        self.predictor_blocks = nn.ModuleList([
            Block(
                dim=predictor_embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
            )
            for i in range(depth)
        ])
        self.predictor_norm = norm_layer(predictor_embed_dim)
        # Project back to encoder embedding dimension
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)

        self.init_std = init_std
        trunc_normal_(self.mask_token, std=init_std)
        self.apply(self._init_weights)
        self._fix_init_weight()

    def _fix_init_weight(self) -> None:
        for layer_id, layer in enumerate(self.predictor_blocks):
            layer.attn.proj.weight.data.div_(math.sqrt(2.0 * (layer_id + 1)))
            layer.mlp.fc2.weight.data.div_(math.sqrt(2.0 * (layer_id + 1)))

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        masks_x: list[torch.Tensor],
        masks: list[torch.Tensor],
    ) -> torch.Tensor:
        """Predict target representations from context representations.

        Args:
            x: Encoded context patches [B*nenc, N_ctx, D].
            masks_x: List of context mask index tensors.
            masks: List of target mask index tensors (positions to predict).

        Returns:
            Predicted target representations [B*npred*nenc, N_pred, D].
        """
        assert masks is not None and masks_x is not None, (
            "Cannot run predictor without mask indices"
        )

        if not isinstance(masks_x, list):
            masks_x = [masks_x]
        if not isinstance(masks, list):
            masks = [masks]

        B = len(x) // len(masks_x)

        # Map from encoder-dim to predictor-dim
        x = self.predictor_embed(x)

        # Add positional embedding to context tokens
        x_pos_embed = self.predictor_pos_embed.repeat(B, 1, 1)
        x += apply_masks(x_pos_embed, masks_x)

        _, N_ctxt, D = x.shape

        # Prepare mask tokens with positional embeddings for target positions
        pos_embs = self.predictor_pos_embed.repeat(B, 1, 1)
        pos_embs = apply_masks(pos_embs, masks)
        pos_embs = repeat_interleave_batch(pos_embs, B, repeat=len(masks_x))

        pred_tokens = self.mask_token.repeat(pos_embs.size(0), pos_embs.size(1), 1)
        pred_tokens += pos_embs

        # Concatenate context and mask tokens
        x = x.repeat(len(masks), 1, 1)
        x = torch.cat([x, pred_tokens], dim=1)

        # Forward through predictor transformer
        for blk in self.predictor_blocks:
            x = blk(x)
        x = self.predictor_norm(x)

        # Return only predictions for mask token positions
        x = x[:, N_ctxt:]
        x = self.predictor_proj(x)
        return x


def vit_predictor(**kwargs) -> VisionTransformerPredictor:
    """Factory function for creating a predictor with standard settings."""
    return VisionTransformerPredictor(
        mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
