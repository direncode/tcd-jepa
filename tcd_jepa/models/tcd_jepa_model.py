"""Full TCD-JEPA model — production grade.

Combines context encoder, target encoder, predictor, and optional TCD
dynamic predictor with proper loss computation, target normalization,
and collapse prevention.
"""

from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from tcd_jepa.models.context_encoder import ContextEncoder
from tcd_jepa.models.target_encoder import TargetEncoder
from tcd_jepa.models.vision_transformer import VisionTransformer
from tcd_jepa.modules.predictor import VisionTransformerPredictor
from tcd_jepa.training.losses import jepa_loss, variance_covariance_loss
from tcd_jepa.utils.tensors import apply_masks


class TCDJEPAModel(nn.Module):
    """TCD-JEPA model combining context encoder, target encoder, and predictor.

    Forward pass computes:
    1. Context encoding (masked patches)
    2. Target encoding (no gradient, different masks)
    3. Prediction of targets from context
    4. Loss = JEPA prediction loss + optional collapse prevention
    """

    def __init__(
        self,
        context_encoder: ContextEncoder,
        target_encoder: TargetEncoder,
        predictor: VisionTransformerPredictor,
        use_dynamic_predictor: bool = False,
        embed_dim: int = 192,
        collapse_weight: float = 0.0,
    ):
        super().__init__()
        self.context_encoder = context_encoder
        self.target_encoder = target_encoder
        self.collapse_weight = collapse_weight

        if use_dynamic_predictor:
            from tcd_jepa.modules.dynamic_predictor import DynamicPredictor
            self.predictor = DynamicPredictor(
                base_predictor=predictor,
                embed_dim=embed_dim,
            )
            self._has_dynamic_predictor = True
        else:
            self.predictor = predictor
            self._has_dynamic_predictor = False

    def forward(
        self,
        images: torch.Tensor,
        masks_enc: list[torch.Tensor],
        masks_pred: list[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Run the full JEPA forward pass.

        Args:
            images: Input images [B, C, H, W].
            masks_enc: List of context mask index tensors.
            masks_pred: List of target mask index tensors.

        Returns:
            Dictionary with 'loss', 'predictions', 'targets', and diagnostics.
        """
        # Encode context patches
        z = self.context_encoder(images, masks=masks_enc)

        # Predict target representations
        z_pred = self.predictor(z, masks_enc, masks_pred)

        # Encode targets (no gradient, no compile needed)
        with torch.no_grad():
            h = self.target_encoder(images, masks=masks_pred)
            h = h.repeat(len(masks_enc), 1, 1)

        # JEPA loss with target normalization
        loss = jepa_loss(z_pred, h.detach())

        result = {
            "loss": loss,
            "predictions": z_pred,
            "targets": h,
        }

        # Optional collapse prevention
        if self.collapse_weight > 0.0 and self.training:
            # Pool context representations for collapse detection
            z_pooled = z.mean(dim=1)  # [B*nenc, D]
            collapse = variance_covariance_loss(z_pooled)
            result["loss"] = loss + self.collapse_weight * collapse["total"]
            result["var_loss"] = collapse["var_loss"]
            result["cov_loss"] = collapse["cov_loss"]

        return result

    @torch.no_grad()
    def update_target_encoder(self, momentum: float) -> None:
        """Update target encoder via EMA."""
        self.target_encoder.update_ema(self.context_encoder, momentum)

    def set_module_registry(self, registry) -> None:
        """Connect crystallizer's registry to dynamic predictor."""
        if self._has_dynamic_predictor:
            self.predictor.registry = registry


def build_tcd_jepa(
    img_size: int = 224,
    patch_size: int = 16,
    in_chans: int = 3,
    embed_dim: int = 384,
    depth: int = 12,
    num_heads: int = 6,
    mlp_ratio: float = 4.0,
    predictor_embed_dim: int = 192,
    predictor_depth: int = 6,
    predictor_num_heads: int = 6,
    drop_path_rate: float = 0.1,
    use_dynamic_predictor: bool = False,
    collapse_weight: float = 0.0,
) -> TCDJEPAModel:
    """Build a TCD-JEPA model from hyperparameters.

    Args:
        img_size: Input image size.
        patch_size: Patch size for ViT.
        in_chans: Number of input channels.
        embed_dim: Encoder embedding dimension.
        depth: Number of encoder transformer blocks.
        num_heads: Number of attention heads in encoder.
        mlp_ratio: MLP hidden dim ratio.
        predictor_embed_dim: Predictor embedding dimension.
        predictor_depth: Number of predictor transformer blocks.
        predictor_num_heads: Number of attention heads in predictor.
        drop_path_rate: Stochastic depth rate for encoder.
        use_dynamic_predictor: Enable TCD dynamic predictor.
        collapse_weight: Weight for collapse prevention loss (0=disabled).

    Returns:
        Initialized TCDJEPAModel.
    """
    norm_layer = partial(nn.LayerNorm, eps=1e-6)

    encoder = VisionTransformer(
        img_size=[img_size],
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=True,
        drop_path_rate=drop_path_rate,
        norm_layer=norm_layer,
    )

    context_encoder = ContextEncoder(encoder)
    target_encoder = TargetEncoder(encoder)

    num_patches = encoder.patch_embed.num_patches
    predictor = VisionTransformerPredictor(
        num_patches=num_patches,
        embed_dim=embed_dim,
        predictor_embed_dim=predictor_embed_dim,
        depth=predictor_depth,
        num_heads=predictor_num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=True,
        norm_layer=norm_layer,
    )

    return TCDJEPAModel(
        context_encoder, target_encoder, predictor,
        use_dynamic_predictor=use_dynamic_predictor,
        embed_dim=embed_dim,
        collapse_weight=collapse_weight,
    )
