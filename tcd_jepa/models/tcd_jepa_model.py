"""Full TCD-JEPA model combining all components."""

from functools import partial
from typing import Optional

import torch
import torch.nn as nn

from tcd_jepa.models.context_encoder import ContextEncoder
from tcd_jepa.models.target_encoder import TargetEncoder
from tcd_jepa.models.vision_transformer import VisionTransformer
from tcd_jepa.modules.predictor import VisionTransformerPredictor
from tcd_jepa.training.losses import jepa_loss
from tcd_jepa.utils.tensors import apply_masks


class TCDJEPAModel(nn.Module):
    """TCD-JEPA model combining context encoder, target encoder, and predictor.

    When use_dynamic_predictor=True, wraps the base predictor in a
    DynamicPredictor that incorporates crystallized modules from System 3.
    """

    def __init__(
        self,
        context_encoder: ContextEncoder,
        target_encoder: TargetEncoder,
        predictor: VisionTransformerPredictor,
        use_dynamic_predictor: bool = False,
        embed_dim: int = 192,
    ):
        super().__init__()
        self.context_encoder = context_encoder
        self.target_encoder = target_encoder

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
            masks_enc: List of context mask index tensors (patches to keep for encoder).
            masks_pred: List of target mask index tensors (patches to predict).

        Returns:
            Dictionary with 'loss', 'predictions', and 'targets'.
        """
        # Encode context patches
        z = self.context_encoder(images, masks=masks_enc)

        # Predict target representations
        z_pred = self.predictor(z, masks_enc, masks_pred)

        # Encode targets (no gradient)
        with torch.no_grad():
            h = self.target_encoder(images, masks=masks_pred)
            # Repeat targets to match predictor output shape
            h = h.repeat(len(masks_enc), 1, 1)

        # Compute loss
        loss = jepa_loss(z_pred, h.detach())

        return {
            "loss": loss,
            "predictions": z_pred,
            "targets": h,
        }

    @torch.no_grad()
    def update_target_encoder(self, momentum: float) -> None:
        """Update target encoder via EMA."""
        self.target_encoder.update_ema(self.context_encoder, momentum)

    def set_module_registry(self, registry) -> None:
        """Share a ModuleRegistry with the dynamic predictor.

        Call this to connect the crystallizer's registry so that modules
        created by System 3 are automatically used by the predictor.
        """
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
    use_dynamic_predictor: bool = False,
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

    Returns:
        Initialized TCDJEPAModel.
    """
    norm_layer = partial(nn.LayerNorm, eps=1e-6)

    # Build encoder
    encoder = VisionTransformer(
        img_size=[img_size],
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=True,
        norm_layer=norm_layer,
    )

    # Wrap in context encoder (System 1)
    context_encoder = ContextEncoder(encoder)

    # Create EMA target encoder
    target_encoder = TargetEncoder(encoder)

    # Build predictor
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
    )
