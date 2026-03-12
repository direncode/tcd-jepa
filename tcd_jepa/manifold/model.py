"""ManifoldJEPA: TCD-JEPA model for Latent Ocean Layer 3 Intelligence.

Drop-in replacement for TCDJEPAModel that uses ManifoldTransformer instead of
VisionTransformer. The JEPA objective is the same — predict target representations
from context — but now operates on manifold entity tokens (384D fingerprints +
S² positions + velocities + causal links) instead of image patches.

The entire TCD pipeline (System 2 exploration, System 3 crystallization, persistent
homology, module factory) works unchanged because it operates on [N, D] point clouds.
"""

import copy
from functools import partial
from typing import Optional

import torch
import torch.nn as nn

from tcd_jepa.manifold.encoder import ManifoldTransformer
from tcd_jepa.modules.predictor import VisionTransformerPredictor
from tcd_jepa.training.losses import jepa_loss
from tcd_jepa.utils.tensors import apply_masks


class ManifoldContextEncoder(nn.Module):
    """Context encoder wrapping ManifoldTransformer with monitoring hooks."""

    def __init__(self, encoder: ManifoldTransformer):
        super().__init__()
        self.encoder = encoder
        self._layer_stats: list[dict[str, float]] = []
        self._hooks: list = []
        self._register_hooks()

    @property
    def embed_dim(self) -> int:
        return self.encoder.embed_dim

    @property
    def num_heads(self) -> int:
        return self.encoder.num_heads

    @property
    def patch_embed(self) -> nn.Module:
        return self.encoder.patch_embed

    def _register_hooks(self) -> None:
        for i, block in enumerate(self.encoder.blocks):
            hook = block.register_forward_hook(self._make_hook(i))
            self._hooks.append(hook)

    def _make_hook(self, layer_idx: int):
        def hook_fn(module, input, output):
            with torch.no_grad():
                if isinstance(output, torch.Tensor):
                    stats = {
                        "mean": output.mean().item(),
                        "std": output.std().item(),
                        "norm": output.norm(dim=-1).mean().item(),
                    }
                    while len(self._layer_stats) <= layer_idx:
                        self._layer_stats.append({})
                    self._layer_stats[layer_idx] = stats
        return hook_fn

    def get_layer_stats(self) -> list[dict]:
        return list(self._layer_stats)

    def forward(
        self,
        x: torch.Tensor,
        masks: Optional[list[torch.Tensor]] = None,
        coords: Optional[torch.Tensor] = None,
        adjacency: Optional[torch.Tensor] = None,
        velocity: Optional[torch.Tensor] = None,
        causal_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self._layer_stats.clear()
        return self.encoder(
            x, masks=masks, coords=coords, adjacency=adjacency,
            velocity=velocity, causal_features=causal_features,
        )

    def parameters(self, recurse=True):
        return self.encoder.parameters(recurse=recurse)

    def named_parameters(self, prefix="", recurse=True):
        return self.encoder.named_parameters(prefix=prefix, recurse=recurse)

    def state_dict(self, *args, **kwargs):
        return self.encoder.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True):
        return self.encoder.load_state_dict(state_dict, strict=strict)

    def modules(self):
        return self.encoder.modules()


class ManifoldTargetEncoder(nn.Module):
    """EMA target encoder for manifold data."""

    def __init__(self, encoder: ManifoldTransformer):
        super().__init__()
        self.encoder = copy.deepcopy(encoder)
        for p in self.encoder.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update_ema(self, context_encoder: nn.Module, momentum: float) -> None:
        source = context_encoder
        if hasattr(context_encoder, "encoder"):
            source = context_encoder.encoder
        for param_q, param_k in zip(source.parameters(), self.encoder.parameters()):
            param_k.data.mul_(momentum).add_((1.0 - momentum) * param_q.detach().data)

    def forward(
        self,
        x: torch.Tensor,
        masks: Optional[list[torch.Tensor]] = None,
        coords: Optional[torch.Tensor] = None,
        adjacency: Optional[torch.Tensor] = None,
        velocity: Optional[torch.Tensor] = None,
        causal_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            return self.encoder(
                x, masks=masks, coords=coords, adjacency=adjacency,
                velocity=velocity, causal_features=causal_features,
            )

    @property
    def embed_dim(self) -> int:
        return self.encoder.embed_dim

    @property
    def patch_embed(self) -> nn.Module:
        return self.encoder.patch_embed


class ManifoldJEPAModel(nn.Module):
    """TCD-JEPA model for Latent Ocean manifold data.

    Same architecture as TCDJEPAModel but with ManifoldTransformer replacing
    VisionTransformer. The predictor and dynamic predictor are reused unchanged.

    The JEPA objective: given context entity tokens (fingerprints at scattered S²
    positions), predict the latent representations of target entity tokens in a
    geodesic neighborhood. This forces the encoder to learn manifold structure,
    causal topology, and temporal dynamics jointly.
    """

    def __init__(
        self,
        context_encoder: ManifoldContextEncoder,
        target_encoder: ManifoldTargetEncoder,
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
                base_predictor=predictor, embed_dim=embed_dim,
            )
            self._has_dynamic_predictor = True
        else:
            self.predictor = predictor
            self._has_dynamic_predictor = False

    def forward(
        self,
        batch_data: dict,
        masks_enc: list[torch.Tensor],
        masks_pred: list[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Run the manifold JEPA forward pass.

        Args:
            batch_data: Dict with 'fingerprints', 'coords', 'velocity', 'adjacency'.
            masks_enc: Context mask indices.
            masks_pred: Target mask indices.

        Returns:
            Dictionary with 'loss', 'predictions', 'targets'.
        """
        fp = batch_data["fingerprints"]
        coords = batch_data.get("coords")
        adj = batch_data.get("adjacency")
        vel = batch_data.get("velocity")
        cf = batch_data.get("causal_features")

        # Encode context
        z = self.context_encoder(fp, masks=masks_enc, coords=coords,
                                  adjacency=adj, velocity=vel, causal_features=cf)

        # Predict targets
        z_pred = self.predictor(z, masks_enc, masks_pred)

        # Encode targets (no gradient)
        with torch.no_grad():
            h = self.target_encoder(fp, masks=masks_pred, coords=coords,
                                     adjacency=adj, velocity=vel, causal_features=cf)
            h = h.repeat(len(masks_enc), 1, 1)

        loss = jepa_loss(z_pred, h.detach())

        return {"loss": loss, "predictions": z_pred, "targets": h}

    @torch.no_grad()
    def update_target_encoder(self, momentum: float) -> None:
        self.target_encoder.update_ema(self.context_encoder, momentum)

    def set_module_registry(self, registry) -> None:
        if self._has_dynamic_predictor:
            self.predictor.registry = registry


def build_manifold_jepa(
    fingerprint_dim: int = 384,
    num_tokens: int = 64,
    coord_dim: int = 3,
    embed_dim: int = 192,
    depth: int = 6,
    num_heads: int = 3,
    mlp_ratio: float = 4.0,
    predictor_embed_dim: int = 96,
    predictor_depth: int = 4,
    predictor_num_heads: int = 3,
    use_dynamic_predictor: bool = True,
    use_causal_encoding: bool = True,
    use_velocity_encoding: bool = True,
    sphere_radius: float = 4.5,
) -> ManifoldJEPAModel:
    """Build a ManifoldJEPA model for Latent Ocean data."""
    norm_layer = partial(nn.LayerNorm, eps=1e-6)

    encoder = ManifoldTransformer(
        fingerprint_dim=fingerprint_dim,
        num_tokens=num_tokens,
        coord_dim=coord_dim,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=True,
        norm_layer=norm_layer,
        use_causal_encoding=use_causal_encoding,
        use_velocity_encoding=use_velocity_encoding,
        sphere_radius=sphere_radius,
    )

    context_encoder = ManifoldContextEncoder(encoder)
    target_encoder = ManifoldTargetEncoder(encoder)

    predictor = VisionTransformerPredictor(
        num_patches=num_tokens,
        embed_dim=embed_dim,
        predictor_embed_dim=predictor_embed_dim,
        depth=predictor_depth,
        num_heads=predictor_num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=True,
        norm_layer=norm_layer,
    )

    return ManifoldJEPAModel(
        context_encoder, target_encoder, predictor,
        use_dynamic_predictor=use_dynamic_predictor,
        embed_dim=embed_dim,
    )
