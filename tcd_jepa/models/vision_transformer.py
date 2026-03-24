"""Vision Transformer for JEPA — production-grade implementation.

Architecture matches I-JEPA (Meta/FAIR) with enhancements:
- Flash Attention (F.scaled_dot_product_attention) for O(N) memory
- Compile-friendly stochastic depth (no data-dependent branching)
- Proper weight initialization matching DeIT/I-JEPA
- Sinusoidal 2D positional embeddings (frozen)
"""

import math
from functools import partial
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tcd_jepa.utils.tensors import apply_masks, trunc_normal_


# ---------------------------------------------------------------------------
# Positional embeddings
# ---------------------------------------------------------------------------

def get_2d_sincos_pos_embed(
    embed_dim: int, grid_size: int, cls_token: bool = False
) -> np.ndarray:
    """Generate 2D sinusoidal positional embeddings."""
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])
    pos_embed = _get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def _get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0
    emb_h = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def _get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


# ---------------------------------------------------------------------------
# Core components
# ---------------------------------------------------------------------------

class DropPath(nn.Module):
    """Stochastic depth — compile-friendly implementation.

    Uses a fixed random mask per sample rather than data-dependent branching,
    which avoids torch.compile guard recompilations.
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor_(random_tensor + keep_prob)
        return x * random_tensor / keep_prob


class MLP(nn.Module):
    """Feed-forward network with GELU activation."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: type = nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """Multi-head self-attention with Flash Attention support.

    Uses F.scaled_dot_product_attention which automatically dispatches
    to Flash Attention 2 on H100/A100 GPUs with bf16/fp16.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop_p = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, N, D]
        q, k, v = qkv.unbind(0)

        # Flash Attention via PyTorch SDPA
        dropout_p = self.attn_drop_p if self.training else 0.0
        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=dropout_p,
            scale=self.scale,
        )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        # Return None for attn weights (not computed with flash attention)
        return x, None


class Block(nn.Module):
    """Transformer block with pre-norm (LayerNorm → Attention/MLP)."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: type = nn.GELU,
        norm_layer: type = nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> torch.Tensor:
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """Image to patch embedding via convolution."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


# ---------------------------------------------------------------------------
# Vision Transformer
# ---------------------------------------------------------------------------

class VisionTransformer(nn.Module):
    """Vision Transformer encoder matching I-JEPA's interface.

    forward(x, masks=None) where masks is an optional list of index
    tensors for patch selection during masked prediction.
    """

    def __init__(
        self,
        img_size: list[int] | int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
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
        self.num_features = self.embed_dim = embed_dim
        self.num_heads = num_heads

        _img_size = img_size[0] if isinstance(img_size, list) else img_size
        self.patch_embed = PatchEmbed(
            img_size=_img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        # Sinusoidal positional embeddings (frozen)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, embed_dim), requires_grad=False
        )
        pos_embed = get_2d_sincos_pos_embed(embed_dim, int(num_patches**0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Stochastic depth schedule (linearly increasing)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
            )
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        self.init_std = init_std
        self.apply(self._init_weights)
        self._fix_init_weight()

    def _fix_init_weight(self) -> None:
        """Rescale attention/MLP projection weights by 1/sqrt(2*layer_depth).

        This is critical for training stability in deep transformers.
        Matches DeIT/I-JEPA initialization.
        """
        for layer_id, layer in enumerate(self.blocks):
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

    def interpolate_pos_encoding(
        self, x: torch.Tensor, pos_embed: torch.Tensor
    ) -> torch.Tensor:
        """Interpolate positional embeddings for different input sizes."""
        npatch = x.shape[1]
        N = pos_embed.shape[1]
        if npatch == N:
            return pos_embed
        dim = x.shape[-1]
        pos_embed = nn.functional.interpolate(
            pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=math.sqrt(npatch / N),
            mode="bicubic",
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return pos_embed

    def forward(
        self, x: torch.Tensor, masks: Optional[list[torch.Tensor]] = None
    ) -> torch.Tensor:
        """Encode images, optionally applying masks for patch selection.

        Args:
            x: Input images [B, C, H, W].
            masks: Optional list of index tensors for patch selection.

        Returns:
            Encoded patch representations [B', N', D].
        """
        if masks is not None and not isinstance(masks, list):
            masks = [masks]

        x = self.patch_embed(x)
        pos_embed = self.interpolate_pos_encoding(x, self.pos_embed)
        x = x + pos_embed

        if masks is not None:
            x = apply_masks(x, masks)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def vit_tiny(patch_size: int = 16, **kwargs) -> VisionTransformer:
    return VisionTransformer(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs,
    )


def vit_small(patch_size: int = 16, **kwargs) -> VisionTransformer:
    return VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs,
    )


def vit_base(patch_size: int = 16, **kwargs) -> VisionTransformer:
    return VisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs,
    )


def vit_large(patch_size: int = 16, **kwargs) -> VisionTransformer:
    return VisionTransformer(
        patch_size=patch_size, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs,
    )


def vit_huge(patch_size: int = 14, **kwargs) -> VisionTransformer:
    return VisionTransformer(
        patch_size=patch_size, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs,
    )


VIT_EMBED_DIMS = {
    "vit_tiny": 192,
    "vit_small": 384,
    "vit_base": 768,
    "vit_large": 1024,
    "vit_huge": 1280,
}
