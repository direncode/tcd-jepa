"""Tests for System 1 (context encoder) and related components."""

import torch
import pytest

from tcd_jepa.models.vision_transformer import VisionTransformer, vit_small, vit_tiny
from tcd_jepa.models.context_encoder import ContextEncoder
from tcd_jepa.models.target_encoder import TargetEncoder, momentum_schedule
from tcd_jepa.utils.tensors import apply_masks, trunc_normal_


class TestVisionTransformer:
    """Test the ViT encoder."""

    def test_forward_no_mask(self):
        """ViT produces correct output shape without masks."""
        model = VisionTransformer(img_size=[32], patch_size=4, embed_dim=64, depth=2, num_heads=2)
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        num_patches = (32 // 4) ** 2  # 64
        assert out.shape == (2, num_patches, 64)

    def test_forward_with_mask(self):
        """ViT produces correct output shape with masks."""
        model = VisionTransformer(img_size=[32], patch_size=4, embed_dim=64, depth=2, num_heads=2)
        x = torch.randn(2, 3, 32, 32)
        # Keep 10 patches per sample
        masks = [torch.randint(0, 64, (2, 10))]
        out = model(x, masks=masks)
        assert out.shape == (2, 10, 64)

    def test_factory_functions(self):
        """Factory functions create models with correct dimensions."""
        model = vit_tiny(img_size=[32], patch_size=4)
        assert model.embed_dim == 192
        x = torch.randn(1, 3, 32, 32)
        out = model(x)
        assert out.shape[2] == 192


class TestContextEncoder:
    """Test the instrumented context encoder."""

    def test_forward_delegates_to_vit(self):
        """ContextEncoder forward pass produces same shape as raw ViT."""
        vit = VisionTransformer(img_size=[32], patch_size=4, embed_dim=64, depth=2, num_heads=2)
        encoder = ContextEncoder(vit)
        x = torch.randn(2, 3, 32, 32)
        out = encoder(x)
        assert out.shape == (2, 64, 64)

    def test_collects_layer_stats(self):
        """ContextEncoder records statistics for each layer."""
        vit = VisionTransformer(img_size=[32], patch_size=4, embed_dim=64, depth=3, num_heads=2)
        encoder = ContextEncoder(vit, collect_stats=True)
        x = torch.randn(2, 3, 32, 32)
        encoder(x)
        stats = encoder.get_layer_stats()
        assert len(stats) == 3  # one per block
        for s in stats:
            assert "mean" in s
            assert "std" in s
            assert "norm" in s

    def test_clear_stats(self):
        """Stats are cleared between forward passes."""
        vit = VisionTransformer(img_size=[32], patch_size=4, embed_dim=64, depth=2, num_heads=2)
        encoder = ContextEncoder(vit, collect_stats=True)
        x = torch.randn(2, 3, 32, 32)
        encoder(x)
        assert len(encoder.get_layer_stats()) == 2
        encoder.clear_stats()
        assert len(encoder.get_layer_stats()) == 0


class TestTargetEncoder:
    """Test the EMA target encoder."""

    def test_ema_update(self):
        """EMA update moves target encoder toward context encoder."""
        vit = VisionTransformer(img_size=[32], patch_size=4, embed_dim=64, depth=2, num_heads=2)
        context = ContextEncoder(vit)
        target = TargetEncoder(vit)

        # Save initial target params
        initial_params = [p.clone() for p in target.encoder.parameters()]

        # Mutate context encoder
        with torch.no_grad():
            for p in context.parameters():
                p.add_(torch.randn_like(p))

        # EMA update with momentum 0.5
        target.update_ema(context, momentum=0.5)

        # Verify parameters changed
        for p_init, p_new in zip(initial_params, target.encoder.parameters()):
            assert not torch.equal(p_init, p_new.data)

    def test_no_grad(self):
        """Target encoder parameters have no gradient."""
        vit = VisionTransformer(img_size=[32], patch_size=4, embed_dim=64, depth=2, num_heads=2)
        target = TargetEncoder(vit)
        for p in target.encoder.parameters():
            assert not p.requires_grad

    def test_momentum_schedule(self):
        """Momentum schedule produces correct range."""
        schedule = list(momentum_schedule(0.996, 1.0, 100))
        assert len(schedule) == 100
        assert abs(schedule[0] - 0.996) < 1e-6
        assert abs(schedule[-1] - 1.0) < 1e-6


class TestTensorUtils:
    """Test tensor utility functions."""

    def test_apply_masks(self):
        """apply_masks selects correct patches."""
        x = torch.randn(2, 10, 8)  # [B, N, D]
        masks = [torch.arange(5).unsqueeze(0).repeat(2, 1)]  # keep first 5
        out = apply_masks(x, masks)
        assert out.shape == (2, 5, 8)
        assert torch.equal(out, x[:, :5, :])

    def test_trunc_normal(self):
        """trunc_normal_ produces values within bounds."""
        t = torch.empty(1000)
        trunc_normal_(t, mean=0, std=1, a=-2, b=2)
        assert t.min() >= -2
        assert t.max() <= 2
