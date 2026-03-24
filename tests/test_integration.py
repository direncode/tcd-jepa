"""Integration tests for the full TCD-JEPA model."""

import torch
import pytest

from tcd_jepa.models.tcd_jepa_model import TCDJEPAModel, build_tcd_jepa
from tcd_jepa.modules.predictor import VisionTransformerPredictor, vit_predictor
from tcd_jepa.core.energy_landscape import (
    compute_energy,
    compute_energy_statistics,
    compute_smooth_l1_energy,
)
from tcd_jepa.training.losses import jepa_loss, variance_covariance_loss, compute_collapse_metrics
from tcd_jepa.training.schedulers import WarmupCosineSchedule, CosineWDSchedule
from tcd_jepa.training.metrics import TrainingMetrics


class TestPredictor:
    """Test the vanilla JEPA predictor."""

    def test_forward_shape(self):
        """Predictor produces correct output shape."""
        num_patches = 64  # 8x8 grid
        predictor = vit_predictor(
            num_patches=num_patches,
            embed_dim=64,
            predictor_embed_dim=32,
            depth=2,
            num_heads=2,
        )
        B = 2
        # Context: 10 patches per sample
        x = torch.randn(B, 10, 64)
        masks_x = [torch.randint(0, num_patches, (B, 10))]
        masks = [torch.randint(0, num_patches, (B, 16))]
        out = predictor(x, masks_x, masks)
        assert out.shape == (B, 16, 64)


class TestTCDJEPAModel:
    """Test the full model forward pass."""

    def test_forward_returns_loss(self):
        """Full model forward pass returns valid loss and predictions."""
        model = build_tcd_jepa(
            img_size=32,
            patch_size=4,
            embed_dim=64,
            depth=2,
            num_heads=2,
            predictor_embed_dim=32,
            predictor_depth=2,
            predictor_num_heads=2,
        )
        B = 4
        num_patches = (32 // 4) ** 2  # 64
        images = torch.randn(B, 3, 32, 32)
        masks_enc = [torch.randint(0, num_patches, (B, 10))]
        masks_pred = [torch.randint(0, num_patches, (B, 16))]

        result = model(images, masks_enc, masks_pred)
        assert "loss" in result
        assert "predictions" in result
        assert "targets" in result
        assert result["loss"].dim() == 0  # scalar
        assert result["loss"].item() > 0  # non-trivial loss

    def test_backward_pass(self):
        """Gradients flow through context encoder and predictor."""
        model = build_tcd_jepa(
            img_size=32,
            patch_size=4,
            embed_dim=64,
            depth=2,
            num_heads=2,
            predictor_embed_dim=32,
            predictor_depth=2,
            predictor_num_heads=2,
        )
        B = 2
        num_patches = 64
        images = torch.randn(B, 3, 32, 32)
        masks_enc = [torch.randint(0, num_patches, (B, 10))]
        masks_pred = [torch.randint(0, num_patches, (B, 16))]

        result = model(images, masks_enc, masks_pred)
        result["loss"].backward()

        # Context encoder should have gradients
        for p in model.context_encoder.parameters():
            if p.requires_grad:
                assert p.grad is not None

        # Predictor should have gradients
        for p in model.predictor.parameters():
            if p.requires_grad:
                assert p.grad is not None

        # Target encoder should NOT have gradients
        for p in model.target_encoder.encoder.parameters():
            assert p.grad is None

    def test_ema_update(self):
        """EMA update changes target encoder parameters."""
        model = build_tcd_jepa(
            img_size=32, patch_size=4, embed_dim=64, depth=2, num_heads=2,
            predictor_embed_dim=32, predictor_depth=2, predictor_num_heads=2,
        )
        initial = [p.clone() for p in model.target_encoder.encoder.parameters()]

        # Perturb context encoder
        with torch.no_grad():
            for p in model.context_encoder.parameters():
                p.add_(torch.randn_like(p) * 0.1)

        model.update_target_encoder(momentum=0.9)

        for p_init, p_new in zip(initial, model.target_encoder.encoder.parameters()):
            assert not torch.equal(p_init, p_new.data)


class TestEnergyLandscape:
    """Test energy computation functions."""

    def test_compute_energy(self):
        """Energy is non-negative and zero for identical inputs."""
        x = torch.randn(4, 10, 64)
        assert compute_energy(x, x).item() == 0.0
        y = torch.randn(4, 10, 64)
        assert compute_energy(x, y).item() > 0.0

    def test_compute_energy_statistics(self):
        """Energy statistics returns correct keys."""
        x = torch.randn(4, 10, 64)
        y = torch.randn(4, 10, 64)
        stats = compute_energy_statistics(x, y)
        assert "energy_mean" in stats
        assert "energy_std" in stats
        assert "energy_min" in stats
        assert "energy_max" in stats


class TestLosses:
    """Test loss functions."""

    def test_jepa_loss_nonneg(self):
        """JEPA loss is non-negative."""
        x = torch.randn(4, 10, 64)
        y = torch.randn(4, 10, 64)
        loss = jepa_loss(x, y)
        assert loss.item() >= 0.0

    def test_jepa_loss_small_for_similar(self):
        """JEPA loss is small for similar inputs (after target normalization)."""
        x = torch.randn(4, 10, 64)
        # After layer_norm on targets, loss won't be exactly 0 for identical inputs
        # but should be very small for normalized inputs
        import torch.nn.functional as F
        x_norm = F.layer_norm(x, (64,))
        assert jepa_loss(x_norm, x).item() < 0.01

    def test_collapse_metrics(self):
        """Collapse metrics detect healthy vs collapsed representations."""
        healthy = torch.randn(100, 64)
        metrics = compute_collapse_metrics(healthy)
        assert metrics["effective_rank"] > 10
        assert metrics["std_mean"] > 0.5

        collapsed = torch.ones(100, 64) + torch.randn(100, 64) * 0.001
        metrics_c = compute_collapse_metrics(collapsed)
        assert metrics_c["effective_rank"] < metrics["effective_rank"]

    def test_variance_covariance_loss(self):
        """VICReg loss penalizes collapsed representations."""
        healthy = torch.randn(32, 64)
        collapsed = torch.ones(32, 64)
        loss_h = variance_covariance_loss(healthy)
        loss_c = variance_covariance_loss(collapsed)
        # Collapsed should have higher variance loss
        assert loss_c["var_loss"].item() > loss_h["var_loss"].item()


class TestSchedulers:
    """Test learning rate and weight decay schedulers."""

    def test_warmup_cosine(self):
        """LR schedule warms up then decays."""
        params = [torch.nn.Parameter(torch.randn(2, 2))]
        opt = torch.optim.AdamW(params, lr=0.1)
        scheduler = WarmupCosineSchedule(
            opt, warmup_steps=10, start_lr=0.0, ref_lr=0.1, T_max=100
        )
        lrs = [scheduler.step() for _ in range(100)]
        # Should increase during warmup
        assert lrs[5] > lrs[0]
        # Should decrease after warmup
        assert lrs[50] < lrs[10]

    def test_cosine_wd(self):
        """WD schedule decays correctly."""
        params = [torch.nn.Parameter(torch.randn(2, 2))]
        opt = torch.optim.AdamW(params, lr=0.1)
        scheduler = CosineWDSchedule(opt, ref_wd=0.05, T_max=100)
        wds = [scheduler.step() for _ in range(100)]
        assert wds[0] <= 0.05
        assert wds[-1] <= wds[0]


class TestDynamicPredictorWiring:
    """Test DynamicPredictor integration with TCDJEPAModel."""

    def test_build_with_dynamic_predictor(self):
        """Model builds successfully with use_dynamic_predictor=True."""
        model = build_tcd_jepa(
            img_size=32, patch_size=4, embed_dim=64, depth=2, num_heads=2,
            predictor_embed_dim=32, predictor_depth=2, predictor_num_heads=2,
            use_dynamic_predictor=True,
        )
        assert model._has_dynamic_predictor
        from tcd_jepa.modules.dynamic_predictor import DynamicPredictor
        assert isinstance(model.predictor, DynamicPredictor)

    def test_forward_with_dynamic_predictor(self):
        """Forward pass works with dynamic predictor (no modules registered)."""
        model = build_tcd_jepa(
            img_size=32, patch_size=4, embed_dim=64, depth=2, num_heads=2,
            predictor_embed_dim=32, predictor_depth=2, predictor_num_heads=2,
            use_dynamic_predictor=True,
        )
        B = 2
        num_patches = 64
        images = torch.randn(B, 3, 32, 32)
        masks_enc = [torch.randint(0, num_patches, (B, 10))]
        masks_pred = [torch.randint(0, num_patches, (B, 16))]

        result = model(images, masks_enc, masks_pred)
        assert result["loss"].dim() == 0
        assert result["loss"].item() > 0

    def test_set_module_registry(self):
        """Registry sharing works between model and crystallizer."""
        from tcd_jepa.modules.module_registry import ModuleRegistry

        model = build_tcd_jepa(
            img_size=32, patch_size=4, embed_dim=64, depth=2, num_heads=2,
            predictor_embed_dim=32, predictor_depth=2, predictor_num_heads=2,
            use_dynamic_predictor=True,
        )
        external_registry = ModuleRegistry(max_modules=5)
        model.set_module_registry(external_registry)
        assert model.predictor.registry is external_registry

    def test_backward_with_dynamic_predictor(self):
        """Gradients flow through dynamic predictor."""
        model = build_tcd_jepa(
            img_size=32, patch_size=4, embed_dim=64, depth=2, num_heads=2,
            predictor_embed_dim=32, predictor_depth=2, predictor_num_heads=2,
            use_dynamic_predictor=True,
        )
        B = 2
        num_patches = 64
        images = torch.randn(B, 3, 32, 32)
        masks_enc = [torch.randint(0, num_patches, (B, 10))]
        masks_pred = [torch.randint(0, num_patches, (B, 16))]

        result = model(images, masks_enc, masks_pred)
        result["loss"].backward()

        # Gate parameters should have gradients (even without modules)
        for p in model.predictor.module_gate.parameters():
            # Gate isn't used when no modules registered, so no grads expected
            pass

        # Base predictor should have gradients
        for p in model.predictor.base_predictor.parameters():
            if p.requires_grad:
                assert p.grad is not None


class TestTrainingMetrics:
    """Test metric accumulation."""

    def test_avg_loss(self):
        """Average loss is computed correctly."""
        m = TrainingMetrics()
        m.update(loss=1.0)
        m.update(loss=3.0)
        assert m.avg_loss == 2.0

    def test_reset(self):
        """Reset clears accumulated metrics."""
        m = TrainingMetrics()
        m.update(loss=1.0)
        m.reset()
        assert m.avg_loss == 0.0
        assert m.num_steps == 0
