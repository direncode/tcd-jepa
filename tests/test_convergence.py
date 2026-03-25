"""Convergence and training stability tests.

Verifies that training actually decreases loss over multiple steps,
EMA converges, and the full pipeline is numerically stable.
"""

import torch

from tcd_jepa.models.tcd_jepa_model import build_tcd_jepa
from tcd_jepa.training.guards import check_loss_health, compute_gradient_stats


class TestLossDecreases:
    """Verify that the JEPA loss decreases over multiple training steps."""

    def test_loss_decreases_over_10_steps(self):
        """Loss should decrease when training on repeated data."""
        torch.manual_seed(42)

        model = build_tcd_jepa(
            img_size=32, patch_size=4, embed_dim=64, depth=2, num_heads=2,
            predictor_embed_dim=32, predictor_depth=2, predictor_num_heads=2,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Fixed batch for overfitting test
        images = torch.randn(4, 3, 32, 32)
        masks_enc = [torch.arange(32).unsqueeze(0).expand(4, -1)]
        masks_pred = [torch.arange(32).unsqueeze(0).expand(4, -1)]

        losses = []
        for step in range(20):
            result = model(images, masks_enc, masks_pred)
            loss = result["loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss at end should be lower than at start
        assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
        # Loss should have decreased by at least 10%
        assert losses[-1] < losses[0] * 0.9, f"Loss decrease too small: {losses[0]:.4f} -> {losses[-1]:.4f}"

    def test_no_nan_during_training(self):
        """Training should never produce NaN loss over 20 steps."""
        torch.manual_seed(123)

        model = build_tcd_jepa(
            img_size=32, patch_size=4, embed_dim=64, depth=2, num_heads=2,
            predictor_embed_dim=32, predictor_depth=2, predictor_num_heads=2,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        images = torch.randn(4, 3, 32, 32)
        masks_enc = [torch.arange(32).unsqueeze(0).expand(4, -1)]
        masks_pred = [torch.arange(32).unsqueeze(0).expand(4, -1)]

        for step in range(20):
            result = model(images, masks_enc, masks_pred)
            loss = result["loss"]
            assert check_loss_health(loss), f"Unhealthy loss at step {step}: {loss.item()}"
            optimizer.zero_grad()
            loss.backward()
            stats = compute_gradient_stats(model)
            assert stats.is_healthy, f"Unhealthy gradients at step {step}: nan={stats.nan_count}, inf={stats.inf_count}"
            optimizer.step()


class TestEMAConvergence:
    """Verify EMA target encoder converges toward context encoder."""

    def test_ema_approaches_context(self):
        """With momentum < 1, target params should approach context params after divergence."""
        torch.manual_seed(42)

        model = build_tcd_jepa(
            img_size=32, patch_size=4, embed_dim=64, depth=2, num_heads=2,
            predictor_embed_dim=32, predictor_depth=2, predictor_num_heads=2,
        )

        # Perturb context encoder so it differs from target
        with torch.no_grad():
            for p in model.context_encoder.parameters():
                p.add_(torch.randn_like(p) * 0.1)

        # Measure distance
        def param_distance():
            ctx = model.context_encoder.encoder if hasattr(model.context_encoder, "encoder") else model.context_encoder
            total = 0.0
            count = 0
            for p_c, p_t in zip(ctx.parameters(), model.target_encoder.encoder.parameters()):
                total += (p_c.data - p_t.data).pow(2).sum().item()
                count += p_c.numel()
            return total / count

        initial_dist = param_distance()
        assert initial_dist > 0, "Context and target should differ after perturbation"

        # Apply many EMA updates with low momentum (fast convergence)
        for _ in range(100):
            model.update_target_encoder(momentum=0.9)

        final_dist = param_distance()

        # Distance should decrease significantly
        assert final_dist < initial_dist * 0.5, (
            f"EMA did not converge: {initial_dist:.6f} -> {final_dist:.6f}"
        )

    def test_ema_with_momentum_1_preserves_target(self):
        """With momentum=1.0, target should be unchanged."""
        model = build_tcd_jepa(
            img_size=32, patch_size=4, embed_dim=64, depth=2, num_heads=2,
            predictor_embed_dim=32, predictor_depth=2, predictor_num_heads=2,
        )

        # Save target params
        target_params_before = [p.data.clone() for p in model.target_encoder.parameters()]

        # Update with momentum=1 (no update)
        model.update_target_encoder(momentum=1.0)

        for p_before, p_after in zip(target_params_before, model.target_encoder.parameters()):
            assert torch.allclose(p_before, p_after.data), "momentum=1.0 should not change target"


class TestGradientFlow:
    """Verify gradients flow correctly through the model."""

    def test_all_trained_params_get_gradients(self):
        """Context encoder and predictor params should all receive gradients."""
        model = build_tcd_jepa(
            img_size=32, patch_size=4, embed_dim=64, depth=2, num_heads=2,
            predictor_embed_dim=32, predictor_depth=2, predictor_num_heads=2,
        )

        images = torch.randn(2, 3, 32, 32)
        masks_enc = [torch.arange(32).unsqueeze(0).expand(2, -1)]
        masks_pred = [torch.arange(32).unsqueeze(0).expand(2, -1)]

        result = model(images, masks_enc, masks_pred)
        result["loss"].backward()

        # Check context encoder
        for name, p in model.context_encoder.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for context_encoder.{name}"
                assert not torch.all(p.grad == 0), f"Zero gradient for context_encoder.{name}"

        # Check predictor
        for name, p in model.predictor.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for predictor.{name}"

    def test_target_encoder_no_gradients(self):
        """Target encoder should never have gradients."""
        model = build_tcd_jepa(
            img_size=32, patch_size=4, embed_dim=64, depth=2, num_heads=2,
            predictor_embed_dim=32, predictor_depth=2, predictor_num_heads=2,
        )

        images = torch.randn(2, 3, 32, 32)
        masks_enc = [torch.arange(32).unsqueeze(0).expand(2, -1)]
        masks_pred = [torch.arange(32).unsqueeze(0).expand(2, -1)]

        result = model(images, masks_enc, masks_pred)
        result["loss"].backward()

        for name, p in model.target_encoder.named_parameters():
            assert not p.requires_grad, f"Target encoder param {name} should not require grad"
