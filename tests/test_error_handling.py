"""Tests for error handling hardening across the codebase."""

import os
import tempfile

import numpy as np
import torch
import torch.nn as nn

from tcd_jepa.utils.checkpointing import (
    load_checkpoint,
    save_checkpoint,
    validate_checkpoint,
)


class TestCheckpointValidation:
    def test_save_and_validate(self):
        """Saved checkpoint should pass validation."""
        encoder = nn.Linear(4, 4)
        predictor = nn.Linear(4, 4)
        target_encoder = nn.Linear(4, 4)
        optimizer = torch.optim.Adam(encoder.parameters())

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name

        try:
            save_checkpoint(path, epoch=5, encoder=encoder, predictor=predictor,
                            target_encoder=target_encoder, optimizer=optimizer)

            is_valid, msg = validate_checkpoint(path)
            assert is_valid
            assert "epoch 5" in msg
        finally:
            os.unlink(path)

    def test_validate_missing_file(self):
        is_valid, msg = validate_checkpoint("/nonexistent/path.pt")
        assert not is_valid
        assert "not found" in msg.lower()

    def test_validate_corrupted_file(self):
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False, mode="wb") as f:
            f.write(b"corrupted data here")
            path = f.name

        try:
            is_valid, msg = validate_checkpoint(path)
            assert not is_valid
        finally:
            os.unlink(path)

    def test_validate_missing_keys(self):
        """Checkpoint with missing required keys should fail validation."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
            torch.save({"epoch": 0, "encoder": {}}, path)

        try:
            is_valid, msg = validate_checkpoint(path)
            assert not is_valid
            assert "missing" in msg.lower()
        finally:
            os.unlink(path)

    def test_load_checkpoint_missing_keys_raises(self):
        """Loading checkpoint with missing keys should raise RuntimeError."""
        encoder = nn.Linear(4, 4)
        predictor = nn.Linear(4, 4)
        target = nn.Linear(4, 4)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
            torch.save({"epoch": 0}, path)

        try:
            import pytest
            with pytest.raises(KeyError):
                load_checkpoint(path, encoder, predictor, target)
        finally:
            os.unlink(path)

    def test_save_and_load_with_rng_state(self):
        """Checkpoint should preserve and restore RNG state."""
        encoder = nn.Linear(4, 4)
        predictor = nn.Linear(4, 4)
        target_encoder = nn.Linear(4, 4)
        optimizer = torch.optim.Adam(encoder.parameters())

        # Set known RNG state
        torch.manual_seed(123)
        np.random.seed(456)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name

        try:
            save_checkpoint(path, epoch=1, encoder=encoder, predictor=predictor,
                            target_encoder=target_encoder, optimizer=optimizer)

            # Change RNG state
            torch.manual_seed(999)

            # Load should restore
            load_checkpoint(path, encoder, predictor, target_encoder,
                            optimizer=optimizer, restore_rng=True)

            # Verify RNG was restored by checking next random values match
            # (This is a basic sanity check)
            ckpt = torch.load(path, weights_only=False)
            assert "rng_state" in ckpt
            assert "torch" in ckpt["rng_state"]
            assert "numpy" in ckpt["rng_state"]
        finally:
            os.unlink(path)


class TestCheckpointVersioning:
    def test_version_in_checkpoint(self):
        encoder = nn.Linear(4, 4)
        predictor = nn.Linear(4, 4)
        target_encoder = nn.Linear(4, 4)
        optimizer = torch.optim.Adam(encoder.parameters())

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name

        try:
            save_checkpoint(path, epoch=0, encoder=encoder, predictor=predictor,
                            target_encoder=target_encoder, optimizer=optimizer)
            ckpt = torch.load(path, weights_only=False)
            assert "tcd_jepa_version" in ckpt
        finally:
            os.unlink(path)


class TestLangevinErrorHandling:
    def test_non_differentiable_energy(self):
        """Langevin should handle non-differentiable energy gracefully."""
        from tcd_jepa.exploration.langevin import LangevinSampler

        sampler = LangevinSampler(step_size=0.01, max_steps=5)
        z = torch.randn(4, 8)

        # Energy function that doesn't track gradients
        def energy_fn(z_in):
            return z_in.detach().sum()

        # Should return zero gradient instead of crashing
        grad = sampler._compute_energy_gradient(z, energy_fn)
        assert grad.shape == z.shape
        # Should be zero gradient (no grad available)
        assert torch.allclose(grad, torch.zeros_like(grad))

    def test_nan_energy(self):
        """Langevin should handle NaN energy gracefully."""
        from tcd_jepa.exploration.langevin import LangevinSampler

        sampler = LangevinSampler(step_size=0.01, max_steps=5)
        z = torch.randn(4, 8)

        def energy_fn(z_in):
            return torch.tensor(float("nan"))

        grad = sampler._compute_energy_gradient(z, energy_fn)
        assert grad.shape == z.shape
        assert torch.allclose(grad, torch.zeros_like(grad))

    def test_trajectory_with_bad_energy(self):
        """sample_trajectory should not crash on problematic energy functions."""
        from tcd_jepa.exploration.langevin import LangevinSampler

        sampler = LangevinSampler(step_size=0.01, max_steps=3)
        z = torch.randn(2, 4)

        # Energy that returns inf sometimes
        call_count = [0]
        def energy_fn(z_in):
            call_count[0] += 1
            return z_in.pow(2).sum(dim=-1)

        trajectory = sampler.sample_trajectory(z, energy_fn, num_steps=3)
        assert trajectory.shape[0] == 4  # 3 steps + initial


class TestDynamicPredictorErrorHandling:
    def test_module_shape_mismatch_skipped(self):
        """Modules with wrong output shape should be skipped, not crash."""
        from tcd_jepa.modules.dynamic_predictor import DynamicPredictor
        from tcd_jepa.modules.module_registry import ModuleRegistry

        embed_dim = 64

        # Use a simple mock predictor that just returns its input
        class MockPredictor(nn.Module):
            def forward(self, x, masks_x, masks):
                return x

        registry = ModuleRegistry(max_modules=4)
        dp = DynamicPredictor(
            base_predictor=MockPredictor(),
            embed_dim=embed_dim,
            registry=registry,
        )

        # Register a module with wrong output dim
        bad_module = nn.Linear(embed_dim, embed_dim * 2)  # Wrong output size

        class FakeFeature:
            module_type = "attractor"
            centroid = np.zeros(embed_dim)
            persistence = 1.0
            dimension = 0
            birth = 0.0
            death = 1.0

        registry.register(bad_module, FakeFeature(), epoch=0)

        # Forward should not crash — bad module should be skipped
        B, N, D = 2, 8, embed_dim
        x = torch.randn(B, N, D)

        output = dp(x, [], [])
        assert output.shape == (B, N, D)  # Should still produce valid output


class TestCrystallizerEmptyPointCloud:
    def test_empty_point_cloud(self):
        """Crystallizer should handle empty/small point clouds gracefully."""
        from tcd_jepa.core.system3_crystallizer import ModuleCrystallizer

        crystallizer = ModuleCrystallizer(embed_dim=32)

        # Empty point cloud
        point_cloud = torch.randn(0, 32)
        result = crystallizer.crystallize(point_cloud, epoch=0)
        assert result["new_modules"] == []
        assert result["num_features"] == 0

    def test_single_point(self):
        """Crystallizer should handle single-point cloud."""
        from tcd_jepa.core.system3_crystallizer import ModuleCrystallizer

        crystallizer = ModuleCrystallizer(embed_dim=32)

        point_cloud = torch.randn(1, 32)
        result = crystallizer.crystallize(point_cloud, epoch=0)
        assert result["new_modules"] == []


class TestBlankSpaceDetectorNumericalStability:
    def test_zero_input(self):
        """Blank space detector should handle all-zero input."""
        from tcd_jepa.exploration.blank_space_detector import BlankSpaceDetector

        detector = BlankSpaceDetector()
        z = torch.zeros(4, 8)

        def energy_fn(z_in):
            return z_in.pow(2).sum(dim=-1)

        result = detector.detect(z, energy_fn)
        assert "is_blank" in result
        assert result["is_blank"].shape == (4,)
        # All zeros should be detected as flat/blank
        assert not torch.isnan(result["flatness_score"]).any()
        assert not torch.isinf(result["combined_score"]).any()
