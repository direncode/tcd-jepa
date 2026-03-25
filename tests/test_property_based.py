"""Property-based tests using Hypothesis for TCD-JEPA.

Tests mathematical invariants and properties that should hold for any valid input.
"""

import pytest
import torch

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from tcd_jepa.training.guards import (  # noqa: E402
    GradientStats,
    LossTracker,
    check_loss_health,
)
from tcd_jepa.training.losses import jepa_loss  # noqa: E402
from tcd_jepa.utils.config_validation import validate_config  # noqa: E402

# --- Loss function properties ---

class TestJEPALossProperties:
    @given(
        batch_size=st.integers(min_value=1, max_value=8),
        seq_len=st.integers(min_value=1, max_value=16),
        dim=st.integers(min_value=4, max_value=32),
    )
    @settings(max_examples=30, deadline=5000)
    def test_loss_non_negative(self, batch_size, seq_len, dim):
        """JEPA loss (smooth L1) should always be non-negative."""
        preds = torch.randn(batch_size, seq_len, dim)
        targets = torch.randn(batch_size, seq_len, dim)
        loss = jepa_loss(preds, targets)
        assert loss.item() >= 0.0

    @given(
        batch_size=st.integers(min_value=1, max_value=8),
        dim=st.integers(min_value=4, max_value=32),
    )
    @settings(max_examples=20, deadline=5000)
    def test_loss_zero_for_identical(self, batch_size, dim):
        """Loss should be zero when predictions equal targets."""
        x = torch.randn(batch_size, dim)
        loss = jepa_loss(x, x)
        assert abs(loss.item()) < 1e-6

    @given(
        batch_size=st.integers(min_value=1, max_value=8),
        dim=st.integers(min_value=4, max_value=32),
    )
    @settings(max_examples=20, deadline=5000)
    def test_loss_symmetric(self, batch_size, dim):
        """Smooth L1 loss should be symmetric: L(a,b) == L(b,a)."""
        a = torch.randn(batch_size, dim)
        b = torch.randn(batch_size, dim)
        assert abs(jepa_loss(a, b).item() - jepa_loss(b, a).item()) < 1e-6


# --- Loss health check properties ---

class TestLossHealthProperties:
    @given(val=st.floats(min_value=-1e5, max_value=1e5, allow_nan=False, allow_infinity=False))
    @settings(max_examples=50, deadline=5000)
    def test_finite_loss_is_healthy(self, val):
        """Any finite loss within threshold should be healthy."""
        loss = torch.tensor(val)
        assert check_loss_health(loss, threshold=1e6) is True

    @given(val=st.floats(allow_nan=True, allow_infinity=True))
    @settings(max_examples=30, deadline=5000)
    def test_nan_inf_always_unhealthy(self, val):
        """NaN and Inf should always be detected as unhealthy."""
        import math
        loss = torch.tensor(val)
        if math.isnan(val) or math.isinf(val):
            assert check_loss_health(loss) is False


# --- Gradient stats properties ---

class TestGradientStatsProperties:
    def test_healthy_iff_no_nan_inf(self):
        """GradientStats.is_healthy should be True iff nan_count==0 and inf_count==0."""
        for nan_count in [0, 1, 5]:
            for inf_count in [0, 1, 5]:
                stats = GradientStats(nan_count=nan_count, inf_count=inf_count)
                expected = (nan_count == 0 and inf_count == 0)
                assert stats.is_healthy == expected


# --- Loss tracker properties ---

class TestLossTrackerProperties:
    @given(
        values=st.lists(st.floats(min_value=0.1, max_value=10.0), min_size=1, max_size=50),
    )
    @settings(max_examples=30, deadline=5000)
    def test_rolling_average_bounded(self, values):
        """Rolling average should always be between min and max of window."""
        tracker = LossTracker(window_size=100)
        for v in values:
            tracker.update(v)
        avg = tracker.rolling_avg
        assert min(values) <= avg <= max(values) + 1e-8

    @given(
        n=st.integers(min_value=15, max_value=30),
    )
    @settings(max_examples=10, deadline=5000)
    def test_spike_detection_consistent(self, n):
        """After stable history, a 100x spike should always be detected."""
        tracker = LossTracker(window_size=20, spike_factor=5.0)
        for _ in range(n):
            tracker.update(1.0)
        is_spike = tracker.update(1000.0)
        assert is_spike is True


# --- Config validation properties ---

class TestConfigValidationProperties:
    @given(
        img_size=st.sampled_from([16, 32, 64, 128]),
        patch_size=st.sampled_from([2, 4, 8, 16]),
        embed_dim=st.sampled_from([32, 64, 128, 192, 256]),
        num_heads=st.sampled_from([1, 2, 4, 8]),
    )
    @settings(max_examples=30, deadline=5000)
    def test_valid_configs_pass(self, img_size, patch_size, embed_dim, num_heads):
        """Configs with valid constraints should pass validation."""
        if img_size % patch_size != 0 or embed_dim % num_heads != 0:
            return  # Skip invalid combinations

        cfg = {
            "model": {
                "encoder": {
                    "img_size": img_size,
                    "patch_size": patch_size,
                    "embed_dim": embed_dim,
                    "depth": 2,
                    "num_heads": num_heads,
                },
            },
            "training": {
                "epochs": 10,
                "batch_size": 32,
                "learning_rate": 0.001,
            },
        }
        errors = validate_config(cfg)
        assert errors == [], f"Valid config rejected: {errors}"

    @given(
        img_size=st.integers(min_value=1, max_value=256),
        patch_size=st.integers(min_value=1, max_value=256),
    )
    @settings(max_examples=30, deadline=5000)
    def test_indivisible_img_patch_detected(self, img_size, patch_size):
        """img_size not divisible by patch_size should be caught."""
        if img_size % patch_size == 0:
            return

        cfg = {
            "model": {
                "encoder": {
                    "img_size": img_size,
                    "patch_size": patch_size,
                    "embed_dim": 64,
                    "depth": 2,
                    "num_heads": 2,
                },
            },
            "training": {
                "epochs": 10,
                "batch_size": 32,
                "learning_rate": 0.001,
            },
        }
        errors = validate_config(cfg)
        assert any("divisible" in e for e in errors), f"Should detect indivisible: {img_size}/{patch_size}"


# --- Model output properties ---

class TestModelOutputProperties:
    @given(batch_size=st.integers(min_value=1, max_value=4))
    @settings(max_examples=5, deadline=30000)
    def test_output_finite_for_any_batch_size(self, batch_size):
        """Model output should be finite for any valid batch size."""
        from tcd_jepa.models.tcd_jepa_model import build_tcd_jepa

        torch.manual_seed(42)
        model = build_tcd_jepa(
            img_size=32, patch_size=4, embed_dim=64, depth=2, num_heads=2,
            predictor_embed_dim=32, predictor_depth=2, predictor_num_heads=2,
        )

        images = torch.randn(batch_size, 3, 32, 32)
        masks_enc = [torch.arange(32).unsqueeze(0).expand(batch_size, -1)]
        masks_pred = [torch.arange(32).unsqueeze(0).expand(batch_size, -1)]

        result = model(images, masks_enc, masks_pred)
        assert torch.isfinite(result["loss"]), f"Non-finite loss for batch_size={batch_size}"
        assert torch.isfinite(result["predictions"]).all(), "Non-finite predictions"
