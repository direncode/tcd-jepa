"""Tests for checkpoint save/load round-trips."""

import tempfile
from pathlib import Path

import torch

from tcd_jepa.models.tcd_jepa_model import build_tcd_jepa
from tcd_jepa.training.trainer import build_optimizer
from tcd_jepa.utils.checkpointing import load_checkpoint, save_checkpoint


def _build_small_model(use_dynamic_predictor: bool = False):
    """Build a minimal model for testing."""
    return build_tcd_jepa(
        img_size=32,
        patch_size=8,
        embed_dim=64,
        depth=2,
        num_heads=2,
        predictor_embed_dim=32,
        predictor_depth=2,
        predictor_num_heads=2,
        use_dynamic_predictor=use_dynamic_predictor,
    )


class TestCheckpointRoundTrip:
    def test_save_load_reproduces_output(self):
        """Saved and loaded model should produce identical outputs."""
        model = _build_small_model()
        optimizer = build_optimizer(model)
        torch.manual_seed(42)
        x = torch.randn(2, 3, 32, 32)

        # Get reference output
        model.eval()
        with torch.no_grad():
            ref_z = model.context_encoder(x)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "ckpt.pt")
            save_checkpoint(
                path=path,
                epoch=5,
                encoder=model.context_encoder,
                predictor=model.predictor,
                target_encoder=model.target_encoder,
                optimizer=optimizer,
            )

            # Build fresh model and load
            model2 = _build_small_model()
            epoch = load_checkpoint(
                path,
                model2.context_encoder,
                model2.predictor,
                model2.target_encoder,
            )

            assert epoch == 5

            model2.eval()
            with torch.no_grad():
                loaded_z = model2.context_encoder(x)

            assert torch.allclose(ref_z, loaded_z, atol=1e-6), (
                "Loaded model should produce identical output"
            )

    def test_optimizer_state_preserved(self):
        """Optimizer state should survive checkpoint round-trip."""
        model = _build_small_model()
        optimizer = build_optimizer(model, lr=0.01)

        # Run a fake step to populate optimizer state
        x = torch.randn(2, 3, 32, 32)
        masks_enc = [torch.arange(16)]
        masks_pred = [torch.arange(16)]
        result = model(x, masks_enc, masks_pred)
        result["loss"].backward()
        optimizer.step()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "ckpt.pt")
            save_checkpoint(
                path=path,
                epoch=1,
                encoder=model.context_encoder,
                predictor=model.predictor,
                target_encoder=model.target_encoder,
                optimizer=optimizer,
            )

            # Load into fresh optimizer
            model2 = _build_small_model()
            optimizer2 = build_optimizer(model2, lr=0.01)
            load_checkpoint(
                path,
                model2.context_encoder,
                model2.predictor,
                model2.target_encoder,
                optimizer=optimizer2,
            )

            # Check optimizer state is populated
            assert len(optimizer2.state) > 0, "Optimizer state should be loaded"

    def test_dynamic_predictor_checkpoint(self):
        """DynamicPredictor state should survive checkpoint."""
        model = _build_small_model(use_dynamic_predictor=True)
        optimizer = build_optimizer(model)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "ckpt.pt")
            save_checkpoint(
                path=path,
                epoch=3,
                encoder=model.context_encoder,
                predictor=model.predictor,
                target_encoder=model.target_encoder,
                optimizer=optimizer,
            )

            model2 = _build_small_model(use_dynamic_predictor=True)
            epoch = load_checkpoint(
                path,
                model2.context_encoder,
                model2.predictor,
                model2.target_encoder,
            )
            assert epoch == 3

            # Verify predictor state matches
            for (n1, p1), (n2, p2) in zip(
                model.predictor.state_dict().items(),
                model2.predictor.state_dict().items(),
            ):
                assert n1 == n2
                assert torch.allclose(p1, p2), f"Mismatch in {n1}"

    def test_scaler_checkpoint(self):
        """GradScaler state should be preserved."""
        model = _build_small_model()
        optimizer = build_optimizer(model)
        scaler = torch.amp.GradScaler()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "ckpt.pt")
            save_checkpoint(
                path=path,
                epoch=2,
                encoder=model.context_encoder,
                predictor=model.predictor,
                target_encoder=model.target_encoder,
                optimizer=optimizer,
                scaler=scaler,
            )

            scaler2 = torch.amp.GradScaler()
            model2 = _build_small_model()
            load_checkpoint(
                path,
                model2.context_encoder,
                model2.predictor,
                model2.target_encoder,
                scaler=scaler2,
            )
            assert scaler2.get_scale() == scaler.get_scale()
