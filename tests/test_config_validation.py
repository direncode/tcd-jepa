"""Tests for config schema validation."""

from tcd_jepa.utils.config_validation import validate_config


class TestConfigValidation:
    def _make_valid_config(self) -> dict:
        return {
            "model": {
                "encoder": {
                    "img_size": 32,
                    "patch_size": 4,
                    "embed_dim": 64,
                    "depth": 2,
                    "num_heads": 2,
                },
                "predictor": {
                    "predictor_embed_dim": 32,
                    "predictor_depth": 2,
                    "num_heads": 2,
                },
            },
            "training": {
                "epochs": 10,
                "batch_size": 32,
                "learning_rate": 0.001,
                "warmup_epochs": 2,
            },
            "masking": {
                "enc_mask_scale": [0.15, 0.2],
            },
        }

    def test_valid_config(self):
        cfg = self._make_valid_config()
        errors = validate_config(cfg)
        assert errors == []

    def test_missing_required_field(self):
        cfg = self._make_valid_config()
        del cfg["model"]["encoder"]["embed_dim"]
        errors = validate_config(cfg)
        assert any("embed_dim" in e for e in errors)

    def test_img_size_not_divisible_by_patch_size(self):
        cfg = self._make_valid_config()
        cfg["model"]["encoder"]["img_size"] = 30
        cfg["model"]["encoder"]["patch_size"] = 4
        errors = validate_config(cfg)
        assert any("divisible" in e for e in errors)

    def test_warmup_exceeds_epochs(self):
        cfg = self._make_valid_config()
        cfg["training"]["warmup_epochs"] = 20
        cfg["training"]["epochs"] = 10
        errors = validate_config(cfg)
        assert any("warmup" in e.lower() for e in errors)

    def test_negative_batch_size(self):
        cfg = self._make_valid_config()
        cfg["training"]["batch_size"] = -1
        errors = validate_config(cfg)
        assert any("batch_size" in e for e in errors)

    def test_negative_learning_rate(self):
        cfg = self._make_valid_config()
        cfg["training"]["learning_rate"] = -0.01
        errors = validate_config(cfg)
        assert any("learning_rate" in e for e in errors)

    def test_embed_dim_not_divisible_by_heads(self):
        cfg = self._make_valid_config()
        cfg["model"]["encoder"]["embed_dim"] = 65
        cfg["model"]["encoder"]["num_heads"] = 4
        errors = validate_config(cfg)
        assert any("divisible" in e for e in errors)

    def test_missing_entire_section(self):
        cfg = {"model": {"encoder": {}}}
        errors = validate_config(cfg)
        # Should detect multiple missing required fields
        assert len(errors) >= 3

    def test_wrong_type(self):
        cfg = self._make_valid_config()
        cfg["training"]["epochs"] = "ten"
        errors = validate_config(cfg)
        assert any("wrong type" in e.lower() for e in errors)
