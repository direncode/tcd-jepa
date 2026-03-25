"""Config schema validation for TCD-JEPA training configs.

Validates required fields, types, constraints, and provides warnings
for suspicious values.
"""

import logging
from typing import Any

logger = logging.getLogger("tcd_jepa")

REQUIRED_FIELDS = {
    "model.encoder.img_size": (int, float),
    "model.encoder.patch_size": (int, float),
    "model.encoder.embed_dim": (int, float),
    "model.encoder.depth": (int, float),
    "model.encoder.num_heads": (int, float),
    "training.epochs": (int, float),
    "training.batch_size": (int, float),
    "training.learning_rate": (int, float),
}


def _get_nested(cfg: dict, dotpath: str) -> Any:
    """Get a nested value from a dict using dot notation."""
    parts = dotpath.split(".")
    d = cfg
    for part in parts:
        if not isinstance(d, dict) or part not in d:
            return None
        d = d[part]
    return d


def validate_config(cfg: dict) -> list[str]:
    """Validate a TCD-JEPA config.

    Returns:
        List of error strings. Empty list means config is valid.
        Warnings are logged but not returned as errors.
    """
    errors = []
    warnings = []

    # Check required fields
    for field, expected_types in REQUIRED_FIELDS.items():
        val = _get_nested(cfg, field)
        if val is None:
            errors.append(f"Missing required field: {field}")
        elif not isinstance(val, expected_types):
            errors.append(f"Field {field} has wrong type: {type(val).__name__} (expected {expected_types})")

    # Constraint checks (only if fields exist)
    img_size = _get_nested(cfg, "model.encoder.img_size")
    patch_size = _get_nested(cfg, "model.encoder.patch_size")
    if img_size is not None and patch_size is not None:
        if isinstance(img_size, (int, float)) and isinstance(patch_size, (int, float)):
            if int(img_size) % int(patch_size) != 0:
                errors.append(
                    f"img_size ({img_size}) must be divisible by patch_size ({patch_size})"
                )

    epochs = _get_nested(cfg, "training.epochs")
    warmup = _get_nested(cfg, "training.warmup_epochs")
    if epochs is not None and warmup is not None:
        if isinstance(epochs, (int, float)) and isinstance(warmup, (int, float)) and warmup >= epochs:
            errors.append(f"warmup_epochs ({warmup}) must be less than epochs ({epochs})")

    batch_size = _get_nested(cfg, "training.batch_size")
    if batch_size is not None and isinstance(batch_size, (int, float)):
        if batch_size <= 0:
            errors.append(f"batch_size must be positive, got {batch_size}")

    lr = _get_nested(cfg, "training.learning_rate")
    if lr is not None and isinstance(lr, (int, float)):
        if lr <= 0:
            errors.append(f"learning_rate must be positive, got {lr}")
        elif lr > 0.1:
            warnings.append(f"learning_rate ({lr}) is unusually high — check if this is intentional")

    depth = _get_nested(cfg, "model.encoder.depth")
    if depth is not None and isinstance(depth, (int, float)):
        if depth > 48:
            warnings.append(f"encoder depth ({depth}) is very deep — ensure sufficient GPU memory")

    embed_dim = _get_nested(cfg, "model.encoder.embed_dim")
    num_heads = _get_nested(cfg, "model.encoder.num_heads")
    if embed_dim is not None and num_heads is not None:
        if isinstance(embed_dim, (int, float)) and isinstance(num_heads, (int, float)):
            if int(embed_dim) % int(num_heads) != 0:
                errors.append(
                    f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
                )

    # Masking scale validation
    enc_scale = _get_nested(cfg, "masking.enc_mask_scale")
    if enc_scale is not None and isinstance(enc_scale, list):
        if len(enc_scale) != 2 or enc_scale[0] > enc_scale[1]:
            warnings.append(f"enc_mask_scale should be [min, max] with min <= max, got {enc_scale}")

    # Log warnings
    for w in warnings:
        logger.warning(f"Config warning: {w}")

    return errors
