"""Save and load model checkpoints with validation and RNG state preservation."""

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

logger = logging.getLogger("tcd_jepa")

CHECKPOINT_VERSION = "1.0"
REQUIRED_KEYS = {"epoch", "encoder", "predictor", "target_encoder"}


def save_checkpoint(
    path: str,
    epoch: int,
    encoder: torch.nn.Module,
    predictor: torch.nn.Module,
    target_encoder: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[Any] = None,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    """Save a training checkpoint with version info and RNG state."""
    checkpoint = {
        "tcd_jepa_version": CHECKPOINT_VERSION,
        "epoch": epoch,
        "encoder": encoder.state_dict(),
        "predictor": predictor.state_dict(),
        "target_encoder": target_encoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng_state": {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
        },
    }

    # Include CUDA RNG state if available
    if torch.cuda.is_available():
        checkpoint["rng_state"]["cuda"] = torch.cuda.get_rng_state_all()

    if scaler is not None:
        checkpoint["scaler"] = scaler.state_dict()
    if extra is not None:
        checkpoint.update(extra)

    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        torch.save(checkpoint, save_path)
        logger.info(f"Saved checkpoint to {save_path} (epoch {epoch})")
    except Exception as e:
        logger.error(f"Failed to save checkpoint to {save_path}: {e}")
        raise


def load_checkpoint(
    path: str,
    encoder: torch.nn.Module,
    predictor: torch.nn.Module,
    target_encoder: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[Any] = None,
    device: str = "cpu",
    restore_rng: bool = True,
) -> int:
    """Load a training checkpoint with validation. Returns the epoch number."""
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint from {path}: {e}") from e

    # Validate required keys
    missing = REQUIRED_KEYS - set(checkpoint.keys())
    if missing:
        raise KeyError(f"Checkpoint missing required keys: {missing}")

    # Version check
    version = checkpoint.get("tcd_jepa_version", "unknown")
    if version != CHECKPOINT_VERSION:
        logger.warning(f"Checkpoint version mismatch: {version} (expected {CHECKPOINT_VERSION})")

    # Load state dicts with error handling
    try:
        encoder.load_state_dict(checkpoint["encoder"])
    except Exception as e:
        raise RuntimeError(f"Failed to load encoder state: {e}") from e

    try:
        predictor.load_state_dict(checkpoint["predictor"])
    except Exception as e:
        raise RuntimeError(f"Failed to load predictor state: {e}") from e

    try:
        target_encoder.load_state_dict(checkpoint["target_encoder"])
    except Exception as e:
        raise RuntimeError(f"Failed to load target_encoder state: {e}") from e

    if optimizer is not None and "optimizer" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
        except Exception as e:
            logger.warning(f"Failed to restore optimizer state (continuing without it): {e}")

    if scaler is not None and "scaler" in checkpoint:
        try:
            scaler.load_state_dict(checkpoint["scaler"])
        except Exception as e:
            logger.warning(f"Failed to restore scaler state: {e}")

    # Restore RNG states for reproducibility
    if restore_rng and "rng_state" in checkpoint:
        rng = checkpoint["rng_state"]
        try:
            if "torch" in rng:
                torch.set_rng_state(rng["torch"])
            if "numpy" in rng:
                np.random.set_state(rng["numpy"])
            if "cuda" in rng and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng["cuda"])
        except Exception as e:
            logger.warning(f"Failed to restore RNG state: {e}")

    epoch = checkpoint.get("epoch", 0)
    logger.info(f"Loaded checkpoint from {path} (epoch {epoch}, version {version})")
    return epoch


def validate_checkpoint(path: str) -> tuple[bool, str]:
    """Validate a checkpoint file without loading into a model.

    Returns:
        Tuple of (is_valid, message).
    """
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        return False, f"File not found: {path}"

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        return False, f"Failed to load: {e}"

    if not isinstance(checkpoint, dict):
        return False, "Checkpoint is not a dictionary"

    missing = REQUIRED_KEYS - set(checkpoint.keys())
    if missing:
        return False, f"Missing required keys: {missing}"

    return True, f"Valid checkpoint (epoch {checkpoint.get('epoch', '?')})"
