"""Save and load model checkpoints including dynamic module state."""

import logging
from pathlib import Path
from typing import Any, Optional

import torch

logger = logging.getLogger("tcd_jepa")


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
    """Save a training checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "encoder": encoder.state_dict(),
        "predictor": predictor.state_dict(),
        "target_encoder": target_encoder.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    if scaler is not None:
        checkpoint["scaler"] = scaler.state_dict()
    if extra is not None:
        checkpoint.update(extra)

    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, save_path)
    logger.info(f"Saved checkpoint to {save_path} (epoch {epoch})")


def load_checkpoint(
    path: str,
    encoder: torch.nn.Module,
    predictor: torch.nn.Module,
    target_encoder: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[Any] = None,
    device: str = "cpu",
) -> int:
    """Load a training checkpoint. Returns the epoch number."""
    checkpoint = torch.load(path, map_location=device)

    encoder.load_state_dict(checkpoint["encoder"])
    predictor.load_state_dict(checkpoint["predictor"])
    target_encoder.load_state_dict(checkpoint["target_encoder"])

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scaler is not None and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])

    epoch = checkpoint.get("epoch", 0)
    logger.info(f"Loaded checkpoint from {path} (epoch {epoch})")
    return epoch
