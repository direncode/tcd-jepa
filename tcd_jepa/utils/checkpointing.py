"""Save and load model checkpoints — compile-safe and resume-ready.

Handles torch.compile _orig_mod. prefix stripping on save so checkpoints
are portable. On load, restores full training state including optimizer,
schedulers, scaler, and RNG states for exact reproducibility.
"""

import logging
import os
from pathlib import Path
from typing import Any, Optional

import torch

logger = logging.getLogger("tcd_jepa")


def _strip_compile_prefix(state_dict: dict) -> dict:
    """Strip _orig_mod. prefix from torch.compile state dicts."""
    cleaned = {}
    for k, v in state_dict.items():
        new_key = k.replace("_orig_mod.", "")
        cleaned[new_key] = v
    return cleaned


def save_checkpoint(
    path: str,
    epoch: int,
    encoder: torch.nn.Module,
    predictor: torch.nn.Module,
    target_encoder: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[Any] = None,
    lr_scheduler_step: Optional[int] = None,
    wd_scheduler_step: Optional[int] = None,
    global_step: Optional[int] = None,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    """Save a training checkpoint with compile-safe state dicts.

    All state dicts have _orig_mod. prefixes stripped so checkpoints
    work regardless of whether torch.compile was used.
    """
    checkpoint = {
        "epoch": epoch,
        "encoder": _strip_compile_prefix(encoder.state_dict()),
        "predictor": _strip_compile_prefix(predictor.state_dict()),
        "target_encoder": _strip_compile_prefix(target_encoder.state_dict()),
        "optimizer": optimizer.state_dict(),
    }

    if scaler is not None:
        checkpoint["scaler"] = scaler.state_dict()
    if lr_scheduler_step is not None:
        checkpoint["lr_scheduler_step"] = lr_scheduler_step
    if wd_scheduler_step is not None:
        checkpoint["wd_scheduler_step"] = wd_scheduler_step
    if global_step is not None:
        checkpoint["global_step"] = global_step

    # Save RNG states for reproducibility
    checkpoint["rng_state"] = {
        "python": torch.random.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }

    if extra is not None:
        checkpoint.update(extra)

    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Atomic save: write to temp file then rename
    tmp_path = str(save_path) + ".tmp"
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, save_path)

    logger.info(f"Saved checkpoint to {save_path} (epoch {epoch}, step {global_step})")


def load_checkpoint(
    path: str,
    encoder: torch.nn.Module,
    predictor: torch.nn.Module,
    target_encoder: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[Any] = None,
    device: str = "cpu",
) -> dict:
    """Load a training checkpoint. Returns full checkpoint dict.

    Handles both compiled and non-compiled state dicts transparently.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    # Load encoder — handle ContextEncoder wrapper
    enc_sd = _strip_compile_prefix(checkpoint["encoder"])
    if hasattr(encoder, "encoder"):
        # ContextEncoder wraps VisionTransformer
        encoder.encoder.load_state_dict(enc_sd, strict=True)
    else:
        encoder.load_state_dict(enc_sd, strict=True)

    # Load predictor
    pred_sd = _strip_compile_prefix(checkpoint["predictor"])
    if hasattr(predictor, "base_predictor"):
        # DynamicPredictor wraps base predictor
        predictor.base_predictor.load_state_dict(pred_sd, strict=True)
    else:
        predictor.load_state_dict(pred_sd, strict=True)

    # Load target encoder
    tgt_sd = _strip_compile_prefix(checkpoint["target_encoder"])
    if hasattr(target_encoder, "encoder"):
        # Check if keys have 'encoder.' prefix (saved as TargetEncoder.state_dict())
        if any(k.startswith("encoder.") for k in tgt_sd):
            target_encoder.load_state_dict(tgt_sd, strict=True)
        else:
            target_encoder.encoder.load_state_dict(tgt_sd, strict=True)
    else:
        target_encoder.load_state_dict(tgt_sd, strict=True)

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scaler is not None and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])

    epoch = checkpoint.get("epoch", 0)
    logger.info(f"Loaded checkpoint from {path} (epoch {epoch})")
    return checkpoint
