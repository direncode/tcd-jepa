"""Profiling utilities for model analysis: FLOPs counting, model summary, timers."""

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger("tcd_jepa")


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Count parameters per top-level module and total.

    Returns:
        Dict mapping module names to parameter counts, plus 'total'.
    """
    counts = {}
    for name, child in model.named_children():
        n = sum(p.numel() for p in child.parameters())
        counts[name] = n
    counts["total"] = sum(p.numel() for p in model.parameters())
    counts["trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return counts


def log_model_summary(model: nn.Module, name: str = "Model") -> str:
    """Generate and log a summary of model parameter counts.

    Returns:
        Formatted summary string.
    """
    counts = count_parameters(model)
    lines = [f"{name} Parameter Summary:"]
    for key, count in counts.items():
        lines.append(f"  {key}: {count:,}")

    # Memory estimate (assuming float32)
    total_bytes = counts["total"] * 4
    lines.append(f"  Memory (float32): {total_bytes / 1e6:.1f} MB")

    summary = "\n".join(lines)
    logger.info(summary)
    return summary


def estimate_flops(
    embed_dim: int,
    depth: int,
    num_heads: int,
    num_patches: int,
    mlp_ratio: float = 4.0,
    batch_size: int = 1,
) -> dict[str, int]:
    """Estimate FLOPs for a Vision Transformer forward pass.

    Provides a rough estimate based on the major operations (attention, MLP).

    Returns:
        Dict with per-component and total FLOP estimates.
    """
    B = batch_size
    N = num_patches
    D = embed_dim
    mlp_dim = int(D * mlp_ratio)

    # Per-layer FLOPs
    # Attention: QKV projection (3 * 2*N*D*D) + attention scores (2*N*N*D) + output proj (2*N*D*D)
    qkv_flops = 3 * 2 * N * D * D
    attn_scores_flops = 2 * N * N * D
    attn_proj_flops = 2 * N * D * D
    attn_total = qkv_flops + attn_scores_flops + attn_proj_flops

    # MLP: two linear layers
    mlp_flops = 2 * N * D * mlp_dim + 2 * N * mlp_dim * D

    per_layer = attn_total + mlp_flops
    encoder_flops = depth * per_layer * B

    return {
        "per_layer_flops": per_layer,
        "encoder_flops": encoder_flops,
        "total_gflops": encoder_flops / 1e9,
    }


def get_gpu_memory_stats() -> Optional[dict[str, float]]:
    """Get current GPU memory statistics in MB.

    Returns:
        Dict with allocated, reserved, peak stats, or None if no GPU.
    """
    if not torch.cuda.is_available():
        return None
    return {
        "allocated_mb": torch.cuda.memory_allocated() / 1e6,
        "reserved_mb": torch.cuda.memory_reserved() / 1e6,
        "peak_allocated_mb": torch.cuda.max_memory_allocated() / 1e6,
        "peak_reserved_mb": torch.cuda.max_memory_reserved() / 1e6,
    }
