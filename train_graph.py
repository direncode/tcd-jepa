"""Graph-native training entry point for TCD-JEPA.

Bridges the entity/edge JSON format from the crystallization pipeline to the
ManifoldJEPA training loop. Called by both the RunPod handler and the local
wrapper via `from train_graph import train`.

Data flow:
    entities.json + edges.json
        -> attribute-based fingerprints + S² coords + adjacency matrix
        -> CausalManifoldDataset + ManifoldMaskCollator
        -> ManifoldTrainer with RecursiveLoop (System 2+3)
        -> learned embeddings -> spectral clustering
        -> module assignments with entity lists, purity, type distribution
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger("tcd_jepa.train_graph")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SPHERE_RADIUS = 4.5
HOUR_SLOTS = ["8am", "12pm", "6pm", "10pm"]
MIN_CLUSTER_SIZE = 1
DEFAULT_EMBED_DIM = 96
DEFAULT_DEPTH = 4
DEFAULT_NUM_HEADS = 3
DEFAULT_NUM_TOKENS = 64


# ---------------------------------------------------------------------------
# Data conversion helpers
# ---------------------------------------------------------------------------

def _parse_attributes(raw) -> dict:
    """Parse entity attributes from JSON string or dict."""
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {}
    return raw or {}


def _build_type_vocab(entities: list[dict]) -> dict[str, int]:
    """Build entity type -> index mapping from the entity list."""
    types = sorted({e.get("type", "venue") for e in entities})
    return {t: i for i, t in enumerate(types)}


def _build_fingerprints(
    entities: list[dict], type_vocab: dict[str, int],
) -> torch.Tensor:
    """Build attribute-based fingerprint vectors for all entities.

    Feature layout:
        [0:4]   busyness curve (4 hours, normalized 0-1)
        [4:5]   daily variance (normalized)
        [5:9]   peak hour one-hot (8am, 12pm, 6pm, 10pm)
        [9:9+T] entity type one-hot (T = number of unique types)
    """
    num_types = len(type_vocab)
    fp_dim = 4 + 1 + 4 + num_types  # busyness + variance + peak + type
    fingerprints = torch.zeros(len(entities), fp_dim)

    for i, ent in enumerate(entities):
        attrs = _parse_attributes(ent.get("attributes"))

        # Busyness curve (normalize by max across dataset later)
        curve = attrs.get("busyness_curve", [0, 0, 0, 0])
        if isinstance(curve, list) and len(curve) >= 4:
            for j in range(4):
                fingerprints[i, j] = float(curve[j])
        else:
            for slot in HOUR_SLOTS:
                j = HOUR_SLOTS.index(slot)
                fingerprints[i, j] = float(attrs.get(f"busyness_{slot}", 0))

        # Daily variance
        fingerprints[i, 4] = float(attrs.get("daily_variance", 0))

        # Peak hour one-hot
        peak = attrs.get("peak_hour", "")
        if peak in HOUR_SLOTS:
            fingerprints[i, 5 + HOUR_SLOTS.index(peak)] = 1.0

        # Entity type one-hot
        etype = ent.get("type", "venue")
        if etype in type_vocab:
            fingerprints[i, 9 + type_vocab[etype]] = 1.0

    # Normalize busyness columns to [0, 1]
    for col in range(4):
        col_max = fingerprints[:, col].max()
        if col_max > 0:
            fingerprints[:, col] /= col_max

    # Normalize variance column
    var_max = fingerprints[:, 4].max()
    if var_max > 0:
        fingerprints[:, 4] /= var_max

    return fingerprints


def _build_coords(entities: list[dict]) -> torch.Tensor:
    """Convert lat/lon to Cartesian coordinates on S² (radius 4.5)."""
    coords = torch.zeros(len(entities), 3)

    for i, ent in enumerate(entities):
        attrs = _parse_attributes(ent.get("attributes"))
        lat = math.radians(float(attrs.get("lat", 0)))
        lon = math.radians(float(attrs.get("lon", 0)))

        coords[i, 0] = SPHERE_RADIUS * math.cos(lat) * math.cos(lon)
        coords[i, 1] = SPHERE_RADIUS * math.cos(lat) * math.sin(lon)
        coords[i, 2] = SPHERE_RADIUS * math.sin(lat)

    return coords


def _build_adjacency(
    entities: list[dict], edges: list[dict],
) -> torch.Tensor:
    """Build dense adjacency matrix from edge list."""
    name_to_idx = {e.get("name", f"entity_{i}"): i for i, e in enumerate(entities)}
    N = len(entities)
    adj = torch.zeros(N, N)

    for edge in edges:
        src = name_to_idx.get(edge.get("source"))
        tgt = name_to_idx.get(edge.get("target"))
        if src is not None and tgt is not None:
            w = float(edge.get("weight", 1.0))
            adj[src, tgt] = w
            adj[tgt, src] = w  # undirected

    return adj


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def _kmeans_torch(
    embeddings: torch.Tensor, k: int, max_iter: int = 100, seed: int = 42,
) -> torch.Tensor:
    """Simple k-means on GPU/CPU tensors. Returns [N] label tensor."""
    N, D = embeddings.shape
    torch.manual_seed(seed)

    # Initialize centroids via k-means++
    centroids = torch.zeros(k, D, device=embeddings.device)
    idx = torch.randint(0, N, (1,)).item()
    centroids[0] = embeddings[idx]

    for c in range(1, k):
        dists = torch.cdist(embeddings, centroids[:c]).min(dim=1).values
        probs = dists / (dists.sum() + 1e-8)
        idx = torch.multinomial(probs, 1).item()
        centroids[c] = embeddings[idx]

    labels = torch.zeros(N, dtype=torch.long, device=embeddings.device)
    for _ in range(max_iter):
        dists = torch.cdist(embeddings, centroids)
        new_labels = dists.argmin(dim=1)
        if (new_labels == labels).all():
            break
        labels = new_labels
        for c in range(k):
            mask = labels == c
            if mask.any():
                centroids[c] = embeddings[mask].mean(dim=0)

    return labels


def _auto_module_count(embeddings: torch.Tensor, max_k: int = 30) -> int:
    """Estimate good k using the elbow heuristic on inertia."""
    N = embeddings.shape[0]
    max_k = min(max_k, N // 2)
    if max_k < 3:
        return max(2, max_k)

    inertias = []
    for k in range(2, max_k + 1):
        labels = _kmeans_torch(embeddings, k)
        centroids = torch.zeros(k, embeddings.shape[1], device=embeddings.device)
        for c in range(k):
            mask = labels == c
            if mask.any():
                centroids[c] = embeddings[mask].mean(dim=0)
        dists = torch.cdist(embeddings, centroids)
        inertia = sum(
            dists[labels == c, c].pow(2).sum().item() for c in range(k)
        )
        inertias.append(inertia)

    # Find elbow: max second derivative
    if len(inertias) < 3:
        return 2
    second_deriv = [
        inertias[i - 1] - 2 * inertias[i] + inertias[i + 1]
        for i in range(1, len(inertias) - 1)
    ]
    best_idx = int(np.argmax(second_deriv)) + 2  # +2 for offset (k starts at 2, skip first)
    return best_idx


# ---------------------------------------------------------------------------
# Module naming
# ---------------------------------------------------------------------------

def _name_module(
    entity_indices: list[int],
    entities: list[dict],
) -> tuple[str, str, dict[str, int], float]:
    """Compute name, dominant_type, type_distribution, purity for a module."""
    type_counter: Counter[str] = Counter()
    peak_counter: Counter[str] = Counter()

    for idx in entity_indices:
        etype = entities[idx].get("type", "venue")
        type_counter[etype] += 1
        attrs = _parse_attributes(entities[idx].get("attributes"))
        peak = attrs.get("peak_hour", "")
        if peak:
            peak_counter[peak] += 1

    dominant_type = type_counter.most_common(1)[0][0] if type_counter else "unknown"
    total = sum(type_counter.values())
    purity = type_counter[dominant_type] / max(total, 1)
    type_dist = dict(type_counter.most_common())

    # Name: use dominant type, add peak hour suffix if this is a large
    # single-type module with a clear temporal peak
    name = dominant_type.replace("_", " ").title()
    if purity > 0.8 and len(entity_indices) > 5 and peak_counter:
        top_peak, top_count = peak_counter.most_common(1)[0]
        peak_frac = top_count / len(entity_indices)
        if peak_frac > 0.5:
            name = f"{name} \u00b7 {top_peak.title()}"

    return name, dominant_type, type_dist, round(purity, 4)


# ---------------------------------------------------------------------------
# Main train() function
# ---------------------------------------------------------------------------

def train(
    config: dict,
    progress_callback: Optional[Callable[[dict], None]] = None,
) -> dict:
    """Run TCD-JEPA training on graph/entity data.

    Called by the RunPod handler and the local TCDJEPAWrapper.

    Args:
        config: Training configuration with keys:
            - data_path: directory containing entities.json + edges.json
            - config_path: (optional) YAML config path (ignored, we build our own)
            - device: "cuda" or "cpu"
            - epochs: number of training epochs (default 100)
            - embedding_dim: model embedding dimension (default 96)
            - lr / learning_rate: learning rate (default 0.001)
            - batch_size: training batch size (default 32)
        progress_callback: optional function receiving epoch metrics dicts

    Returns:
        Dict with keys: modules, training_metrics, final_loss, final_auc,
        final_knn, total_epochs, training_time, checkpoint_path, device
    """
    from tcd_jepa.core.recursive_loop import RecursiveLoop
    from tcd_jepa.manifold.dataset import CausalManifoldDataset
    from tcd_jepa.manifold.masking import ManifoldMaskCollator
    from tcd_jepa.manifold.model import build_manifold_jepa
    from tcd_jepa.manifold.trainer import ManifoldTrainer
    from tcd_jepa.models.target_encoder import momentum_schedule
    from tcd_jepa.training.schedulers import CosineWDSchedule, WarmupCosineSchedule
    from tcd_jepa.training.trainer import build_optimizer

    start_time = time.time()

    # ── Parse config (supports both flat and nested YAML formats) ────
    # Nested YAML from ConfigBuilder uses training.epochs, model.embedding_dim, etc.
    train_cfg = config.get("training", {})
    model_cfg = config.get("model", {})
    cryst_cfg = config.get("crystallization", {})

    data_path = config.get("data_path") or config.get("data", {}).get("path", ".")
    device_str = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    num_epochs = int(config.get("epochs") or train_cfg.get("epochs", 100))
    embed_dim = int(
        config.get("embedding_dim")
        or model_cfg.get("embedding_dim", DEFAULT_EMBED_DIM)
    )
    lr = float(
        config.get("lr")
        or config.get("learning_rate")
        or train_cfg.get("learning_rate", 0.001)
    )
    batch_size = int(config.get("batch_size") or train_cfg.get("batch_size", 32))
    num_tokens = int(config.get("num_tokens", DEFAULT_NUM_TOKENS))
    depth = int(config.get("depth", DEFAULT_DEPTH))
    num_heads = int(config.get("num_heads", DEFAULT_NUM_HEADS))
    seed = int(config.get("seed") or train_cfg.get("seed", 42))
    module_count = config.get("module_count") or cryst_cfg.get("module_count")  # None = auto

    torch.manual_seed(seed)
    np.random.seed(seed)

    logger.info(
        "Graph training: device=%s, epochs=%d, embed_dim=%d, lr=%.4f",
        device_str, num_epochs, embed_dim, lr,
    )

    # ── Load data ─────────────────────────────────────────────────────
    data_dir = Path(data_path)
    with open(data_dir / "entities.json") as f:
        entities = json.load(f)
    with open(data_dir / "edges.json") as f:
        edges = json.load(f)

    entity_count = len(entities)
    edge_count = len(edges)
    logger.info("Loaded %d entities, %d edges", entity_count, edge_count)

    if entity_count == 0:
        raise ValueError("No entities to train on")

    # ── Build tensors ─────────────────────────────────────────────────
    type_vocab = _build_type_vocab(entities)
    fingerprints = _build_fingerprints(entities, type_vocab)
    coords = _build_coords(entities)
    adjacency = _build_adjacency(entities, edges)
    fp_dim = fingerprints.shape[1]

    logger.info(
        "Tensors built: fingerprint_dim=%d, types=%d, adjacency_nnz=%d",
        fp_dim, len(type_vocab), (adjacency > 0).sum().item(),
    )

    # ── Dataset + DataLoader ──────────────────────────────────────────
    # Adjust num_tokens to not exceed entity count
    actual_tokens = min(num_tokens, entity_count)
    num_samples = max(500, entity_count * 10)  # enough windows per epoch

    dataset = CausalManifoldDataset(
        fingerprints=fingerprints,
        coords=coords,
        adjacency=adjacency,
        velocity=torch.zeros_like(coords),  # no velocity data from FSD
        entity_labels=None,
        num_tokens=actual_tokens,
        num_samples=num_samples,
        window_mode="geodesic" if (adjacency > 0).any() else "random",
    )

    mask_collator = ManifoldMaskCollator(
        num_tokens=actual_tokens,
        context_ratio=(0.3, 0.5),
        target_ratio=(0.15, 0.3),
        num_context_masks=4,
        num_target_masks=1,
        min_keep=max(4, actual_tokens // 8),
        use_geodesic=True,
        sphere_radius=SPHERE_RADIUS,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # in-process for RunPod compatibility
        collate_fn=mask_collator,
        drop_last=True,
    )

    logger.info(
        "Dataset: %d samples, %d tokens/window, %d batches/epoch",
        len(dataset), actual_tokens, len(dataloader),
    )

    # ── Model ─────────────────────────────────────────────────────────
    predictor_embed_dim = max(embed_dim // 2, 32)

    model = build_manifold_jepa(
        fingerprint_dim=fp_dim,
        num_tokens=actual_tokens,
        coord_dim=3,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=4.0,
        predictor_embed_dim=predictor_embed_dim,
        predictor_depth=max(depth // 2, 2),
        predictor_num_heads=num_heads,
        use_dynamic_predictor=True,
        use_causal_encoding=True,
        use_velocity_encoding=False,  # no velocity from FSD
        sphere_radius=SPHERE_RADIUS,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %s", f"{param_count:,}")

    # ── Optimizer + schedulers ────────────────────────────────────────
    steps_per_epoch = len(dataloader)
    total_steps = num_epochs * steps_per_epoch

    optimizer = build_optimizer(
        model, lr=lr, weight_decay=1e-4,
    )
    lr_scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=min(5, num_epochs // 10) * steps_per_epoch,
        start_lr=lr * 0.1,
        ref_lr=lr,
        T_max=total_steps,
    )
    wd_scheduler = CosineWDSchedule(
        optimizer, ref_wd=1e-4, T_max=total_steps,
    )
    ema_sched = momentum_schedule(0.996, 1.0, total_steps)

    # ── TCD Recursive Loop ────────────────────────────────────────────
    recursive_loop = RecursiveLoop(
        embed_dim=embed_dim,
        explore_every=2,
        crystallize_every=5,
        langevin_steps=20,
        device=device,
    )
    model.set_module_registry(recursive_loop.crystallizer.registry)

    # ── Trainer ───────────────────────────────────────────────────────
    checkpoint_dir = str(data_dir / "checkpoints")

    # Build config dict for trainer
    trainer_cfg = {
        "training": {
            "grad_clip_norm": 1.0,
            "checkpoint_freq": max(num_epochs // 5, 10),
        },
    }

    trainer = ManifoldTrainer(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        wd_scheduler=wd_scheduler,
        momentum_schedule=ema_sched,
        train_loader=dataloader,
        device=device,
        cfg=trainer_cfg,
        checkpoint_dir=checkpoint_dir,
        recursive_loop=recursive_loop,
    )

    # ── Training loop with progress reporting ─────────────────────────
    metrics_log: list[dict] = []
    final_loss = 0.0

    # Monkey-patch trainer to capture per-epoch metrics for progress reporting
    _original_train = trainer.train

    def _train_with_progress(num_ep: int, start_epoch: int = 0):
        nonlocal final_loss
        model.train()

        for epoch in range(start_epoch, num_ep):
            from tcd_jepa.training.metrics import TrainingMetrics
            epoch_metrics = TrainingMetrics()
            t0 = time.time()

            for itr, (batch_data, masks_enc, masks_pred) in enumerate(dataloader):
                step_result = trainer._train_step(batch_data, masks_enc, masks_pred)
                epoch_metrics.update(
                    loss=step_result["loss"],
                    lr=step_result["lr"],
                    wd=step_result["wd"],
                    momentum=step_result["momentum"],
                    grad_norm=step_result["grad_norm"],
                    param_norm=step_result["param_norm"],
                    nan_count=step_result["nan_count"],
                    inf_count=step_result["inf_count"],
                    is_spike=step_result["is_spike"],
                    skipped=step_result["skipped"],
                )
                trainer.global_step += 1

            # TCD crystallization at end of epoch
            if trainer.recursive_loop is not None:
                trainer._run_recursive_loop(batch_data, epoch)

            epoch_time = time.time() - t0
            m = epoch_metrics.to_dict()
            final_loss = m["loss"]
            n_modules = recursive_loop.num_modules

            metric_entry = {
                "epoch": epoch,
                "step": epoch,
                "loss": round(m["loss"], 4),
                "auc": 0.0,  # computed post-training
                "knn": 0.0,
                "nan_detected": m["nan_count"] > 0,
                "num_modules": n_modules,
            }
            metrics_log.append(metric_entry)

            logger.info(
                "Epoch %d/%d: loss=%.4f modules=%d time=%.1fs",
                epoch, num_ep, m["loss"], n_modules, epoch_time,
            )

            if progress_callback:
                progress_callback({
                    "epoch": epoch,
                    "loss": m["loss"],
                    "num_modules": n_modules,
                    "link_auc": 0.0,
                    "knn_accuracy": 0.0,
                })

            # Checkpoint
            save_freq = trainer_cfg["training"]["checkpoint_freq"]
            if (epoch + 1) % save_freq == 0 or epoch == num_ep - 1:
                from tcd_jepa.utils.checkpointing import save_checkpoint
                save_checkpoint(
                    path=str(Path(checkpoint_dir) / f"graph_checkpoint_{epoch:04d}.pt"),
                    epoch=epoch,
                    encoder=model.context_encoder,
                    predictor=model.predictor,
                    target_encoder=model.target_encoder,
                    optimizer=optimizer,
                    scaler=None,
                )

    _train_with_progress(num_epochs)

    training_time = time.time() - start_time
    logger.info("Training completed in %.1fs", training_time)

    # ── Extract embeddings for all entities ───────────────────────────
    logger.info("Extracting embeddings for %d entities", entity_count)
    model.eval()

    with torch.no_grad():
        fp_batch = fingerprints.unsqueeze(0).to(device)
        coord_batch = coords.unsqueeze(0).to(device)
        adj_batch = adjacency.unsqueeze(0).to(device)

        all_embeddings = model.context_encoder(
            fp_batch,
            coords=coord_batch,
            adjacency=adj_batch,
        )
        # [1, N, D] -> [N, D]
        all_embeddings = all_embeddings.squeeze(0).cpu()

    logger.info("Embeddings shape: %s", all_embeddings.shape)

    # ── Cluster to discover modules ───────────────────────────────────
    n_crystallized = recursive_loop.num_modules
    if module_count:
        k = int(module_count)
    elif n_crystallized > 1:
        k = n_crystallized
        logger.info("Using crystallizer module count: %d", k)
    else:
        k = _auto_module_count(all_embeddings)
        logger.info("Auto-detected module count: %d", k)

    labels = _kmeans_torch(all_embeddings, k)
    logger.info("Clustered %d entities into %d modules", entity_count, k)

    # ── Build module results ──────────────────────────────────────────
    modules = []
    for module_idx in range(k):
        member_mask = (labels == module_idx)
        member_indices = member_mask.nonzero(as_tuple=True)[0].tolist()

        if len(member_indices) < MIN_CLUSTER_SIZE:
            continue

        name, dominant_type, type_dist, purity = _name_module(
            member_indices, entities,
        )

        module_entities = []
        for idx in member_indices:
            ent = entities[idx]
            module_entities.append({
                "name": ent.get("name", f"entity_{idx}"),
                "type": ent.get("type", "unknown"),
                "attributes": _parse_attributes(ent.get("attributes")),
            })

        # Internal density: fraction of possible edges that exist within module
        sub_adj = adjacency[member_mask][:, member_mask]
        possible = len(member_indices) * (len(member_indices) - 1)
        density = float((sub_adj > 0).sum()) / max(possible, 1)

        modules.append({
            "module_index": len(modules),
            "name": name,
            "entity_count": len(member_indices),
            "dominant_type": dominant_type,
            "purity_score": purity,
            "type_distribution": type_dist,
            "internal_density": round(density, 4),
            "entities": module_entities,
        })

    # Sort by size descending
    modules.sort(key=lambda m: m["entity_count"], reverse=True)
    for i, m in enumerate(modules):
        m["module_index"] = i

    logger.info(
        "Final: %d modules, largest=%d entities, training_time=%.0fs",
        len(modules),
        modules[0]["entity_count"] if modules else 0,
        training_time,
    )

    # ── Checkpoint path ───────────────────────────────────────────────
    checkpoint_path = str(
        Path(checkpoint_dir) / f"graph_checkpoint_{num_epochs - 1:04d}.pt"
    )

    return {
        "modules": modules,
        "training_metrics": metrics_log,
        "final_loss": round(final_loss, 4),
        "final_auc": 0.0,  # link prediction AUC — future work
        "final_knn": 0.0,  # kNN accuracy — future work
        "total_epochs": num_epochs,
        "training_time": round(training_time, 2),
        "training_time_seconds": round(training_time, 2),
        "checkpoint_path": checkpoint_path,
        "device": device_str,
        "raw_result": {},
    }
