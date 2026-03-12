"""Training entry point for Latent Ocean TCD-JEPA (Layer 3 Intelligence).

Usage:
    # Synthetic data (testing)
    python train_manifold.py --config configs/manifold.yaml --tcd --eval

    # Real Latent Ocean data
    python train_manifold.py --config configs/manifold.yaml --tcd --eval \
        data.dataset=latent_ocean data.data_dir=./data/latent_ocean

For Latent Ocean data, export from DuckDB to a directory containing:
    fingerprints.pt  — [num_entities, 384]  (sentence-transformer embeddings)
    coords.pt        — [num_entities, 3]    (S² positions, radius 4.5)
    velocity.pt      — [num_entities, 3]    (tangent-space velocities)
    adjacency.pt     — [num_entities, num_entities]  (weighted directed links)
    labels.pt        — [num_entities]        (entity_type indices, optional)

Or as latent_ocean_export.npz with the same keys.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from tcd_jepa.utils.config import load_config_with_overrides
from tcd_jepa.manifold.model import build_manifold_jepa
from tcd_jepa.manifold.dataset import CausalManifoldDataset, SyntheticManifoldDataset, LatentOceanDataset
from tcd_jepa.manifold.masking import ManifoldMaskCollator
from tcd_jepa.manifold.trainer import ManifoldTrainer
from tcd_jepa.manifold.evaluation import run_full_manifold_evaluation
from tcd_jepa.models.target_encoder import momentum_schedule
from tcd_jepa.training.trainer import build_optimizer
from tcd_jepa.training.schedulers import WarmupCosineSchedule, CosineWDSchedule
from tcd_jepa.core.recursive_loop import RecursiveLoop

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("tcd_jepa.manifold")


def build_dataset(cfg: dict):
    """Build manifold dataset from config."""
    data_cfg = cfg.get("data", {})
    name = data_cfg.get("dataset", "synthetic_manifold")

    if name in ("synthetic_manifold", "synthetic"):
        return SyntheticManifoldDataset(
            num_entities=data_cfg.get("num_entities", 256),
            fingerprint_dim=data_cfg.get("fingerprint_dim", 384),
            num_clusters=data_cfg.get("num_clusters", 6),
            num_tokens=data_cfg.get("num_tokens", 64),
            num_samples=data_cfg.get("num_samples", 500),
            causal_link_prob=data_cfg.get("causal_link_prob", 0.15),
            cross_cluster_prob=data_cfg.get("cross_cluster_prob", 0.03),
            sphere_radius=data_cfg.get("sphere_radius", 4.5),
            seed=cfg.get("training", {}).get("seed", 42),
        )

    elif name in ("latent_ocean", "spherical_db"):
        return LatentOceanDataset(
            data_dir=data_cfg.get("data_dir", "./data/latent_ocean"),
            num_tokens=data_cfg.get("num_tokens", 64),
            num_samples=data_cfg.get("num_samples", 1000),
        )

    else:
        raise ValueError(f"Unknown dataset: {name}")


def main():
    parser = argparse.ArgumentParser(description="Train Latent Ocean TCD-JEPA")
    parser.add_argument("--config", default="configs/manifold.yaml")
    parser.add_argument("--tcd", action="store_true", help="Enable TCD crystallization")
    parser.add_argument("--eval", action="store_true", help="Run evaluation after training")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = load_config_with_overrides(args.config, args.overrides)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    seed = cfg.get("training", {}).get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    enc_cfg = cfg["model"]["encoder"]
    pred_cfg = cfg["model"]["predictor"]
    train_cfg = cfg["training"]
    mask_cfg = cfg["masking"]
    tcd_cfg = cfg.get("tcd", {})

    use_tcd = args.tcd or tcd_cfg.get("use_dynamic_predictor", False)

    # Build model
    model = build_manifold_jepa(
        fingerprint_dim=enc_cfg.get("fingerprint_dim", 384),
        num_tokens=enc_cfg.get("num_tokens", 64),
        coord_dim=enc_cfg.get("coord_dim", 3),
        embed_dim=enc_cfg["embed_dim"],
        depth=enc_cfg["depth"],
        num_heads=enc_cfg["num_heads"],
        mlp_ratio=enc_cfg.get("mlp_ratio", 4.0),
        predictor_embed_dim=pred_cfg["predictor_embed_dim"],
        predictor_depth=pred_cfg["predictor_depth"],
        predictor_num_heads=pred_cfg["num_heads"],
        use_dynamic_predictor=use_tcd,
        use_causal_encoding=enc_cfg.get("use_causal_encoding", True),
        use_velocity_encoding=enc_cfg.get("use_velocity_encoding", True),
        sphere_radius=enc_cfg.get("sphere_radius", 4.5),
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {param_count:,}")

    # Build dataset
    dataset = build_dataset(cfg)
    logger.info(f"Dataset: {len(dataset)} samples, {dataset.num_entities} entities, "
                f"{dataset.num_clusters} clusters")

    # Build mask collator
    mask_collator = ManifoldMaskCollator(
        num_tokens=enc_cfg.get("num_tokens", 64),
        context_ratio=tuple(mask_cfg.get("context_ratio", [0.3, 0.5])),
        target_ratio=tuple(mask_cfg.get("target_ratio", [0.15, 0.3])),
        num_context_masks=mask_cfg.get("num_context_masks", 4),
        num_target_masks=mask_cfg.get("num_target_masks", 1),
        min_keep=mask_cfg.get("min_keep", 4),
        use_geodesic=mask_cfg.get("use_geodesic", True),
        sphere_radius=mask_cfg.get("sphere_radius", 4.5),
    )

    dataloader = DataLoader(
        dataset, batch_size=train_cfg.get("batch_size", 32),
        shuffle=True, num_workers=0, collate_fn=mask_collator, drop_last=True,
    )
    logger.info(f"DataLoader: {len(dataloader)} batches/epoch")

    # Optimizer and schedulers
    num_epochs = train_cfg["epochs"]
    steps_per_epoch = len(dataloader)
    total_steps = num_epochs * steps_per_epoch

    optimizer = build_optimizer(model, lr=train_cfg["learning_rate"], weight_decay=train_cfg["weight_decay"])
    lr_scheduler = WarmupCosineSchedule(
        optimizer, warmup_steps=train_cfg.get("warmup_epochs", 5) * steps_per_epoch,
        start_lr=train_cfg.get("start_lr", 1e-4), ref_lr=train_cfg["learning_rate"], T_max=total_steps,
    )
    wd_scheduler = CosineWDSchedule(optimizer, ref_wd=train_cfg["weight_decay"], T_max=total_steps)
    ema_schedule = momentum_schedule(
        cfg["model"]["ema"]["start"], cfg["model"]["ema"]["end"], total_steps,
    )

    # TCD recursive loop
    recursive_loop = None
    if use_tcd:
        recursive_loop = RecursiveLoop(
            embed_dim=enc_cfg["embed_dim"],
            explore_every=tcd_cfg.get("explore_every", 5),
            crystallize_every=tcd_cfg.get("crystallize_every", 10),
            langevin_steps=tcd_cfg.get("langevin_steps", 50),
            langevin_step_size=tcd_cfg.get("langevin_step_size", 0.01),
            persistence_threshold=tcd_cfg.get("persistence_threshold", 0.3),
            max_modules=tcd_cfg.get("max_modules", 10),
            device=device,
        )
        model.set_module_registry(recursive_loop.crystallizer.registry)
        logger.info("TCD recursive loop enabled")

    # Build trainer
    log_dir = cfg.get("logging", {}).get("log_dir", "./logs/manifold")
    trainer = ManifoldTrainer(
        model=model, optimizer=optimizer, lr_scheduler=lr_scheduler,
        wd_scheduler=wd_scheduler, momentum_schedule=ema_schedule,
        train_loader=dataloader, device=device, cfg=cfg,
        checkpoint_dir=str(Path(log_dir) / "checkpoints"),
        recursive_loop=recursive_loop,
    )

    logger.info(f"Starting Latent Ocean TCD-JEPA training for {num_epochs} epochs")
    trainer.train(num_epochs)
    logger.info("Training complete")

    # Evaluation
    if args.eval:
        logger.info("=" * 70)
        logger.info("LATENT OCEAN TCD-JEPA EVALUATION")
        logger.info("=" * 70)

        results = run_full_manifold_evaluation(
            model, dataset, device, num_classes=dataset.num_clusters,
            max_samples=300, recursive_loop=recursive_loop,
        )

        # Print results by category
        logger.info("\n--- Standard SSL Metrics ---")
        for key in ["linear_probe_train", "linear_probe_test", "knn_k1", "knn_k5", "knn_k20"]:
            if key in results:
                logger.info(f"  {key:35s}: {results[key]:.4f}")

        logger.info("\n--- Causal Link Prediction ---")
        for key in ["link_auc", "link_pos_sim", "link_neg_sim", "link_sim_gap"]:
            if key in results:
                logger.info(f"  {key:35s}: {results[key]:.4f}")

        logger.info("\n--- Manifold Geometry ---")
        for key in ["neighborhood_preservation", "preservation_std", "effective_rank", "feature_std", "uniformity"]:
            if key in results:
                logger.info(f"  {key:35s}: {results[key]:.4f}")

        logger.info("\n--- Latent Ocean KPIs ---")
        for key in ["clarity_score", "drift_velocity", "opportunity_surface", "risk_horizon"]:
            if key in results:
                logger.info(f"  {key:35s}: {results[key]:.4f}" if isinstance(results[key], float) else f"  {key:35s}: {results[key]}")

        logger.info("\n--- TCD Topology ---")
        for key in ["tcd_num_modules", "tcd_converged", "tcd_attractors_h0", "tcd_cycles_h1", "tcd_boundaries_h2", "tcd_total_persistence"]:
            if key in results:
                logger.info(f"  {key:35s}: {results[key]}")

        # Save
        results_dir = Path(log_dir) / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        with open(results_dir / "latent_ocean_eval.json", "w") as f:
            # Convert non-serializable values
            serializable = {k: (bool(v) if isinstance(v, (np.bool_,)) else v) for k, v in results.items()}
            json.dump(serializable, f, indent=2, default=str)
        logger.info(f"\nResults saved to {results_dir / 'latent_ocean_eval.json'}")


if __name__ == "__main__":
    main()
