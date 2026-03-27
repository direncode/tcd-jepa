#!/usr/bin/env python
"""TCD-JEPA ablation study: isolate contribution of each component.

Tests four conditions to determine what drives TCD's performance:
1. Baseline JEPA (no modules, no PH)
2. JEPA + random modules (modules assigned randomly, not from PH)
3. JEPA + PH modules but no crystallization routing (modules exist but frozen)
4. Full TCD-JEPA (PH → crystallization → dynamic routing)

This isolates whether the gains come from:
- Having extra parameters (random modules would show this)
- The topological feature extraction (PH vs random assignment)
- The dynamic routing (crystallized vs frozen modules)

Usage:
    python scripts/run_ablation_tcd.py --data-dir ./data/semiconductor
    python scripts/run_ablation_tcd.py --data-dir ./data/gdelt --epochs 100
"""

import argparse
import json
import logging
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, ".")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("ablation")

from tcd_jepa.core.recursive_loop import RecursiveLoop  # noqa: E402
from tcd_jepa.manifold.dataset import LatentOceanDataset  # noqa: E402
from tcd_jepa.manifold.evaluation import run_full_manifold_evaluation  # noqa: E402
from tcd_jepa.manifold.masking import ManifoldMaskCollator  # noqa: E402
from tcd_jepa.manifold.model import build_manifold_jepa  # noqa: E402
from tcd_jepa.manifold.trainer import ManifoldTrainer  # noqa: E402
from tcd_jepa.models.target_encoder import momentum_schedule  # noqa: E402
from tcd_jepa.modules.module_factory import ModuleFactory  # noqa: E402
from tcd_jepa.training.schedulers import CosineWDSchedule, WarmupCosineSchedule  # noqa: E402
from tcd_jepa.training.trainer import build_optimizer  # noqa: E402


def build_trainer(model, dataset, cfg_epochs, recursive_loop=None, log_dir="./logs/ablation"):
    """Build a ManifoldTrainer with standard settings."""
    collator = ManifoldMaskCollator(
        num_tokens=64, context_ratio=(0.3, 0.5), target_ratio=(0.15, 0.3),
        num_context_masks=4, num_target_masks=1, min_keep=4,
        use_geodesic=True, sphere_radius=4.5,
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0,
                        collate_fn=collator, drop_last=True)

    steps = len(loader)
    total = cfg_epochs * steps
    opt = build_optimizer(model, lr=0.001, weight_decay=0.05)
    lr_s = WarmupCosineSchedule(opt, 5 * steps, 1e-4, 0.001, total)
    wd_s = CosineWDSchedule(opt, 0.05, total)
    ema_s = momentum_schedule(0.996, 1.0, total)

    return ManifoldTrainer(
        model=model, optimizer=opt, lr_scheduler=lr_s,
        wd_scheduler=wd_s, momentum_schedule=ema_s,
        train_loader=loader, device=device, cfg={"training": {"grad_clip_norm": 1.0, "checkpoint_freq": 999}},
        checkpoint_dir=f"{log_dir}/ckpt", recursive_loop=recursive_loop,
    )


def run_condition(name, dataset, epochs, use_dynamic=False, use_tcd=False, use_random_modules=False):
    """Run one ablation condition and return results."""
    logger.info(f"\n{'=' * 60}")
    logger.info(f"  CONDITION: {name}")
    logger.info(f"{'=' * 60}")

    model = build_manifold_jepa(
        fingerprint_dim=384, num_tokens=64, coord_dim=3,
        embed_dim=192, depth=6, num_heads=3,
        predictor_embed_dim=96, predictor_depth=4, predictor_num_heads=3,
        use_dynamic_predictor=use_dynamic,
        use_causal_encoding=True, use_velocity_encoding=True, sphere_radius=4.5,
    ).to(device)

    recursive_loop = None
    if use_tcd:
        recursive_loop = RecursiveLoop(
            embed_dim=192, explore_every=3, crystallize_every=6,
            langevin_steps=30, persistence_threshold=0.2, max_modules=16, device=device,
        )
        model.set_module_registry(recursive_loop.crystallizer.registry)

    if use_random_modules and use_dynamic:
        # Add random modules directly to the registry (bypassing PH)
        factory = ModuleFactory(embed_dim=192)
        for i in range(8):
            # Create dummy topological features
            class DummyFeature:
                module_type = ["attractor", "cycle", "boundary"][i % 3]
                centroid = torch.randn(192).numpy()
                persistence = 1.0
                dimension = i % 3
                dim = i % 3
                birth = 0.0
                death = 1.0

            module = factory.create_module(DummyFeature(), device)
            model.predictor.registry.register(module, DummyFeature(), epoch=0)
        logger.info("  Added 8 random modules (not from PH)")

    trainer = build_trainer(model, dataset, epochs, recursive_loop)

    logger.info(f"  Training {epochs} epochs...")
    trainer.train(epochs)

    # Evaluate
    results = run_full_manifold_evaluation(
        model, dataset, device, num_classes=dataset.num_clusters, max_samples=300,
        recursive_loop=recursive_loop,
    )

    # Log key metrics
    for key in ["knn_k1", "knn_k5", "link_auc", "link_sim_gap", "linear_probe_test", "clarity_score"]:
        if key in results:
            logger.info(f"  {key}: {results[key]:.4f}" if isinstance(results[key], float) else f"  {key}: {results[key]}")

    results["condition"] = name
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TCD-JEPA ablation study")
    parser.add_argument("--data-dir", default="./data/semiconductor")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset = LatentOceanDataset(args.data_dir, num_tokens=64, num_samples=1000)
    logger.info(f"Dataset: {dataset.num_entities} entities, {dataset.num_clusters} clusters")

    all_results = {}

    # Condition 1: Baseline JEPA (no modules)
    all_results["baseline"] = run_condition(
        "Baseline JEPA (no modules)", dataset, args.epochs,
        use_dynamic=False, use_tcd=False, use_random_modules=False,
    )

    # Condition 2: JEPA + random modules (extra params, no PH)
    all_results["random_modules"] = run_condition(
        "JEPA + Random Modules (no PH)", dataset, args.epochs,
        use_dynamic=True, use_tcd=False, use_random_modules=True,
    )

    # Condition 3: Full TCD-JEPA
    all_results["tcd_full"] = run_condition(
        "Full TCD-JEPA (PH + crystallization)", dataset, args.epochs,
        use_dynamic=True, use_tcd=True, use_random_modules=False,
    )

    # Summary table
    logger.info(f"\n{'=' * 80}")
    logger.info("ABLATION SUMMARY")
    logger.info(f"{'=' * 80}")
    logger.info(f"{'Condition':40s} {'Link AUC':>10s} {'kNN-1':>10s} {'LP Test':>10s}")
    logger.info("-" * 74)
    for name, r in all_results.items():
        logger.info(f"{r['condition']:40s} "
                     f"{r.get('link_auc', 0):10.4f} "
                     f"{r.get('knn_k1', 0):10.4f} "
                     f"{r.get('linear_probe_test', 0):10.4f}")

    # Save
    output_path = args.output or f"{args.data_dir}/ablation_results.json"
    with open(output_path, "w") as f:
        serializable = {}
        for k, v in all_results.items():
            serializable[k] = {kk: (bool(vv) if isinstance(vv, bool) else vv)
                                for kk, vv in v.items() if not isinstance(vv, torch.Tensor)}
        json.dump(serializable, f, indent=2, default=str)
    logger.info(f"\nResults saved to {output_path}")
