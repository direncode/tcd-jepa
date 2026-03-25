#!/usr/bin/env python
"""Automated ablation study runner for TCD-JEPA.

Runs a matrix of experiments varying one factor at a time against a baseline.
Results are saved to a structured directory and optionally logged to wandb.

Usage:
    python scripts/run_ablation.py --base-config configs/ablation.yaml
    python scripts/run_ablation.py --base-config configs/ablation.yaml --ablations depth,masking --seeds 3
    python scripts/run_ablation.py --base-config configs/ablation.yaml --dry-run
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("ablation")

# Ablation factor definitions: name -> list of (description, overrides)
ABLATION_FACTORS = {
    "tcd": {
        "description": "TCD recursive loop on/off",
        "runs": [
            {"name": "no_tcd", "args": [], "description": "Baseline without TCD"},
            {"name": "with_tcd", "args": ["--tcd"], "description": "With TCD enabled"},
        ],
    },
    "depth": {
        "description": "Encoder depth",
        "runs": [
            {"name": "depth_2", "overrides": ["model.encoder.depth=2", "model.encoder.num_heads=2", "model.encoder.embed_dim=64"], "description": "Tiny (2 layers)"},
            {"name": "depth_4", "overrides": ["model.encoder.depth=4"], "description": "Shallow (4 layers)"},
            {"name": "depth_6", "overrides": [], "description": "Baseline (6 layers)"},
            {"name": "depth_12", "overrides": ["model.encoder.depth=12", "model.encoder.num_heads=6", "model.encoder.embed_dim=384"], "description": "Deep (12 layers)"},
        ],
    },
    "masking": {
        "description": "Masking strategy",
        "runs": [
            {"name": "mask_small", "overrides": ["masking.enc_mask_scale=[0.1,0.15]", "masking.pred_mask_scale=[0.15,0.25]"], "description": "Small masks"},
            {"name": "mask_baseline", "overrides": [], "description": "Baseline masks"},
            {"name": "mask_large", "overrides": ["masking.enc_mask_scale=[0.2,0.35]", "masking.pred_mask_scale=[0.3,0.5]"], "description": "Large masks"},
        ],
    },
    "lr": {
        "description": "Learning rate",
        "runs": [
            {"name": "lr_low", "overrides": ["training.learning_rate=0.0003"], "description": "Low LR"},
            {"name": "lr_baseline", "overrides": [], "description": "Baseline LR"},
            {"name": "lr_high", "overrides": ["training.learning_rate=0.003"], "description": "High LR"},
        ],
    },
    "ema": {
        "description": "EMA momentum schedule",
        "runs": [
            {"name": "ema_slow", "overrides": ["model.ema.start=0.999"], "description": "Slow EMA (0.999)"},
            {"name": "ema_baseline", "overrides": [], "description": "Baseline EMA (0.996)"},
            {"name": "ema_fast", "overrides": ["model.ema.start=0.99"], "description": "Fast EMA (0.99)"},
        ],
    },
    "modules": {
        "description": "TCD module count limit (requires --tcd)",
        "runs": [
            {"name": "modules_4", "args": ["--tcd"], "overrides": ["tcd.max_modules=4"], "description": "Max 4 modules"},
            {"name": "modules_8", "args": ["--tcd"], "overrides": ["tcd.max_modules=8"], "description": "Max 8 modules"},
            {"name": "modules_16", "args": ["--tcd"], "overrides": ["tcd.max_modules=16"], "description": "Max 16 modules"},
        ],
    },
}


def run_experiment(
    base_config: str,
    name: str,
    overrides: list[str],
    extra_args: list[str],
    log_dir: str,
    seed: int,
    dry_run: bool = False,
) -> dict:
    """Run a single training experiment."""
    run_dir = Path(log_dir) / name / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "train.py",
        "--config", base_config,
        *extra_args,
        *overrides,
        f"training.seed={seed}",
        f"logging.log_dir={run_dir}",
        "logging.use_wandb=false",
    ]

    logger.info(f"  [{name}/seed_{seed}] {' '.join(cmd)}")

    if dry_run:
        return {"name": name, "seed": seed, "status": "dry_run", "cmd": cmd}

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    elapsed = time.time() - t0

    status = "success" if result.returncode == 0 else "failed"
    if status == "failed":
        logger.error(f"  [{name}/seed_{seed}] FAILED (exit {result.returncode})")
        logger.error(f"  stderr: {result.stderr[-500:]}")

    run_result = {
        "name": name,
        "seed": seed,
        "status": status,
        "elapsed_seconds": elapsed,
        "return_code": result.returncode,
    }

    # Save run metadata
    with open(run_dir / "run_info.json", "w") as f:
        json.dump(run_result, f, indent=2)

    return run_result


def main():
    parser = argparse.ArgumentParser(description="Run TCD-JEPA ablation study")
    parser.add_argument("--base-config", default="configs/ablation.yaml",
                        help="Base config for ablation experiments")
    parser.add_argument("--ablations", default=None,
                        help="Comma-separated ablation factors to run (default: all)")
    parser.add_argument("--seeds", type=int, default=3, help="Number of seeds per experiment")
    parser.add_argument("--log-dir", default="./logs/ablation", help="Root log directory")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    parser.add_argument("--list", action="store_true", help="List available ablation factors")
    args = parser.parse_args()

    if args.list:
        print("Available ablation factors:")
        for name, factor in ABLATION_FACTORS.items():
            print(f"  {name}: {factor['description']}")
            for run in factor["runs"]:
                print(f"    - {run['name']}: {run['description']}")
        return

    # Select factors
    if args.ablations:
        factors = args.ablations.split(",")
        for f in factors:
            if f not in ABLATION_FACTORS:
                logger.error(f"Unknown ablation factor: {f}")
                logger.info(f"Available: {list(ABLATION_FACTORS.keys())}")
                sys.exit(1)
    else:
        factors = list(ABLATION_FACTORS.keys())

    # Count total runs
    total_runs = sum(
        len(ABLATION_FACTORS[f]["runs"]) * args.seeds
        for f in factors
    )
    logger.info(f"Ablation study: {len(factors)} factors, {total_runs} total runs")
    logger.info(f"Factors: {factors}")
    logger.info(f"Seeds: {args.seeds}")
    logger.info(f"Base config: {args.base_config}")
    logger.info(f"Log dir: {args.log_dir}")

    all_results = []

    for factor_name in factors:
        factor = ABLATION_FACTORS[factor_name]
        logger.info(f"\n{'='*60}")
        logger.info(f"Factor: {factor_name} — {factor['description']}")
        logger.info(f"{'='*60}")

        for run in factor["runs"]:
            for seed in range(args.seeds):
                result = run_experiment(
                    base_config=args.base_config,
                    name=f"{factor_name}/{run['name']}",
                    overrides=run.get("overrides", []),
                    extra_args=run.get("args", []),
                    log_dir=args.log_dir,
                    seed=seed + 42,
                    dry_run=args.dry_run,
                )
                all_results.append(result)

    # Summary
    summary_path = Path(args.log_dir) / "ablation_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    succeeded = sum(1 for r in all_results if r["status"] == "success")
    failed = sum(1 for r in all_results if r["status"] == "failed")
    logger.info(f"\nAblation complete: {succeeded} succeeded, {failed} failed")
    logger.info(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
