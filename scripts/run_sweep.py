#!/usr/bin/env python
"""Launch a wandb hyperparameter sweep for TCD-JEPA.

Usage:
    python scripts/run_sweep.py --config configs/sweep.yaml --count 50
    python scripts/run_sweep.py --config configs/sweep.yaml --count 20 --base-config configs/small_scale.yaml
"""

import argparse
import subprocess
import sys

import yaml


def main():
    parser = argparse.ArgumentParser(description="Run TCD-JEPA hyperparameter sweep")
    parser.add_argument("--config", required=True, help="Path to wandb sweep YAML config")
    parser.add_argument("--base-config", default="configs/small_scale.yaml",
                        help="Base training config to sweep over")
    parser.add_argument("--count", type=int, default=50, help="Number of sweep runs")
    parser.add_argument("--project", default="tcd-jepa-sweep", help="wandb project name")
    parser.add_argument("--entity", default=None, help="wandb entity/team name")
    args = parser.parse_args()

    try:
        import wandb  # noqa: F401
    except ImportError:
        print("Error: wandb not installed. Run: pip install wandb")
        sys.exit(1)

    # Load and validate sweep config
    with open(args.config) as f:
        sweep_cfg = yaml.safe_load(f)

    print(f"Sweep config: {args.config}")
    print(f"Base config: {args.base_config}")
    print(f"Method: {sweep_cfg.get('method', 'bayes')}")
    print(f"Parameters: {len(sweep_cfg.get('parameters', {}))}")
    print(f"Runs: {args.count}")

    # Initialize sweep
    sweep_args = ["wandb", "sweep", args.config]
    if args.project:
        sweep_args.extend(["--project", args.project])
    if args.entity:
        sweep_args.extend(["--entity", args.entity])

    print("\nTo launch manually:")
    print(f"  wandb sweep {args.config}")
    print(f"  wandb agent <SWEEP_ID> --count {args.count}")

    # Launch sweep
    result = subprocess.run(sweep_args, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"wandb sweep failed:\n{result.stderr}")
        sys.exit(1)

    # Extract sweep ID from output
    for line in result.stdout.splitlines() + result.stderr.splitlines():
        if "wandb agent" in line:
            print(f"\nLaunching agent: {line.strip()}")
            sweep_id = line.strip().split()[-1]
            subprocess.run(["wandb", "agent", sweep_id, "--count", str(args.count)])
            return

    print("Could not extract sweep ID from wandb output.")
    print(result.stdout)


if __name__ == "__main__":
    main()
