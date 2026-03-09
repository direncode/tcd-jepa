"""Two Rooms experiment result analysis and plotting."""

import json
from pathlib import Path
from typing import Optional

import numpy as np


def load_results(results_dir: str) -> dict:
    """Load experiment results from JSON."""
    path = Path(results_dir) / "results.json"
    with open(path) as f:
        return json.load(f)


def compare_runs(results_dir: str) -> str:
    """Generate a text summary comparing vanilla JEPA and TCD-JEPA."""
    results = load_results(results_dir)

    vanilla_loss = results.get("vanilla_final_loss", float("inf"))
    tcd_loss = results.get("tcd_final_loss", float("inf"))
    num_modules = results.get("num_modules_final", 0)
    converged = results.get("converged", False)

    improvement = (vanilla_loss - tcd_loss) / vanilla_loss * 100 if vanilla_loss > 0 else 0

    summary = [
        "=== Two Rooms Experiment Results ===",
        f"Vanilla JEPA final loss: {vanilla_loss:.4f}",
        f"TCD-JEPA final loss:     {tcd_loss:.4f}",
        f"Improvement:             {improvement:.1f}%",
        f"Modules formed:          {num_modules}",
        f"Converged:               {converged}",
    ]
    return "\n".join(summary)


if __name__ == "__main__":
    import sys
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "./logs/two_rooms"
    print(compare_runs(results_dir))
