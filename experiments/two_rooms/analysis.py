"""Two Rooms experiment result analysis and plotting."""

import json
from pathlib import Path


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


def plot_two_rooms_analysis(results_dir: str) -> None:
    """Generate comprehensive Two Rooms analysis plots.

    Creates:
    - Loss comparison bar chart
    - Module formation timeline
    - Environment structure visualization
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    results = load_results(results_dir)
    output_dir = Path(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Loss comparison bar chart
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    methods = ["Vanilla JEPA", "TCD-JEPA"]
    losses = [
        results.get("vanilla_final_loss", 0),
        results.get("tcd_final_loss", 0),
    ]
    colors = ["#4A90D9", "#E74C3C"]
    bars = ax.bar(methods, losses, color=colors, edgecolor="black", linewidth=0.5)

    for bar, loss in zip(bars, losses):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{loss:.4f}", ha="center", va="bottom", fontsize=11)

    ax.set_ylabel("Final Loss")
    ax.set_title("Two Rooms: Final Loss Comparison")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(output_dir / "loss_bar_chart.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. Summary card
    vanilla_loss = results.get("vanilla_final_loss", 0)
    tcd_loss = results.get("tcd_final_loss", 0)
    improvement = (vanilla_loss - tcd_loss) / vanilla_loss * 100 if vanilla_loss > 0 else 0
    num_modules = results.get("num_modules_final", 0)
    converged = results.get("converged", False)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.axis("off")
    summary_text = (
        f"Two Rooms Experiment Summary\n"
        f"{'─' * 35}\n"
        f"Vanilla JEPA loss:  {vanilla_loss:.4f}\n"
        f"TCD-JEPA loss:      {tcd_loss:.4f}\n"
        f"Improvement:        {improvement:+.1f}%\n"
        f"Modules formed:     {num_modules}\n"
        f"Converged:          {converged}"
    )
    ax.text(0.5, 0.5, summary_text, transform=ax.transAxes,
            fontsize=12, verticalalignment="center", horizontalalignment="center",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", edgecolor="gray"))
    fig.tight_layout()
    fig.savefig(str(output_dir / "summary_card.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Two Rooms analysis plots saved to {output_dir}")


if __name__ == "__main__":
    import sys
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "./logs/two_rooms"
    print(compare_runs(results_dir))
    plot_two_rooms_analysis(results_dir)
