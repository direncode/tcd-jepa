"""Visualization utilities for energy landscapes, trajectories, and modules.

All functions save to files rather than displaying interactively.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch

# Lazy imports to avoid matplotlib import overhead when not visualizing
_MPL_AVAILABLE = None


def _check_matplotlib():
    global _MPL_AVAILABLE
    if _MPL_AVAILABLE is None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            _MPL_AVAILABLE = True
        except ImportError:
            _MPL_AVAILABLE = False
    return _MPL_AVAILABLE


def plot_energy_heatmap(
    z_grid: torch.Tensor,
    energies: torch.Tensor,
    save_path: str,
    title: str = "Energy Landscape",
) -> None:
    """Plot 2D energy landscape heatmap.

    If z_grid is high-dimensional, projects to 2D via PCA.

    Args:
        z_grid: Grid points [N, D].
        energies: Energy values [N].
        save_path: Path to save the figure.
        title: Plot title.
    """
    if not _check_matplotlib():
        return
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    points = z_grid.detach().cpu().numpy()
    e = energies.detach().cpu().numpy()

    # Project to 2D if needed
    if points.shape[1] > 2:
        pca = PCA(n_components=2)
        points_2d = pca.fit_transform(points)
    else:
        points_2d = points

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    scatter = ax.scatter(
        points_2d[:, 0], points_2d[:, 1],
        c=e, cmap="viridis", s=10, alpha=0.7
    )
    plt.colorbar(scatter, ax=ax, label="Energy")
    ax.set_xlabel("PC1" if points.shape[1] > 2 else "z1")
    ax.set_ylabel("PC2" if points.shape[1] > 2 else "z2")
    ax.set_title(title)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_trajectory(
    trajectory: torch.Tensor,
    save_path: str,
    energies: Optional[torch.Tensor] = None,
    title: str = "Exploration Trajectory",
) -> None:
    """Plot exploration trajectory in 2D projection.

    Args:
        trajectory: [T, B, D] or [T, D] trajectory tensor.
        save_path: Path to save figure.
        energies: Optional energy values along trajectory.
        title: Plot title.
    """
    if not _check_matplotlib():
        return
    import matplotlib.pyplot as plt

    traj = trajectory.detach().cpu().numpy()
    if traj.ndim == 3:
        # Take first sample from batch
        traj = traj[:, 0, :]

    T, D = traj.shape

    # Project to 2D
    if D > 2:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        traj_2d = pca.fit_transform(traj)
    else:
        traj_2d = traj

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Color by time step
    colors = np.linspace(0, 1, T)
    scatter = ax.scatter(
        traj_2d[:, 0], traj_2d[:, 1],
        c=colors, cmap="coolwarm", s=15, alpha=0.8
    )
    # Draw path
    ax.plot(traj_2d[:, 0], traj_2d[:, 1], 'k-', alpha=0.2, linewidth=0.5)
    # Mark start and end
    ax.plot(traj_2d[0, 0], traj_2d[0, 1], 'go', markersize=10, label="Start")
    ax.plot(traj_2d[-1, 0], traj_2d[-1, 1], 'rs', markersize=10, label="End")

    plt.colorbar(scatter, ax=ax, label="Time step")
    ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_persistence_diagram(
    diagram: np.ndarray,
    save_path: str,
    title: str = "Persistence Diagram",
) -> None:
    """Plot persistence diagram (birth vs death).

    Args:
        diagram: [K, 3] array with (birth, death, dimension) per feature.
        save_path: Path to save figure.
        title: Plot title.
    """
    if not _check_matplotlib():
        return
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))

    dim_colors = {0: "tab:blue", 1: "tab:orange", 2: "tab:green"}
    dim_labels = {0: "$H_0$ (components)", 1: "$H_1$ (loops)", 2: "$H_2$ (voids)"}

    if len(diagram) > 0:
        max_val = max(diagram[:, 1].max(), diagram[:, 0].max()) * 1.1
    else:
        max_val = 1.0

    # Diagonal line
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label="diagonal")

    for dim in [0, 1, 2]:
        mask = diagram[:, 2] == dim if len(diagram) > 0 else np.array([])
        if not np.any(mask):
            continue
        points = diagram[mask]
        ax.scatter(
            points[:, 0], points[:, 1],
            c=dim_colors[dim], label=dim_labels[dim],
            s=30, alpha=0.7, edgecolors="black", linewidth=0.5
        )

    ax.set_xlabel("Birth")
    ax.set_ylabel("Death")
    ax.set_title(title)
    ax.legend()
    ax.set_aspect("equal")
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_module_formation(
    module_counts: list[int],
    convergence_scores: list[float],
    save_path: str,
    title: str = "Module Formation Over Time",
) -> None:
    """Plot module count and convergence score over training.

    Args:
        module_counts: Number of active modules at each checkpoint.
        convergence_scores: Convergence C(t) at each checkpoint.
        save_path: Path to save figure.
        title: Plot title.
    """
    if not _check_matplotlib():
        return
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    steps = range(len(module_counts))
    ax1.plot(steps, module_counts, 'b-', linewidth=2)
    ax1.set_ylabel("Active Modules")
    ax1.set_title(title)

    ax2.plot(range(len(convergence_scores)), convergence_scores, 'r-', linewidth=2)
    ax2.set_ylabel("Convergence Score C(t)")
    ax2.set_xlabel("Recursive Loop Iteration")
    ax2.axhline(y=0.01, color='g', linestyle='--', alpha=0.5, label="$\\epsilon$ threshold")
    ax2.legend()

    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_loss_curves(
    losses: dict[str, list[float]],
    save_path: str,
    title: str = "Training Loss",
) -> None:
    """Plot training loss curves for multiple runs/ablations.

    Args:
        losses: Dict mapping run name -> list of loss values.
        save_path: Path to save figure.
        title: Plot title.
    """
    if not _check_matplotlib():
        return
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    for name, loss_values in losses.items():
        ax.plot(loss_values, label=name, linewidth=2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
