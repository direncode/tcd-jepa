#!/usr/bin/env python
"""Convert ogbn-arxiv citation network to Latent Ocean format.

Downloads the OGB arxiv dataset (111K papers, 1.2M citation edges, 40 subject areas)
and converts it into Latent Ocean manifold format for TCD-JEPA training at scale.

- Entities = academic papers (111,059)
- Fingerprints = 128D Word2Vec embeddings (projected to 384D)
- Coordinates = projected onto S² via spectral embedding of the citation graph
- Causal links = directed citation edges (paper A cites paper B)
- Entity types = 40 arXiv subject areas
- Velocities = temporal signal from publication year

Requirements:
    pip install ogb torch numpy scipy

Usage:
    python scripts/data_adapters/ogbn_arxiv.py --output ./data/arxiv
    python train_manifold.py --config configs/manifold_large.yaml --tcd --eval \\
        data.dataset=latent_ocean data.data_dir=./data/arxiv

Source: https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("arxiv_adapter")


def load_ogbn_arxiv() -> dict:
    """Load ogbn-arxiv via the OGB package."""
    try:
        from ogb.nodeproppred import NodePropPredDataset
    except ImportError:
        raise ImportError(
            "ogb package required. Install with: pip install ogb\n"
            "Dataset will be downloaded automatically (~200MB)"
        )

    logger.info("Loading ogbn-arxiv dataset (will download on first run)...")
    dataset = NodePropPredDataset(name="ogbn-arxiv", root="./data/ogb")
    graph, labels = dataset[0]

    logger.info(f"  Nodes: {graph['num_nodes']:,}")
    logger.info(f"  Edges: {graph['edge_index'].shape[1]:,}")
    logger.info(f"  Features: {graph['node_feat'].shape}")
    logger.info(f"  Classes: {labels.max().item() + 1}")

    return {
        "node_feat": graph["node_feat"],  # [N, 128] Word2Vec
        "edge_index": graph["edge_index"],  # [2, E]
        "labels": labels.flatten(),  # [N]
        "num_nodes": graph["num_nodes"],
    }


def build_fingerprints(node_feat: np.ndarray, target_dim: int = 384) -> np.ndarray:
    """Project 128D Word2Vec features to 384D via random projection + padding."""
    n, d = node_feat.shape
    rng = np.random.RandomState(42)

    if d < target_dim:
        # Random projection to fill remaining dims
        projection = rng.randn(d, target_dim - d).astype(np.float32) * 0.1
        projected = node_feat @ projection
        fingerprints = np.concatenate([node_feat, projected], axis=1)
    else:
        projection = rng.randn(d, target_dim).astype(np.float32) / np.sqrt(d)
        fingerprints = node_feat @ projection

    # L2 normalize
    norms = np.linalg.norm(fingerprints, axis=1, keepdims=True)
    fingerprints = fingerprints / np.clip(norms, 1e-8, None)

    logger.info(f"Fingerprints: {fingerprints.shape}")
    return fingerprints.astype(np.float32)


def build_coordinates(
    edge_index: np.ndarray, n: int, sphere_radius: float = 4.5
) -> np.ndarray:
    """Project papers onto S² using approximate spectral embedding.

    For 111K nodes, exact spectral embedding is too slow. We use
    randomized SVD on the normalized adjacency for speed.
    """
    logger.info("Computing spectral embedding for S² coordinates...")

    from scipy.sparse import csr_matrix

    # Build sparse symmetric adjacency
    rows = np.concatenate([edge_index[0], edge_index[1]])
    cols = np.concatenate([edge_index[1], edge_index[0]])
    data = np.ones(len(rows), dtype=np.float32)
    adj = csr_matrix((data, (rows, cols)), shape=(n, n))

    # Normalized adjacency: D^{-1/2} A D^{-1/2}
    degree = np.array(adj.sum(axis=1)).flatten()
    degree_inv_sqrt = np.where(degree > 0, 1.0 / np.sqrt(degree), 0)

    # Randomized SVD for top eigenvectors (fast for large sparse matrices)
    from scipy.sparse import diags
    from scipy.sparse.linalg import eigsh

    D_inv_sqrt = diags(degree_inv_sqrt)
    L_norm = diags(np.ones(n)) - D_inv_sqrt @ adj @ D_inv_sqrt

    logger.info("  Running eigsh for 3 smallest eigenvectors...")
    try:
        eigenvalues, eigenvectors = eigsh(L_norm, k=3, which="SM", maxiter=500)
        embed_2d = eigenvectors[:, 1:3]  # Skip trivial first eigenvector
    except Exception as e:
        logger.warning(f"  eigsh failed ({e}), using random projection")
        rng = np.random.RandomState(42)
        embed_2d = rng.randn(n, 2).astype(np.float32)

    # Map to S²
    norms = np.linalg.norm(embed_2d, axis=1, keepdims=True)
    embed_2d = embed_2d / np.clip(norms, 1e-8, None)

    theta = np.arccos(np.clip(embed_2d[:, 0], -1, 1))
    phi = np.arctan2(embed_2d[:, 1], embed_2d[:, 0]) + np.pi

    # Add small jitter
    rng = np.random.RandomState(42)
    theta += rng.normal(0, 0.02, n)
    phi += rng.normal(0, 0.02, n)
    theta = np.clip(theta, 0.01, np.pi - 0.01)

    x = sphere_radius * np.sin(theta) * np.cos(phi)
    y = sphere_radius * np.sin(theta) * np.sin(phi)
    z = sphere_radius * np.cos(theta)
    coords = np.stack([x, y, z], axis=1)

    logger.info(f"Coordinates: {coords.shape}")
    return coords.astype(np.float32)


def build_edges_tensor(edge_index: np.ndarray) -> torch.Tensor:
    """Build [E, 3] edge tensor: (source, target, weight)."""
    E = edge_index.shape[1]
    edges = torch.zeros(E, 3)
    edges[:, 0] = torch.from_numpy(edge_index[0])
    edges[:, 1] = torch.from_numpy(edge_index[1])
    edges[:, 2] = 0.8  # Uniform citation weight
    logger.info(f"Edges tensor: {edges.shape}")
    return edges


def build_velocities(n: int, coords: np.ndarray) -> np.ndarray:
    """Build small random tangent-space velocities."""
    rng = np.random.RandomState(42)
    velocity = rng.randn(n, 3).astype(np.float32) * 0.02
    for i in range(n):
        normal = coords[i] / (np.linalg.norm(coords[i]) + 1e-8)
        velocity[i] -= np.dot(velocity[i], normal) * normal
    logger.info(f"Velocities: {velocity.shape}")
    return velocity


def main():
    parser = argparse.ArgumentParser(description="Convert ogbn-arxiv to Latent Ocean format")
    parser.add_argument("--output", default="./data/arxiv", help="Output directory")
    parser.add_argument("--sphere-radius", type=float, default=4.5)
    parser.add_argument("--fingerprint-dim", type=int, default=384)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    data = load_ogbn_arxiv()
    n = data["num_nodes"]

    # Build all tensors
    fingerprints = build_fingerprints(data["node_feat"], target_dim=args.fingerprint_dim)
    coords = build_coordinates(data["edge_index"], n, sphere_radius=args.sphere_radius)
    velocity = build_velocities(n, coords)
    edges = build_edges_tensor(data["edge_index"])
    labels = data["labels"]

    # Save in Latent Ocean format (sparse for adjacency)
    torch.save(torch.from_numpy(fingerprints), output_dir / "fingerprints.pt")
    torch.save(torch.from_numpy(coords), output_dir / "coords.pt")
    torch.save(torch.from_numpy(velocity), output_dir / "velocity.pt")
    torch.save(edges, output_dir / "edges.pt")  # Sparse edge format
    torch.save(torch.from_numpy(labels.astype(np.int64)), output_dir / "labels.pt")

    # Save metadata
    import json
    metadata = {
        "source": "ogbn-arxiv",
        "num_entities": n,
        "num_edges": int(edges.shape[0]),
        "num_classes": int(labels.max()) + 1,
        "fingerprint_dim": args.fingerprint_dim,
        "sphere_radius": args.sphere_radius,
        "format": "sparse_edges",
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"\nSaved to {output_dir}:")
    logger.info(f"  fingerprints.pt: [{n:,}, {args.fingerprint_dim}]")
    logger.info(f"  coords.pt:      [{n:,}, 3]")
    logger.info(f"  velocity.pt:    [{n:,}, 3]")
    logger.info(f"  edges.pt:       [{edges.shape[0]:,}, 3] (sparse)")
    logger.info(f"  labels.pt:      [{n:,}] ({int(labels.max()) + 1} classes)")
    logger.info(f"\nTotal memory: ~{(n * args.fingerprint_dim * 4 + n * 3 * 4 * 2 + edges.shape[0] * 3 * 4) / 1e6:.0f} MB")
    logger.info(f"\nReady for: python train_manifold.py --config configs/manifold_large.yaml --tcd --eval "
                f"data.dataset=latent_ocean data.data_dir={output_dir}")


if __name__ == "__main__":
    main()
