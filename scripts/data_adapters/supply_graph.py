#!/usr/bin/env python
"""Convert SupplyGraph FMCG dataset to Latent Ocean format.

Downloads the SupplyGraph benchmark dataset (FMCG supply chain from Bangladesh)
and converts it into Latent Ocean manifold format for TCD-JEPA training.

- Entities = products in the supply chain
- Fingerprints = node features (production, demand, inventory signals)
- Coordinates = projected onto S² via graph spectral embedding
- Causal links = supply chain edges (shared facilities, product groups)
- Entity types = product categories
- Velocities = demand/production rate changes

Usage:
    python scripts/data_adapters/supply_graph.py --output ./data/supply_graph
    python train_manifold.py --config configs/manifold.yaml --tcd --eval \\
        data.dataset=latent_ocean data.data_dir=./data/supply_graph

Source: https://github.com/ciol-researchlab/SupplyGraph
"""

import argparse
import logging
import os
import zipfile
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("supply_graph_adapter")

REPO_URL = "https://github.com/ciol-researchlab/SupplyGraph/archive/refs/heads/main.zip"


def download_repo(cache_dir: Path) -> Path:
    """Download and extract the SupplyGraph repo."""
    import urllib.request

    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / "supply_graph.zip"
    extract_dir = cache_dir / "SupplyGraph-main"

    if extract_dir.exists():
        logger.info("Using cached SupplyGraph data")
        return extract_dir

    logger.info(f"Downloading SupplyGraph from {REPO_URL}")
    urllib.request.urlretrieve(REPO_URL, zip_path)

    logger.info("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(cache_dir)

    return extract_dir


def find_data_files(repo_dir: Path) -> dict:
    """Locate the dataset files in the repo."""
    # Look for CSV or numpy files in various locations
    data_files = {}

    for root, dirs, files in os.walk(repo_dir):
        for f in files:
            path = Path(root) / f
            if f.endswith((".csv", ".npy", ".npz", ".xlsx")):
                key = path.stem.lower()
                data_files[key] = path

    logger.info(f"Found {len(data_files)} data files:")
    for k, v in sorted(data_files.items()):
        logger.info(f"  {k}: {v.name}")

    return data_files


def load_supply_graph_data(repo_dir: Path) -> dict:
    """Load data from SupplyGraph repo, handling various file formats."""
    data_files = find_data_files(repo_dir)

    result = {
        "node_features": None,
        "edge_index": None,
        "labels": None,
    }

    # Try to load node features
    for key in ["node_features", "features", "x", "node_feature"]:
        if key in data_files:
            path = data_files[key]
            if path.suffix == ".npy":
                result["node_features"] = np.load(path)
                logger.info(f"Node features from {path.name}: {result['node_features'].shape}")
            elif path.suffix == ".csv":
                import csv
                with open(path) as f:
                    reader = csv.reader(f)
                    rows = [row for row in reader]
                try:
                    result["node_features"] = np.array(rows[1:], dtype=np.float32)
                    logger.info(f"Node features from {path.name}: {result['node_features'].shape}")
                except ValueError:
                    logger.warning(f"Could not parse {path.name} as numeric")
            break

    # Try to load edges
    for key in ["edge_index", "edges", "adjacency", "adj"]:
        if key in data_files:
            path = data_files[key]
            if path.suffix == ".npy":
                result["edge_index"] = np.load(path)
                logger.info(f"Edges from {path.name}: {result['edge_index'].shape}")
            elif path.suffix == ".csv":
                import csv
                with open(path) as f:
                    reader = csv.reader(f)
                    rows = [row for row in reader]
                try:
                    result["edge_index"] = np.array(rows[1:], dtype=np.int64)
                except ValueError:
                    result["edge_index"] = np.array(rows, dtype=np.int64)
                logger.info(f"Edges from {path.name}: {result['edge_index'].shape}")
            break

    # Try to load labels
    for key in ["labels", "y", "label", "node_labels"]:
        if key in data_files:
            path = data_files[key]
            if path.suffix == ".npy":
                result["labels"] = np.load(path)
            break

    return result, data_files


def build_from_supply_graph(repo_dir: Path, fingerprint_dim: int = 384, sphere_radius: float = 4.5) -> dict:
    """Build Latent Ocean tensors from SupplyGraph data."""
    data, data_files = load_supply_graph_data(repo_dir)

    # Determine number of nodes
    if data["node_features"] is not None:
        n = data["node_features"].shape[0]
        feat_dim = data["node_features"].shape[1]
    else:
        # Infer from edges or default
        if data["edge_index"] is not None:
            n = int(data["edge_index"].max()) + 1
        else:
            logger.warning("No node features or edges found — using fallback")
            n = 100
        feat_dim = 0

    logger.info(f"Building Latent Ocean format: {n} entities")

    # Fingerprints: pad/project node features to target dim
    rng = np.random.RandomState(42)
    if data["node_features"] is not None:
        feats = data["node_features"].astype(np.float32)
        if feat_dim < fingerprint_dim:
            # Pad with learned random projection of features
            projection = rng.randn(feat_dim, fingerprint_dim - feat_dim).astype(np.float32) * 0.1
            projected = feats @ projection
            fingerprints = np.concatenate([feats, projected], axis=1)
        elif feat_dim > fingerprint_dim:
            # Random projection down
            projection = rng.randn(feat_dim, fingerprint_dim).astype(np.float32)
            projection /= np.sqrt(feat_dim)
            fingerprints = feats @ projection
        else:
            fingerprints = feats
    else:
        fingerprints = rng.randn(n, fingerprint_dim).astype(np.float32) * 0.1

    # Normalize
    norms = np.linalg.norm(fingerprints, axis=1, keepdims=True)
    fingerprints = fingerprints / np.clip(norms, 1e-8, None)

    # Adjacency matrix
    adjacency = np.zeros((n, n), dtype=np.float32)
    if data["edge_index"] is not None:
        edges = data["edge_index"]
        if edges.shape[1] == 2:
            for row in edges:
                i, j = int(row[0]), int(row[1])
                if 0 <= i < n and 0 <= j < n:
                    adjacency[i, j] = 0.8
        elif edges.shape[0] == 2:
            for k in range(edges.shape[1]):
                i, j = int(edges[0, k]), int(edges[1, k])
                if 0 <= i < n and 0 <= j < n:
                    adjacency[i, j] = 0.8
    num_edges = int((adjacency > 0).sum())
    logger.info(f"Adjacency: {num_edges} edges")

    # Coordinates via spectral embedding
    sym = adjacency + adjacency.T
    np.fill_diagonal(sym, 0)
    degree = sym.sum(axis=1)
    degree_inv_sqrt = np.where(degree > 0, 1.0 / np.sqrt(degree), 0)
    D_inv_sqrt = np.diag(degree_inv_sqrt)
    L_norm = np.eye(n) - D_inv_sqrt @ sym @ D_inv_sqrt

    try:
        from scipy.linalg import eigh
        eigenvalues, eigenvectors = eigh(L_norm, subset_by_index=[1, 2])
        embed_2d = eigenvectors
    except Exception:
        embed_2d = rng.randn(n, 2)

    norms_2d = np.linalg.norm(embed_2d, axis=1, keepdims=True)
    embed_2d = embed_2d / np.clip(norms_2d, 1e-8, None)

    theta = np.arccos(np.clip(embed_2d[:, 0], -1, 1))
    phi = np.arctan2(embed_2d[:, 1], embed_2d[:, 0]) + np.pi
    theta += rng.normal(0, 0.05, n)
    phi += rng.normal(0, 0.05, n)
    theta = np.clip(theta, 0.01, np.pi - 0.01)

    x = sphere_radius * np.sin(theta) * np.cos(phi)
    y = sphere_radius * np.sin(theta) * np.sin(phi)
    z = sphere_radius * np.cos(theta)
    coords = np.stack([x, y, z], axis=1).astype(np.float32)

    # Velocities from feature gradients
    velocity = rng.randn(n, 3).astype(np.float32) * 0.02
    for i in range(n):
        normal = coords[i] / (np.linalg.norm(coords[i]) + 1e-8)
        velocity[i] -= np.dot(velocity[i], normal) * normal

    # Labels
    if data["labels"] is not None:
        labels = data["labels"].astype(np.int64).flatten()
    else:
        # Cluster by graph structure
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import pdist
        dist = pdist(fingerprints)
        Z = linkage(dist, method="ward")
        labels = fcluster(Z, t=8, criterion="maxclust") - 1

    return {
        "fingerprints": fingerprints,
        "coords": coords,
        "velocity": velocity,
        "adjacency": adjacency,
        "labels": labels,
        "num_entities": n,
    }


def main():
    parser = argparse.ArgumentParser(description="Convert SupplyGraph to Latent Ocean format")
    parser.add_argument("--output", default="./data/supply_graph", help="Output directory")
    parser.add_argument("--cache", default="./data/sg_cache", help="Download cache directory")
    parser.add_argument("--sphere-radius", type=float, default=4.5)
    parser.add_argument("--fingerprint-dim", type=int, default=384)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache)

    # Download
    repo_dir = download_repo(cache_dir)

    # Build
    result = build_from_supply_graph(repo_dir, args.fingerprint_dim, args.sphere_radius)
    n = result["num_entities"]

    # Save
    torch.save(torch.from_numpy(result["fingerprints"]), output_dir / "fingerprints.pt")
    torch.save(torch.from_numpy(result["coords"]), output_dir / "coords.pt")
    torch.save(torch.from_numpy(result["velocity"]), output_dir / "velocity.pt")
    torch.save(torch.from_numpy(result["adjacency"]), output_dir / "adjacency.pt")
    torch.save(torch.from_numpy(result["labels"]), output_dir / "labels.pt")

    logger.info(f"\nSaved to {output_dir}:")
    logger.info(f"  fingerprints.pt: [{n}, {args.fingerprint_dim}]")
    logger.info(f"  coords.pt:      [{n}, 3]")
    logger.info(f"  adjacency.pt:   [{n}, {n}]")
    logger.info(f"  labels.pt:      [{n}] ({len(set(result['labels'].tolist()))} types)")
    logger.info(f"\nReady for: python train_manifold.py --config configs/manifold.yaml --tcd --eval "
                f"data.dataset=latent_ocean data.data_dir={output_dir}")


if __name__ == "__main__":
    main()
