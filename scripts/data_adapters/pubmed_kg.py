#!/usr/bin/env python
"""Convert PubMed citation network to Latent Ocean format.

Downloads PubMed article metadata and citation links, builds a biomedical
knowledge graph for TCD-JEPA training.

Uses the SNAP PubMed/Cora citation datasets or the ogbn-arxiv-style
approach with PubMed data from PyTorch Geometric / OGB.

Entities = biomedical papers
Fingerprints = TF-IDF of titles/abstracts (or pre-computed node features)
Coordinates = spectral embedding on S²
Causal links = citation edges
Labels = MeSH subject categories

Usage:
    # Using Planetoid PubMed (19.7K nodes, 88K edges — quick)
    python scripts/data_adapters/pubmed_kg.py --output ./data/pubmed

    # Large scale via OGB (if available)
    python scripts/data_adapters/pubmed_kg.py --output ./data/pubmed --source ogbn

Source: https://pubmedkg.github.io/
"""

import argparse
import logging
import urllib.request
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("pubmed_adapter")

# Planetoid PubMed dataset (19.7K nodes, 88K edges, 3 classes)
PUBMED_URL = "https://github.com/kimiyoung/planetoid/raw/master/data/"
PUBMED_FILES = [
    "ind.pubmed.allx", "ind.pubmed.ally", "ind.pubmed.graph",
    "ind.pubmed.tx", "ind.pubmed.ty", "ind.pubmed.x", "ind.pubmed.y",
    "ind.pubmed.test.index",
]


def download_planetoid_pubmed(cache_dir: Path) -> Path:
    """Download Planetoid PubMed dataset."""
    cache_dir.mkdir(parents=True, exist_ok=True)

    for fname in PUBMED_FILES:
        local = cache_dir / fname
        if not local.exists():
            url = PUBMED_URL + fname
            logger.info(f"  Downloading {fname}...")
            try:
                urllib.request.urlretrieve(url, local)
            except Exception as e:
                logger.warning(f"  Failed: {e}")

    return cache_dir


def load_planetoid_pubmed(cache_dir: Path) -> dict:
    """Load Planetoid PubMed dataset using pickle."""
    import pickle

    def load_pkl(name):
        path = cache_dir / f"ind.pubmed.{name}"
        if not path.exists():
            return None
        with open(path, "rb") as f:
            try:
                return pickle.load(f, encoding="latin1")
            except Exception:
                return None

    # Load features
    allx = load_pkl("allx")  # scipy sparse matrix
    ally = load_pkl("ally")  # numpy array of labels
    graph = load_pkl("graph")  # dict of adjacency lists
    tx = load_pkl("tx")
    ty = load_pkl("ty")

    if allx is None or graph is None:
        raise FileNotFoundError("Could not load PubMed data files")

    # Combine train + test features
    import scipy.sparse as sp

    # Load test indices
    test_idx_path = cache_dir / "ind.pubmed.test.index"
    if test_idx_path.exists():
        test_indices = []
        with open(test_idx_path) as f:
            for line in f:
                test_indices.append(int(line.strip()))
        test_indices = sorted(test_indices)
    else:
        test_indices = []

    if tx is not None:
        features = sp.vstack([allx, tx]).toarray()
    else:
        features = allx.toarray() if sp.issparse(allx) else np.array(allx)

    if ty is not None:
        labels = np.vstack([ally, ty]).argmax(axis=1)
    else:
        labels = ally.argmax(axis=1) if ally.ndim > 1 else ally

    # Build edge list from adjacency dict
    edges = []
    for src, neighbors in graph.items():
        for tgt in neighbors:
            edges.append((src, tgt))

    num_nodes = features.shape[0]
    logger.info(f"PubMed: {num_nodes:,} nodes, {len(edges):,} edges, {features.shape[1]} features")

    return {
        "features": features.astype(np.float32),
        "labels": labels.astype(np.int64),
        "edges": edges,
        "num_nodes": num_nodes,
    }


def main():
    parser = argparse.ArgumentParser(description="Convert PubMed to Latent Ocean format")
    parser.add_argument("--output", default="./data/pubmed")
    parser.add_argument("--cache", default="./data/pubmed_cache")
    parser.add_argument("--sphere-radius", type=float, default=4.5)
    parser.add_argument("--fingerprint-dim", type=int, default=384)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache)

    # Download and load
    download_planetoid_pubmed(cache_dir)
    data = load_planetoid_pubmed(cache_dir)

    n = data["num_nodes"]
    feat_dim = data["features"].shape[1]

    # Project features to target dim
    if feat_dim < args.fingerprint_dim:
        rng = np.random.RandomState(42)
        proj = rng.randn(feat_dim, args.fingerprint_dim - feat_dim).astype(np.float32) * 0.1
        projected = data["features"] @ proj
        fingerprints = np.concatenate([data["features"], projected], axis=1)
    elif feat_dim > args.fingerprint_dim:
        rng = np.random.RandomState(42)
        proj = rng.randn(feat_dim, args.fingerprint_dim).astype(np.float32) / np.sqrt(feat_dim)
        fingerprints = data["features"] @ proj
    else:
        fingerprints = data["features"]

    # Normalize
    norms = np.linalg.norm(fingerprints, axis=1, keepdims=True)
    fingerprints = fingerprints / np.clip(norms, 1e-8, None)

    # Build adjacency
    adjacency = np.zeros((n, n), dtype=np.float32)
    for src, tgt in data["edges"]:
        if src < n and tgt < n:
            adjacency[src, tgt] = 0.8

    # Spectral coordinates (sparse for speed on 19.7K nodes)
    from scipy.sparse import csr_matrix, diags
    from scipy.sparse.linalg import eigsh

    adj_sparse = csr_matrix(adjacency + adjacency.T)
    degree = np.array(adj_sparse.sum(axis=1)).flatten()
    d_inv = np.where(degree > 0, 1.0 / np.sqrt(degree), 0)
    D = diags(d_inv)
    L = diags(np.ones(n)) - D @ adj_sparse @ D

    logger.info(f"Computing spectral embedding for {n:,} nodes...")
    try:
        eigenvalues, eigenvectors = eigsh(L, k=3, which="SM", maxiter=500)
        embed_2d = eigenvectors[:, 1:3]
    except Exception as e:
        logger.warning(f"  eigsh failed ({e}), using random layout")
        embed_2d = np.random.RandomState(42).randn(n, 2)

    norms_2d = np.linalg.norm(embed_2d, axis=1, keepdims=True)
    embed_2d = embed_2d / np.clip(norms_2d, 1e-8, None)
    rng = np.random.RandomState(42)
    r = args.sphere_radius
    theta = np.arccos(np.clip(embed_2d[:, 0], -1, 1)) + rng.normal(0, 0.03, n)
    phi = np.arctan2(embed_2d[:, 1], embed_2d[:, 0]) + np.pi + rng.normal(0, 0.03, n)
    theta = np.clip(theta, 0.01, np.pi - 0.01)
    coords = np.stack([r * np.sin(theta) * np.cos(phi), r * np.sin(theta) * np.sin(phi), r * np.cos(theta)], axis=1).astype(np.float32)

    velocity = rng.randn(n, 3).astype(np.float32) * 0.02
    for i in range(n):
        normal = coords[i] / (np.linalg.norm(coords[i]) + 1e-8)
        velocity[i] -= np.dot(velocity[i], normal) * normal

    # Save
    torch.save(torch.from_numpy(fingerprints), output_dir / "fingerprints.pt")
    torch.save(torch.from_numpy(coords), output_dir / "coords.pt")
    torch.save(torch.from_numpy(velocity), output_dir / "velocity.pt")
    torch.save(torch.from_numpy(adjacency), output_dir / "adjacency.pt")
    torch.save(torch.from_numpy(data["labels"]), output_dir / "labels.pt")

    import json
    with open(output_dir / "metadata.json", "w") as f:
        json.dump({
            "source": "Planetoid PubMed",
            "num_papers": n,
            "num_citations": len(data["edges"]),
            "num_classes": int(data["labels"].max()) + 1,
        }, f, indent=2)

    logger.info(f"\nSaved to {output_dir}: {n:,} papers, {len(data['edges']):,} citations, "
                f"{int(data['labels'].max()) + 1} classes")


if __name__ == "__main__":
    main()
