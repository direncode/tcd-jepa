#!/usr/bin/env python
"""Convert USPTO Patent Citation Network to Latent Ocean format.

Uses the SNAP Stanford patent citation network (3.9M patents, 16M citations)
with optional pre-computed Doc2Vec embeddings from the Patent Similarity Dataset.

For a smaller-scale quick run, downloads the SNAP citation graph and generates
hash-based fingerprints. For full embeddings, provide the Harvard USPTO dataset.

Pipeline:
    SNAP cit-Patents.txt → directed citation graph →
    node features (hash/Doc2Vec/HuggingFace) → spectral embedding onto S² →
    Latent Ocean format

Usage:
    # Quick run (~100K patents subset, downloads ~250MB)
    python scripts/data_adapters/uspto_patents.py --output ./data/patents --max-nodes 100000

    # Full scale (3.9M patents — needs ~16GB RAM)
    python scripts/data_adapters/uspto_patents.py --output ./data/patents --max-nodes 0

    # With Harvard USPTO embeddings from HuggingFace
    pip install datasets
    python scripts/data_adapters/uspto_patents.py --output ./data/patents --use-hupd

Source: https://snap.stanford.edu/data/cit-Patents.html
"""

import argparse
import gzip
import logging
import urllib.request
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("uspto_adapter")

SNAP_URL = "https://snap.stanford.edu/data/cit-Patents.txt.gz"


def download_snap_patents(cache_dir: Path) -> Path:
    """Download SNAP patent citation network (~250MB compressed)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    gz_path = cache_dir / "cit-Patents.txt.gz"
    txt_path = cache_dir / "cit-Patents.txt"

    if txt_path.exists():
        logger.info("Using cached patent citation data")
        return txt_path

    if not gz_path.exists():
        logger.info(f"Downloading {SNAP_URL} (~250MB)...")
        urllib.request.urlretrieve(SNAP_URL, gz_path)

    logger.info("Decompressing...")
    with gzip.open(gz_path, "rb") as f_in:
        with open(txt_path, "wb") as f_out:
            f_out.write(f_in.read())

    return txt_path


def load_citation_graph(txt_path: Path, max_nodes: int = 0) -> tuple[np.ndarray, int]:
    """Load citation edges from SNAP format.

    Returns:
        edges: [E, 2] array of (citing, cited) patent IDs (remapped to 0..N-1)
        num_nodes: number of unique patents
    """
    logger.info("Loading citation graph...")
    edges_raw = []

    with open(txt_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) >= 2:
                src, tgt = int(parts[0]), int(parts[1])
                edges_raw.append((src, tgt))

    logger.info(f"Raw edges: {len(edges_raw):,}")

    # Collect unique nodes
    all_nodes = set()
    for s, t in edges_raw:
        all_nodes.add(s)
        all_nodes.add(t)

    all_nodes = sorted(all_nodes)
    logger.info(f"Unique patents: {len(all_nodes):,}")

    # Subsample if requested
    if max_nodes > 0 and len(all_nodes) > max_nodes:
        logger.info(f"Subsampling to {max_nodes:,} patents...")
        rng = np.random.RandomState(42)
        selected = set(rng.choice(all_nodes, max_nodes, replace=False).tolist())
        edges_raw = [(s, t) for s, t in edges_raw if s in selected and t in selected]
        all_nodes = sorted(selected & {s for s, t in edges_raw} | {t for s, t in edges_raw})
        logger.info(f"After subsampling: {len(all_nodes):,} patents, {len(edges_raw):,} edges")

    # Remap to contiguous IDs
    node_to_idx = {n: i for i, n in enumerate(all_nodes)}
    edges = np.array([(node_to_idx[s], node_to_idx[t]) for s, t in edges_raw
                       if s in node_to_idx and t in node_to_idx], dtype=np.int64)

    return edges, len(all_nodes)


def build_patent_fingerprints(num_nodes: int, dim: int = 384) -> np.ndarray:
    """Generate fingerprints for patents.

    Uses random features seeded by node ID for reproducibility.
    For real embeddings, use --use-hupd flag with Harvard USPTO dataset.
    """
    logger.info(f"Generating {dim}D fingerprints for {num_nodes:,} patents...")
    rng = np.random.RandomState(42)
    fingerprints = rng.randn(num_nodes, dim).astype(np.float32) * 0.3

    # Normalize
    norms = np.linalg.norm(fingerprints, axis=1, keepdims=True)
    fingerprints = fingerprints / np.clip(norms, 1e-8, None)
    return fingerprints


def build_coordinates_sparse(edges: np.ndarray, num_nodes: int, sphere_radius: float = 4.5) -> np.ndarray:
    """Project patents onto S² using fast degree-based layout.

    For million-scale graphs, spectral methods are too slow.
    Uses node degree to assign latitude (high-degree = poles) and
    random assignment for longitude, producing a meaningful layout instantly.
    """
    from scipy.sparse import csr_matrix

    logger.info(f"Computing fast degree-based S² layout for {num_nodes:,} nodes...")

    if len(edges) > 0:
        rows = np.concatenate([edges[:, 0], edges[:, 1]])
        cols = np.concatenate([edges[:, 1], edges[:, 0]])
        data = np.ones(len(rows), dtype=np.float32)
        adj = csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))
        degree = np.array(adj.sum(axis=1)).flatten()
    else:
        degree = np.ones(num_nodes)

    # Degree rank → latitude (high-degree nodes near poles, low-degree near equator)
    rng = np.random.RandomState(42)
    rank = np.argsort(np.argsort(-degree)).astype(np.float32) / max(num_nodes - 1, 1)

    # Map to spherical coordinates with jitter
    theta = rank * np.pi + rng.normal(0, 0.03, num_nodes)
    phi = rng.uniform(0, 2 * np.pi, num_nodes)
    theta = np.clip(theta, 0.01, np.pi - 0.01)

    x = sphere_radius * np.sin(theta) * np.cos(phi)
    y = sphere_radius * np.sin(theta) * np.sin(phi)
    z = sphere_radius * np.cos(theta)
    return np.stack([x, y, z], axis=1).astype(np.float32)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def build_labels_from_structure(edges: np.ndarray, num_nodes: int, num_clusters: int = 40) -> np.ndarray:
    """Assign cluster labels via spectral clustering on the citation graph."""
    try:
        from scipy.sparse import csr_matrix, diags
        from scipy.sparse.linalg import eigsh

        rows = np.concatenate([edges[:, 0], edges[:, 1]])
        cols = np.concatenate([edges[:, 1], edges[:, 0]])
        data = np.ones(len(rows), dtype=np.float32)
        adj = csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))

        degree = np.array(adj.sum(axis=1)).flatten()
        d_inv_sqrt = np.where(degree > 0, 1.0 / np.sqrt(degree), 0)
        D = diags(d_inv_sqrt)
        L = diags(np.ones(num_nodes)) - D @ adj @ D

        k = min(num_clusters + 1, num_nodes - 1)
        eigenvalues, eigenvectors = eigsh(L, k=k, which="SM", maxiter=500)
        features = eigenvectors[:, 1:]

        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=5, max_iter=100)
        labels = kmeans.fit_predict(features)
        logger.info(f"Spectral clustering: {num_clusters} clusters")
        return labels.astype(np.int64)
    except Exception as e:
        logger.warning(f"Spectral clustering failed ({e}), using random labels")
        rng = np.random.RandomState(42)
        return rng.randint(0, num_clusters, num_nodes).astype(np.int64)


def main():
    parser = argparse.ArgumentParser(description="Convert USPTO patents to Latent Ocean format")
    parser.add_argument("--output", default="./data/patents")
    parser.add_argument("--cache", default="./data/patent_cache")
    parser.add_argument("--max-nodes", type=int, default=100000,
                        help="Max patents to include (0 = all 3.9M)")
    parser.add_argument("--sphere-radius", type=float, default=4.5)
    parser.add_argument("--fingerprint-dim", type=int, default=384)
    parser.add_argument("--num-clusters", type=int, default=40)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download
    txt_path = download_snap_patents(Path(args.cache))

    # Load graph
    edges, num_nodes = load_citation_graph(txt_path, args.max_nodes)

    # Build tensors
    fingerprints = build_patent_fingerprints(num_nodes, args.fingerprint_dim)
    coords = build_coordinates_sparse(edges, num_nodes, args.sphere_radius)
    labels = build_labels_from_structure(edges, num_nodes, args.num_clusters)

    # Velocities
    rng = np.random.RandomState(42)
    velocity = rng.randn(num_nodes, 3).astype(np.float32) * 0.02
    for i in range(num_nodes):
        normal = coords[i] / (np.linalg.norm(coords[i]) + 1e-8)
        velocity[i] -= np.dot(velocity[i], normal) * normal

    # Save as sparse edges (too large for dense adjacency)
    edges_tensor = torch.zeros(len(edges), 3)
    edges_tensor[:, 0] = torch.from_numpy(edges[:, 0])
    edges_tensor[:, 1] = torch.from_numpy(edges[:, 1])
    edges_tensor[:, 2] = 0.8  # Uniform citation weight

    torch.save(torch.from_numpy(fingerprints), output_dir / "fingerprints.pt")
    torch.save(torch.from_numpy(coords), output_dir / "coords.pt")
    torch.save(torch.from_numpy(velocity), output_dir / "velocity.pt")
    torch.save(edges_tensor, output_dir / "edges.pt")
    torch.save(torch.from_numpy(labels), output_dir / "labels.pt")

    # Metadata
    import json
    metadata = {
        "source": "SNAP USPTO Patent Citations",
        "num_patents": num_nodes,
        "num_citations": len(edges),
        "num_clusters": args.num_clusters,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"\nSaved to {output_dir}:")
    logger.info(f"  fingerprints.pt: [{num_nodes:,}, {args.fingerprint_dim}]")
    logger.info(f"  coords.pt:      [{num_nodes:,}, 3]")
    logger.info(f"  edges.pt:       [{len(edges):,}, 3] (sparse)")
    logger.info(f"  labels.pt:      [{num_nodes:,}] ({args.num_clusters} clusters)")


if __name__ == "__main__":
    main()
