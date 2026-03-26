#!/usr/bin/env python
"""Convert Common Crawl Host-Level Web Graph to Latent Ocean format.

Downloads the Common Crawl host-level web graph (270M nodes, 9B edges)
and subsamples to a manageable size for TCD-JEPA training.

Entities = web hosts (domains)
Fingerprints = hash-based features from domain names
Coordinates = spectral embedding on S²
Causal links = hyperlink edges between hosts
Labels = TLD-based clustering (.com, .org, .edu, country codes)

Usage:
    # 500K host subset
    python scripts/data_adapters/commoncrawl_webgraph.py --output ./data/webgraph --max-nodes 500000

    # 1M hosts
    python scripts/data_adapters/commoncrawl_webgraph.py --output ./data/webgraph --max-nodes 1000000

Source: https://commoncrawl.org/ and https://data.commoncrawl.org/
"""

import argparse
import gzip
import logging
import urllib.request
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("webgraph_adapter")

# Common Crawl host-level graph vertices and edges
CC_VERTICES_URL = "https://data.commoncrawl.org/projects/hyperlinkgraph/cc-main-2024-nov-dec-jan/host/vertices.txt.gz"
CC_EDGES_URL = "https://data.commoncrawl.org/projects/hyperlinkgraph/cc-main-2024-nov-dec-jan/host/edges.txt.gz"


def download_file(url: str, path: Path) -> None:
    """Download with progress."""
    if path.exists():
        logger.info(f"Using cached {path.name}")
        return
    logger.info(f"Downloading {url}...")
    req = urllib.request.Request(url, headers={"User-Agent": "TCD-JEPA-Research"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        with open(path, "wb") as f:
            total = 0
            while True:
                chunk = resp.read(8192 * 1024)  # 8MB chunks
                if not chunk:
                    break
                f.write(chunk)
                total += len(chunk)
                if total % (100 * 1024 * 1024) == 0:
                    logger.info(f"  Downloaded {total / 1e9:.1f} GB")
    logger.info(f"  Done: {total / 1e9:.2f} GB")


def load_vertices(path: Path, max_nodes: int) -> tuple[list[str], dict[str, int]]:
    """Load host names from vertices file."""
    logger.info(f"Loading vertices (max {max_nodes:,})...")
    hosts = []
    host_to_idx = {}

    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                host = parts[1] if len(parts) > 1 else parts[0]
            else:
                host = parts[0]

            if host and host not in host_to_idx:
                host_to_idx[host] = len(hosts)
                hosts.append(host)

            if len(hosts) >= max_nodes:
                break

    logger.info(f"  Loaded {len(hosts):,} hosts")
    return hosts, host_to_idx


def load_edges(path: Path, host_to_idx: dict[str, int], max_edges: int = 10000000) -> list[tuple[int, int]]:
    """Load edges, filtering to only include known hosts."""
    logger.info(f"Loading edges (max {max_edges:,})...")
    edges = []
    n_hosts = len(host_to_idx)

    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                try:
                    src_idx = int(parts[0])
                    tgt_idx = int(parts[1])
                    if src_idx < n_hosts and tgt_idx < n_hosts:
                        edges.append((src_idx, tgt_idx))
                except ValueError:
                    continue

            if len(edges) >= max_edges:
                break

    logger.info(f"  Loaded {len(edges):,} edges")
    return edges


def build_domain_fingerprints(hosts: list[str], dim: int = 384) -> np.ndarray:
    """Build fingerprints from domain name features."""
    n = len(hosts)
    fingerprints = np.zeros((n, dim), dtype=np.float32)

    for i, host in enumerate(hosts):
        # Extract features from domain name
        parts = host.split(".")
        tld = parts[-1] if parts else ""

        # Hash-based embedding of domain parts
        for part in parts:
            for char_idx, char in enumerate(part):
                h = (hash(part) * 31 + hash(char) + char_idx) % dim
                fingerprints[i, h] += 1.0

        # TLD one-hot region
        tld_hash = hash(tld) % (dim // 4)
        fingerprints[i, tld_hash] += 2.0

        # Length features
        fingerprints[i, dim - 1] = len(host) / 50.0
        fingerprints[i, dim - 2] = len(parts) / 5.0

    # Normalize
    norms = np.linalg.norm(fingerprints, axis=1, keepdims=True)
    fingerprints = fingerprints / np.clip(norms, 1e-8, None)
    return fingerprints


def build_tld_labels(hosts: list[str]) -> np.ndarray:
    """Assign labels by top-level domain."""
    tld_counts = {}
    for host in hosts:
        tld = host.split(".")[-1] if "." in host else "unknown"
        tld_counts[tld] = tld_counts.get(tld, 0) + 1

    # Top TLDs as clusters
    top_tlds = sorted(tld_counts, key=lambda t: -tld_counts[t])[:50]
    tld_to_idx = {t: i for i, t in enumerate(top_tlds)}

    labels = []
    for host in hosts:
        tld = host.split(".")[-1] if "." in host else "unknown"
        labels.append(tld_to_idx.get(tld, len(top_tlds)))

    logger.info(f"Labels: {len(top_tlds) + 1} TLD clusters")
    return np.array(labels, dtype=np.int64)


def main():
    parser = argparse.ArgumentParser(description="Convert Common Crawl web graph to Latent Ocean format")
    parser.add_argument("--output", default="./data/webgraph")
    parser.add_argument("--cache", default="./data/webgraph_cache")
    parser.add_argument("--max-nodes", type=int, default=500000)
    parser.add_argument("--max-edges", type=int, default=5000000)
    parser.add_argument("--sphere-radius", type=float, default=4.5)
    parser.add_argument("--fingerprint-dim", type=int, default=384)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Download
    vertices_path = cache_dir / "vertices.txt.gz"
    edges_path = cache_dir / "edges.txt.gz"

    download_file(CC_VERTICES_URL, vertices_path)
    download_file(CC_EDGES_URL, edges_path)

    # Load
    hosts, host_to_idx = load_vertices(vertices_path, args.max_nodes)
    edge_list = load_edges(edges_path, host_to_idx, args.max_edges)
    n = len(hosts)

    # Fingerprints
    fingerprints = build_domain_fingerprints(hosts, args.fingerprint_dim)
    logger.info(f"Fingerprints: {fingerprints.shape}")

    # Coordinates via randomized SVD for large graphs
    logger.info("Computing spectral embedding...")
    edges_np = np.array(edge_list, dtype=np.int64) if edge_list else np.zeros((0, 2), dtype=np.int64)

    from scipy.sparse import csr_matrix
    if len(edges_np) > 0:
        rows = np.concatenate([edges_np[:, 0], edges_np[:, 1]])
        cols = np.concatenate([edges_np[:, 1], edges_np[:, 0]])
        data = np.ones(len(rows), dtype=np.float32)
        adj_sparse = csr_matrix((data, (rows, cols)), shape=(n, n))

        try:
            if n > 500000:
                from scipy.sparse import diags
                from sklearn.utils.extmath import randomized_svd
                degree = np.array(adj_sparse.sum(axis=1)).flatten()
                d_inv = np.where(degree > 0, 1.0 / np.sqrt(degree), 0)
                D = diags(d_inv)
                A_norm = D @ adj_sparse @ D
                logger.info(f"  Randomized SVD on {n:,} nodes...")
                U, S, Vt = randomized_svd(A_norm, n_components=3, random_state=42)
                embed_2d = U[:, 1:3]
            else:
                from scipy.sparse import diags
                from scipy.sparse.linalg import eigsh
                degree = np.array(adj_sparse.sum(axis=1)).flatten()
                d_inv = np.where(degree > 0, 1.0 / np.sqrt(degree), 0)
                D = diags(d_inv)
                L = diags(np.ones(n)) - D @ adj_sparse @ D
                eigenvalues, eigenvectors = eigsh(L, k=3, which="SM", maxiter=1000)
                embed_2d = eigenvectors[:, 1:3]
        except Exception as e:
            logger.warning(f"  Spectral failed ({e}), using degree layout")
            degree = np.array(adj_sparse.sum(axis=1)).flatten()
            rng = np.random.RandomState(42)
            rank = np.argsort(np.argsort(-degree)).astype(np.float32) / max(n - 1, 1)
            embed_2d = np.stack([rank + rng.normal(0, 0.05, n), rng.uniform(-1, 1, n)], axis=1)
    else:
        embed_2d = np.random.RandomState(42).randn(n, 2)

    # Map to S²
    norms_2d = np.linalg.norm(embed_2d, axis=1, keepdims=True)
    embed_2d = embed_2d / np.clip(norms_2d, 1e-8, None)
    rng = np.random.RandomState(42)
    theta = np.arccos(np.clip(embed_2d[:, 0], -1, 1)) + rng.normal(0, 0.02, n)
    phi = np.arctan2(embed_2d[:, 1], embed_2d[:, 0]) + np.pi + rng.normal(0, 0.02, n)
    theta = np.clip(theta, 0.01, np.pi - 0.01)
    r = args.sphere_radius
    coords = np.stack([r * np.sin(theta) * np.cos(phi), r * np.sin(theta) * np.sin(phi), r * np.cos(theta)], axis=1).astype(np.float32)

    # Velocity
    velocity = rng.randn(n, 3).astype(np.float32) * 0.02
    for i in range(n):
        normal = coords[i] / (np.linalg.norm(coords[i]) + 1e-8)
        velocity[i] -= np.dot(velocity[i], normal) * normal

    # Labels
    labels = build_tld_labels(hosts)

    # Save
    torch.save(torch.from_numpy(fingerprints), output_dir / "fingerprints.pt")
    torch.save(torch.from_numpy(coords), output_dir / "coords.pt")
    torch.save(torch.from_numpy(velocity), output_dir / "velocity.pt")
    torch.save(torch.from_numpy(labels), output_dir / "labels.pt")

    edges_tensor = torch.zeros(len(edge_list), 3)
    for idx, (s, t) in enumerate(edge_list):
        edges_tensor[idx] = torch.tensor([s, t, 0.8])
    torch.save(edges_tensor, output_dir / "edges.pt")

    import json
    with open(output_dir / "metadata.json", "w") as f:
        json.dump({"source": "Common Crawl Host Web Graph", "num_hosts": n, "num_edges": len(edge_list)}, f, indent=2)

    logger.info(f"\nSaved to {output_dir}: {n:,} hosts, {len(edge_list):,} edges")


if __name__ == "__main__":
    main()
