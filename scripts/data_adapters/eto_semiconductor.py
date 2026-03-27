#!/usr/bin/env python
"""Convert ETO Semiconductor Supply Chain dataset to Latent Ocean format.

Downloads the Georgetown CSET ETO Chip Explorer dataset and converts it into
the Latent Ocean manifold format for TCD-JEPA training:

- Entities = inputs (chips, tools, materials, processes) + providers (companies, countries)
- Fingerprints = TF-IDF or sentence-transformer embeddings of descriptions
- Coordinates = projected onto S² via spectral embedding of the supply graph
- Causal links = "goes_into" supply chain relationships + "provides" market share links
- Entity types = stage (Design, Fabrication, ATP) or provider type (organization, country)
- Velocities = market share change signals (where available)

Usage:
    python scripts/data_adapters/eto_semiconductor.py --output ./data/semiconductor
    python train_manifold.py --config configs/manifold.yaml --tcd --eval \\
        data.dataset=latent_ocean data.data_dir=./data/semiconductor

Source: https://github.com/georgetown-cset/eto-chip-explorer
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("eto_adapter")

ETO_BASE_URL = "https://raw.githubusercontent.com/georgetown-cset/eto-chip-explorer/main/data"


def download_csv(filename: str, cache_dir: Path) -> Path:
    """Download a CSV file from the ETO repo if not already cached."""
    import urllib.request

    cache_dir.mkdir(parents=True, exist_ok=True)
    local_path = cache_dir / filename

    if local_path.exists():
        logger.info(f"Using cached {filename}")
        return local_path

    url = f"{ETO_BASE_URL}/{filename}"
    logger.info(f"Downloading {url}")
    urllib.request.urlretrieve(url, local_path)
    return local_path


def load_csvs(cache_dir: Path) -> dict:
    """Download and load all ETO CSV files."""
    import csv

    data = {}
    for name in ["inputs", "providers", "provision", "sequence", "stages"]:
        path = download_csv(f"{name}.csv", cache_dir)
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            data[name] = list(reader)
        logger.info(f"  {name}: {len(data[name])} rows")
    return data


def build_entity_index(data: dict) -> tuple[list[dict], dict[str, int]]:
    """Build unified entity list from inputs + providers."""
    entities = []
    id_to_idx = {}

    # Add inputs (chips, tools, materials, processes)
    for row in data["inputs"]:
        eid = row["input_id"]
        if eid not in id_to_idx:
            id_to_idx[eid] = len(entities)
            entities.append({
                "id": eid,
                "name": row["input_name"],
                "type": row.get("type", "unknown"),
                "description": row.get("description", row["input_name"]),
                "stage": row.get("stage_name", ""),
                "source": "input",
            })

    # Add providers (companies, countries)
    for row in data["providers"]:
        eid = row["provider_id"]
        if eid not in id_to_idx:
            id_to_idx[eid] = len(entities)
            entities.append({
                "id": eid,
                "name": row["provider_name"],
                "type": row.get("provider_type", "organization"),
                "description": f"{row['provider_name']} ({row.get('country', 'global')})",
                "stage": "",
                "source": "provider",
            })

    logger.info(f"Total entities: {len(entities)}")
    return entities, id_to_idx


def build_fingerprints(entities: list[dict], dim: int = 384) -> np.ndarray:
    """Build entity fingerprints from descriptions using TF-IDF + random projection.

    Falls back to bag-of-words if sklearn not available.
    """
    descriptions = [e["description"] for e in entities]
    n = len(entities)

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.random_projection import GaussianRandomProjection

        vectorizer = TfidfVectorizer(max_features=2000, stop_words="english")
        tfidf = vectorizer.fit_transform(descriptions).toarray()
        logger.info(f"TF-IDF features: {tfidf.shape}")

        # Project to target dim
        if tfidf.shape[1] < dim:
            # Pad with zeros
            padded = np.zeros((n, dim))
            padded[:, :tfidf.shape[1]] = tfidf
            fingerprints = padded
        else:
            projector = GaussianRandomProjection(n_components=dim, random_state=42)
            fingerprints = projector.fit_transform(tfidf)

    except ImportError:
        logger.warning("sklearn not available — using hash-based fingerprints")
        rng = np.random.RandomState(42)
        fingerprints = np.zeros((n, dim))
        for i, desc in enumerate(descriptions):
            words = desc.lower().split()
            for word in words:
                h = hash(word) % dim
                fingerprints[i, h] += 1.0
            fingerprints[i] /= max(len(words), 1)
            fingerprints[i] += rng.normal(0, 0.01, dim)

    # Normalize
    norms = np.linalg.norm(fingerprints, axis=1, keepdims=True)
    fingerprints = fingerprints / np.clip(norms, 1e-8, None)

    logger.info(f"Fingerprints: {fingerprints.shape}")
    return fingerprints.astype(np.float32)


def build_adjacency(data: dict, id_to_idx: dict, n: int) -> np.ndarray:
    """Build directed adjacency matrix from supply chain relationships."""
    adjacency = np.zeros((n, n), dtype=np.float32)

    # Supply chain flow: "goes_into" relationships (sequence.csv)
    links_added = 0
    for row in data["sequence"]:
        src_id = row.get("input_id", "")
        dst_id = row.get("goes_into_id", "")
        if src_id in id_to_idx and dst_id in id_to_idx:
            i, j = id_to_idx[src_id], id_to_idx[dst_id]
            adjacency[i, j] = 0.9  # Strong causal link (supply dependency)
            links_added += 1

        # "is_type_of" relationships (taxonomy)
        parent_id = row.get("is_type_of_id", "")
        if src_id in id_to_idx and parent_id in id_to_idx:
            i, j = id_to_idx[src_id], id_to_idx[parent_id]
            adjacency[i, j] = 0.7  # Taxonomic link
            links_added += 1

    # Provider→input relationships (provision.csv: who provides what)
    for row in data["provision"]:
        provider_id = row.get("provider_id", "")
        input_id = row.get("provided_id", "")
        if provider_id in id_to_idx and input_id in id_to_idx:
            i, j = id_to_idx[provider_id], id_to_idx[input_id]
            # Market share as link weight
            share_str = row.get("share_provided", "0")
            try:
                share = float(share_str.replace("%", "")) / 100.0
            except (ValueError, AttributeError):
                share = 0.5
            adjacency[i, j] = max(0.1, min(1.0, share * 2))  # Scale to [0.1, 1.0]
            links_added += 1

    logger.info(f"Adjacency: {links_added} directed links")
    return adjacency


def build_coordinates(adjacency: np.ndarray, n: int, sphere_radius: float = 4.5) -> np.ndarray:
    """Project entities onto S² using spectral embedding of the supply graph."""
    # Build symmetric adjacency for spectral embedding
    sym = adjacency + adjacency.T
    np.fill_diagonal(sym, 0)

    # Degree matrix and Laplacian
    degree = sym.sum(axis=1)
    degree_inv_sqrt = np.where(degree > 0, 1.0 / np.sqrt(degree), 0)
    D_inv_sqrt = np.diag(degree_inv_sqrt)
    L_norm = np.eye(n) - D_inv_sqrt @ sym @ D_inv_sqrt

    # Eigenvectors for 2D embedding (skip first trivial eigenvector)
    try:
        from scipy.linalg import eigh
        eigenvalues, eigenvectors = eigh(L_norm, subset_by_index=[1, 2])
        embed_2d = eigenvectors
    except Exception:
        # Fallback: random projection
        rng = np.random.RandomState(42)
        embed_2d = rng.randn(n, 2)

    # Normalize to unit circle, then map to S²
    norms = np.linalg.norm(embed_2d, axis=1, keepdims=True)
    embed_2d = embed_2d / np.clip(norms, 1e-8, None)

    # Map 2D to spherical coordinates (spread across S²)
    theta = np.arccos(np.clip(embed_2d[:, 0], -1, 1))  # [0, pi]
    phi = np.arctan2(embed_2d[:, 1], embed_2d[:, 0]) + np.pi  # [0, 2pi]

    # Add jitter to avoid exact overlaps
    rng = np.random.RandomState(42)
    theta += rng.normal(0, 0.05, n)
    phi += rng.normal(0, 0.05, n)
    theta = np.clip(theta, 0.01, np.pi - 0.01)

    # Cartesian on S²
    x = sphere_radius * np.sin(theta) * np.cos(phi)
    y = sphere_radius * np.sin(theta) * np.sin(phi)
    z = sphere_radius * np.cos(theta)
    coords = np.stack([x, y, z], axis=1)

    logger.info(f"Coordinates: {coords.shape} on S² (radius {sphere_radius})")
    return coords.astype(np.float32)


def build_velocities(data: dict, id_to_idx: dict, coords: np.ndarray, n: int) -> np.ndarray:
    """Build velocity vectors from market share data (approximating dynamics)."""
    velocity = np.zeros((n, 3), dtype=np.float32)

    # Use provision data to create "influence" velocity toward supplied inputs
    for row in data["provision"]:
        provider_id = row.get("provider_id", "")
        input_id = row.get("provided_id", "")
        if provider_id in id_to_idx and input_id in id_to_idx:
            i = id_to_idx[provider_id]
            j = id_to_idx[input_id]
            # Drift toward what you supply
            drift = coords[j] - coords[i]
            # Project onto tangent plane
            normal = coords[i] / (np.linalg.norm(coords[i]) + 1e-8)
            drift_tangent = drift - np.dot(drift, normal) * normal
            velocity[i] += drift_tangent * 0.01

    # Normalize
    norms = np.linalg.norm(velocity, axis=1, keepdims=True)
    velocity = velocity / np.clip(norms, 1e-8, None) * 0.1

    # Replace NaN with zero
    velocity = np.nan_to_num(velocity, 0.0)

    logger.info(f"Velocities: {velocity.shape}")
    return velocity


def build_labels(entities: list[dict]) -> np.ndarray:
    """Assign cluster labels based on entity type + stage."""
    label_map = {}
    labels = []
    for e in entities:
        key = f"{e['source']}_{e['type']}_{e['stage']}" if e["stage"] else f"{e['source']}_{e['type']}"
        if key not in label_map:
            label_map[key] = len(label_map)
        labels.append(label_map[key])

    logger.info(f"Labels: {len(label_map)} unique types: {list(label_map.keys())}")
    return np.array(labels, dtype=np.int64)


def main():
    parser = argparse.ArgumentParser(description="Convert ETO Semiconductor data to Latent Ocean format")
    parser.add_argument("--output", default="./data/semiconductor", help="Output directory")
    parser.add_argument("--cache", default="./data/eto_cache", help="Download cache directory")
    parser.add_argument("--sphere-radius", type=float, default=4.5)
    parser.add_argument("--fingerprint-dim", type=int, default=384)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache)

    # Download and load
    logger.info("Loading ETO Semiconductor Supply Chain dataset...")
    data = load_csvs(cache_dir)

    # Build entities
    entities, id_to_idx = build_entity_index(data)
    n = len(entities)

    # Build all tensors
    fingerprints = build_fingerprints(entities, dim=args.fingerprint_dim)
    adjacency = build_adjacency(data, id_to_idx, n)
    coords = build_coordinates(adjacency, n, sphere_radius=args.sphere_radius)
    velocity = build_velocities(data, id_to_idx, coords, n)
    labels = build_labels(entities)

    # Save in Latent Ocean format
    torch.save(torch.from_numpy(fingerprints), output_dir / "fingerprints.pt")
    torch.save(torch.from_numpy(coords), output_dir / "coords.pt")
    torch.save(torch.from_numpy(velocity), output_dir / "velocity.pt")
    torch.save(torch.from_numpy(adjacency), output_dir / "adjacency.pt")
    torch.save(torch.from_numpy(labels), output_dir / "labels.pt")

    # Save entity metadata for interpretability
    import json
    with open(output_dir / "entities.json", "w") as f:
        json.dump(entities, f, indent=2)

    logger.info(f"\nSaved to {output_dir}:")
    logger.info(f"  fingerprints.pt: [{n}, {args.fingerprint_dim}]")
    logger.info(f"  coords.pt:      [{n}, 3] (S² radius {args.sphere_radius})")
    logger.info(f"  velocity.pt:    [{n}, 3]")
    logger.info(f"  adjacency.pt:   [{n}, {n}] ({int(adjacency.sum())} total weight)")
    logger.info(f"  labels.pt:      [{n}] ({len(set(labels))} types)")
    logger.info(f"  entities.json:  {n} entity records")
    logger.info(f"\nReady for: python train_manifold.py --config configs/manifold.yaml --tcd --eval "
                f"data.dataset=latent_ocean data.data_dir={output_dir}")


if __name__ == "__main__":
    main()
