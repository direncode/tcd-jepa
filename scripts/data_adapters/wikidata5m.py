#!/usr/bin/env python
"""Convert Wikidata5m knowledge graph to Latent Ocean format.

Downloads the Wikidata5m dataset (4.6M entities, 20.6M triples, 822 relations)
from the MilaGraph project and converts to Latent Ocean manifold format.

Wikidata5m pairs each entity with a Wikipedia page description, enabling
text-based embeddings. The knowledge graph provides directed typed relations.

Pipeline:
    Wikidata5m download → entity descriptions → sentence embeddings (384D) →
    directed relation graph → spectral embedding onto S² → Latent Ocean format

Usage:
    # Subset (100K entities — recommended for first run)
    python scripts/data_adapters/wikidata5m.py --output ./data/wikidata --max-entities 100000

    # Full scale (4.6M entities — needs ~32GB RAM and sentence-transformers)
    python scripts/data_adapters/wikidata5m.py --output ./data/wikidata --max-entities 0

Source: https://deepgraphlearning.github.io/project/wikidata5m
"""

import argparse
import json
import logging
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("wikidata5m_adapter")

# Wikidata5m transductive split (smaller, has all entities)
WIKIDATA5M_URL = "https://huggingface.co/datasets/wikidata5m/resolve/main/wikidata5m_transductive.tar.gz"
WIKIDATA5M_ALIAS_URL = "https://huggingface.co/datasets/wikidata5m/resolve/main/wikidata5m_alias.tar.gz"


def download_file(url: str, path: Path) -> None:
    """Download a file with progress logging."""
    if path.exists():
        logger.info(f"Using cached {path.name}")
        return
    logger.info(f"Downloading {url}...")
    urllib.request.urlretrieve(url, path)
    logger.info(f"  Saved to {path}")


def download_and_extract(cache_dir: Path) -> Path:
    """Download and extract Wikidata5m dataset."""
    cache_dir.mkdir(parents=True, exist_ok=True)

    tar_path = cache_dir / "wikidata5m_transductive.tar.gz"
    alias_tar_path = cache_dir / "wikidata5m_alias.tar.gz"

    download_file(WIKIDATA5M_URL, tar_path)
    download_file(WIKIDATA5M_ALIAS_URL, alias_tar_path)

    # Extract
    for tar_p in [tar_path, alias_tar_path]:
        if tar_p.exists():
            logger.info(f"Extracting {tar_p.name}...")
            with tarfile.open(tar_p, "r:gz") as tar:
                tar.extractall(cache_dir)

    return cache_dir


def load_entity_aliases(cache_dir: Path) -> dict[str, str]:
    """Load entity ID → human-readable name mapping."""
    alias_file = cache_dir / "wikidata5m_entity.txt"
    if not alias_file.exists():
        # Try alternate paths
        for candidate in cache_dir.rglob("*entity*"):
            if candidate.suffix == ".txt":
                alias_file = candidate
                break

    aliases = {}
    if alias_file.exists():
        logger.info(f"Loading entity aliases from {alias_file}...")
        with open(alias_file, encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    aliases[parts[0]] = parts[1]
        logger.info(f"  Loaded {len(aliases):,} entity names")
    return aliases


def load_triples(cache_dir: Path, max_entities: int = 0) -> tuple[list, set, set]:
    """Load knowledge graph triples."""
    triple_file = None
    for candidate in cache_dir.rglob("*train*"):
        if candidate.suffix == ".txt":
            triple_file = candidate
            break

    if triple_file is None:
        raise FileNotFoundError("Could not find triple file in downloaded data")

    logger.info(f"Loading triples from {triple_file}...")
    triples = []
    entities = set()
    relations = set()

    with open(triple_file, encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                h, r, t = parts[0], parts[1], parts[2]
                triples.append((h, r, t))
                entities.add(h)
                entities.add(t)
                relations.add(r)

    logger.info(f"  Triples: {len(triples):,}")
    logger.info(f"  Entities: {len(entities):,}")
    logger.info(f"  Relations: {len(relations):,}")

    # Subsample if needed
    if max_entities > 0 and len(entities) > max_entities:
        logger.info(f"Subsampling to {max_entities:,} entities...")
        # Select high-degree entities preferentially
        entity_degree = {}
        for h, r, t in triples:
            entity_degree[h] = entity_degree.get(h, 0) + 1
            entity_degree[t] = entity_degree.get(t, 0) + 1

        sorted_entities = sorted(entity_degree.keys(), key=lambda e: -entity_degree[e])
        selected = set(sorted_entities[:max_entities])

        triples = [(h, r, t) for h, r, t in triples if h in selected and t in selected]
        entities = selected & ({h for h, r, t in triples} | {t for h, r, t in triples})
        logger.info(f"  After subsampling: {len(entities):,} entities, {len(triples):,} triples")

    return triples, entities, relations


def build_fingerprints_from_names(
    entity_list: list[str],
    aliases: dict[str, str],
    dim: int = 384,
) -> np.ndarray:
    """Build fingerprints from entity names via sentence-transformers or TF-IDF."""
    descriptions = []
    for eid in entity_list:
        name = aliases.get(eid, eid)
        descriptions.append(name)

    n = len(descriptions)

    try:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Encoding {n:,} entity names with all-MiniLM-L6-v2...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(descriptions, show_progress_bar=True, batch_size=256)
        return embeddings.astype(np.float32)
    except ImportError:
        logger.warning("sentence-transformers not installed — using TF-IDF")
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.random_projection import GaussianRandomProjection

            vectorizer = TfidfVectorizer(max_features=2000, stop_words="english")
            tfidf = vectorizer.fit_transform(descriptions).toarray()

            if tfidf.shape[1] < dim:
                padded = np.zeros((n, dim), dtype=np.float32)
                padded[:, :tfidf.shape[1]] = tfidf
                return padded
            proj = GaussianRandomProjection(n_components=dim, random_state=42)
            return proj.fit_transform(tfidf).astype(np.float32)
        except ImportError:
            logger.warning("sklearn not available — using hash embeddings")
            embeddings = np.zeros((n, dim), dtype=np.float32)
            for i, desc in enumerate(descriptions):
                for w in desc.lower().split():
                    embeddings[i, hash(w) % dim] += 1.0
                embeddings[i] /= max(len(desc.split()), 1)
            return embeddings


def build_relation_labels(
    triples: list,
    entity_to_idx: dict[str, int],
    relations: set,
    num_nodes: int,
) -> np.ndarray:
    """Assign cluster labels based on most common relation type per entity."""
    relation_list = sorted(relations)
    rel_to_idx = {r: i for i, r in enumerate(relation_list)}

    # Count relation types per entity
    # Cap relation types to top 100 most common for memory efficiency at scale
    if len(relation_list) > 100:
        rel_counts_global = {}
        for h, r, t in triples:
            rel_counts_global[r] = rel_counts_global.get(r, 0) + 1
        top_rels = sorted(rel_counts_global, key=lambda r: -rel_counts_global[r])[:100]
        relation_list = top_rels
        rel_to_idx = {r: i for i, r in enumerate(relation_list)}

    entity_rel_counts = np.zeros((num_nodes, len(relation_list)), dtype=np.int32)
    for h, r, t in triples:
        if h in entity_to_idx and r in rel_to_idx:
            entity_rel_counts[entity_to_idx[h], rel_to_idx[r]] += 1
        if t in entity_to_idx and r in rel_to_idx:
            entity_rel_counts[entity_to_idx[t], rel_to_idx[r]] += 1

    # Assign label = most common relation
    labels = entity_rel_counts.argmax(axis=1)

    # Reduce to manageable number of clusters
    unique_labels = np.unique(labels)
    if len(unique_labels) > 50:
        # Map to top-50 most common
        label_counts = np.bincount(labels, minlength=len(relation_list))
        top50 = np.argsort(-label_counts)[:50]
        top50_set = set(top50.tolist())
        labels = np.array([l if l in top50_set else -1 for l in labels])
        # Remap -1 to nearest valid label
        valid_mask = labels >= 0
        if not valid_mask.all():
            from scipy.spatial.distance import cdist
            invalid_idx = np.where(~valid_mask)[0]
            valid_idx = np.where(valid_mask)[0]
            if len(valid_idx) > 0:
                dists = cdist(
                    entity_rel_counts[invalid_idx].astype(float),
                    entity_rel_counts[valid_idx].astype(float),
                    metric="cosine",
                )
                nearest = dists.argmin(axis=1)
                labels[invalid_idx] = labels[valid_idx[nearest]]

    # Remap to contiguous
    unique = np.unique(labels)
    remap = {old: new for new, old in enumerate(unique)}
    labels = np.array([remap[l] for l in labels], dtype=np.int64)

    logger.info(f"Labels: {len(unique)} clusters from relation types")
    return labels


def main():
    parser = argparse.ArgumentParser(description="Convert Wikidata5m to Latent Ocean format")
    parser.add_argument("--output", default="./data/wikidata")
    parser.add_argument("--cache", default="./data/wikidata_cache")
    parser.add_argument("--max-entities", type=int, default=100000,
                        help="Max entities (0 = all 4.6M)")
    parser.add_argument("--sphere-radius", type=float, default=4.5)
    parser.add_argument("--fingerprint-dim", type=int, default=384)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache)

    # Download
    download_and_extract(cache_dir)

    # Load aliases
    aliases = load_entity_aliases(cache_dir)

    # Load triples
    triples, entities, relations = load_triples(cache_dir, args.max_entities)

    # Build entity index
    entity_list = sorted(entities)
    entity_to_idx = {e: i for i, e in enumerate(entity_list)}
    n = len(entity_list)
    logger.info(f"Building Latent Ocean format for {n:,} entities...")

    # Fingerprints
    fingerprints = build_fingerprints_from_names(entity_list, aliases, args.fingerprint_dim)
    norms = np.linalg.norm(fingerprints, axis=1, keepdims=True)
    fingerprints = fingerprints / np.clip(norms, 1e-8, None)

    # Edges (sparse)
    edge_list = []
    for h, r, t in triples:
        if h in entity_to_idx and t in entity_to_idx:
            edge_list.append((entity_to_idx[h], entity_to_idx[t]))

    edges = np.array(edge_list, dtype=np.int64) if edge_list else np.zeros((0, 2), dtype=np.int64)
    logger.info(f"Edges: {len(edges):,}")

    # Coordinates
    if n <= 50000:
        # Dense spectral for smaller graphs
        from scipy.linalg import eigh
        adj = np.zeros((n, n), dtype=np.float32)
        for src, tgt in edges:
            adj[src, tgt] = 1.0
            adj[tgt, src] = 1.0

        degree = adj.sum(axis=1)
        d_inv = np.where(degree > 0, 1.0 / np.sqrt(degree), 0)
        D = np.diag(d_inv)
        L = np.eye(n) - D @ adj @ D

        try:
            eigenvalues, eigenvectors = eigh(L, subset_by_index=[1, 2])
            embed_2d = eigenvectors
        except Exception:
            embed_2d = np.random.RandomState(42).randn(n, 2)
    else:
        # Sparse spectral for larger graphs
        from scipy.sparse import csr_matrix, diags
        from scipy.sparse.linalg import eigsh

        rows = np.concatenate([edges[:, 0], edges[:, 1]])
        cols = np.concatenate([edges[:, 1], edges[:, 0]])
        data = np.ones(len(rows), dtype=np.float32)
        adj = csr_matrix((data, (rows, cols)), shape=(n, n))

        degree = np.array(adj.sum(axis=1)).flatten()
        d_inv = np.where(degree > 0, 1.0 / np.sqrt(degree), 0)
        D = diags(d_inv)
        L = diags(np.ones(n)) - D @ adj @ D

        try:
            if n > 500000:
                from sklearn.utils.extmath import randomized_svd
                logger.info(f"  Using randomized SVD for {n:,} nodes...")
                A_norm = D @ adj @ D
                U, S, Vt = randomized_svd(A_norm, n_components=3, random_state=42)
                embed_2d = U[:, 1:3]
            else:
                logger.info(f"  Running eigsh for {n:,} nodes...")
                eigenvalues, eigenvectors = eigsh(L, k=3, which="SM", maxiter=1000)
                embed_2d = eigenvectors[:, 1:3]
        except Exception as e:
            logger.warning(f"  Spectral embedding failed ({e}), using degree-based layout")
            degree_arr = np.array(adj.sum(axis=1)).flatten()
            rng_fb = np.random.RandomState(42)
            rank = np.argsort(np.argsort(-degree_arr)).astype(np.float32) / max(n - 1, 1)
            embed_2d = np.stack([rank + rng_fb.normal(0, 0.05, n), rng_fb.uniform(-1, 1, n)], axis=1).astype(np.float32)

    # Map to S²
    norms_2d = np.linalg.norm(embed_2d, axis=1, keepdims=True)
    embed_2d = embed_2d / np.clip(norms_2d, 1e-8, None)

    rng = np.random.RandomState(42)
    theta = np.arccos(np.clip(embed_2d[:, 0], -1, 1)) + rng.normal(0, 0.02, n)
    phi = np.arctan2(embed_2d[:, 1], embed_2d[:, 0]) + np.pi + rng.normal(0, 0.02, n)
    theta = np.clip(theta, 0.01, np.pi - 0.01)

    r = args.sphere_radius
    coords = np.stack([
        r * np.sin(theta) * np.cos(phi),
        r * np.sin(theta) * np.sin(phi),
        r * np.cos(theta),
    ], axis=1).astype(np.float32)

    # Velocities
    velocity = rng.randn(n, 3).astype(np.float32) * 0.02
    for i in range(n):
        normal = coords[i] / (np.linalg.norm(coords[i]) + 1e-8)
        velocity[i] -= np.dot(velocity[i], normal) * normal

    # Labels
    labels = build_relation_labels(triples, entity_to_idx, relations, n)

    # Save
    torch.save(torch.from_numpy(fingerprints), output_dir / "fingerprints.pt")
    torch.save(torch.from_numpy(coords), output_dir / "coords.pt")
    torch.save(torch.from_numpy(velocity), output_dir / "velocity.pt")
    torch.save(torch.from_numpy(labels), output_dir / "labels.pt")

    if n <= 10000:
        # Dense adjacency for small graphs
        adj_dense = np.zeros((n, n), dtype=np.float32)
        for src, tgt in edges:
            adj_dense[src, tgt] = 0.8
        torch.save(torch.from_numpy(adj_dense), output_dir / "adjacency.pt")
    else:
        # Sparse edges
        edges_tensor = torch.zeros(len(edges), 3)
        edges_tensor[:, 0] = torch.from_numpy(edges[:, 0])
        edges_tensor[:, 1] = torch.from_numpy(edges[:, 1])
        edges_tensor[:, 2] = 0.8
        torch.save(edges_tensor, output_dir / "edges.pt")

    # Save entity metadata
    entity_meta = [{"id": eid, "name": aliases.get(eid, eid)} for eid in entity_list]
    with open(output_dir / "entities.json", "w") as f:
        json.dump(entity_meta[:10000], f, indent=2)  # Cap JSON size

    metadata = {
        "source": "Wikidata5m",
        "num_entities": n,
        "num_triples": len(edges),
        "num_relations": len(relations),
        "num_clusters": int(labels.max()) + 1,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"\nSaved to {output_dir}:")
    logger.info(f"  fingerprints.pt: [{n:,}, {args.fingerprint_dim}]")
    logger.info(f"  coords.pt:      [{n:,}, 3]")
    logger.info(f"  edges/adjacency: {len(edges):,} triples")
    logger.info(f"  labels.pt:      [{n:,}] ({int(labels.max()) + 1} clusters)")


if __name__ == "__main__":
    main()
