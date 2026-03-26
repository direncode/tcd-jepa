#!/usr/bin/env python
"""Convert GDELT Global Knowledge Graph events to Latent Ocean format.

Downloads recent GDELT event data (who did what to whom, where, when)
and converts to manifold format. GDELT records every news event on Earth
with actors, actions, geographic coordinates, and emotional tone.

Entities = actors (people, orgs, countries) + events
Fingerprints = TF-IDF of event descriptions + actor metadata
Coordinates = real geographic lat/lon projected onto S²
Causal links = actor co-occurrence in events + temporal sequence
Labels = event type (CAMEO code categories)

Usage:
    python scripts/data_adapters/gdelt_events.py --output ./data/gdelt
    python scripts/data_adapters/gdelt_events.py --output ./data/gdelt --days 30 --max-events 500000

Source: https://www.gdeltproject.org/data.html
"""

import argparse
import csv
import io
import logging
import urllib.request
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("gdelt_adapter")

# GDELT 2.0 Events CSV columns (selected)
# Full spec: https://www.gdeltproject.org/data/documentation/GDELT-Event_Codebook-V2.0.pdf
GDELT_EVENTS_BASE = "http://data.gdeltproject.org/gdeltv2"


def get_gdelt_file_list(days: int = 7) -> list[str]:
    """Get list of GDELT 2.0 event file URLs for the past N days."""
    urls = []
    end = datetime.utcnow()
    start = end - timedelta(days=days)

    # GDELT master file list
    master_url = f"{GDELT_EVENTS_BASE}/masterfilelist.txt"
    logger.info("Fetching GDELT master file list...")

    try:
        req = urllib.request.Request(master_url, headers={"User-Agent": "TCD-JEPA-Research"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            content = resp.read().decode("utf-8", errors="ignore")

        for line in content.splitlines():
            parts = line.strip().split()
            if len(parts) >= 3 and parts[2].endswith(".export.CSV.zip"):
                url = parts[2]
                # Extract date from filename (YYYYMMDDHHMMSS)
                fname = url.split("/")[-1]
                try:
                    date_str = fname[:8]
                    file_date = datetime.strptime(date_str, "%Y%m%d")
                    if start <= file_date <= end:
                        urls.append(url)
                except (ValueError, IndexError):
                    continue
    except Exception as e:
        logger.warning(f"Could not fetch master list: {e}")
        # Fallback: construct URLs for last N days
        for d in range(days):
            date = end - timedelta(days=d)
            date_str = date.strftime("%Y%m%d")
            url = f"{GDELT_EVENTS_BASE}/{date_str}000000.export.CSV.zip"
            urls.append(url)

    logger.info(f"Found {len(urls)} event files for past {days} days")
    return urls[:days * 4]  # Cap at ~4 files per day


def download_and_parse_events(urls: list[str], max_events: int = 100000) -> list[dict]:
    """Download GDELT event CSVs and parse into structured records."""
    events = []

    for url in urls:
        if len(events) >= max_events:
            break

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "TCD-JEPA-Research"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                zip_data = resp.read()

            with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
                for name in zf.namelist():
                    if name.endswith(".CSV"):
                        with zf.open(name) as csvfile:
                            reader = csv.reader(io.TextIOWrapper(csvfile, encoding="utf-8", errors="ignore"), delimiter="\t")
                            for row in reader:
                                if len(row) < 58 or len(events) >= max_events:
                                    continue
                                try:
                                    event = {
                                        "global_event_id": row[0],
                                        "date": row[1],
                                        "actor1_name": row[6] if len(row) > 6 else "",
                                        "actor1_country": row[7] if len(row) > 7 else "",
                                        "actor1_type": row[12] if len(row) > 12 else "",
                                        "actor2_name": row[16] if len(row) > 16 else "",
                                        "actor2_country": row[17] if len(row) > 17 else "",
                                        "actor2_type": row[22] if len(row) > 22 else "",
                                        "event_code": row[26] if len(row) > 26 else "",
                                        "event_base_code": row[27] if len(row) > 27 else "",
                                        "event_root_code": row[28] if len(row) > 28 else "",
                                        "quad_class": row[29] if len(row) > 29 else "",
                                        "goldstein_scale": float(row[30]) if len(row) > 30 and row[30] else 0.0,
                                        "num_mentions": int(row[31]) if len(row) > 31 and row[31] else 1,
                                        "avg_tone": float(row[34]) if len(row) > 34 and row[34] else 0.0,
                                        "lat": float(row[53]) if len(row) > 53 and row[53] else 0.0,
                                        "lon": float(row[54]) if len(row) > 54 and row[54] else 0.0,
                                    }
                                    # Skip events with no actors
                                    if event["actor1_name"] or event["actor2_name"]:
                                        events.append(event)
                                except (ValueError, IndexError):
                                    continue

            logger.info(f"  {url.split('/')[-1]}: {len(events):,} events so far")
        except Exception as e:
            logger.warning(f"  Failed to download {url}: {e}")
            continue

    logger.info(f"Total events parsed: {len(events):,}")
    return events


def build_actor_entities(events: list[dict]) -> tuple[list[dict], dict[str, int]]:
    """Extract unique actors from events."""
    actors = {}
    for ev in events:
        for prefix in ["actor1", "actor2"]:
            name = ev[f"{prefix}_name"]
            if not name:
                continue
            if name not in actors:
                actors[name] = {
                    "name": name,
                    "country": ev[f"{prefix}_country"],
                    "type": ev[f"{prefix}_type"],
                    "event_count": 0,
                    "avg_tone": 0.0,
                    "avg_lat": 0.0,
                    "avg_lon": 0.0,
                }
            actors[name]["event_count"] += 1
            actors[name]["avg_tone"] += ev["avg_tone"]
            if ev["lat"] != 0:
                actors[name]["avg_lat"] += ev["lat"]
                actors[name]["avg_lon"] += ev["lon"]

    # Normalize averages
    for a in actors.values():
        n = max(a["event_count"], 1)
        a["avg_tone"] /= n
        a["avg_lat"] /= n
        a["avg_lon"] /= n

    entity_list = sorted(actors.values(), key=lambda x: -x["event_count"])
    entity_to_idx = {e["name"]: i for i, e in enumerate(entity_list)}

    logger.info(f"Unique actors: {len(entity_list):,}")
    return entity_list, entity_to_idx


def build_fingerprints(entity_list: list[dict], dim: int = 384) -> np.ndarray:
    """Build fingerprints from actor metadata."""
    descriptions = []
    for e in entity_list:
        desc = f"{e['name']} {e['country']} {e['type']} events:{e['event_count']} tone:{e['avg_tone']:.1f}"
        descriptions.append(desc)

    try:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Encoding {len(descriptions):,} actors with sentence-transformers...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(descriptions, show_progress_bar=True, batch_size=256)
        return embeddings.astype(np.float32)
    except ImportError:
        logger.warning("sentence-transformers not installed — using TF-IDF fallback")
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.random_projection import GaussianRandomProjection
            vec = TfidfVectorizer(max_features=2000)
            tfidf = vec.fit_transform(descriptions).toarray()
            if tfidf.shape[1] < dim:
                padded = np.zeros((len(descriptions), dim), dtype=np.float32)
                padded[:, :tfidf.shape[1]] = tfidf
                return padded
            proj = GaussianRandomProjection(n_components=dim, random_state=42)
            return proj.fit_transform(tfidf).astype(np.float32)
        except ImportError:
            embeddings = np.zeros((len(descriptions), dim), dtype=np.float32)
            for i, desc in enumerate(descriptions):
                for w in desc.lower().split():
                    embeddings[i, hash(w) % dim] += 1.0
                embeddings[i] /= max(len(desc.split()), 1)
            return embeddings


def build_coordinates(entity_list: list[dict], sphere_radius: float = 4.5) -> np.ndarray:
    """Use real geographic coordinates projected onto S²."""
    n = len(entity_list)
    coords = np.zeros((n, 3), dtype=np.float32)
    rng = np.random.RandomState(42)

    for i, e in enumerate(entity_list):
        lat = e["avg_lat"]
        lon = e["avg_lon"]
        if abs(lat) < 0.01 and abs(lon) < 0.01:
            # No location — random placement
            lat = rng.uniform(-80, 80)
            lon = rng.uniform(-180, 180)

        # Geographic lat/lon to S² Cartesian
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        coords[i, 0] = sphere_radius * np.cos(lat_rad) * np.cos(lon_rad)
        coords[i, 1] = sphere_radius * np.cos(lat_rad) * np.sin(lon_rad)
        coords[i, 2] = sphere_radius * np.sin(lat_rad)

    return coords


def build_causal_graph(events: list[dict], entity_to_idx: dict[str, int], n: int) -> np.ndarray:
    """Build directed causal graph from actor co-occurrence in events."""
    if n <= 50000:
        adjacency = np.zeros((n, n), dtype=np.float32)
        for ev in events:
            a1 = ev["actor1_name"]
            a2 = ev["actor2_name"]
            if a1 in entity_to_idx and a2 in entity_to_idx:
                i, j = entity_to_idx[a1], entity_to_idx[a2]
                if i != j:
                    # Goldstein scale as weight (-10 to +10, normalize to 0-1)
                    weight = (ev["goldstein_scale"] + 10) / 20.0
                    weight = max(0.1, min(1.0, weight))
                    adjacency[i, j] = max(adjacency[i, j], weight)
        num_edges = int((adjacency > 0).sum())
        logger.info(f"Dense adjacency: {num_edges:,} edges")
        return adjacency
    else:
        # Return empty — will use edges.pt instead
        return np.zeros((0, 0), dtype=np.float32)


def build_edges_sparse(events: list[dict], entity_to_idx: dict[str, int]) -> torch.Tensor:
    """Build sparse edge tensor for large graphs."""
    edge_dict = {}
    for ev in events:
        a1 = ev["actor1_name"]
        a2 = ev["actor2_name"]
        if a1 in entity_to_idx and a2 in entity_to_idx:
            i, j = entity_to_idx[a1], entity_to_idx[a2]
            if i != j:
                weight = (ev["goldstein_scale"] + 10) / 20.0
                weight = max(0.1, min(1.0, weight))
                key = (i, j)
                edge_dict[key] = max(edge_dict.get(key, 0), weight)

    if not edge_dict:
        return torch.zeros(0, 3)

    edges = torch.zeros(len(edge_dict), 3)
    for idx, ((i, j), w) in enumerate(edge_dict.items()):
        edges[idx] = torch.tensor([i, j, w])

    logger.info(f"Sparse edges: {len(edge_dict):,}")
    return edges


def build_labels(entity_list: list[dict]) -> np.ndarray:
    """Assign labels by actor country."""
    countries = sorted(set(e["country"] for e in entity_list if e["country"]))
    country_to_idx = {c: i for i, c in enumerate(countries)}
    # Entities with no country get their own cluster
    unknown_idx = len(countries)
    labels = np.array([
        country_to_idx.get(e["country"], unknown_idx) for e in entity_list
    ], dtype=np.int64)
    logger.info(f"Labels: {len(countries) + 1} country clusters")
    return labels


def main():
    parser = argparse.ArgumentParser(description="Convert GDELT events to Latent Ocean format")
    parser.add_argument("--output", default="./data/gdelt")
    parser.add_argument("--days", type=int, default=7, help="Days of events to download")
    parser.add_argument("--max-events", type=int, default=200000)
    parser.add_argument("--sphere-radius", type=float, default=4.5)
    parser.add_argument("--fingerprint-dim", type=int, default=384)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download events
    urls = get_gdelt_file_list(args.days)
    events = download_and_parse_events(urls, args.max_events)

    if len(events) < 100:
        logger.error("Too few events downloaded. Check network connectivity.")
        return

    # Build entities
    entity_list, entity_to_idx = build_actor_entities(events)
    n = len(entity_list)

    # Build tensors
    fingerprints = build_fingerprints(entity_list, args.fingerprint_dim)
    norms = np.linalg.norm(fingerprints, axis=1, keepdims=True)
    fingerprints = fingerprints / np.clip(norms, 1e-8, None)

    coords = build_coordinates(entity_list, args.sphere_radius)

    rng = np.random.RandomState(42)
    velocity = rng.randn(n, 3).astype(np.float32) * 0.02
    for i in range(n):
        normal = coords[i] / (np.linalg.norm(coords[i]) + 1e-8)
        velocity[i] -= np.dot(velocity[i], normal) * normal

    labels = build_labels(entity_list)

    # Save
    torch.save(torch.from_numpy(fingerprints), output_dir / "fingerprints.pt")
    torch.save(torch.from_numpy(coords), output_dir / "coords.pt")
    torch.save(torch.from_numpy(velocity), output_dir / "velocity.pt")
    torch.save(torch.from_numpy(labels), output_dir / "labels.pt")

    if n <= 50000:
        adjacency = build_causal_graph(events, entity_to_idx, n)
        torch.save(torch.from_numpy(adjacency), output_dir / "adjacency.pt")
    else:
        edges = build_edges_sparse(events, entity_to_idx)
        torch.save(edges, output_dir / "edges.pt")

    # Metadata
    import json
    with open(output_dir / "entities.json", "w") as f:
        json.dump(entity_list[:10000], f, indent=2, default=str)

    with open(output_dir / "metadata.json", "w") as f:
        json.dump({
            "source": "GDELT 2.0 Global Knowledge Graph",
            "days": args.days,
            "num_events_raw": len(events),
            "num_actors": n,
            "num_countries": len(set(e["country"] for e in entity_list)),
        }, f, indent=2)

    logger.info(f"\nSaved to {output_dir}:")
    logger.info(f"  {n:,} actors from {len(events):,} global events ({args.days} days)")
    logger.info(f"  fingerprints.pt: [{n}, {args.fingerprint_dim}]")
    logger.info(f"  coords.pt:      [{n}, 3] (real geographic S²)")
    logger.info(f"  labels.pt:      [{n}] ({len(set(labels.tolist()))} countries)")


if __name__ == "__main__":
    main()
