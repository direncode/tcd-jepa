"""Dataset classes for Latent Ocean manifold data.

Provides ingestion for spherical databases that store entity fingerprints on S²
with causal links, velocities, and entity types.

Data sources:
1. LatentOceanDataset — loads from exported DuckDB data (.pt or .npz files)
2. SyntheticManifoldDataset — generates Latent Ocean-like data for testing
3. CausalManifoldDataset — base class for windowed manifold sampling
"""

import math
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from tcd_jepa.manifold.sparse_graph import SparseAdjacency


class CausalManifoldDataset(Dataset):
    """Base dataset for manifold-structured data with causal links.

    Handles windowed sampling of entities from a large manifold database,
    preserving local structure via geodesic (BFS) or random windows.

    Fields per entity:
    - fingerprints: [num_entities, 384] — sentence-transformer embeddings
    - coords: [num_entities, 3] — S² positions (Cartesian, radius ~4.5)
    - velocity: [num_entities, 3] — tangent-space velocities
    - adjacency: [num_entities, num_entities] — weighted directed causal links
    - entity_labels: [num_entities] — entity type indices (for evaluation)
    """

    def __init__(
        self,
        fingerprints: torch.Tensor,
        coords: torch.Tensor,
        adjacency: torch.Tensor,
        velocity: Optional[torch.Tensor] = None,
        entity_labels: Optional[torch.Tensor] = None,
        num_tokens: int = 64,
        num_samples: int = 1000,
        window_mode: str = "geodesic",
        sparse_adjacency: Optional[SparseAdjacency] = None,
    ):
        self.fingerprints = fingerprints
        self.coords = coords
        self.adjacency = adjacency
        self.velocity = velocity if velocity is not None else torch.zeros_like(coords)
        self.entity_labels = entity_labels
        self.num_tokens = num_tokens
        self.num_samples = num_samples
        self.window_mode = window_mode
        self.num_entities = fingerprints.shape[0]
        self.sparse_adjacency = sparse_adjacency

        # Precompute COO arrays for fast sparse window extraction
        self._sparse_coo = None
        if sparse_adjacency is not None:
            src_list, tgt_list, val_list = [], [], []
            for link in sparse_adjacency.all_links():
                src_list.append(link.source)
                tgt_list.append(link.target)
                val_list.append(link.strength)
            if src_list:
                self._sparse_coo = (
                    torch.tensor(src_list, dtype=torch.long),
                    torch.tensor(tgt_list, dtype=torch.long),
                    torch.tensor(val_list, dtype=torch.float),
                )

    def __len__(self) -> int:
        return self.num_samples

    def _sample_geodesic_window(self, seed: int) -> torch.Tensor:
        """Sample a connected window via BFS on the causal graph.

        Uses sparse BFS when a SparseAdjacency is available (O(E) instead of O(N²)).
        """
        if self.sparse_adjacency is not None:
            # Use sparse BFS — much faster for large graphs
            reachable = self.sparse_adjacency.bfs(seed, max_depth=10)
            selected = list(reachable)[:self.num_tokens]
            if len(selected) < self.num_tokens:
                all_nodes = set(range(self.num_entities))
                remaining = list(all_nodes - set(selected))
                import random
                random.shuffle(remaining)
                selected.extend(remaining[:self.num_tokens - len(selected)])
            return torch.tensor(selected[:self.num_tokens], dtype=torch.long)

        # Fallback: dense BFS
        adj_binary = (self.adjacency.abs() > 1e-6)
        visited = torch.zeros(self.num_entities, dtype=torch.bool)
        visited[seed] = True
        frontier = [seed]
        selected = [seed]

        while len(selected) < self.num_tokens and frontier:
            next_frontier = []
            for node in frontier:
                neighbors_out = torch.where(adj_binary[node])[0]
                neighbors_in = torch.where(adj_binary[:, node])[0]
                neighbors = torch.cat([neighbors_out, neighbors_in]).unique()
                perm = torch.randperm(len(neighbors))
                for idx in perm:
                    nb = neighbors[idx].item()
                    if not visited[nb]:
                        visited[nb] = True
                        selected.append(nb)
                        next_frontier.append(nb)
                        if len(selected) >= self.num_tokens:
                            break
                if len(selected) >= self.num_tokens:
                    break
            frontier = next_frontier

        # Pad with random if not enough connected
        if len(selected) < self.num_tokens:
            remaining = torch.where(~visited)[0]
            if len(remaining) > 0:
                extra = remaining[torch.randperm(len(remaining))[:self.num_tokens - len(selected)]]
                selected.extend(extra.tolist())

        return torch.tensor(selected[:self.num_tokens], dtype=torch.long)

    def __getitem__(self, idx: int) -> dict:
        seed = torch.randint(0, self.num_entities, (1,)).item()

        if self.window_mode == "geodesic":
            indices = self._sample_geodesic_window(seed)
        else:
            indices = torch.randperm(self.num_entities)[:self.num_tokens]

        # Build adjacency submatrix for the window
        if self.sparse_adjacency is not None and self._sparse_coo is not None:
            # Fast vectorized sparse lookup using precomputed COO tensor
            adj_window = self._fast_sparse_window(indices)
        elif self.sparse_adjacency is not None:
            adj_window = self.sparse_adjacency.to_dense(indices.tolist())
        elif self.adjacency.numel() > 0:
            adj_window = self.adjacency[indices][:, indices]
        else:
            adj_window = torch.zeros(len(indices), len(indices))

        result = {
            "fingerprints": self.fingerprints[indices],
            "coords": self.coords[indices],
            "velocity": self.velocity[indices],
            "adjacency": adj_window,
            "indices": indices,
        }

        if self.entity_labels is not None:
            result["labels"] = self.entity_labels[indices]

        return result

    def _fast_sparse_window(self, indices: torch.Tensor) -> torch.Tensor:
        """Extract K×K adjacency submatrix using vectorized sparse ops.

        Uses precomputed COO arrays with torch.isin for O(E) filtering
        instead of O(K*avg_degree) Python loops.
        """
        K = len(indices)
        src_all, tgt_all, val_all = self._sparse_coo

        # Vectorized: find edges where both src and tgt are in the window
        src_mask = torch.isin(src_all, indices)
        tgt_mask = torch.isin(tgt_all, indices)
        both_mask = src_mask & tgt_mask

        if not both_mask.any():
            return torch.zeros(K, K)

        # Map global indices to local positions
        # Create lookup: global_idx -> local_pos
        local_map = torch.full((self.num_entities,), -1, dtype=torch.long)
        local_map[indices] = torch.arange(K)

        matched_src = local_map[src_all[both_mask]]
        matched_tgt = local_map[tgt_all[both_mask]]
        matched_val = val_all[both_mask]

        mat = torch.zeros(K, K)
        mat[matched_src, matched_tgt] = matched_val
        return mat


class LatentOceanDataset(CausalManifoldDataset):
    """Loads entity data exported from Latent Ocean's DuckDB.

    Expected file format (directory containing .pt files):
        fingerprints.pt  — [num_entities, 384]  (all-MiniLM-L6-v2 embeddings)
        coords.pt        — [num_entities, 3]    (S² positions, radius 4.5)
        velocity.pt      — [num_entities, 3]    (tangent-space velocities)
        adjacency.pt     — [num_entities, num_entities]  (weighted directed links)
        labels.pt        — [num_entities]        (entity_type indices, optional)

    For large-scale graphs (>10K entities), use sparse format:
        adjacency_sparse.pt — sparse COO tensor (saves O(N²) → O(E) memory)
        OR edges.pt — [E, 3] tensor of (source, target, weight) triples

    Or .npz format:
        latent_ocean_export.npz with keys: fingerprints, coords, velocity, adjacency, labels
    """

    def __init__(
        self,
        data_dir: str,
        num_tokens: int = 64,
        num_samples: int = 1000,
        window_mode: str = "geodesic",
    ):
        from pathlib import Path
        data_path = Path(data_dir)

        if (data_path / "latent_ocean_export.npz").exists():
            data = np.load(data_path / "latent_ocean_export.npz")
            fingerprints = torch.from_numpy(data["fingerprints"]).float()
            coords = torch.from_numpy(data["coords"]).float()
            velocity = torch.from_numpy(data["velocity"]).float()
            adjacency = torch.from_numpy(data["adjacency"]).float()
            labels = torch.from_numpy(data["labels"]).long() if "labels" in data else None
        else:
            fingerprints = torch.load(data_path / "fingerprints.pt", weights_only=True)
            coords = torch.load(data_path / "coords.pt", weights_only=True)
            velocity = torch.load(data_path / "velocity.pt", weights_only=True) if (data_path / "velocity.pt").exists() else torch.zeros_like(coords)
            labels = torch.load(data_path / "labels.pt", weights_only=True) if (data_path / "labels.pt").exists() else None

            # Load adjacency — prefer sparse for large graphs
            if (data_path / "edges.pt").exists():
                adjacency = torch.zeros(0, 0)  # Placeholder — use sparse
            elif (data_path / "adjacency_sparse.pt").exists():
                adjacency = torch.zeros(0, 0)  # Placeholder — use sparse
            else:
                adjacency = torch.load(data_path / "adjacency.pt", weights_only=True)

        # Build SparseAdjacency for large-scale graphs
        sparse_adj = None
        if (data_path / "edges.pt").exists():
            edges = torch.load(data_path / "edges.pt", weights_only=True)
            sparse_adj = SparseAdjacency()
            for row in edges:
                src, tgt = int(row[0].item()), int(row[1].item())
                weight = float(row[2].item()) if row.shape[0] > 2 else 0.8
                sparse_adj.add_link(src, tgt, weight)
            import logging
            logging.getLogger("tcd_jepa").info(
                f"Loaded sparse adjacency: {sparse_adj.num_nodes} nodes, {sparse_adj.num_links} edges"
            )
        elif (data_path / "adjacency_sparse.pt").exists():
            sparse_tensor = torch.load(data_path / "adjacency_sparse.pt", weights_only=True)
            indices = sparse_tensor.coalesce().indices()
            values = sparse_tensor.coalesce().values()
            sparse_adj = SparseAdjacency()
            for k in range(indices.shape[1]):
                sparse_adj.add_link(int(indices[0, k]), int(indices[1, k]), float(values[k]))
        elif adjacency.numel() > 0 and adjacency.shape[0] > 5000:
            # Auto-convert dense to sparse for large graphs
            sparse_adj = SparseAdjacency.from_dense(adjacency)
            adjacency = torch.zeros(0, 0)  # Free memory

        super().__init__(
            fingerprints=fingerprints,
            coords=coords,
            adjacency=adjacency,
            velocity=velocity,
            entity_labels=labels,
            num_tokens=num_tokens,
            num_samples=num_samples,
            window_mode=window_mode,
            sparse_adjacency=sparse_adj,
        )

        self.num_clusters = int(labels.max().item()) + 1 if labels is not None else 0


class SyntheticManifoldDataset(Dataset):
    """Synthetic dataset mimicking Latent Ocean's data structure.

    Generates entities on S² (radius 4.5) with:
    - Clusters of entities at different regions (like entity_types)
    - 4 types of causal links with different strengths (FK=0.9, semantic=0.7, URL=0.6, cosine=0.65)
    - Tangent-space velocities (drifting toward cluster centroids)
    - 384D fingerprints combining cluster identity + positional encoding + noise
    """

    def __init__(
        self,
        num_entities: int = 256,
        fingerprint_dim: int = 384,
        num_clusters: int = 6,
        num_tokens: int = 64,
        num_samples: int = 500,
        causal_link_prob: float = 0.15,
        cross_cluster_prob: float = 0.03,
        noise_std: float = 0.05,
        sphere_radius: float = 4.5,
        seed: int = 42,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.num_samples = num_samples
        self.num_clusters = num_clusters
        self.sphere_radius = sphere_radius

        rng = np.random.RandomState(seed)

        # Cluster centers via golden spiral on S²
        golden_ratio = (1 + math.sqrt(5)) / 2
        cluster_theta = np.zeros(num_clusters)
        cluster_phi = np.zeros(num_clusters)
        for i in range(num_clusters):
            cluster_theta[i] = math.acos(1 - 2 * (i + 0.5) / num_clusters)
            cluster_phi[i] = 2 * math.pi * i / golden_ratio

        labels = rng.randint(0, num_clusters, num_entities)

        # Positions near cluster centers
        theta = np.zeros(num_entities)
        phi = np.zeros(num_entities)
        for i in range(num_entities):
            c = labels[i]
            theta[i] = cluster_theta[c] + rng.normal(0, 0.25)
            phi[i] = cluster_phi[c] + rng.normal(0, 0.25)
        theta = np.clip(theta, 0.01, math.pi - 0.01)
        phi = phi % (2 * math.pi)

        # Cartesian on S² at radius 4.5
        x = sphere_radius * np.sin(theta) * np.cos(phi)
        y = sphere_radius * np.sin(theta) * np.sin(phi)
        z = sphere_radius * np.cos(theta)
        coords = np.stack([x, y, z], axis=1)

        # Cluster centroids
        cluster_centers = np.zeros((num_clusters, 3))
        for c in range(num_clusters):
            cluster_centers[c] = [
                sphere_radius * math.sin(cluster_theta[c]) * math.cos(cluster_phi[c]),
                sphere_radius * math.sin(cluster_theta[c]) * math.sin(cluster_phi[c]),
                sphere_radius * math.cos(cluster_theta[c]),
            ]

        # Tangent-space velocities (drift toward centroid)
        velocity = np.zeros((num_entities, 3))
        for i in range(num_entities):
            drift = cluster_centers[labels[i]] - coords[i]
            normal = coords[i] / (np.linalg.norm(coords[i]) + 1e-8)
            drift_tangent = drift - np.dot(drift, normal) * normal
            velocity[i] = drift_tangent * 0.1 + rng.normal(0, 0.02, 3)
            velocity[i] -= np.dot(velocity[i], normal) * normal

        # 384D fingerprints
        fingerprints = np.zeros((num_entities, fingerprint_dim))
        for i in range(num_entities):
            c = labels[i]
            cluster_dim = fingerprint_dim // num_clusters
            base = np.zeros(fingerprint_dim)
            start = c * cluster_dim
            end = min((c + 1) * cluster_dim, fingerprint_dim)
            base[start:end] = rng.uniform(0.3, 0.7, end - start)

            pos_signal = np.zeros(fingerprint_dim)
            for freq in range(1, 6):
                pos_signal += np.sin(np.arange(fingerprint_dim) * theta[i] * freq / math.pi) / freq
                pos_signal += np.cos(np.arange(fingerprint_dim) * phi[i] * freq / (2 * math.pi)) / freq
            pos_signal /= 5.0

            fingerprints[i] = base * 0.5 + pos_signal * 0.3 + rng.normal(0, noise_std, fingerprint_dim)

        # 4-type causal links (FK=0.9, semantic=0.7, URL=0.6, cosine=0.65)
        adjacency = np.zeros((num_entities, num_entities))
        for i in range(num_entities):
            for j in range(num_entities):
                if i == j:
                    continue
                same = labels[i] == labels[j]
                if same:
                    r = rng.random()
                    if r < causal_link_prob * 0.3:
                        adjacency[i, j] = 0.9
                    elif r < causal_link_prob * 0.6:
                        adjacency[i, j] = 0.7
                    elif r < causal_link_prob:
                        adjacency[i, j] = 0.65
                else:
                    r = rng.random()
                    if r < cross_cluster_prob * 0.3:
                        adjacency[i, j] = 0.6
                    elif r < cross_cluster_prob:
                        adjacency[i, j] = 0.65

        # Causal direction bias
        for i in range(num_entities):
            for j in range(i):
                if adjacency[i, j] > 0 and rng.random() < 0.7:
                    adjacency[i, j] = 0.0

        self.fingerprints = torch.from_numpy(fingerprints).float()
        self.coords = torch.from_numpy(coords).float()
        self.velocity = torch.from_numpy(velocity).float()
        self.adjacency = torch.from_numpy(adjacency).float()
        self.entity_labels = torch.from_numpy(labels).long()
        self.num_entities = num_entities

        self._inner = CausalManifoldDataset(
            fingerprints=self.fingerprints,
            coords=self.coords,
            adjacency=self.adjacency,
            velocity=self.velocity,
            entity_labels=self.entity_labels,
            num_tokens=num_tokens,
            num_samples=num_samples,
            window_mode="geodesic",
        )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        return self._inner[idx]
