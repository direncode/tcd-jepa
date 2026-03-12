"""Manifold-native masking for Latent Ocean JEPA training.

Instead of rectangular block masks on a 2D grid (vision), we mask
neighborhoods on S² using great-circle distance and the causal graph.

Masking modes:
- Geodesic: BFS on the causal graph (respects causal structure)
- Spherical: great-circle distance neighborhoods on S² (respects geometry)
- Combined: intersection of both (nearby on sphere AND causally connected)

The JEPA objective becomes: given context entities at scattered S² positions,
predict the latent representations of entities in a target geodesic neighborhood.
"""

import math
from multiprocessing import Value
from typing import Optional

import torch


def great_circle_distance(coords1: torch.Tensor, coords2: torch.Tensor, radius: float = 4.5) -> torch.Tensor:
    """Compute great-circle distance between points on S².

    Args:
        coords1: [N, 3] Cartesian coordinates.
        coords2: [M, 3] Cartesian coordinates.
        radius: Sphere radius (4.5 in Latent Ocean).

    Returns:
        [N, M] pairwise great-circle distances.
    """
    # Normalize to unit sphere
    c1 = coords1 / (coords1.norm(dim=-1, keepdim=True) + 1e-8)
    c2 = coords2 / (coords2.norm(dim=-1, keepdim=True) + 1e-8)

    # Cosine of angular distance
    cos_angle = torch.mm(c1, c2.t()).clamp(-1, 1)

    # Great-circle distance = R * arccos(dot product)
    return radius * torch.acos(cos_angle)


class ManifoldMaskCollator:
    """Generates context and target masks for manifold JEPA training.

    Target masks: geodesic neighborhoods (connected subgraphs around seed nodes)
    or spherical neighborhoods (nearby on S² by great-circle distance).
    Context masks: remaining nodes, subsampled.
    """

    def __init__(
        self,
        num_tokens: int = 64,
        context_ratio: tuple[float, float] = (0.3, 0.5),
        target_ratio: tuple[float, float] = (0.15, 0.3),
        num_context_masks: int = 4,
        num_target_masks: int = 1,
        min_keep: int = 4,
        use_geodesic: bool = True,
        sphere_radius: float = 4.5,
    ):
        self.num_tokens = num_tokens
        self.context_ratio = context_ratio
        self.target_ratio = target_ratio
        self.num_context_masks = num_context_masks
        self.num_target_masks = num_target_masks
        self.min_keep = min_keep
        self.use_geodesic = use_geodesic
        self.sphere_radius = sphere_radius
        self._itr_counter = Value("i", -1)

    def step(self) -> int:
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def _sample_geodesic_neighborhood(
        self, adjacency, seed: int, target_size: int,
    ) -> torch.Tensor:
        """Grow neighborhood via BFS on the causal graph. Pads with random if needed.

        Accepts either a dense tensor or a SparseAdjacency object.
        """
        from tcd_jepa.manifold.sparse_graph import SparseAdjacency

        if isinstance(adjacency, SparseAdjacency):
            reachable = adjacency.bfs(seed, max_depth=10)
            selected = list(reachable)[:target_size]
            N = adjacency.num_nodes
            if len(selected) < target_size:
                all_nodes = set(range(N))
                remaining = list(all_nodes - set(selected))
                import random as _rng
                _rng.shuffle(remaining)
                selected.extend(remaining[:target_size - len(selected)])
            return torch.tensor(selected[:target_size], dtype=torch.long)

        # Dense tensor path
        N = adjacency.shape[0]
        adj_binary = (adjacency.abs() > 1e-6)
        visited = torch.zeros(N, dtype=torch.bool)
        visited[seed] = True
        frontier = [seed]
        selected = [seed]

        while len(selected) < target_size and frontier:
            next_frontier = []
            for node in frontier:
                neighbors = torch.where(adj_binary[node])[0]
                for nb in neighbors:
                    nb_idx = nb.item()
                    if not visited[nb_idx]:
                        visited[nb_idx] = True
                        selected.append(nb_idx)
                        next_frontier.append(nb_idx)
                        if len(selected) >= target_size:
                            break
                if len(selected) >= target_size:
                    break
            frontier = next_frontier

        # Pad with random nodes if BFS didn't reach target_size
        if len(selected) < target_size:
            remaining = torch.where(~visited)[0]
            if len(remaining) > 0:
                extra = remaining[torch.randperm(len(remaining))[:target_size - len(selected)]]
                selected.extend(extra.tolist())

        return torch.tensor(selected[:target_size], dtype=torch.long)

    def _sample_spherical_neighborhood(
        self, coords: torch.Tensor, seed: int, target_size: int,
    ) -> torch.Tensor:
        """Select nearest neighbors by great-circle distance on S²."""
        seed_coord = coords[seed:seed+1]
        dists = great_circle_distance(seed_coord, coords, self.sphere_radius).squeeze(0)
        _, sorted_idx = dists.sort()
        return sorted_idx[:target_size]

    def _sample_random_subset(
        self, N: int, size: int, exclude: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if exclude is not None:
            mask = torch.ones(N, dtype=torch.bool)
            mask[exclude] = False
            available = torch.where(mask)[0]
        else:
            available = torch.arange(N)

        if len(available) >= size:
            perm = torch.randperm(len(available))[:size]
            return available[perm]

        # Not enough available — pad by repeating
        if len(available) == 0:
            return torch.randperm(N)[:size]
        repeats = (size + len(available) - 1) // len(available)
        padded = available.repeat(repeats)[:size]
        return padded

    def __call__(
        self, batch: list,
    ) -> tuple[dict, list[torch.Tensor], list[torch.Tensor]]:
        """Create manifold-aware masks when collating a batch.

        Each item is a dict with fingerprints, coords, velocity, adjacency, etc.
        """
        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)

        B = len(batch)

        # Extract components
        fp_list, coord_list, vel_list, adj_list = [], [], [], []
        labels_list, indices_list = [], []

        for item in batch:
            fp_list.append(item["fingerprints"])
            coord_list.append(item.get("coords"))
            vel_list.append(item.get("velocity"))
            adj_list.append(item.get("adjacency"))
            if "labels" in item:
                labels_list.append(item["labels"])
            if "indices" in item:
                indices_list.append(item["indices"])

        N = fp_list[0].shape[0]

        # Sample mask sizes
        target_frac = self.target_ratio[0] + torch.rand(1, generator=g).item() * (
            self.target_ratio[1] - self.target_ratio[0])
        context_frac = self.context_ratio[0] + torch.rand(1, generator=g).item() * (
            self.context_ratio[1] - self.context_ratio[0])
        target_size = max(int(N * target_frac), self.min_keep)
        context_size = max(int(N * context_frac), self.min_keep)

        all_target_masks, all_context_masks = [], []
        min_target, min_context = N, N

        for b_idx in range(B):
            adj = adj_list[b_idx]
            coords = coord_list[b_idx]

            # Target masks
            sample_targets = []
            all_target_nodes = set()
            for _ in range(self.num_target_masks):
                seed_node = torch.randint(0, N, (1,), generator=g).item()
                if self.use_geodesic and adj is not None:
                    target_mask = self._sample_geodesic_neighborhood(adj, seed_node, target_size)
                elif coords is not None:
                    target_mask = self._sample_spherical_neighborhood(coords, seed_node, target_size)
                else:
                    target_mask = self._sample_random_subset(N, target_size)
                sample_targets.append(target_mask)
                all_target_nodes.update(target_mask.tolist())
                min_target = min(min_target, len(target_mask))

            # Context masks (from non-target nodes)
            target_indices = torch.tensor(list(all_target_nodes), dtype=torch.long)
            sample_contexts = []
            for _ in range(self.num_context_masks):
                context_mask = self._sample_random_subset(N, context_size, exclude=target_indices)
                sample_contexts.append(context_mask)
                min_context = min(min_context, len(context_mask))

            all_target_masks.append(sample_targets)
            all_context_masks.append(sample_contexts)

        # Trim and collate
        min_target = max(min_target, self.min_keep)
        min_context = max(min_context, self.min_keep)

        collated_target = [[m[:min_target] for m in masks] for masks in all_target_masks]
        collated_context = [[m[:min_context] for m in masks] for masks in all_context_masks]

        collated_target = torch.utils.data.default_collate(collated_target)
        collated_context = torch.utils.data.default_collate(collated_context)

        # Collate batch data
        batch_data = {"fingerprints": torch.stack(fp_list)}
        if coord_list[0] is not None:
            batch_data["coords"] = torch.stack([c for c in coord_list if c is not None])
        if vel_list[0] is not None:
            batch_data["velocity"] = torch.stack([v for v in vel_list if v is not None])
        if adj_list[0] is not None:
            batch_data["adjacency"] = torch.stack([a for a in adj_list if a is not None])
        if labels_list:
            batch_data["labels"] = torch.stack(labels_list)
        if indices_list:
            batch_data["indices"] = torch.stack(indices_list)

        return batch_data, collated_context, collated_target
