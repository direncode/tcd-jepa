"""Sparse adjacency graph for scaling to millions of causal links.

Replaces dense [N, N] tensors with an edge-list representation that uses O(E)
memory instead of O(N²).  Provides a ``to_dense(indices)`` bridge so existing
training code that indexes ``adjacency[i, j]`` still works on small subsets.
"""

from dataclasses import dataclass, field
from collections import deque
from typing import Iterator, Optional

import torch

from tcd_jepa.manifold.lineage import make_lineage_id


@dataclass
class CausalLink:
    """A single directed edge in the causal graph."""
    link_id: str = field(default_factory=lambda: make_lineage_id("link"))
    source: int = 0
    target: int = 0
    strength: float = 0.0
    link_type: str = ""       # structural, semantic, reference, cooccurrence
    metadata: dict = field(default_factory=dict)


class SparseAdjacency:
    """Edge-list sparse graph with O(1) neighbor lookup.

    Storage: ``_out[source] = [CausalLink, ...]`` and
             ``_in[target]  = [CausalLink, ...]``.
    """

    def __init__(self) -> None:
        self._out: dict[int, list[CausalLink]] = {}
        self._in: dict[int, list[CausalLink]] = {}
        self._nodes: set[int] = set()
        self._num_links: int = 0

    # -- mutation --

    def add_link(
        self,
        source: int,
        target: int,
        strength: float,
        link_type: str = "",
        link_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> CausalLink:
        """Add a directed edge and return the CausalLink object."""
        link = CausalLink(
            link_id=link_id or make_lineage_id("link"),
            source=source,
            target=target,
            strength=strength,
            link_type=link_type,
            metadata=metadata or {},
        )
        self._out.setdefault(source, []).append(link)
        self._in.setdefault(target, []).append(link)
        self._nodes.add(source)
        self._nodes.add(target)
        self._num_links += 1
        return link

    # -- query --

    def neighbors(
        self, node_id: int, direction: str = "out",
    ) -> Iterator[CausalLink]:
        """Iterate over edges from / to *node_id*."""
        if direction in ("out", "both"):
            yield from self._out.get(node_id, [])
        if direction in ("in", "both"):
            yield from self._in.get(node_id, [])

    def get_strength(self, source: int, target: int) -> float:
        """Return the maximum link strength between source→target, or 0."""
        best = 0.0
        for link in self._out.get(source, []):
            if link.target == target and link.strength > best:
                best = link.strength
        return best

    def bfs(self, seed: int, max_depth: int = 5) -> set[int]:
        """Breadth-first search returning reachable node IDs."""
        visited: set[int] = set()
        queue: deque[tuple[int, int]] = deque([(seed, 0)])
        while queue:
            node, depth = queue.popleft()
            if node in visited or depth > max_depth:
                continue
            visited.add(node)
            for link in self._out.get(node, []):
                queue.append((link.target, depth + 1))
            for link in self._in.get(node, []):
                queue.append((link.source, depth + 1))
        return visited

    def top_k_similar(
        self,
        node: int,
        k: int,
        embeddings: torch.Tensor,
        batch_size: int = 1000,
    ) -> list[tuple[int, float]]:
        """Batch-wise top-k cosine similarity, avoiding full N×N matrix."""
        if node >= embeddings.shape[0]:
            return []
        query = embeddings[node].unsqueeze(0)  # [1, D]
        query_norm = query / (query.norm(dim=-1, keepdim=True) + 1e-8)

        N = embeddings.shape[0]
        all_scores: list[tuple[int, float]] = []

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch = embeddings[start:end]
            batch_norm = batch / (batch.norm(dim=-1, keepdim=True) + 1e-8)
            sims = (query_norm @ batch_norm.t()).squeeze(0)  # [batch]
            for offset, sim_val in enumerate(sims.tolist()):
                idx = start + offset
                if idx != node:
                    all_scores.append((idx, sim_val))

        all_scores.sort(key=lambda x: -x[1])
        return all_scores[:k]

    # -- conversion --

    def to_dense(self, indices: Optional[list[int]] = None) -> torch.Tensor:
        """Materialise a dense [K, K] submatrix for a subset of K nodes.

        If *indices* is None, builds the full [N, N] matrix (use with care).
        """
        if indices is None:
            if not self._nodes:
                return torch.zeros(0, 0)
            N = max(self._nodes) + 1
            indices = list(range(N))

        K = len(indices)
        idx_to_pos = {idx: pos for pos, idx in enumerate(indices)}
        mat = torch.zeros(K, K)

        for pos_i, src in enumerate(indices):
            for link in self._out.get(src, []):
                pos_j = idx_to_pos.get(link.target)
                if pos_j is not None:
                    mat[pos_i, pos_j] = max(mat[pos_i, pos_j].item(), link.strength)

        return mat

    def to_torch_sparse(self) -> torch.Tensor:
        """Return a ``torch.sparse_coo_tensor`` of the full graph."""
        if self._num_links == 0:
            return torch.sparse_coo_tensor(
                torch.empty(2, 0, dtype=torch.long),
                torch.empty(0),
                size=(0, 0),
            )
        N = max(self._nodes) + 1
        rows, cols, vals = [], [], []
        for src, links in self._out.items():
            for link in links:
                rows.append(src)
                cols.append(link.target)
                vals.append(link.strength)
        indices = torch.tensor([rows, cols], dtype=torch.long)
        values = torch.tensor(vals, dtype=torch.float)
        return torch.sparse_coo_tensor(indices, values, size=(N, N))

    @classmethod
    def from_dense(cls, tensor: torch.Tensor, min_strength: float = 1e-6) -> "SparseAdjacency":
        """Create SparseAdjacency from a dense [N, N] tensor."""
        graph = cls()
        rows, cols = torch.where(tensor.abs() > min_strength)
        for r, c in zip(rows.tolist(), cols.tolist()):
            graph.add_link(r, c, float(tensor[r, c].item()))
        return graph

    # -- stats --

    def stats(self) -> dict:
        N = len(self._nodes)
        E = self._num_links
        density = E / max(N * N, 1) if N else 0.0
        degrees = {n: len(self._out.get(n, [])) for n in self._nodes}
        avg_degree = sum(degrees.values()) / max(N, 1)
        return {
            "num_nodes": N,
            "num_links": E,
            "density": density,
            "avg_out_degree": avg_degree,
            "max_out_degree": max(degrees.values()) if degrees else 0,
        }

    @property
    def num_nodes(self) -> int:
        return len(self._nodes)

    @property
    def num_links(self) -> int:
        return self._num_links

    def all_links(self) -> Iterator[CausalLink]:
        """Iterate over every link in the graph."""
        for links in self._out.values():
            yield from links
