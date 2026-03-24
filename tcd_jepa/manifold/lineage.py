"""Lineage tracking system for enterprise-grade provenance.

Every artifact in the pipeline (chunk, link, cluster, insight, module) gets a
unique LineageID.  The LineageGraph stores the full DAG so any insight can be
traced back to the source documents that produced it.
"""

import json
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

# ---------------------------------------------------------------------------
# Lineage ID
# ---------------------------------------------------------------------------

_PREFIX_MAP = {
    "chunk": "chk",
    "link": "lnk",
    "cluster": "clu",
    "insight": "ins",
    "module": "mod",
}


def make_lineage_id(kind: str) -> str:
    """Create a prefixed UUID lineage ID.

    Args:
        kind: One of 'chunk', 'link', 'cluster', 'insight', 'module'.

    Returns:
        String like ``chk_a1b2c3d4``.
    """
    prefix = _PREFIX_MAP.get(kind, kind[:3])
    short = uuid.uuid4().hex[:12]
    return f"{prefix}_{short}"


# ---------------------------------------------------------------------------
# Lineage node hierarchy
# ---------------------------------------------------------------------------

@dataclass
class LineageNode:
    """Base node in the lineage DAG."""
    lineage_id: str
    kind: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source_ids: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class ChunkNode(LineageNode):
    """A document chunk with provenance."""
    doc_id: str = ""
    doc_title: str = ""
    chunk_idx: int = 0
    text_hash: str = ""
    char_start: int = 0
    char_end: int = 0
    entity_spans: list[dict] = field(default_factory=list)  # [{entity, start, end}]


@dataclass
class LinkNode(LineageNode):
    """A causal / semantic link between two chunks."""
    source_chunk_id: str = ""
    target_chunk_id: str = ""
    link_type: str = ""       # structural, semantic, reference, cooccurrence
    strength: float = 0.0
    method: str = ""


@dataclass
class ClusterNode(LineageNode):
    """A discovered topic cluster."""
    member_chunk_ids: list[str] = field(default_factory=list)
    centroid_chunk_id: str = ""
    coherence: float = 0.0
    method_params: dict = field(default_factory=dict)


@dataclass
class InsightNode(LineageNode):
    """A generated insight with full derivation chain."""
    parent_insight_ids: list[str] = field(default_factory=list)
    source_chunk_ids: list[str] = field(default_factory=list)
    source_link_ids: list[str] = field(default_factory=list)
    source_cluster_ids: list[str] = field(default_factory=list)
    derivation_steps: list[str] = field(default_factory=list)
    confidence_breakdown: dict = field(default_factory=dict)
    contradicts: list[str] = field(default_factory=list)
    supports: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Lineage graph
# ---------------------------------------------------------------------------

class LineageGraph:
    """Directed acyclic graph of lineage nodes.

    Supports forward (children) and backward (parents) traversal so any
    insight can be traced to the exact source chunks that produced it.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, LineageNode] = {}

    # -- mutation --

    def add(self, node: LineageNode) -> None:
        """Register a node (idempotent)."""
        self._nodes[node.lineage_id] = node

    def add_edge(self, parent_id: str, child_id: str) -> None:
        """Record that *child* derives from *parent*."""
        child = self._nodes.get(child_id)
        if child is not None and parent_id not in child.source_ids:
            child.source_ids.append(parent_id)

    # -- query --

    def get(self, lineage_id: str) -> Optional[LineageNode]:
        return self._nodes.get(lineage_id)

    def ancestors(self, lineage_id: str, max_depth: int = 50) -> list[LineageNode]:
        """BFS backward through source_ids."""
        visited: set[str] = set()
        result: list[LineageNode] = []
        queue: deque[tuple[str, int]] = deque([(lineage_id, 0)])
        while queue:
            nid, depth = queue.popleft()
            if nid in visited or depth > max_depth:
                continue
            visited.add(nid)
            node = self._nodes.get(nid)
            if node is None:
                continue
            if nid != lineage_id:
                result.append(node)
            for parent_id in node.source_ids:
                queue.append((parent_id, depth + 1))
        return result

    def descendants(self, lineage_id: str, max_depth: int = 50) -> list[LineageNode]:
        """BFS forward to find everything derived from *lineage_id*."""
        children_map: dict[str, list[str]] = {}
        for nid, node in self._nodes.items():
            for pid in node.source_ids:
                children_map.setdefault(pid, []).append(nid)

        visited: set[str] = set()
        result: list[LineageNode] = []
        queue: deque[tuple[str, int]] = deque([(lineage_id, 0)])
        while queue:
            nid, depth = queue.popleft()
            if nid in visited or depth > max_depth:
                continue
            visited.add(nid)
            node = self._nodes.get(nid)
            if node is None:
                continue
            if nid != lineage_id:
                result.append(node)
            for child_id in children_map.get(nid, []):
                queue.append((child_id, depth + 1))
        return result

    def trace_lineage(self, insight_id: str) -> dict:
        """Full chain from an insight back to source chunks.

        Returns dict with keys: chunks, links, clusters, insights, modules.
        """
        chain: dict[str, list[LineageNode]] = {
            "chunks": [],
            "links": [],
            "clusters": [],
            "insights": [],
            "modules": [],
        }
        for node in self.ancestors(insight_id):
            bucket = {
                "chunk": "chunks",
                "link": "links",
                "cluster": "clusters",
                "insight": "insights",
                "module": "modules",
            }.get(node.kind, "modules")
            chain[bucket].append(node)
        return chain

    def find_contradictions(self, insight_id: str) -> list[str]:
        """Return insight IDs that contradict the given insight."""
        node = self._nodes.get(insight_id)
        if node is None or not isinstance(node, InsightNode):
            return []
        return list(node.contradicts)

    # -- serialisation --

    def export_lineage(self, insight_id: str) -> dict:
        """Serialise the full lineage of an insight to a JSON-friendly dict."""
        chain = self.trace_lineage(insight_id)
        root = self._nodes.get(insight_id)

        def _ser(n: LineageNode) -> dict:
            d = {"lineage_id": n.lineage_id, "kind": n.kind, "source_ids": n.source_ids}
            d.update(n.metadata)
            return d

        return {
            "insight_id": insight_id,
            "insight": _ser(root) if root else None,
            "chunks": [_ser(n) for n in chain["chunks"]],
            "links": [_ser(n) for n in chain["links"]],
            "clusters": [_ser(n) for n in chain["clusters"]],
            "parent_insights": [_ser(n) for n in chain["insights"]],
        }

    def to_json(self) -> str:
        """Dump entire graph as JSON."""
        nodes = {}
        for nid, node in self._nodes.items():
            d = {"kind": node.kind, "source_ids": node.source_ids,
                 "created_at": node.created_at, "metadata": node.metadata}
            # Add type-specific fields
            if isinstance(node, ChunkNode):
                d.update(doc_id=node.doc_id, doc_title=node.doc_title,
                         chunk_idx=node.chunk_idx, text_hash=node.text_hash,
                         char_start=node.char_start, char_end=node.char_end)
            elif isinstance(node, LinkNode):
                d.update(source_chunk_id=node.source_chunk_id,
                         target_chunk_id=node.target_chunk_id,
                         link_type=node.link_type, strength=node.strength)
            elif isinstance(node, ClusterNode):
                d.update(member_chunk_ids=node.member_chunk_ids,
                         coherence=node.coherence)
            elif isinstance(node, InsightNode):
                d.update(source_chunk_ids=node.source_chunk_ids,
                         source_link_ids=node.source_link_ids,
                         derivation_steps=node.derivation_steps,
                         contradicts=node.contradicts, supports=node.supports,
                         confidence_breakdown=node.confidence_breakdown)
            nodes[nid] = d
        return json.dumps(nodes, indent=2, default=str)

    @property
    def num_nodes(self) -> int:
        return len(self._nodes)

    def nodes_by_kind(self, kind: str) -> list[LineageNode]:
        return [n for n in self._nodes.values() if n.kind == kind]
