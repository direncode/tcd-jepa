"""Document processor: raw NL text → manifold entities for TCD-JEPA.

Pipeline:
1. Document chunking — split documents into semantic units (paragraphs, sections)
2. Embedding — sentence-transformer (384D) or fallback hash-based embedding
3. Manifold placement — project embeddings onto S² via PCA + spherical normalization
4. Causal link discovery — extract relationships between chunks via:
   - Structural links (same document, sequential sections)
   - Semantic links (embedding cosine similarity above threshold)
   - Reference links (shared entities, citations, URLs)
   - Co-occurrence links (entities appearing in same context window)
5. Velocity estimation — temporal signal from document ordering/timestamps

Output: manifold-ready tensors (fingerprints, coords, velocity, adjacency, metadata)
compatible with CausalManifoldDataset and the full TCD-JEPA training pipeline.
"""

import hashlib
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger("tcd_jepa.document_processor")


@dataclass
class DocumentChunk:
    """A semantic unit extracted from a document."""
    text: str
    doc_id: str
    chunk_idx: int
    doc_title: str = ""
    section: str = ""
    timestamp: Optional[float] = None
    entities: list[str] = field(default_factory=list)
    urls: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class ProcessedCorpus:
    """Output of document processing, ready for manifold JEPA."""
    fingerprints: torch.Tensor    # [N, 384] embeddings
    coords: torch.Tensor          # [N, 3] S² positions
    velocity: torch.Tensor        # [N, 3] temporal signal
    adjacency: torch.Tensor       # [N, N] causal link weights
    chunks: list[DocumentChunk]   # Original text chunks (for insight generation)
    entity_labels: torch.Tensor   # [N] cluster assignments
    doc_ids: list[str]            # Document ID per chunk
    chunk_texts: list[str]        # Raw text per chunk


class DocumentChunker:
    """Splits documents into semantic chunks for embedding."""

    def __init__(
        self,
        chunk_size: int = 256,
        chunk_overlap: int = 64,
        min_chunk_length: int = 20,
        respect_paragraphs: bool = True,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_length = min_chunk_length
        self.respect_paragraphs = respect_paragraphs

    def chunk_document(
        self,
        text: str,
        doc_id: str = "",
        doc_title: str = "",
        timestamp: Optional[float] = None,
    ) -> list[DocumentChunk]:
        """Split a document into semantic chunks."""
        if not text.strip():
            return []

        if self.respect_paragraphs:
            raw_chunks = self._paragraph_split(text)
        else:
            raw_chunks = self._sliding_window(text)

        chunks = []
        for i, chunk_text in enumerate(raw_chunks):
            chunk_text = chunk_text.strip()
            if len(chunk_text) < self.min_chunk_length:
                continue

            entities = self._extract_entities(chunk_text)
            urls = self._extract_urls(chunk_text)
            section = self._detect_section(chunk_text, text)

            chunks.append(DocumentChunk(
                text=chunk_text,
                doc_id=doc_id or f"doc_{hash(text[:100]) % 10000:04d}",
                chunk_idx=i,
                doc_title=doc_title,
                section=section,
                timestamp=timestamp,
                entities=entities,
                urls=urls,
            ))

        return chunks

    def _paragraph_split(self, text: str) -> list[str]:
        """Split on paragraph boundaries, merging short paragraphs."""
        paragraphs = re.split(r'\n\s*\n', text)
        merged = []
        current = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if len(current) + len(para) <= self.chunk_size:
                current = f"{current}\n\n{para}" if current else para
            else:
                if current:
                    merged.append(current)
                if len(para) > self.chunk_size:
                    # Split long paragraphs with sliding window
                    merged.extend(self._sliding_window(para))
                else:
                    current = para
                    continue
                current = ""
        if current:
            merged.append(current)

        return merged

    def _sliding_window(self, text: str) -> list[str]:
        """Split into overlapping windows by word count."""
        words = text.split()
        if len(words) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            if end >= len(words):
                break
            start += self.chunk_size - self.chunk_overlap

        return chunks

    def _extract_entities(self, text: str) -> list[str]:
        """Extract named entities via capitalization patterns."""
        # Proper nouns: sequences of capitalized words (excluding sentence starts)
        entities = set()
        sentences = re.split(r'[.!?]\s+', text)
        for sent in sentences:
            words = sent.split()
            if len(words) < 2:
                continue
            i = 1  # Skip first word (sentence start)
            while i < len(words):
                if words[i][0:1].isupper() and words[i].isalpha():
                    entity = [words[i]]
                    j = i + 1
                    while j < len(words) and words[j][0:1].isupper() and words[j].isalpha():
                        entity.append(words[j])
                        j += 1
                    name = " ".join(entity)
                    if len(name) > 2:
                        entities.add(name)
                    i = j
                else:
                    i += 1
        return list(entities)

    def _extract_urls(self, text: str) -> list[str]:
        """Extract URLs from text."""
        return re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', text)

    def _detect_section(self, chunk: str, full_text: str) -> str:
        """Detect which section a chunk belongs to."""
        pos = full_text.find(chunk[:50])
        if pos < 0:
            return ""
        before = full_text[:pos]
        # Find last heading-like line
        lines = before.split('\n')
        for line in reversed(lines):
            line = line.strip()
            if line and (line.startswith('#') or (line.isupper() and len(line) < 100) or
                         re.match(r'^\d+\.\s+\w', line)):
                return line.lstrip('#').strip()
        return ""


class DocumentEmbedder:
    """Embeds text chunks into 384D vectors."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        fingerprint_dim: int = 384,
        device: str = "cpu",
        batch_size: int = 64,
    ):
        self.model_name = model_name
        self.fingerprint_dim = fingerprint_dim
        self.device = device
        self.batch_size = batch_size
        self._model = None

    def _load_model(self):
        """Lazy load sentence transformer."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
            actual_dim = self._model.get_sentence_embedding_dimension()
            if actual_dim != self.fingerprint_dim:
                logger.warning(
                    f"Model {self.model_name} produces {actual_dim}D, expected {self.fingerprint_dim}D. "
                    f"Will pad/truncate."
                )
            logger.info(f"Loaded sentence-transformer: {self.model_name} ({actual_dim}D)")
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. Using deterministic hash embeddings. "
                "Install with: pip install sentence-transformers"
            )
            self._model = "hash_fallback"

    def embed(self, texts: list[str]) -> torch.Tensor:
        """Embed texts into fingerprint vectors [N, 384]."""
        self._load_model()

        if self._model == "hash_fallback":
            return self._hash_embed(texts)

        embeddings = self._model.encode(
            texts, batch_size=self.batch_size, show_progress_bar=len(texts) > 100,
            convert_to_numpy=True,
        )
        result = torch.from_numpy(embeddings).float()

        # Pad/truncate to fingerprint_dim
        if result.shape[1] < self.fingerprint_dim:
            pad = torch.zeros(result.shape[0], self.fingerprint_dim - result.shape[1])
            result = torch.cat([result, pad], dim=1)
        elif result.shape[1] > self.fingerprint_dim:
            result = result[:, :self.fingerprint_dim]

        return result

    def _hash_embed(self, texts: list[str]) -> torch.Tensor:
        """Deterministic hash-based embedding fallback.

        Produces consistent 384D vectors from text content using SHA-256.
        Not semantically meaningful but preserves identity and allows pipeline testing.
        """
        embeddings = torch.zeros(len(texts), self.fingerprint_dim)
        for i, text in enumerate(texts):
            # Generate deterministic pseudo-random vector from text hash
            h = hashlib.sha256(text.encode('utf-8')).digest()
            rng = np.random.RandomState(int.from_bytes(h[:4], 'big'))

            # Word-level features for basic semantic signal
            words = text.lower().split()
            word_hashes = []
            for w in words[:50]:  # Cap at 50 words
                wh = hashlib.md5(w.encode()).digest()
                word_hashes.append(int.from_bytes(wh[:4], 'big'))

            # Build embedding from word hash mixture
            vec = np.zeros(self.fingerprint_dim)
            for j, wh in enumerate(word_hashes):
                wrng = np.random.RandomState(wh)
                word_vec = wrng.normal(0, 1, self.fingerprint_dim)
                weight = 1.0 / (1 + j * 0.1)  # Decay for later words
                vec += weight * word_vec
            vec /= (np.linalg.norm(vec) + 1e-8)

            # Add document-level hash for uniqueness
            doc_vec = rng.normal(0, 0.1, self.fingerprint_dim)
            vec = vec * 0.8 + doc_vec * 0.2
            vec /= (np.linalg.norm(vec) + 1e-8)

            embeddings[i] = torch.from_numpy(vec).float()

        return embeddings


class ManifoldProjector:
    """Projects high-dimensional embeddings onto S² manifold."""

    def __init__(self, sphere_radius: float = 4.5, method: str = "pca"):
        self.sphere_radius = sphere_radius
        self.method = method

    def project(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Map [N, D] embeddings to [N, 3] S² coordinates.

        Uses PCA to extract 3 principal directions, then normalizes to S².
        """
        N, D = embeddings.shape

        if N < 3:
            # Not enough points for PCA
            coords = torch.randn(N, 3)
            coords = coords / (coords.norm(dim=-1, keepdim=True) + 1e-8) * self.sphere_radius
            return coords

        # Center
        centered = embeddings - embeddings.mean(0)

        if self.method == "pca":
            # SVD for PCA
            U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
            # Project to 3D using top-3 components
            coords_3d = U[:, :3] * S[:3].unsqueeze(0)
        else:
            # Random projection fallback
            proj_matrix = torch.randn(D, 3) / math.sqrt(D)
            coords_3d = centered @ proj_matrix

        # Normalize to sphere surface
        coords_3d = coords_3d / (coords_3d.norm(dim=-1, keepdim=True) + 1e-8) * self.sphere_radius

        return coords_3d


class CausalLinkDiscoverer:
    """Discovers causal/relational links between document chunks."""

    def __init__(
        self,
        semantic_threshold: float = 0.65,
        structural_weight: float = 0.8,
        semantic_weight: float = 0.7,
        reference_weight: float = 0.9,
        cooccurrence_weight: float = 0.6,
    ):
        self.semantic_threshold = semantic_threshold
        self.structural_weight = structural_weight
        self.semantic_weight = semantic_weight
        self.reference_weight = reference_weight
        self.cooccurrence_weight = cooccurrence_weight

    def discover_links(
        self,
        chunks: list[DocumentChunk],
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Build weighted directed adjacency matrix from document chunks.

        Link types (matching Latent Ocean):
        - Structural (weight ~0.8): sequential chunks in same document
        - Semantic (weight ~0.7): cosine similarity above threshold
        - Reference (weight ~0.9): shared entities or URLs
        - Co-occurrence (weight ~0.6): shared entities across documents
        """
        N = len(chunks)
        adjacency = torch.zeros(N, N)

        # 1. Structural links (same document, sequential)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                if chunks[i].doc_id == chunks[j].doc_id:
                    gap = abs(chunks[i].chunk_idx - chunks[j].chunk_idx)
                    if gap == 1:
                        # Direct sequence: strong directed link (i→j if j follows i)
                        if chunks[j].chunk_idx > chunks[i].chunk_idx:
                            adjacency[i, j] = max(adjacency[i, j], self.structural_weight)
                    elif gap <= 3:
                        # Same section proximity
                        weight = self.structural_weight * 0.5 / gap
                        adjacency[i, j] = max(adjacency[i, j], weight)

        # 2. Semantic links (cosine similarity)
        emb_norm = embeddings / (embeddings.norm(dim=-1, keepdim=True) + 1e-8)
        sim = emb_norm @ emb_norm.t()
        semantic_mask = (sim > self.semantic_threshold) & (torch.eye(N) == 0)
        semantic_links = sim * semantic_mask.float() * self.semantic_weight
        adjacency = torch.maximum(adjacency, semantic_links)

        # 3. Reference links (shared entities)
        for i in range(N):
            entities_i = set(chunks[i].entities)
            urls_i = set(chunks[i].urls)
            if not entities_i and not urls_i:
                continue
            for j in range(i + 1, N):
                entities_j = set(chunks[j].entities)
                urls_j = set(chunks[j].urls)

                # Shared entities
                shared_entities = entities_i & entities_j
                if shared_entities:
                    strength = min(1.0, len(shared_entities) * 0.3) * self.reference_weight
                    adjacency[i, j] = max(adjacency[i, j], strength)
                    adjacency[j, i] = max(adjacency[j, i], strength * 0.8)

                # Shared URLs
                shared_urls = urls_i & urls_j
                if shared_urls:
                    strength = min(1.0, len(shared_urls) * 0.4) * self.reference_weight
                    adjacency[i, j] = max(adjacency[i, j], strength)
                    adjacency[j, i] = max(adjacency[j, i], strength)

        # 4. Co-occurrence links (cross-document entity overlap)
        for i in range(N):
            for j in range(N):
                if i == j or chunks[i].doc_id == chunks[j].doc_id:
                    continue
                shared = set(chunks[i].entities) & set(chunks[j].entities)
                if shared:
                    strength = min(1.0, len(shared) * 0.2) * self.cooccurrence_weight
                    adjacency[i, j] = max(adjacency[i, j], strength)

        return adjacency


class VelocityEstimator:
    """Estimates tangent-space velocities from temporal/ordering information."""

    def __init__(self, sphere_radius: float = 4.5):
        self.sphere_radius = sphere_radius

    def estimate(
        self,
        chunks: list[DocumentChunk],
        coords: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate velocity vectors tangent to S².

        Uses:
        - Document ordering (later chunks drift from earlier ones)
        - Adjacency structure (entities drift toward strongly linked neighbors)
        - Timestamps if available
        """
        N = coords.shape[0]
        velocity = torch.zeros(N, 3)

        for i in range(N):
            # Weighted drift toward strongly connected neighbors
            neighbors = adjacency[i]
            if neighbors.sum() > 0:
                weights = neighbors / (neighbors.sum() + 1e-8)
                target = (weights.unsqueeze(-1) * coords).sum(dim=0)
                drift = target - coords[i]
            else:
                drift = torch.zeros(3)

            # Add temporal ordering signal
            if chunks[i].timestamp is not None:
                # Scale drift by recency
                drift *= 0.1
            else:
                # Use chunk ordering as temporal proxy
                progress = chunks[i].chunk_idx / max(len(chunks), 1)
                drift *= 0.05 * (1 + progress)

            # Project to tangent space (remove radial component)
            normal = coords[i] / (coords[i].norm() + 1e-8)
            drift = drift - (drift @ normal) * normal

            velocity[i] = drift

        return velocity


class DocumentProcessor:
    """Full pipeline: raw documents → manifold-ready tensors.

    This is the main entry point for converting a collection of NL documents
    into the tensor format expected by CausalManifoldDataset and the TCD-JEPA
    training pipeline.
    """

    def __init__(
        self,
        fingerprint_dim: int = 384,
        sphere_radius: float = 4.5,
        chunk_size: int = 256,
        chunk_overlap: int = 64,
        semantic_threshold: float = 0.65,
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_device: str = "cpu",
        min_chunks: int = 16,
    ):
        self.chunker = DocumentChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.embedder = DocumentEmbedder(
            model_name=embedding_model,
            fingerprint_dim=fingerprint_dim,
            device=embedding_device,
        )
        self.projector = ManifoldProjector(sphere_radius=sphere_radius)
        self.link_discoverer = CausalLinkDiscoverer(
            semantic_threshold=semantic_threshold,
        )
        self.velocity_estimator = VelocityEstimator(sphere_radius=sphere_radius)
        self.min_chunks = min_chunks
        self.sphere_radius = sphere_radius

    def process_documents(
        self,
        documents: list[dict],
    ) -> ProcessedCorpus:
        """Process a collection of documents into manifold tensors.

        Args:
            documents: List of dicts with keys:
                - 'text': document content (required)
                - 'id': document identifier (optional)
                - 'title': document title (optional)
                - 'timestamp': numeric timestamp (optional)
                - 'metadata': arbitrary metadata dict (optional)

        Returns:
            ProcessedCorpus with all tensors and metadata.
        """
        # 1. Chunk all documents
        all_chunks = []
        for doc in documents:
            text = doc.get("text", "")
            doc_id = doc.get("id", f"doc_{len(all_chunks)}")
            title = doc.get("title", "")
            timestamp = doc.get("timestamp")

            chunks = self.chunker.chunk_document(
                text, doc_id=doc_id, doc_title=title, timestamp=timestamp,
            )
            all_chunks.extend(chunks)

        if not all_chunks:
            raise ValueError("No chunks produced from documents. Check document content.")

        logger.info(f"Chunked {len(documents)} documents into {len(all_chunks)} chunks")

        # 2. Embed chunks
        texts = [c.text for c in all_chunks]
        fingerprints = self.embedder.embed(texts)
        logger.info(f"Embedded {len(texts)} chunks → [{fingerprints.shape[0]}, {fingerprints.shape[1]}]")

        # 3. Project to S²
        coords = self.projector.project(fingerprints)
        logger.info(f"Projected to S² at radius {self.sphere_radius}")

        # 4. Discover causal links
        adjacency = self.link_discoverer.discover_links(all_chunks, fingerprints)
        link_count = (adjacency > 0).sum().item()
        logger.info(f"Discovered {link_count} causal links ({link_count / max(len(all_chunks)**2, 1) * 100:.1f}% density)")

        # 5. Estimate velocities
        velocity = self.velocity_estimator.estimate(all_chunks, coords, adjacency)

        # 6. Cluster assignment (for evaluation)
        entity_labels = self._cluster_chunks(fingerprints, all_chunks)

        return ProcessedCorpus(
            fingerprints=fingerprints,
            coords=coords,
            velocity=velocity,
            adjacency=adjacency,
            chunks=all_chunks,
            entity_labels=entity_labels,
            doc_ids=[c.doc_id for c in all_chunks],
            chunk_texts=texts,
        )

    def process_text(self, text: str, doc_id: str = "input") -> ProcessedCorpus:
        """Convenience: process a single text document."""
        return self.process_documents([{"text": text, "id": doc_id}])

    def process_files(self, file_paths: list[str]) -> ProcessedCorpus:
        """Process document files (txt, md, etc.)."""
        documents = []
        for path_str in file_paths:
            path = Path(path_str)
            if not path.exists():
                logger.warning(f"File not found: {path}")
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            documents.append({
                "text": text,
                "id": path.stem,
                "title": path.name,
            })
        return self.process_documents(documents)

    def process_directory(self, dir_path: str, extensions: tuple = (".txt", ".md", ".rst")) -> ProcessedCorpus:
        """Process all matching files in a directory."""
        path = Path(dir_path)
        files = []
        for ext in extensions:
            files.extend(sorted(path.glob(f"**/*{ext}")))
        return self.process_files([str(f) for f in files])

    def _cluster_chunks(
        self, embeddings: torch.Tensor, chunks: list[DocumentChunk],
    ) -> torch.Tensor:
        """Assign cluster labels to chunks.

        Uses document ID as primary grouping, with spectral refinement.
        """
        N = len(chunks)

        # Group by document
        doc_ids = list(set(c.doc_id for c in chunks))
        doc_to_label = {doc_id: i for i, doc_id in enumerate(doc_ids)}
        labels = torch.tensor([doc_to_label[c.doc_id] for c in chunks], dtype=torch.long)

        # If too many docs, re-cluster via k-means on embeddings
        num_unique = len(doc_ids)
        if num_unique > 20:
            labels = self._kmeans_cluster(embeddings, k=min(num_unique, 20))

        return labels

    def _kmeans_cluster(self, embeddings: torch.Tensor, k: int = 10, max_iter: int = 50) -> torch.Tensor:
        """Simple k-means clustering on embeddings."""
        N, D = embeddings.shape
        perm = torch.randperm(N)[:k]
        centroids = embeddings[perm].clone()

        for _ in range(max_iter):
            dists = torch.cdist(embeddings, centroids)
            labels = dists.argmin(dim=1)
            new_centroids = torch.zeros_like(centroids)
            for c in range(k):
                mask = labels == c
                if mask.sum() > 0:
                    new_centroids[c] = embeddings[mask].mean(0)
                else:
                    new_centroids[c] = centroids[c]
            if (new_centroids - centroids).norm() < 1e-6:
                break
            centroids = new_centroids

        return labels

    def save(self, corpus: ProcessedCorpus, output_dir: str) -> None:
        """Save processed corpus to disk for later use."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        torch.save(corpus.fingerprints, out / "fingerprints.pt")
        torch.save(corpus.coords, out / "coords.pt")
        torch.save(corpus.velocity, out / "velocity.pt")
        torch.save(corpus.adjacency, out / "adjacency.pt")
        torch.save(corpus.entity_labels, out / "labels.pt")

        # Save text metadata
        import json
        meta = []
        for chunk in corpus.chunks:
            meta.append({
                "text": chunk.text,
                "doc_id": chunk.doc_id,
                "chunk_idx": chunk.chunk_idx,
                "doc_title": chunk.doc_title,
                "section": chunk.section,
                "entities": chunk.entities,
                "urls": chunk.urls,
            })
        with open(out / "chunk_metadata.json", "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved processed corpus ({len(corpus.chunks)} chunks) to {out}")

    def load(self, input_dir: str) -> ProcessedCorpus:
        """Load a previously saved corpus."""
        import json
        inp = Path(input_dir)

        fingerprints = torch.load(inp / "fingerprints.pt", weights_only=True)
        coords = torch.load(inp / "coords.pt", weights_only=True)
        velocity = torch.load(inp / "velocity.pt", weights_only=True)
        adjacency = torch.load(inp / "adjacency.pt", weights_only=True)
        entity_labels = torch.load(inp / "labels.pt", weights_only=True)

        with open(inp / "chunk_metadata.json") as f:
            meta = json.load(f)

        chunks = []
        for m in meta:
            chunks.append(DocumentChunk(
                text=m["text"],
                doc_id=m["doc_id"],
                chunk_idx=m["chunk_idx"],
                doc_title=m.get("doc_title", ""),
                section=m.get("section", ""),
                entities=m.get("entities", []),
                urls=m.get("urls", []),
            ))

        return ProcessedCorpus(
            fingerprints=fingerprints,
            coords=coords,
            velocity=velocity,
            adjacency=adjacency,
            chunks=chunks,
            entity_labels=entity_labels,
            doc_ids=[c.doc_id for c in chunks],
            chunk_texts=[c.text for c in chunks],
        )
