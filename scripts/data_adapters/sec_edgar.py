#!/usr/bin/env python
"""Download and convert SEC EDGAR 10-K filings to Latent Ocean format.

Downloads recent 10-K annual reports from S&P 500 companies via the free
SEC EDGAR API, chunks text into paragraphs, generates sentence-transformer
embeddings, builds a causal link graph from entity co-references, and
projects everything onto S² for TCD-JEPA training.

Pipeline:
    SEC EDGAR API → 10-K HTML → text extraction → paragraph chunks →
    sentence-transformer embeddings (384D) → co-reference causal graph →
    spectral embedding onto S² → Latent Ocean format (.pt files)

Usage:
    pip install sentence-transformers beautifulsoup4 lxml
    python scripts/data_adapters/sec_edgar.py --output ./data/sec_edgar
    python train_manifold.py --config configs/manifold_large.yaml --tcd --eval \\
        data.dataset=latent_ocean data.data_dir=./data/sec_edgar

Requirements:
    - sentence-transformers (for all-MiniLM-L6-v2 embeddings)
    - beautifulsoup4 + lxml (for HTML parsing)
    - scipy (for spectral embedding)
    - No API key needed — SEC EDGAR is free

Source: https://data.sec.gov/
"""

import argparse
import json
import logging
import re
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("sec_edgar")

# SEC requires User-Agent header with contact info
USER_AGENT = "TCD-JEPA-Research research@example.com"
SEC_BASE = "https://data.sec.gov"
EFTS_BASE = "https://efts.sec.gov/LATEST/search-index?q=%2210-K%22&dateRange=custom"

# Top companies by market cap — guaranteed to have 10-K filings
SP500_CIKS = {
    "Apple": "0000320193",
    "Microsoft": "0000789019",
    "Amazon": "0001018724",
    "Alphabet": "0001652044",
    "Meta": "0001326801",
    "Tesla": "0001318605",
    "Nvidia": "0001045810",
    "Berkshire Hathaway": "0001067983",
    "JPMorgan Chase": "0000019617",
    "Johnson & Johnson": "0000200406",
    "Visa": "0001403161",
    "Procter & Gamble": "0000080424",
    "UnitedHealth": "0000731766",
    "Exxon Mobil": "0000034088",
    "Walmart": "0000104169",
    "Mastercard": "0001141391",
    "Bank of America": "0000070858",
    "Chevron": "0000093410",
    "Home Depot": "0000354950",
    "Coca-Cola": "0000021344",
    "Pfizer": "0000078003",
    "AbbVie": "0001551152",
    "Costco": "0000909832",
    "Merck": "0000310158",
    "PepsiCo": "0000077476",
    "Broadcom": "0001649338",
    "Eli Lilly": "0000059478",
    "Cisco": "0000858877",
    "Adobe": "0000796343",
    "Salesforce": "0001108524",
    "Netflix": "0001065280",
    "Intel": "0000050863",
    "AMD": "0000002488",
    "Qualcomm": "0000804328",
    "Goldman Sachs": "0000886982",
    "Morgan Stanley": "0000895421",
    "Lockheed Martin": "0000936468",
    "Boeing": "0000012927",
    "Caterpillar": "0000018230",
    "3M": "0000066740",
    "Disney": "0001744489",
    "IBM": "0000051143",
    "Nike": "0000320187",
    "McDonald's": "0000063908",
    "Starbucks": "0000829224",
    "AT&T": "0000732717",
    "Verizon": "0000732712",
    "General Electric": "0000040545",
    "Ford": "0000037996",
    "General Motors": "0001467858",
}


def sec_request(url: str) -> bytes:
    """Make a rate-limited request to SEC EDGAR."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    time.sleep(0.15)  # Stay well under 10 req/sec limit
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read()
    except Exception as e:
        logger.warning(f"Request failed: {url}: {e}")
        return b""


def get_latest_10k_url(cik: str) -> str | None:
    """Find the URL of the most recent 10-K filing for a company."""
    url = f"{SEC_BASE}/submissions/CIK{cik}.json"
    data = sec_request(url)
    if not data:
        return None

    try:
        submissions = json.loads(data)
    except json.JSONDecodeError:
        return None

    recent = submissions.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accessions = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])

    for i, form in enumerate(forms):
        if form == "10-K" and i < len(accessions) and i < len(primary_docs):
            accession = accessions[i].replace("-", "")
            doc = primary_docs[i]
            return f"{SEC_BASE}/Archives/edgar/data/{cik}/{accession}/{doc}"

    return None


def extract_text_from_html(html_bytes: bytes) -> str:
    """Extract clean text from SEC filing HTML."""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html_bytes, "lxml")
        # Remove script and style elements
        for tag in soup(["script", "style", "table"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
    except ImportError:
        # Fallback: regex-based extraction
        text = html_bytes.decode("utf-8", errors="ignore")
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"&[a-z]+;", " ", text)

    # Clean up whitespace
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    text = "\n".join(lines)
    return text


def chunk_text(text: str, chunk_size: int = 256, overlap: int = 64) -> list[str]:
    """Split text into overlapping word chunks."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if len(chunk) > 50:  # Skip very short chunks
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def download_and_process_filings(
    companies: dict[str, str],
    max_chunks_per_company: int = 100,
) -> tuple[list[dict], list[str]]:
    """Download 10-K filings and extract text chunks."""
    all_chunks = []
    all_metadata = []

    for company_name, cik in companies.items():
        logger.info(f"  {company_name} (CIK {cik})...")

        url = get_latest_10k_url(cik)
        if url is None:
            logger.warning(f"    No 10-K found for {company_name}")
            continue

        html = sec_request(url)
        if not html:
            continue

        text = extract_text_from_html(html)
        if len(text) < 1000:
            logger.warning(f"    Too little text for {company_name}: {len(text)} chars")
            continue

        chunks = chunk_text(text)[:max_chunks_per_company]
        logger.info(f"    {len(chunks)} chunks from {len(text):,} chars")

        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadata.append({
                "company": company_name,
                "cik": cik,
                "chunk_idx": i,
                "filing_url": url,
            })

    return all_metadata, all_chunks


def build_embeddings(chunks: list[str], dim: int = 384) -> np.ndarray:
    """Generate sentence-transformer embeddings for all chunks."""
    try:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Encoding {len(chunks)} chunks with all-MiniLM-L6-v2...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(chunks, show_progress_bar=True, batch_size=64)
        return embeddings.astype(np.float32)
    except ImportError:
        logger.warning("sentence-transformers not installed — using TF-IDF fallback")
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.random_projection import GaussianRandomProjection

            vectorizer = TfidfVectorizer(max_features=2000, stop_words="english")
            tfidf = vectorizer.fit_transform(chunks).toarray()

            if tfidf.shape[1] < dim:
                padded = np.zeros((len(chunks), dim), dtype=np.float32)
                padded[:, :tfidf.shape[1]] = tfidf
                return padded
            else:
                proj = GaussianRandomProjection(n_components=dim, random_state=42)
                return proj.fit_transform(tfidf).astype(np.float32)
        except ImportError:
            logger.warning("sklearn not available — using hash embeddings")
            embeddings = np.zeros((len(chunks), dim), dtype=np.float32)
            for i, chunk in enumerate(chunks):
                words = chunk.lower().split()
                for w in words:
                    h = hash(w) % dim
                    embeddings[i, h] += 1.0
                embeddings[i] /= max(len(words), 1)
            return embeddings


def build_causal_graph(
    metadata: list[dict],
    embeddings: np.ndarray,
    similarity_threshold: float = 0.5,
) -> np.ndarray:
    """Build directed causal link graph from entity co-references and similarity."""
    n = len(metadata)
    adjacency = np.zeros((n, n), dtype=np.float32)

    # 1. Same-company sequential links (paragraph flow)
    for i in range(n - 1):
        if (metadata[i]["company"] == metadata[i + 1]["company"]
                and metadata[i]["chunk_idx"] + 1 == metadata[i + 1]["chunk_idx"]):
            adjacency[i, i + 1] = 0.9  # Strong sequential link

    # 2. Cross-company links via embedding similarity
    logger.info("Computing cross-company similarity links...")
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    emb_norm = embeddings / np.clip(norms, 1e-8, None)

    # Chunked similarity computation (avoid N×N memory for large N)
    chunk_size = 500
    for i_start in range(0, n, chunk_size):
        i_end = min(i_start + chunk_size, n)
        sim_block = emb_norm[i_start:i_end] @ emb_norm.T  # [chunk, N]

        for local_i in range(i_end - i_start):
            global_i = i_start + local_i
            for j in range(n):
                if global_i == j:
                    continue
                if metadata[global_i]["company"] == metadata[j]["company"]:
                    continue  # Skip same-company (already handled)
                if sim_block[local_i, j] > similarity_threshold:
                    # Directed: from the earlier-indexed company to later
                    adjacency[global_i, j] = float(sim_block[local_i, j])

    num_edges = int((adjacency > 0).sum())
    logger.info(f"Causal graph: {num_edges} edges ({num_edges / max(n * n, 1) * 100:.2f}% density)")
    return adjacency


def build_coordinates(adjacency: np.ndarray, n: int, sphere_radius: float = 4.5) -> np.ndarray:
    """Project entities onto S² via spectral embedding."""
    from scipy.linalg import eigh

    sym = adjacency + adjacency.T
    np.fill_diagonal(sym, 0)
    degree = sym.sum(axis=1)
    d_inv_sqrt = np.where(degree > 0, 1.0 / np.sqrt(degree), 0)
    D = np.diag(d_inv_sqrt)
    L = np.eye(n) - D @ sym @ D

    try:
        eigenvalues, eigenvectors = eigh(L, subset_by_index=[1, 2])
        embed_2d = eigenvectors
    except Exception:
        rng = np.random.RandomState(42)
        embed_2d = rng.randn(n, 2)

    norms = np.linalg.norm(embed_2d, axis=1, keepdims=True)
    embed_2d = embed_2d / np.clip(norms, 1e-8, None)

    rng = np.random.RandomState(42)
    theta = np.arccos(np.clip(embed_2d[:, 0], -1, 1)) + rng.normal(0, 0.03, n)
    phi = np.arctan2(embed_2d[:, 1], embed_2d[:, 0]) + np.pi + rng.normal(0, 0.03, n)
    theta = np.clip(theta, 0.01, np.pi - 0.01)

    x = sphere_radius * np.sin(theta) * np.cos(phi)
    y = sphere_radius * np.sin(theta) * np.sin(phi)
    z = sphere_radius * np.cos(theta)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def build_labels(metadata: list[dict]) -> np.ndarray:
    """Assign cluster labels by company."""
    companies = sorted(set(m["company"] for m in metadata))
    company_to_idx = {c: i for i, c in enumerate(companies)}
    labels = np.array([company_to_idx[m["company"]] for m in metadata], dtype=np.int64)
    logger.info(f"Labels: {len(companies)} companies")
    return labels


def main():
    parser = argparse.ArgumentParser(description="Convert SEC EDGAR 10-K filings to Latent Ocean format")
    parser.add_argument("--output", default="./data/sec_edgar")
    parser.add_argument("--max-companies", type=int, default=50,
                        help="Number of companies to download (default: 50)")
    parser.add_argument("--max-chunks-per-company", type=int, default=100,
                        help="Max text chunks per company (default: 100)")
    parser.add_argument("--sphere-radius", type=float, default=4.5)
    parser.add_argument("--similarity-threshold", type=float, default=0.5)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select companies
    companies = dict(list(SP500_CIKS.items())[:args.max_companies])
    logger.info(f"Downloading 10-K filings for {len(companies)} companies...")

    # Download and process
    metadata, chunks = download_and_process_filings(companies, args.max_chunks_per_company)
    n = len(chunks)
    logger.info(f"Total: {n} text chunks from {len(set(m['company'] for m in metadata))} companies")

    if n < 10:
        logger.error("Too few chunks extracted. Check network connectivity to data.sec.gov")
        return

    # Build embeddings
    embeddings = build_embeddings(chunks)
    logger.info(f"Embeddings: {embeddings.shape}")

    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.clip(norms, 1e-8, None)

    # Build causal graph
    adjacency = build_causal_graph(metadata, embeddings, args.similarity_threshold)

    # Build S² coordinates
    coords = build_coordinates(adjacency, n, args.sphere_radius)
    logger.info(f"Coordinates: {coords.shape}")

    # Velocities (small random tangent vectors)
    rng = np.random.RandomState(42)
    velocity = rng.randn(n, 3).astype(np.float32) * 0.02
    for i in range(n):
        normal = coords[i] / (np.linalg.norm(coords[i]) + 1e-8)
        velocity[i] -= np.dot(velocity[i], normal) * normal

    # Labels
    labels = build_labels(metadata)

    # Save
    torch.save(torch.from_numpy(embeddings), output_dir / "fingerprints.pt")
    torch.save(torch.from_numpy(coords), output_dir / "coords.pt")
    torch.save(torch.from_numpy(velocity), output_dir / "velocity.pt")
    torch.save(torch.from_numpy(adjacency), output_dir / "adjacency.pt")
    torch.save(torch.from_numpy(labels), output_dir / "labels.pt")

    # Save entity metadata
    with open(output_dir / "entities.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Save company list
    company_names = sorted(set(m["company"] for m in metadata))
    with open(output_dir / "companies.json", "w") as f:
        json.dump(company_names, f, indent=2)

    logger.info(f"\nSaved to {output_dir}:")
    logger.info(f"  fingerprints.pt: [{n}, {embeddings.shape[1]}]")
    logger.info(f"  coords.pt:      [{n}, 3]")
    logger.info(f"  adjacency.pt:   [{n}, {n}] ({int((adjacency > 0).sum())} edges)")
    logger.info(f"  labels.pt:      [{n}] ({len(company_names)} companies)")
    logger.info(f"  entities.json:  {n} chunk records")
    logger.info(f"  companies.json: {len(company_names)} companies")
    logger.info("\nReady for:")
    logger.info("  python train_manifold.py --config configs/manifold.yaml --tcd --eval \\")
    logger.info(f"      data.dataset=latent_ocean data.data_dir={output_dir}")


if __name__ == "__main__":
    main()
