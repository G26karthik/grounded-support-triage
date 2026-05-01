"""One-shot index builder. Hashes corpus content, embeds chunks, persists shards.

Usage:
    python code/main.py rebuild-index

Cache hit logic: each per-company manifest stores
  (corpus_hash, chunker_version, embedder_version)
We skip rebuild if all three match. Bump CHUNKER_VERSION or EMBEDDER_VERSION
in hro/config.py when their respective implementations change.
"""

from __future__ import annotations

import hashlib
import json
import time

import numpy as np

from hro.config import (
    CHUNKER_VERSION,
    COMPANIES,
    DATA_DIR,
    EMBEDDER_VERSION,
    EMBEDDING_MODEL,
    INDEX_DIR,
)
from hro.corpus.chunker import chunk_documents
from hro.corpus.loader import load_company
from hro.index.device import best_device
from hro.index.store import manifest_path, save_shard


def rebuild(force: bool = False) -> None:
    """Rebuild every per-company index. Skip on cache hit unless force=True."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    embedder = None  # lazily loaded — sentence-transformers is heavy

    for company in COMPANIES:
        ch = corpus_hash(company)
        m_path = manifest_path(company)

        if not force and m_path.exists():
            try:
                m = json.loads(m_path.read_text(encoding="utf-8"))
                if (
                    m.get("corpus_hash") == ch
                    and m.get("chunker_version") == CHUNKER_VERSION
                    and m.get("embedder_version") == EMBEDDER_VERSION
                ):
                    print(f"[{company}] cache hit ({m.get('n_chunks')} chunks); skipping")
                    continue
            except (json.JSONDecodeError, OSError):
                pass

        t0 = time.time()
        docs = load_company(company)
        chunks = chunk_documents(docs)
        if not chunks:
            print(f"[{company}] no chunks; skipping")
            continue

        if embedder is None:
            device = best_device()
            print(f"loading embedding model: {EMBEDDING_MODEL} on {device}")
            from sentence_transformers import SentenceTransformer  # local import — slow

            embedder = SentenceTransformer(EMBEDDING_MODEL, device=device)

        print(f"[{company}] embedding {len(chunks)} chunks from {len(docs)} docs...")
        texts = [c.text for c in chunks]
        embeddings = embedder.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        manifest = {
            "company": company,
            "corpus_hash": ch,
            "chunker_version": CHUNKER_VERSION,
            "embedder_version": EMBEDDER_VERSION,
            "embedding_model": EMBEDDING_MODEL,
            "n_chunks": len(chunks),
            "n_docs": len(docs),
            "dim": int(embeddings.shape[1]),
        }
        save_shard(company, embeddings, chunks, manifest)
        print(f"[{company}] {len(chunks)} chunks in {time.time() - t0:.1f}s")


def corpus_hash(company: str) -> str:
    """sha256 of concatenated file contents for one company's corpus.

    Uses file contents — not mtime — because mtime is unstable across git
    operations and CI checkouts.
    """
    company_dir = DATA_DIR / company
    if not company_dir.exists():
        return ""
    h = hashlib.sha256()
    for path in sorted(company_dir.rglob("*.md")):
        # Skip the per-company aggregator (the table-of-contents file).
        if path.parent == company_dir and path.name == "index.md":
            continue
        h.update(path.relative_to(DATA_DIR).as_posix().encode("utf-8"))
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    return h.hexdigest()
