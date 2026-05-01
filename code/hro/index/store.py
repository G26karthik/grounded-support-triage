"""Per-company numpy vector shards + chunk metadata.

Layout under data/index/<company>/:
  embeddings.npy   float32, shape (n_chunks, dim), L2-normalized
  chunks.jsonl     one chunk metadata + text per line, ordered to match
  manifest.json    {corpus_hash, chunker_version, embedder_version, dim, n_chunks}

We deliberately avoid pickle for chunk persistence: jsonl is human-inspectable,
diff-friendly, and avoids the security/compatibility surface of pickle.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from hro.config import INDEX_DIR
from hro.corpus.chunker import Chunk


@dataclass
class CompanyShard:
    """One company's loaded index."""

    company: str
    embeddings: np.ndarray  # shape (n, dim), float32, L2-normalized
    chunks: list[Chunk]
    manifest: dict


def shard_dir(company: str) -> Path:
    return INDEX_DIR / company


def manifest_path(company: str) -> Path:
    return shard_dir(company) / "manifest.json"


def save_shard(
    company: str,
    embeddings: np.ndarray,
    chunks: list[Chunk],
    manifest: dict,
) -> None:
    """Persist a company's shard to data/index/<company>/."""
    d = shard_dir(company)
    d.mkdir(parents=True, exist_ok=True)

    np.save(d / "embeddings.npy", embeddings.astype(np.float32, copy=False))

    with (d / "chunks.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for c in chunks:
            obj = asdict(c)
            # tuples don't survive json round-trip; store as list.
            obj["breadcrumbs"] = list(c.breadcrumbs)
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    (d / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_shard(company: str) -> CompanyShard:
    """Load a company's shard from disk."""
    d = shard_dir(company)
    embeddings = np.load(d / "embeddings.npy")

    chunks: list[Chunk] = []
    with (d / "chunks.jsonl").open(encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            chunks.append(
                Chunk(
                    chunk_id=obj["chunk_id"],
                    company=obj["company"],
                    path=obj["path"],
                    title=obj["title"],
                    breadcrumbs=tuple(obj["breadcrumbs"]),
                    text=obj["text"],
                    body_token_count=obj["body_token_count"],
                )
            )

    manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    return CompanyShard(
        company=company,
        embeddings=embeddings,
        chunks=chunks,
        manifest=manifest,
    )
