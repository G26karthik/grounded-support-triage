"""Hybrid retrieval (BM25 + dense, z-score fused) with conditional rerank.

Lazy-loads per-company shards. The embedder and reranker are process-singletons
to avoid reloading their weights for every query.

Conditional rerank: if the fused-score gap between rank-1 and rank-4 exceeds
config.RERANK_MARGIN, the top results are unambiguous and we skip the
cross-encoder. Otherwise rerank the top-HYBRID_TOP_K candidates and use the
reranker's score as the final ordering.
"""

from __future__ import annotations

import threading

import numpy as np

from hro.config import (
    BM25_WEIGHT,
    DENSE_WEIGHT,
    EMBEDDING_MODEL,
    HYBRID_TOP_K,
    RERANK_MARGIN,
    RERANK_MODEL,
    RERANK_OUTPUT_K,
)
from hro.index.bm25 import BM25Index
from hro.index.device import best_device
from hro.index.store import CompanyShard, load_shard
from hro.schemas import RetrievedChunk

# ---------------------------------------------------------------------------
# Lazy module-level caches.
# ---------------------------------------------------------------------------

# RLock because _get_bm25 calls _get_shard while holding the lock.
_lock = threading.RLock()
_shards: dict[str, CompanyShard] = {}
_bm25s: dict[str, BM25Index] = {}
_embedder = None
_reranker = None


def _get_embedder():
    global _embedder
    with _lock:
        if _embedder is None:
            from sentence_transformers import SentenceTransformer

            _embedder = SentenceTransformer(EMBEDDING_MODEL, device=best_device())
    return _embedder


def _get_reranker():
    global _reranker
    with _lock:
        if _reranker is None:
            from sentence_transformers import CrossEncoder

            _reranker = CrossEncoder(RERANK_MODEL, device=best_device())
    return _reranker


def _get_shard(company: str) -> CompanyShard:
    with _lock:
        if company not in _shards:
            _shards[company] = load_shard(company)
    return _shards[company]


def _get_bm25(company: str) -> BM25Index:
    with _lock:
        if company not in _bm25s:
            shard = _get_shard(company)
            _bm25s[company] = BM25Index([c.text for c in shard.chunks])
    return _bm25s[company]


# ---------------------------------------------------------------------------
# Scoring helpers.
# ---------------------------------------------------------------------------


def _zscore(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    std = float(arr.std())
    if std < 1e-9:
        return np.zeros_like(arr)
    return (arr - arr.mean()) / std


def _embed_query(query: str) -> np.ndarray:
    """Embed a single query using the same model used at build time."""
    embedder = _get_embedder()
    vec = embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True)
    return vec[0].astype(np.float32)


def _candidate_pool(company: str, query: str, top_k: int) -> list[RetrievedChunk]:
    """Top-K candidates from one company's shard, fused but not yet reranked."""
    shard = _get_shard(company)
    bm25 = _get_bm25(company)
    if not shard.chunks:
        return []

    qvec = _embed_query(query)
    # Embeddings are L2-normalized at build time, so cosine == dot product.
    dense = (shard.embeddings @ qvec).astype(np.float32)
    bm25_raw = bm25.scores(query).astype(np.float32)

    fused = DENSE_WEIGHT * _zscore(dense) + BM25_WEIGHT * _zscore(bm25_raw)

    n = len(shard.chunks)
    k = min(top_k, n)
    # argpartition is faster than argsort when we only need the top-k.
    idx = np.argpartition(-fused, k - 1)[:k]
    idx = idx[np.argsort(-fused[idx])]

    out: list[RetrievedChunk] = []
    for i in idx:
        c = shard.chunks[int(i)]
        out.append(
            RetrievedChunk(
                chunk_id=c.chunk_id,
                path=c.path,
                company=c.company,
                score=float(fused[int(i)]),
                text=c.text,
            )
        )
    return out


def _maybe_rerank(query: str, candidates: list[RetrievedChunk], force: bool = False) -> list[RetrievedChunk]:
    """Cross-encoder rerank when top scores are borderline.

    `force=True` is used by union_retrieve where mixing scores across companies
    means the fused score is less reliable.
    """
    if not candidates:
        return candidates

    should_rerank = force
    if not should_rerank and len(candidates) >= 4:
        gap = candidates[0].score - candidates[3].score
        should_rerank = gap < RERANK_MARGIN

    if not should_rerank:
        return candidates

    reranker = _get_reranker()
    pairs = [(query, c.text) for c in candidates]
    scores = reranker.predict(pairs)
    for c, s in zip(candidates, scores):
        c.rerank_score = float(s)
    candidates.sort(key=lambda x: x.rerank_score if x.rerank_score is not None else x.score, reverse=True)
    return candidates


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------


class HybridRetriever:
    """Retrieves from one company's pre-built shard."""

    def __init__(self, company: str) -> None:
        self.company = company

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        candidates = _candidate_pool(self.company, query, HYBRID_TOP_K)
        candidates = _maybe_rerank(query, candidates)
        return candidates[: (top_k or RERANK_OUTPUT_K)]


def union_retrieve(query: str, companies: list[str], top_k: int | None = None) -> list[RetrievedChunk]:
    """Run retrieval against multiple companies and merge results.

    Used when the triage classifier returns inferred_company='none'. We always
    rerank in the cross-company case because z-scores are computed
    per-company and aren't directly comparable across shards.
    """
    if not companies:
        return []
    pool: list[RetrievedChunk] = []
    for company in companies:
        # Pull a slightly larger pool per-company so the union has variety.
        pool.extend(_candidate_pool(company, query, HYBRID_TOP_K))
    if not pool:
        return []
    pool.sort(key=lambda x: x.score, reverse=True)
    pool = pool[:HYBRID_TOP_K]
    pool = _maybe_rerank(query, pool, force=True)
    return pool[: (top_k or RERANK_OUTPUT_K)]
