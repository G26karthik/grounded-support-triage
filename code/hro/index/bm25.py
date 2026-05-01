"""BM25 sparse retrieval per company.

We don't persist the BM25 state to disk — building it from chunks.jsonl at
load time takes <100ms even for the HackerRank shard, and skipping pickle
keeps the cache simpler and safer.
"""

from __future__ import annotations

import re

import numpy as np
from rank_bm25 import BM25Okapi


# Simple word tokenizer. Lowercase, alphanumeric runs only. Deterministic.
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """Tokenized BM25Okapi over a list of texts."""

    def __init__(self, texts: list[str]) -> None:
        # rank_bm25 trips on an empty corpus; guard with a minimal placeholder.
        tokenized = [tokenize(t) for t in texts] if texts else [[""]]
        self._bm25 = BM25Okapi(tokenized)

    def scores(self, query: str) -> np.ndarray:
        """Raw BM25 scores aligned to corpus order."""
        return np.array(self._bm25.get_scores(tokenize(query)), dtype=np.float32)
