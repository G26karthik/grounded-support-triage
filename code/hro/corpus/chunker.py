"""Header-aware semantic chunker.

Strategy: paragraph-aware greedy packing.

  1. Build a per-chunk prefix: "{title}\\n[{breadcrumbs}]\\n\\n".
  2. Split the body on blank-line paragraph boundaries (markdown convention).
  3. Pre-split any oversized paragraph (rare — long tables, code fences) on
     sentence boundaries first, then on newlines (table rows), then on a hard
     token window as a last resort. Each piece is bounded by body_target.
  4. Greedy-pack pieces into chunks targeting <= body_target body tokens.

Chunk IDs are sha256(path + idx)[:12]. Stable as long as file contents are
byte-identical, which is what we want — the index cache key is the corpus
content hash, so chunk IDs are also a function of corpus content.

We do not use overlap on v1. If the eval harness shows retrieval misses where
a claim spans a chunk boundary, revisit and add token-tail overlap.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from functools import lru_cache

import tiktoken

from hro.config import CHUNK_TOKEN_TARGET, CHUNK_TOKENIZER
from hro.corpus.loader import Document


@dataclass(frozen=True)
class Chunk:
    """A retrievable chunk of text with provenance."""

    chunk_id: str
    company: str
    path: str
    title: str
    breadcrumbs: tuple[str, ...]
    text: str  # already includes title + breadcrumb prefix
    body_token_count: int


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def chunk_document(doc: Document) -> list[Chunk]:
    """Header-aware split of one document."""
    if not doc.body.strip():
        return []

    enc = _encoding()
    prefix = _build_prefix(doc.title, doc.breadcrumbs)
    prefix_tokens = len(enc.encode(prefix))
    # Reserve at least 200 tokens for body even if prefix is unusually long.
    body_target = max(CHUNK_TOKEN_TARGET - prefix_tokens, 200)

    # 1. Split body into paragraphs on blank lines.
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", doc.body) if p.strip()]

    # 2. Pre-split oversized paragraphs into bounded pieces.
    pieces: list[str] = []
    for p in paragraphs:
        if len(enc.encode(p)) > body_target:
            pieces.extend(_hard_split(p, enc, body_target))
        else:
            pieces.append(p)

    # 3. Greedy-pack pieces.
    body_chunks: list[str] = []
    current: list[str] = []
    current_count = 0
    for piece in pieces:
        piece_count = len(enc.encode(piece))
        if current and current_count + piece_count > body_target:
            body_chunks.append("\n\n".join(current))
            current = [piece]
            current_count = piece_count
        else:
            current.append(piece)
            current_count += piece_count
    if current:
        body_chunks.append("\n\n".join(current))

    chunks: list[Chunk] = []
    for idx, body_text in enumerate(body_chunks):
        chunk_id = hashlib.sha256((doc.path + ":" + str(idx)).encode("utf-8")).hexdigest()[:12]
        text = prefix + body_text
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                company=doc.company,
                path=doc.path,
                title=doc.title,
                breadcrumbs=doc.breadcrumbs,
                text=text,
                body_token_count=len(enc.encode(body_text)),
            )
        )
    return chunks


def chunk_documents(docs: list[Document]) -> list[Chunk]:
    out: list[Chunk] = []
    for d in docs:
        out.extend(chunk_document(d))
    return out


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _encoding() -> "tiktoken.Encoding":
    return tiktoken.get_encoding(CHUNK_TOKENIZER)


def _build_prefix(title: str, breadcrumbs: tuple[str, ...]) -> str:
    lines = [title]
    if breadcrumbs:
        lines.append("[" + " / ".join(breadcrumbs) + "]")
    return "\n".join(lines) + "\n\n"


_SENT_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def _hard_split(text: str, enc: "tiktoken.Encoding", max_tokens: int) -> list[str]:
    """Split a too-large paragraph into pieces of <= max_tokens tokens.

    Tries three strategies in order:
      1. Sentence boundaries (works for prose).
      2. Newlines (works for tables and lists, where rows are atomic).
      3. Token window (last-resort fallback for rare cases like one giant
         multi-line code block with no internal newlines).
    """
    # Strategy 1: sentence boundaries.
    units = _SENT_BOUNDARY.split(text)
    if len(units) == 1:
        # Strategy 2: newlines.
        units = [line for line in text.split("\n") if line.strip()]
    if len(units) == 1:
        # Strategy 3: token window.
        ids = enc.encode(text)
        return [enc.decode(ids[i : i + max_tokens]) for i in range(0, len(ids), max_tokens)]

    # Pack units (sentences or rows) greedily, recursing if a single unit is
    # itself too big.
    out: list[str] = []
    cur: list[str] = []
    cur_count = 0
    for u in units:
        u_count = len(enc.encode(u))
        if u_count > max_tokens:
            # Flush whatever we have before a recursive split.
            if cur:
                out.append("\n".join(cur))
                cur = []
                cur_count = 0
            out.extend(_hard_split(u, enc, max_tokens))
            continue
        if cur and cur_count + u_count > max_tokens:
            out.append("\n".join(cur))
            cur = [u]
            cur_count = u_count
        else:
            cur.append(u)
            cur_count += u_count
    if cur:
        out.append("\n".join(cur))
    return out
