"""Centralized configuration: paths, model versions, and tuning thresholds.

Bump CHUNKER_VERSION or EMBEDDER_VERSION when their respective implementations
change in a way that invalidates cached artifacts on disk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

# ---------------------------------------------------------------------------
# Paths. Resolved relative to this file so the package is location-independent.
# ---------------------------------------------------------------------------
HRO_PKG_DIR: Final[Path] = Path(__file__).resolve().parent
CODE_DIR: Final[Path] = HRO_PKG_DIR.parent
REPO_ROOT: Final[Path] = CODE_DIR.parent

DATA_DIR: Final[Path] = REPO_ROOT / "data"
INDEX_DIR: Final[Path] = DATA_DIR / "index"
TICKETS_DIR: Final[Path] = REPO_ROOT / "support_tickets"
PROMPTS_DIR: Final[Path] = CODE_DIR / "prompts"
RUNS_DIR: Final[Path] = CODE_DIR / "runs"

INPUT_CSV: Final[Path] = TICKETS_DIR / "support_tickets.csv"
SAMPLE_CSV: Final[Path] = TICKETS_DIR / "sample_support_tickets.csv"
OUTPUT_CSV: Final[Path] = TICKETS_DIR / "output.csv"

PRODUCT_AREA_MAP: Final[Path] = HRO_PKG_DIR / "corpus" / "product_area_map.json"

# ---------------------------------------------------------------------------
# Companies. These are the directory names under data/ and the lowercase slugs
# used internally. Input CSV uses Title-case (HackerRank, Claude, Visa, None);
# see schemas.CompanyLabel for the input enum.
# ---------------------------------------------------------------------------
COMPANIES: Final[tuple[str, ...]] = ("hackerrank", "claude", "visa")

# ---------------------------------------------------------------------------
# Model identifiers. Pin everything; never bump silently.
# ---------------------------------------------------------------------------
# Originally `gemini-3-flash-preview` per the project spec. Free-tier daily
# quota (20 RPD) on the Gemini API turned out to be incompatible with a
# 28-ticket pipeline that does ~50 calls per full eval+run cycle, and
# enabling billing was not an option for this run. Pivoted to Anthropic
# Claude (key supplied by user). The Gemini constants stay so the path can
# be restored quickly by setting HRO_BACKEND=gemini and the right key.
import os as _os

GEMINI_MODEL: Final[str] = _os.environ.get("GEMINI_MODEL_OVERRIDE", "gemini-2.5-flash")

# Anthropic models. Haiku for the high-volume triage/safety classifier;
# Sonnet for the specialists where groundedness reasoning matters more.
ANTHROPIC_HAIKU: Final[str] = _os.environ.get("HRO_ANTHROPIC_HAIKU", "claude-haiku-4-5")
ANTHROPIC_SONNET: Final[str] = _os.environ.get("HRO_ANTHROPIC_SONNET", "claude-sonnet-4-5")

# Backend selector. "anthropic" or "gemini". The LLM client honors this.
LLM_BACKEND: Final[str] = _os.environ.get("HRO_BACKEND", "anthropic").lower()
EMBEDDING_MODEL: Final[str] = "BAAI/bge-small-en-v1.5"
RERANK_MODEL: Final[str] = "mixedbread-ai/mxbai-rerank-xsmall-v1"

CHUNKER_VERSION: Final[str] = "v1"
EMBEDDER_VERSION: Final[str] = "v1"

# ---------------------------------------------------------------------------
# Chunking parameters.
# ---------------------------------------------------------------------------
CHUNK_TOKEN_TARGET: Final[int] = 500
CHUNK_TOKEN_OVERLAP: Final[int] = 60
CHUNK_TOKENIZER: Final[str] = "cl100k_base"  # tiktoken encoding for token counts

# ---------------------------------------------------------------------------
# Retrieval parameters. Calibrate on the sample set; do not hardcode by
# inspection of the test set.
# ---------------------------------------------------------------------------
HYBRID_TOP_K: Final[int] = 20
RERANK_OUTPUT_K: Final[int] = 5
RERANK_MARGIN: Final[float] = 0.5  # if z-score gap (top1 - top4) > this, skip rerank
GROUNDEDNESS_MIN_SCORE: Final[float] = 0.15  # below this, escalate
BM25_WEIGHT: Final[float] = 0.5
DENSE_WEIGHT: Final[float] = 0.5

# ---------------------------------------------------------------------------
# Concurrency. Default of 1 keeps us under the Gemini free-tier rate limit
# (5 req/min on gemini-3-flash). Bump to 4+ once billing is on.
# ---------------------------------------------------------------------------
DEFAULT_CONCURRENCY: Final[int] = 1

# ---------------------------------------------------------------------------
# Output contract (locked per problem_statement.md and the existing output.csv
# header). Do not reorder.
# ---------------------------------------------------------------------------
ALLOWED_STATUS: Final[tuple[str, ...]] = ("replied", "escalated")
ALLOWED_REQUEST_TYPES: Final[tuple[str, ...]] = (
    "product_issue",
    "feature_request",
    "bug",
    "invalid",
)
OUTPUT_COLUMNS: Final[tuple[str, ...]] = (
    "issue",
    "subject",
    "company",
    "response",
    "product_area",
    "status",
    "request_type",
    "justification",
)
