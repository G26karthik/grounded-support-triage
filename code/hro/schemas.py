"""Pydantic + TypedDict schemas shared across the pipeline.

These models are the contract between nodes. Field changes here ripple
everywhere; update intentionally.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Enums and literal types.
# ---------------------------------------------------------------------------

# As they appear in the input CSV (Title case, observed exactly).
CompanyLabel = Literal["HackerRank", "Claude", "Visa", "None"]

# Lowercase slug used internally and by the corpus directory tree.
CompanySlug = Literal["hackerrank", "claude", "visa", "none"]

RequestType = Literal["product_issue", "feature_request", "bug", "invalid"]
Status = Literal["replied", "escalated"]
SafetyVerdict = Literal["allow", "refuse", "escalate"]
TriageScope = Literal["in_corpus", "out_of_corpus", "conversational"]

FastPathOutcome = Literal[
    "pass_through",
    "trivial_conversational",
    "hard_injection",
    "illegal_request",
    "empty",
]


# ---------------------------------------------------------------------------
# Inputs and intermediate results.
# ---------------------------------------------------------------------------


class TicketInput(BaseModel):
    """One row from support_tickets.csv."""

    idx: int
    issue: str
    subject: str
    company: CompanyLabel


class FastPathDecision(BaseModel):
    """Output of the regex/heuristic pre-classifier (no LLM)."""

    outcome: FastPathOutcome
    reason: str = ""
    canned_response: str = ""


class TriageDecision(BaseModel):
    """Combined safety + triage classifier output (one Gemini call)."""

    verdict: SafetyVerdict
    safety_category: str = Field(default="")
    safety_reason: str = Field(default="")
    request_type: RequestType
    scope: TriageScope
    inferred_company: CompanySlug
    retrieval_query: str = Field(
        description="English-language query optimized for retrieval. Translate if input is non-English."
    )
    language: str = Field(default="en", description="ISO 639-1 of the input language.")
    intent_summary: str = Field(default="")


class RetrievedChunk(BaseModel):
    """One chunk returned by the retriever, with provenance."""

    chunk_id: str
    path: str
    company: str
    score: float
    rerank_score: float | None = None
    text: str


class SpecialistDraft(BaseModel):
    """A specialist's draft output before the critic verifies it."""

    response: str
    citations: list[str] = Field(
        default_factory=list,
        description="chunk_ids supporting the response.",
    )
    internal_status: Status
    justification: str


class CriticVerdict(BaseModel):
    """Programmatic critic decision."""

    passed: bool
    failed_checks: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Output contract — the row written to support_tickets/output.csv.
# ---------------------------------------------------------------------------


class TriageOutput(BaseModel):
    """Final per-ticket prediction."""

    status: Status
    product_area: str
    response: str
    justification: str
    request_type: RequestType


# ---------------------------------------------------------------------------
# Observability.
# ---------------------------------------------------------------------------


class Span(TypedDict, total=False):
    """One node's execution record in the per-ticket trace."""

    node: str
    started_ms: int
    duration_ms: int
    in_tokens: int
    out_tokens: int
    decision: str
    error: str | None
