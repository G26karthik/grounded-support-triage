"""Specialist base — shared chunk-formatting and prompt-building helpers.

Each per-company specialist subclasses or composes around this. The protocol
(below) is what the LangGraph router calls.
"""

from __future__ import annotations

from typing import Protocol

from hro.schemas import RetrievedChunk, SpecialistDraft, TicketInput, TriageDecision


class Specialist(Protocol):
    """One per-company expert that drafts a grounded response."""

    company: str

    async def draft(
        self,
        ticket: TicketInput,
        triage: TriageDecision,
        retrieved: list[RetrievedChunk],
    ) -> SpecialistDraft:
        ...


def format_chunks_block(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks for inclusion in a specialist prompt.

    Each chunk gets a `[chunk_id: ...] [path: ...]` header so the model can
    cite by chunk_id. Limit each chunk's body to ~1200 chars to keep the
    overall prompt reasonable; if a chunk's body is longer, the leading slice
    holds the most-salient title/header content because of how we prefix.
    """
    if not chunks:
        return "(no chunks retrieved)"
    parts: list[str] = []
    for c in chunks:
        body = c.text if len(c.text) <= 1200 else c.text[:1200] + "\n... (truncated)"
        parts.append(f"[chunk_id: {c.chunk_id}] [path: {c.path}]")
        parts.append(body)
        parts.append("")
    return "\n".join(parts).rstrip()
