"""Visa specialist — full-context grounded response generator.

The Visa corpus is small enough (~12K tokens across 13 docs) that we put the
entire thing in the prompt rather than retrieving over it. Citations are
chunk_ids drawn from the same set the index was built from, which lets the
critic do exact set-membership checks.

Context caching is a Gemini-side optimization — we currently don't try to
register a cached_content because the corpus is below the API minimum on the
Flash family; if we measure that the inline path dominates cost during eval,
we'll add a cache-create fallback. For now: pass inline.
"""

from __future__ import annotations

from functools import lru_cache

from hro.config import ANTHROPIC_SONNET
from hro.index.store import load_shard
from hro.llm.client import get_client
from hro.llm.prompts import render
from hro.schemas import RetrievedChunk, SpecialistDraft, TicketInput, TriageDecision


@lru_cache(maxsize=1)
def _visa_corpus_block() -> str:
    """Render every Visa chunk as one big prompt block, with chunk_ids exposed
    so the model can cite them and the critic can do membership checks."""
    shard = load_shard("visa")
    parts: list[str] = []
    for c in shard.chunks:
        parts.append(f"[chunk_id: {c.chunk_id}] [path: {c.path}]")
        parts.append(c.text)
        parts.append("")
    return "\n".join(parts).rstrip()


@lru_cache(maxsize=1)
def visa_chunk_ids() -> set[str]:
    """Set of all Visa chunk_ids — used by the critic for membership checks."""
    return {c.chunk_id for c in load_shard("visa").chunks}


class VisaSpecialist:
    company: str = "visa"

    async def draft(
        self,
        ticket: TicketInput,
        triage: TriageDecision,
        retrieved: list[RetrievedChunk],  # ignored; we always use the full corpus
    ) -> SpecialistDraft:
        prompt = render(
            "specialist_visa.md",
            corpus_block=_visa_corpus_block(),
            subject=ticket.subject or "(none)",
            issue=ticket.issue.strip(),
            request_type=triage.request_type,
            intent_summary=triage.intent_summary or "(unstated)",
        )
        return await get_client().generate_structured(
            prompt,
            SpecialistDraft,
            thinking_level="minimal",
            model=ANTHROPIC_SONNET,
        )
