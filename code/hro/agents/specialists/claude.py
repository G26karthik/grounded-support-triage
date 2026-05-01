"""Claude specialist — retrieval-based grounded response generator."""

from __future__ import annotations

from hro.agents.specialists.base import format_chunks_block
from hro.config import ANTHROPIC_SONNET
from hro.llm.client import get_client
from hro.llm.prompts import render
from hro.schemas import RetrievedChunk, SpecialistDraft, TicketInput, TriageDecision


class ClaudeSpecialist:
    company: str = "claude"

    async def draft(
        self,
        ticket: TicketInput,
        triage: TriageDecision,
        retrieved: list[RetrievedChunk],
    ) -> SpecialistDraft:
        prompt = render(
            "specialist_claude.md",
            chunks_block=format_chunks_block(retrieved),
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
