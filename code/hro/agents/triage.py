"""Combined safety + triage classifier (one Gemini call).

Single round trip returns a TriageDecision covering: safety verdict, request
type, scope, inferred company, English retrieval query, and a one-sentence
intent summary. Rolling these into one call saves a full round trip and shares
context that would otherwise be re-derived twice.

Few-shot examples come from prompts/few_shot.json. They are sourced exclusively
from sample_support_tickets.csv (the labelled set) — never from
support_tickets.csv (the held-out submission set).
"""

from __future__ import annotations

import json
from functools import lru_cache

from hro.config import ANTHROPIC_HAIKU, PROMPTS_DIR
from hro.llm.client import get_client
from hro.llm.prompts import render
from hro.schemas import TicketInput, TriageDecision


@lru_cache(maxsize=1)
def _few_shot_block() -> str:
    """Format prompts/few_shot.json into a prompt-ready block."""
    raw = json.loads((PROMPTS_DIR / "few_shot.json").read_text(encoding="utf-8"))
    examples = raw.get("examples", [])
    lines: list[str] = []
    for i, ex in enumerate(examples, start=1):
        ticket = ex["ticket"]
        expected = ex["expected"]
        lines.append(f"EXAMPLE {i}")
        lines.append(f"  Subject: {ticket.get('subject') or '(none)'}")
        lines.append(f"  Company: {ticket.get('company') or 'None'}")
        lines.append(f"  Issue:   {ticket.get('issue', '').strip()}")
        # Pad the expected dict with the empty safety fields so the model sees
        # the full schema shape.
        full_expected = {
            "verdict": expected.get("verdict", "allow"),
            "safety_category": expected.get("safety_category", ""),
            "safety_reason": expected.get("safety_reason", ""),
            "request_type": expected.get("request_type", "invalid"),
            "scope": expected.get("scope", "in_corpus"),
            "inferred_company": expected.get("inferred_company", "none"),
            "retrieval_query": expected.get("retrieval_query", ""),
            "language": expected.get("language", "en"),
            "intent_summary": expected.get("intent_summary", ""),
        }
        lines.append(f"  Decision: {json.dumps(full_expected, ensure_ascii=False)}")
        lines.append("")
    return "\n".join(lines).strip()


async def classify(ticket: TicketInput) -> TriageDecision:
    """One Gemini call returning the full triage decision."""
    prompt = render(
        "triage.md",
        few_shot_block=_few_shot_block(),
        subject=ticket.subject or "(none)",
        company=ticket.company or "None",
        issue=ticket.issue.strip(),
    )
    return await get_client().generate_structured(
        prompt,
        TriageDecision,
        thinking_level="minimal",
        model=ANTHROPIC_HAIKU,
    )
