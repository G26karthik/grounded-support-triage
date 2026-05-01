"""Deterministic final-row assembler (no LLM).

Inputs come from the upstream nodes:
  - triage:     classifier decision (request_type, scope, inferred_company)
  - draft:      specialist's response + citations
  - retrieved:  the chunks the specialist saw (used for product_area derivation)
  - critic:     verdict; if it failed, we downgrade to escalation here

The composer is the single point that decides the final `status`,
`product_area`, and the customer-facing `response` and `justification`.
"""

from __future__ import annotations

from collections import Counter

from hro.corpus.product_area import slug_for
from hro.schemas import (
    CriticVerdict,
    RetrievedChunk,
    SpecialistDraft,
    TriageDecision,
    TriageOutput,
)


def compose(
    triage: TriageDecision,
    draft: SpecialistDraft,
    retrieved: list[RetrievedChunk],
    critic: CriticVerdict,
) -> TriageOutput:
    """Build the final row from the verified specialist output.

    Downgrade rules:
      - critic failed -> status=escalated, response is replaced with a safe
        message naming the failed checks (without exposing internals).
      - draft.internal_status='escalated' -> trust the specialist's call.
    """
    chunks_by_id = {c.chunk_id: c for c in retrieved}

    # Decide final status.
    if not critic.passed:
        status = "escalated"
        response = (
            "I'm not able to give a fully grounded answer here, so I'm routing this to a human "
            "support agent who can review the details and follow up."
        )
        justification = _justification_for_failed_critic(critic, draft, triage)
    else:
        status = draft.internal_status
        response = draft.response.strip() or _empty_response_fallback(status)
        justification = (draft.justification or "").strip() or _generic_justification(triage, draft)

    # product_area from the most-cited chunk's path. If no usable citations,
    # try the single highest-scored retrieved chunk.
    product_area = _derive_product_area(draft, retrieved, chunks_by_id, status, triage)

    return TriageOutput(
        status=status,
        product_area=product_area,
        response=response,
        justification=justification,
        request_type=triage.request_type,
    )


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _derive_product_area(
    draft: SpecialistDraft,
    retrieved: list[RetrievedChunk],
    chunks_by_id: dict[str, RetrievedChunk],
    status: str,
    triage: TriageDecision,
) -> str:
    # Company=None escalations should not inherit a product bucket from noisy retrieval
    # (labelled sample: site-wide outage → empty product_area).
    if status == "escalated" and triage.inferred_company == "none":
        return ""

    candidates: list[RetrievedChunk] = []
    if draft.citations:
        candidates = [chunks_by_id[c] for c in draft.citations if c in chunks_by_id]
    if not candidates and retrieved:
        candidates = [retrieved[0]]
    if not candidates:
        return ""

    counts: Counter[str] = Counter()
    for c in candidates:
        slug = slug_for(c.company, _strip_company_prefix(c.company, c.path))
        if slug:
            counts[slug] += 1
    if not counts:
        return ""
    return counts.most_common(1)[0][0]


def _strip_company_prefix(company: str, path: str) -> str:
    """Map a corpus path like 'visa/support/consumer/...' to 'support/consumer/...'."""
    prefix = f"{company.lower()}/"
    if path.startswith(prefix):
        return path[len(prefix):]
    return path


def _justification_for_failed_critic(
    critic: CriticVerdict, draft: SpecialistDraft, triage: TriageDecision
) -> str:
    summary = ", ".join(check.split(":")[0] for check in critic.failed_checks[:3])
    return (
        f"Escalated: groundedness check failed ({summary}). "
        f"Routing to a human; the agent could not verify all claims against the corpus."
    )


def _generic_justification(triage: TriageDecision, draft: SpecialistDraft) -> str:
    if not draft.citations:
        return "Replied without retrieved citations; based on safety/triage decision only."
    paths = [c[:8] for c in draft.citations[:3]]
    return f"Grounded in chunk(s) {', '.join(paths)} from the {triage.inferred_company} corpus."


def _empty_response_fallback(status: str) -> str:
    if status == "escalated":
        return (
            "Routing this to a human agent — the corpus does not cover this case in enough "
            "detail to answer safely."
        )
    return "Got it."
