"""LangGraph StateGraph wiring for the per-ticket pipeline.

Flow:
    START -> fast_path
        - trivial / illegal / empty -> early_compose -> END
        - pass_through -> triage
    triage
        - verdict=refuse | scope=out_of_corpus | conversational -> early_compose -> END
        - allow & in_corpus -> retrieve -> specialist -> critic -> compose -> END

We deliberately don't attach a SqliteSaver checkpointer: durability is
provided by row-by-row writes to output.csv in the runner. For a 28-ticket
batch that completes in a few minutes, this is the simpler honest answer.

Each node is async to play nicely with `client.aio.models.generate_content`.
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from hro.agents import composer, critic, fast_path, triage as triage_module
from hro.agents.specialists.claude import ClaudeSpecialist
from hro.agents.specialists.hackerrank import HackerRankSpecialist
from hro.agents.specialists.visa import VisaSpecialist
from hro.config import COMPANIES
from hro.index.retriever import HybridRetriever, union_retrieve
from hro.index.store import load_shard
from hro.observability import span
from hro.schemas import (
    CriticVerdict,
    FastPathDecision,
    RetrievedChunk,
    Span,
    SpecialistDraft,
    TicketInput,
    TriageDecision,
    TriageOutput,
)


# ---------------------------------------------------------------------------
# State.
# ---------------------------------------------------------------------------


class TicketState(TypedDict, total=False):
    """LangGraph state for a single ticket."""

    ticket: TicketInput
    fast_decision: FastPathDecision | None
    triage: TriageDecision | None
    retrieved: list[RetrievedChunk]
    draft: SpecialistDraft | None
    critic: CriticVerdict | None
    output: TriageOutput | None
    spans: list[Span]
    error: str | None


# ---------------------------------------------------------------------------
# Specialist registry.
# ---------------------------------------------------------------------------

_SPECIALISTS = {
    "hackerrank": HackerRankSpecialist(),
    "claude": ClaudeSpecialist(),
    "visa": VisaSpecialist(),
}


# ---------------------------------------------------------------------------
# Nodes. Each returns a partial state update; LangGraph merges into TicketState.
# ---------------------------------------------------------------------------


async def fast_path_node(state: TicketState) -> dict:
    spans: list[Span] = state.get("spans", [])
    with span("fast_path", spans) as s:
        decision = fast_path.classify(state["ticket"])
        s["decision"] = decision.outcome
    return {"fast_decision": decision, "spans": spans}


async def triage_node(state: TicketState) -> dict:
    spans: list[Span] = state.get("spans", [])
    with span("triage", spans) as s:
        try:
            decision = await triage_module.classify(state["ticket"])
            s["decision"] = f"{decision.verdict}/{decision.scope}/{decision.inferred_company}"
        except Exception as e:
            s["error"] = f"{type(e).__name__}: {e}"
            return {"error": str(e), "spans": spans}
    return {"triage": decision, "spans": spans}


async def retrieve_node(state: TicketState) -> dict:
    spans: list[Span] = state.get("spans", [])
    triage = state["triage"]
    assert triage is not None
    with span("retrieve", spans) as s:
        if triage.inferred_company == "visa":
            # Visa is full-context; load the entire shard so the critic can
            # do citation membership checks against any chunk.
            chunks = load_shard("visa").chunks
            retrieved = [
                RetrievedChunk(
                    chunk_id=c.chunk_id,
                    path=c.path,
                    company=c.company,
                    score=0.0,
                    text=c.text,
                )
                for c in chunks
            ]
            s["decision"] = f"visa_full_corpus({len(retrieved)})"
        elif triage.inferred_company in ("hackerrank", "claude"):
            r = HybridRetriever(triage.inferred_company)
            retrieved = r.retrieve(triage.retrieval_query, top_k=5)
            s["decision"] = f"{triage.inferred_company}_top_{len(retrieved)}"
        else:
            # company=none — try to find something across all three; the
            # specialist router will handle the case where nothing meaningful comes back.
            retrieved = union_retrieve(triage.retrieval_query, list(COMPANIES), top_k=5)
            s["decision"] = f"union_top_{len(retrieved)}"
    return {"retrieved": retrieved, "spans": spans}


async def specialist_node(state: TicketState) -> dict:
    spans: list[Span] = state.get("spans", [])
    triage = state["triage"]
    assert triage is not None
    retrieved = state.get("retrieved", [])

    company = triage.inferred_company
    if company == "none":
        # If we got cross-company retrieval, infer the dominant company.
        if retrieved:
            company = max({c.company for c in retrieved}, key=lambda c: sum(1 for x in retrieved if x.company == c))
        else:
            company = "hackerrank"  # arbitrary fallback; critic will catch nothing-grounded

    spec = _SPECIALISTS.get(company)
    if spec is None:
        return {
            "draft": SpecialistDraft(
                response="",
                citations=[],
                internal_status="escalated",
                justification=f"No specialist available for inferred company={company}.",
            ),
            "spans": spans,
        }

    with span(f"specialist:{company}", spans) as s:
        try:
            draft = await spec.draft(state["ticket"], triage, retrieved)
            s["decision"] = draft.internal_status
        except Exception as e:
            s["error"] = f"{type(e).__name__}: {e}"
            draft = SpecialistDraft(
                response="",
                citations=[],
                internal_status="escalated",
                justification=f"Specialist call failed: {type(e).__name__}.",
            )
    return {"draft": draft, "spans": spans}


def critic_node(state: TicketState) -> dict:
    spans: list[Span] = state.get("spans", [])
    draft = state.get("draft")
    retrieved = state.get("retrieved", [])
    if draft is None:
        verdict = CriticVerdict(passed=False, failed_checks=["no_draft"])
        return {"critic": verdict, "spans": spans}
    with span("critic", spans) as s:
        verdict = critic.verify(draft, retrieved)
        s["decision"] = "pass" if verdict.passed else f"fail:{','.join(verdict.failed_checks[:3])}"
    return {"critic": verdict, "spans": spans}


def compose_node(state: TicketState) -> dict:
    spans: list[Span] = state.get("spans", [])
    triage = state["triage"]
    draft = state.get("draft")
    retrieved = state.get("retrieved", [])
    verdict = state.get("critic")
    if draft is None or verdict is None or triage is None:
        # Defensive: shouldn't happen on the main path.
        out = TriageOutput(
            status="escalated",
            product_area="",
            response="Routing this to a human agent.",
            justification="Pipeline produced no draft to compose.",
            request_type=triage.request_type if triage else "invalid",
        )
    else:
        with span("compose", spans):
            out = composer.compose(triage, draft, retrieved, verdict)
    return {"output": out, "spans": spans}


def early_compose_node(state: TicketState) -> dict:
    """Build a TriageOutput from fast_path or triage decision without retrieval."""
    spans: list[Span] = state.get("spans", [])
    fast = state.get("fast_decision")
    triage = state.get("triage")
    ticket = state["ticket"]

    with span("early_compose", spans) as s:
        # Fast-path terminal cases.
        if fast and fast.outcome != "pass_through":
            if fast.outcome == "trivial_conversational":
                out = TriageOutput(
                    status="replied",
                    product_area="",
                    response=fast.canned_response,
                    justification="Conversational acknowledgement; no support corpus needed.",
                    request_type="invalid",
                )
                s["decision"] = "fast:conversational"
            elif fast.outcome == "empty":
                out = TriageOutput(
                    status="escalated",
                    product_area="",
                    response="No issue text provided. A human will follow up.",
                    justification="Empty ticket body; routing to a human.",
                    request_type="invalid",
                )
                s["decision"] = "fast:empty"
            else:  # hard_injection or illegal_request
                out = TriageOutput(
                    status="escalated",
                    product_area="",
                    response=fast.canned_response,
                    justification=f"Pre-LLM safety filter caught {fast.outcome}; {fast.reason}",
                    request_type="invalid",
                )
                s["decision"] = f"fast:{fast.outcome}"
            return {"output": out, "spans": spans}

        # Triage terminal cases.
        if triage:
            if triage.verdict == "refuse":
                out = TriageOutput(
                    status="escalated",
                    product_area="",
                    response=(
                        "This request can't be processed safely. A human reviewer will follow up."
                    ),
                    justification=(
                        f"Safety filter caught {triage.safety_category or 'unsafe content'}; "
                        f"{triage.safety_reason}"
                    ).strip(),
                    request_type="invalid",
                )
                s["decision"] = "triage:refuse"
            elif triage.scope == "conversational":
                out = TriageOutput(
                    status="replied",
                    product_area="",
                    response="Happy to help — let me know if you have a specific support question.",
                    justification="Conversational ticket with no specific support request.",
                    request_type="invalid",
                )
                s["decision"] = "triage:conversational"
            else:  # out_of_corpus
                # Sample rubric tags off-topic user messages as conversation_management
                # (scope boundary / invalid), not a blank product_area.
                ooc_parea = (
                    "conversation_management"
                    if (triage.request_type or "invalid") == "invalid"
                    else ""
                )
                out = TriageOutput(
                    status="replied",
                    product_area=ooc_parea,
                    response="I am sorry, this is out of scope from my capabilities.",
                    justification="Ticket is not related to HackerRank, Claude, or Visa support.",
                    request_type=triage.request_type or "invalid",
                )
                s["decision"] = "triage:out_of_corpus"
            return {"output": out, "spans": spans}

        # Defensive fallback — shouldn't be reachable.
        out = TriageOutput(
            status="escalated",
            product_area="",
            response="Routing this to a human agent.",
            justification="No decision reached; escalating defensively.",
            request_type="invalid",
        )
        s["decision"] = "fallback"
    return {"output": out, "spans": spans}


# ---------------------------------------------------------------------------
# Conditional edges.
# ---------------------------------------------------------------------------


def _route_after_fast_path(state: TicketState) -> str:
    fast = state.get("fast_decision")
    if fast is None or fast.outcome == "pass_through":
        return "triage"
    return "early_compose"


def _route_after_triage(state: TicketState) -> str:
    triage = state.get("triage")
    if triage is None:
        # Triage call failed; treat as escalation.
        return "early_compose"
    if triage.verdict == "refuse":
        return "early_compose"
    if triage.scope in ("conversational", "out_of_corpus"):
        return "early_compose"
    return "retrieve"


# ---------------------------------------------------------------------------
# Graph builder.
# ---------------------------------------------------------------------------


def build_graph():  # type: ignore[no-untyped-def]
    """Construct and compile the LangGraph StateGraph."""
    g = StateGraph(TicketState)
    # Node names deliberately differ from state keys (LangGraph disallows collisions).
    g.add_node("fast_path", fast_path_node)
    g.add_node("triage_step", triage_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("specialist", specialist_node)
    g.add_node("critic_step", critic_node)
    g.add_node("compose", compose_node)
    g.add_node("early_compose", early_compose_node)

    g.add_edge(START, "fast_path")
    g.add_conditional_edges(
        "fast_path",
        _route_after_fast_path,
        {"triage": "triage_step", "early_compose": "early_compose"},
    )
    g.add_conditional_edges(
        "triage_step",
        _route_after_triage,
        {"retrieve": "retrieve", "early_compose": "early_compose"},
    )
    g.add_edge("retrieve", "specialist")
    g.add_edge("specialist", "critic_step")
    g.add_edge("critic_step", "compose")
    g.add_edge("compose", END)
    g.add_edge("early_compose", END)

    return g.compile()
