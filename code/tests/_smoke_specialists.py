"""Phase 7 smoke harness — exercise the three specialists end-to-end.

Pulls retrieval through HybridRetriever for HR/Claude, full-context for Visa.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hro.agents.specialists.claude import ClaudeSpecialist
from hro.agents.specialists.hackerrank import HackerRankSpecialist
from hro.agents.specialists.visa import VisaSpecialist
from hro.agents.triage import classify
from hro.index.retriever import HybridRetriever
from hro.schemas import TicketInput


SPECIALISTS = {
    "hackerrank": HackerRankSpecialist(),
    "claude": ClaudeSpecialist(),
    "visa": VisaSpecialist(),
}


CASES = [
    TicketInput(
        idx=0,
        issue=(
            "Hi, please pause our subscription. We have stopped all hiring efforts for now."
        ),
        subject="Subscription pause",
        company="HackerRank",
    ),
    TicketInput(
        idx=1,
        issue="I want Claude to stop crawling my website",
        subject="Website Data crawl",
        company="Claude",
    ),
    TicketInput(
        idx=2,
        issue="How do I dispute a charge",
        subject="Dispute charge",
        company="Visa",
    ),
    TicketInput(
        idx=3,
        issue=(
            "i am in US Virgin Islands and the merchant is saying i have to spend minimum "
            "10$ on my VISA card, why so?"
        ),
        subject="Visa card minimum spend",
        company="Visa",
    ),
]


async def run_one(ticket: TicketInput) -> None:
    t0 = time.time()
    triage = await classify(ticket)
    t_triage = (time.time() - t0) * 1000

    company = triage.inferred_company
    if company not in SPECIALISTS:
        print(f"[ticket {ticket.idx}] no specialist for inferred_company={company}; skipping draft")
        return

    if company == "visa":
        retrieved = []
    else:
        retrieved = HybridRetriever(company).retrieve(triage.retrieval_query, top_k=5)

    t1 = time.time()
    draft = await SPECIALISTS[company].draft(ticket, triage, retrieved)
    t_draft = (time.time() - t1) * 1000

    print(f"--- ticket {ticket.idx} | triage {t_triage:.0f}ms | draft {t_draft:.0f}ms ---")
    print(f"  ticket:    {ticket.issue[:90]}{'...' if len(ticket.issue) > 90 else ''}")
    print(f"  triage:    company={company}  scope={triage.scope}  request_type={triage.request_type}")
    print(f"  status:    {draft.internal_status}")
    print(f"  citations: {draft.citations}")
    print(f"  response:  {draft.response[:400]}{'...' if len(draft.response) > 400 else ''}")
    print(f"  why:       {draft.justification}")
    print()


async def main() -> None:
    for case in CASES:
        try:
            await run_one(case)
        except Exception as e:
            print(f"[ticket {case.idx}] ERROR: {type(e).__name__}: {e}")
            print()


if __name__ == "__main__":
    asyncio.run(main())
