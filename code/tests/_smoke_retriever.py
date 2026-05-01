"""Phase 3 smoke harness — known-answer retrieval queries.

Run from repo root:
    .venv\\Scripts\\python.exe code\\tests\\_smoke_retriever.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hro.index.retriever import HybridRetriever, union_retrieve


CASES = [
    ("visa", "I bought Visa traveller cheques from Citicorp and they were stolen in Lisbon last night",
     "visa/support/consumer/travelers-cheques.md"),
    ("visa", "Where can I report a lost or stolen Visa card from India", "visa/support.md"),
    ("visa", "US Virgin Islands merchant minimum charge 10 dollars", "visa/support.md"),
    ("hackerrank", "pause my HackerRank subscription stop hiring efforts",
     "hackerrank/settings/user-account-settings-and-preferences/5157311476-pause-subscription.md"),
    ("hackerrank", "how long do tests stay active in the system expiration",
     "hackerrank/screen/managing-tests/2979262079-modify-test-expiration-time.md"),
    ("hackerrank", "how to remove an interviewer from the platform", None),
    ("claude", "block ClaudeBot from crawling my website robots.txt",
     "claude/privacy-and-legal/8896518-does-anthropic-crawl-data-from-the-web-and-how-can-site-owners-block-the-crawler.md"),
    ("claude", "I found a security vulnerability in Claude bug bounty",
     "claude/safeguards/12119250-model-safety-bug-bounty-program.md"),
    ("claude", "how can I delete a Claude conversation",
     "claude/conversation-management/8230524-how-can-i-delete-or-rename-a-conversation.md"),
]


def main() -> None:
    print("--- per-company retrieval ---")
    for company, query, expected_path in CASES:
        r = HybridRetriever(company)
        t0 = time.time()
        hits = r.retrieve(query, top_k=3)
        dt = (time.time() - t0) * 1000
        top = hits[0] if hits else None
        rerank = "RR" if (top and top.rerank_score is not None) else "  "
        if expected_path is None:
            mark = " --"
        elif top and top.path == expected_path:
            mark = " OK"
        else:
            mark = "   "
        path = top.path if top else "(none)"
        score = top.score if top else 0.0
        print(f"{mark} {rerank} {company:10s} {dt:6.0f}ms  {score:6.3f}  {path}")
        if expected_path and top and top.path != expected_path:
            for h in hits:
                rs = h.rerank_score if h.rerank_score is not None else h.score
                print(f"           -> fused={h.score:6.3f}  rr={rs:6.3f}  {h.path}")
    print()
    print("--- union (company=None for site-down case) ---")
    hits = union_retrieve(
        "site is down all pages inaccessible",
        ["hackerrank", "claude", "visa"],
        top_k=3,
    )
    for h in hits:
        rs = h.rerank_score if h.rerank_score is not None else h.score
        print(f"{rs:6.3f}  {h.company:10s}  {h.path}")


if __name__ == "__main__":
    main()
