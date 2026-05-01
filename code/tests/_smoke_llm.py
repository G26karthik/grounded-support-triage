"""Phase 4 smoke harness — verify the Gemini client returns a typed object.

Run from repo root:
    .venv\\Scripts\\python.exe code\\tests\\_smoke_llm.py
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import BaseModel, Field

from hro.llm.client import get_client


class TestDecision(BaseModel):
    company: str = Field(description="One of hackerrank, claude, visa, none")
    confidence: float = Field(description="0.0 to 1.0")
    one_line_reason: str


async def main() -> None:
    client = get_client()
    print(f"model: {client.model}")

    prompt = (
        "Classify this support ticket by company.\n"
        "Ticket: 'My HackerRank assessment was rescheduled, can I re-invite candidates?'\n"
        "Return JSON only."
    )
    t0 = time.time()
    out = await client.generate_structured(
        prompt,
        TestDecision,
        thinking_level="minimal",
        system_instruction="You are a support ticket classifier.",
    )
    dt = (time.time() - t0) * 1000
    print(f"call OK in {dt:.0f}ms")
    print(f"  company={out.company!r}")
    print(f"  confidence={out.confidence}")
    print(f"  reason={out.one_line_reason!r}")
    assert isinstance(out, TestDecision)
    assert out.company.lower() in {"hackerrank", "claude", "visa", "none"}


if __name__ == "__main__":
    asyncio.run(main())
