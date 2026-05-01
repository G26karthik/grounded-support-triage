"""Fast-path pre-classifier (no LLM).

Three classes of input that don't need an LLM at all:
  - empty / whitespace-only
  - trivial conversational (thanks, hi)
  - hard injection or illegal asks

Most tickets fall through to the LLM. The conversational matcher is gated on
length (`_CONV_MAX_LEN`) so a long ticket that incidentally starts with
"Thanks!" still goes through the full pipeline.
"""

from __future__ import annotations

from hro.safety_rules import (
    HARD_INJECTION,
    ILLEGAL_REQUEST,
    TRIVIAL_CONVERSATIONAL,
    matches_any,
)
from hro.schemas import FastPathDecision, TicketInput

# A "thanks" or "hi" longer than this is probably not just an acknowledgement.
_CONV_MAX_LEN = 80


# Canned responses, picked from the matched pattern. Phrasing mirrors the
# sample CSV's tone for invalid-but-friendly tickets.
def _pick_canned_conv_response(text: str) -> str:
    norm = text.lower().strip()
    if "thank" in norm or norm.startswith(("thx", "ty", "appreciate")) or "cheers" in norm:
        return "Happy to help."
    if norm.startswith(("hi", "hello", "hey", "good ")):
        return "Hi there. What can I help you with today?"
    if norm.startswith(("ok", "okay")):
        return "Acknowledged."
    return "Got it."


def classify(ticket: TicketInput) -> FastPathDecision:
    """Run the fast-path pre-classifier."""
    text = ticket.issue.strip()

    if not text:
        return FastPathDecision(
            outcome="empty",
            reason="empty ticket body",
            canned_response="",
        )

    # 1. Conversational acknowledgements (length-gated).
    if len(text) <= _CONV_MAX_LEN:
        for pattern in TRIVIAL_CONVERSATIONAL:
            if pattern.match(text):
                return FastPathDecision(
                    outcome="trivial_conversational",
                    reason=f"conversational pattern matched ({pattern.pattern[:40]}...)",
                    canned_response=_pick_canned_conv_response(text),
                )

    # 2. Hard injection / system-leak attempts.
    matched, pat = matches_any(text, HARD_INJECTION)
    if matched:
        return FastPathDecision(
            outcome="hard_injection",
            reason=f"injection pattern: {pat[:60]}",
            canned_response=(
                "This request can't be processed safely. A human reviewer will follow up."
            ),
        )

    # 3. Illegal / destructive asks.
    matched, pat = matches_any(text, ILLEGAL_REQUEST)
    if matched:
        return FastPathDecision(
            outcome="illegal_request",
            reason=f"illegal-request pattern: {pat[:60]}",
            canned_response=(
                "I can't help with that. Routing to a human reviewer."
            ),
        )

    return FastPathDecision(outcome="pass_through", reason="", canned_response="")
