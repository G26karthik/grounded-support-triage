"""Programmatic citation + claim verifier (no LLM).

Two checks for v1:
  1. Citation membership: every chunk_id in `draft.citations` must exist in
     the retrieval set the specialist saw. (For Visa, the retrieval set is
     the entire Visa shard, since the Visa specialist works full-context.)
  2. Hard-fact membership: every phone number, URL, email, and dollar amount
     in `draft.response` must appear in at least one cited chunk's text,
     after light normalization.

We deliberately skip a "soft policy claim" heuristic for now — it produced too
many false positives during prototyping. The hard-fact check is the
high-leverage anti-hallucination control because phone numbers and URLs are
the facts most expensive to get wrong (the user might dial a fake number).

Failure on any check downgrades to status='escalated' downstream.
"""

from __future__ import annotations

import re

from rapidfuzz import fuzz

from hro.safety_rules import DOLLAR_AMOUNT, EMAIL, URL
from hro.schemas import CriticVerdict, RetrievedChunk, SpecialistDraft


def verify(draft: SpecialistDraft, retrieved: list[RetrievedChunk]) -> CriticVerdict:
    """Run all checks. Return passed=False with named failures on any miss."""
    failed: list[str] = []

    retrieved_by_id = {c.chunk_id: c for c in retrieved}

    # 1. Citation membership.
    for cid in draft.citations:
        if cid not in retrieved_by_id:
            failed.append(f"citation_not_in_retrieval:{cid}")

    # If citations are bogus, hard-fact checks will be unreliable. Stop here.
    if failed:
        return CriticVerdict(passed=False, failed_checks=failed)

    # If the specialist itself decided to escalate, no further checks needed.
    if draft.internal_status == "escalated":
        return CriticVerdict(passed=True, failed_checks=[])

    # Build the haystack from cited chunks. If nothing was cited but the
    # specialist replied, that's itself a failure.
    if not draft.citations:
        return CriticVerdict(passed=False, failed_checks=["replied_without_citations"])

    cited_chunks = [retrieved_by_id[c] for c in draft.citations]
    cited_text = " ".join(c.text for c in cited_chunks)
    cited_lower = cited_text.lower()
    # Markdown escapes ($, %, etc.) appear as `\$10`, `\%`, ... — strip them so
    # the response's `$10` matches the corpus's `\$10`.
    cited_lower_unescaped = cited_lower.replace("\\", "")
    cited_digits_only = re.sub(r"\D", "", cited_lower)

    response = draft.response

    # 2a. Phone numbers — normalize to digit-only and check substring.
    for phone in _find_phones(response):
        digits = re.sub(r"\D", "", phone)
        if not digits:
            continue
        if digits not in cited_digits_only and digits[-7:] not in cited_digits_only:
            failed.append(f"fabricated_phone:{phone}")

    # 2b. URLs — accept full URL OR just the path portion (corpus often has
    # relative paths in markdown links; the model fully-qualifies them).
    for url in URL.findall(response):
        norm = url.lower().rstrip(".,;:)]>!?")
        if norm in cited_lower:
            continue
        m = re.match(r"https?://[^/\s]+(/[^\s]*)", norm)
        path = m.group(1) if m else norm
        if path and path in cited_lower:
            continue
        # Last-ditch: case-insensitive path match (corpus uses /Forms/ but the
        # model may lowercase to /forms/).
        if path and path.lower() in cited_lower:
            continue
        failed.append(f"fabricated_url:{norm}")

    # 2c. Emails — lowercase substring check.
    for email in EMAIL.findall(response):
        if email.lower() not in cited_lower:
            failed.append(f"fabricated_email:{email}")

    # 2d. Dollar amounts — strip whitespace and markdown escapes, then check
    # substring with a fuzzy fallback. The corpus often has `US\$10` (markdown
    # escape) which strips to `US$10` after our normalization.
    cited_norm = re.sub(r"\s+", "", cited_lower_unescaped)
    for amt in DOLLAR_AMOUNT.findall(response):
        norm = re.sub(r"\s+", "", amt.lower())
        if norm in cited_norm:
            continue
        if fuzz.partial_ratio(norm, cited_norm) >= 90:
            continue
        failed.append(f"fabricated_dollar:{amt}")

    return CriticVerdict(passed=not failed, failed_checks=failed)


# ---------------------------------------------------------------------------
# Phone-number extractor — separate from safety_rules because the rule there
# is pattern-only; here we want validated extracts.
# ---------------------------------------------------------------------------

_PHONE_CANDIDATE = re.compile(r"\+?\d[\d\s\-().]{5,}\d")


def _find_phones(text: str) -> list[str]:
    """Return phone-number-shaped substrings with 7-15 digits after separators
    are removed. Skips fragments that look like dates or version numbers.
    """
    out: list[str] = []
    for m in _PHONE_CANDIDATE.finditer(text):
        cand = m.group(0).strip()
        digits = re.sub(r"\D", "", cand)
        if 7 <= len(digits) <= 15:
            # Heuristic: skip if the candidate is just a year (e.g. "2025").
            if len(digits) == 4 and 1900 <= int(digits) <= 2099:
                continue
            out.append(cand)
    return out
