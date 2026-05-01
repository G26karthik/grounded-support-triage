"""Gemini context-caching helpers.

Used by the Visa specialist to avoid resending the full ~12K-token Visa corpus
on every ticket. Falls back to inline corpus if caching fails or the corpus is
below the API's minimum-token threshold.

Built in Phase 7.
"""

from __future__ import annotations


async def get_or_create_visa_cache() -> str | None:
    """Return a cache resource name, or None to indicate fallback to inline corpus."""
    raise NotImplementedError("Phase 7: implement Visa context cache.")
