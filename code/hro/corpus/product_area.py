"""Map a corpus path to a product_area slug.

Loads the JSON map next to this file and applies longest-prefix matching.
"""

from __future__ import annotations

import json
from functools import lru_cache

from hro.config import PRODUCT_AREA_MAP


@lru_cache(maxsize=1)
def _load_map() -> dict[str, dict[str, str]]:
    with PRODUCT_AREA_MAP.open(encoding="utf-8") as f:
        return json.load(f)


def slug_for(company: str, relative_path: str) -> str:
    """Return the product_area slug for a corpus path.

    `relative_path` is the file path relative to data/<company>/, e.g.
    `screen/getting-started/1152916770-ai-assisted-tests.md` for HackerRank or
    `safeguards/12119250-model-safety-bug-bounty-program.md` for Claude.
    Returns an empty string if no mapping matches.
    """
    company = company.lower()
    rules = _load_map().get(company, {})
    # longest-prefix match — gives us "claude/conversation-management" preference
    # over "claude" when both are present.
    best_prefix = ""
    best_slug = ""
    for prefix, slug in rules.items():
        if relative_path.startswith(prefix) and len(prefix) > len(best_prefix):
            best_prefix = prefix
            best_slug = slug
    return best_slug
