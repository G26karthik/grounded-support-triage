"""Regex/heuristic patterns shared by the fast-path and the critic.

Pure data — no LLM here. The fast_path uses these to short-circuit obvious
adversarial or trivial inputs; the critic uses the regex extractors to verify
that response claims are grounded in cited chunks.

Patterns are case-insensitive unless otherwise noted.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Trivial conversational inputs that need no LLM and no retrieval. Use these
# in conjunction with a max-length guard (see hro.agents.fast_path) so a long
# ticket that happens to start with "Thanks!" doesn't get short-circuited.
# ---------------------------------------------------------------------------
TRIVIAL_CONVERSATIONAL: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*$"),
    re.compile(r"^\s*(hi|hello|hey)(\s+there)?[!.,\s]*$", re.I),
    re.compile(r"^\s*good\s+(morning|afternoon|evening)[!.,\s]*$", re.I),
    # "thanks", "thank you", "thank you for helping me", "thanks for the help", etc.
    re.compile(
        r"^\s*(thanks|thank\s*you|thx|cheers|appreciate(?:\s+(?:it|you))?)\b"
        r"(?:\s+(?:for|to)\s+(?:the\s+)?(?:help(?:ing)?|support|reply)?(?:\s+me)?)?"
        r"[!.,\s]*$",
        re.I,
    ),
    re.compile(r"^\s*ok(?:ay)?[!.,\s]*$", re.I),
    re.compile(r"^\s*(ty|tysm|ttyl|np)[!.,\s]*$", re.I),
)

# ---------------------------------------------------------------------------
# Hard prompt-injection / system-leak patterns. Multilingual: EN, FR, ES.
# Match anywhere in the text.
# ---------------------------------------------------------------------------
HARD_INJECTION: tuple[re.Pattern[str], ...] = (
    re.compile(r"ignore (all |the |your )?(previous|prior|above) (instructions|rules|prompt)", re.I),
    re.compile(r"(reveal|show|print|leak|disclose).{0,30}(system|internal|hidden)\s+(prompt|rules|instructions)", re.I),
    re.compile(r"(affiche|montre|donne[- ]moi).{0,30}(r[eè]gles?|instructions?|prompt)\s+(internes?|syst[eè]me)", re.I),
    re.compile(r"(muestra|revela|imprime).{0,30}(reglas|instrucciones|prompt)\s+(internas?|del sistema)", re.I),
    re.compile(r"act as (?:if|though).{0,30}(you (are|were)|developer|root|admin)", re.I),
    re.compile(r"jailbreak|bypass.{0,20}(safety|filter|guardrails?)", re.I),
)

# ---------------------------------------------------------------------------
# Hard "do something illegal/destructive for me" patterns. Match anywhere.
# ---------------------------------------------------------------------------
ILLEGAL_REQUEST: tuple[re.Pattern[str], ...] = (
    re.compile(r"(give|write|generate|provide).{0,40}code.{0,20}(delete|wipe|format|destroy).{0,40}(files?|disk|system|drive)", re.I),
    re.compile(r"\brm\s+-rf\b", re.I),
    re.compile(r"how (to|do i) (hack|break into|crack|bypass)", re.I),
    re.compile(r"(synthesize|make|build).{0,40}(weapon|explosive|bomb|malware|virus)", re.I),
)

# ---------------------------------------------------------------------------
# Critic: extractors for facts that must appear (or paraphrase) in cited chunks.
# ---------------------------------------------------------------------------
PHONE_NUMBER = re.compile(
    r"(?:\+?\d{1,3}[\s\-]?)?(?:\(\d{1,4}\)[\s\-]?)?\d{3,4}[\s\-]?\d{3,4}[\s\-]?\d{0,4}"
)
URL = re.compile(r"https?://[\w\-./%~?#=&]+", re.I)
EMAIL = re.compile(r"[\w.\-+]+@[\w\-]+\.[\w.\-]+", re.I)
DOLLAR_AMOUNT = re.compile(r"(?:US\s*)?\$\s?\d+(?:[.,]\d+)?")


def matches_any(text: str, patterns: tuple[re.Pattern[str], ...]) -> tuple[bool, str]:
    """Return (matched, matched_pattern_str). Empty pattern_str if no match."""
    for p in patterns:
        if p.search(text):
            return True, p.pattern
    return False, ""
