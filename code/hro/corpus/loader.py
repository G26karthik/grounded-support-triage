"""Markdown loader. Walks data/{company}/ and parses YAML frontmatter.

Every article we care about lives at data/<company>/.../<slug>.md and starts
with a YAML frontmatter block. We skip the per-company `index.md` aggregators
because they're tables of contents — useful as a sitemap, not as retrievable
prose. Files without frontmatter (rare) still load, with title inferred from
the first H1 and empty breadcrumbs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from hro.config import COMPANIES, DATA_DIR


@dataclass(frozen=True)
class Document:
    """One source markdown file with parsed metadata."""

    # Path relative to data/, POSIX-style. Example:
    # "claude/safeguards/12119250-model-safety-bug-bounty-program.md".
    path: str
    company: str  # one of config.COMPANIES
    title: str
    breadcrumbs: tuple[str, ...]
    body: str


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_all() -> list[Document]:
    """Load every markdown file under data/, skipping company-level aggregators."""
    docs: list[Document] = []
    for company in COMPANIES:
        docs.extend(load_company(company))
    return docs


def load_company(company: str) -> list[Document]:
    """Load every markdown file under data/<company>/."""
    company_dir = DATA_DIR / company
    if not company_dir.exists():
        return []
    out: list[Document] = []
    for md_path in sorted(company_dir.rglob("*.md")):
        # Skip the per-company aggregator (data/<company>/index.md).
        if md_path.parent == company_dir and md_path.name == "index.md":
            continue
        doc = _load_one(md_path, company)
        if doc is not None:
            out.append(doc)
    return out


def parse_frontmatter(content: str) -> tuple[dict, str]:
    """Split YAML frontmatter from body. Returns ({} , content) if none."""
    if not content.startswith("---"):
        return {}, content
    # Frontmatter is delimited by --- on its own line at the start, then ---
    # on its own line again before the body.
    parts = content.split("---", 2)
    if len(parts) < 3:
        return {}, content
    try:
        metadata = yaml.safe_load(parts[1]) or {}
        if not isinstance(metadata, dict):
            metadata = {}
    except yaml.YAMLError:
        metadata = {}
    body = parts[2].lstrip("\n")
    return metadata, body


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


_H1 = re.compile(r"^\s*#\s+(.+?)\s*$", re.MULTILINE)


def _load_one(path: Path, company: str) -> Document | None:
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    metadata, body = parse_frontmatter(content)

    title = str(metadata.get("title", "")).strip()
    if not title:
        m = _H1.search(body)
        if m:
            title = m.group(1).strip()
        else:
            title = path.stem

    bc_raw = metadata.get("breadcrumbs", [])
    if isinstance(bc_raw, list):
        breadcrumbs = tuple(str(x).strip() for x in bc_raw if str(x).strip())
    else:
        breadcrumbs = ()

    rel_path = path.relative_to(DATA_DIR).as_posix()

    return Document(
        path=rel_path,
        company=company,
        title=title,
        breadcrumbs=breadcrumbs,
        body=body.strip(),
    )
