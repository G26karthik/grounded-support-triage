"""Prompt loader. Reads markdown prompt files from code/prompts/ and renders
{{var}} placeholders.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from hro.config import PROMPTS_DIR


@lru_cache(maxsize=32)
def _read(name: str) -> str:
    path: Path = PROMPTS_DIR / name
    return path.read_text(encoding="utf-8")


def render(name: str, **vars: str) -> str:
    """Substitute {{var}} placeholders in the named prompt file."""
    text = _read(name)
    for key, value in vars.items():
        text = text.replace("{{" + key + "}}", value)
    return text
