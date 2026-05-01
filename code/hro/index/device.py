"""Device detection for torch-backed models (embedder + reranker).

Single source of truth so build-time and run-time paths agree on which device
to use. Honors the `HRO_DEVICE` env var if set (one of `cuda`, `mps`, `cpu`).
"""

from __future__ import annotations

import os
from functools import lru_cache


@lru_cache(maxsize=1)
def best_device() -> str:
    """Return `cuda`, `mps`, or `cpu` based on what's available."""
    forced = os.environ.get("HRO_DEVICE", "").strip().lower()
    if forced in {"cuda", "mps", "cpu"}:
        return forced
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
