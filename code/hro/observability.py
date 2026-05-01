"""Span recording, trace JSONL writer, and run-summary emitter.

Each ticket gets a span tree. `code/runs/<utc-ts>/trace.jsonl` is one JSON
object per ticket; `summary.md` aggregates the run.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from hro.config import RUNS_DIR
from hro.schemas import Span


def new_run_dir() -> Path:
    """Create code/runs/<utc-timestamp>/ and return the path."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = RUNS_DIR / ts
    path.mkdir(parents=True, exist_ok=True)
    return path


def latest_run_dir() -> Path | None:
    """Most recent run directory, or None if none exist."""
    if not RUNS_DIR.exists():
        return None
    candidates = sorted([p for p in RUNS_DIR.iterdir() if p.is_dir()])
    return candidates[-1] if candidates else None


@contextmanager
def span(node: str, spans: list[Span]) -> Iterator[Span]:
    """Time a node's execution and append a Span to `spans`."""
    started = int(time.time() * 1000)
    s: Span = {"node": node, "started_ms": started}
    try:
        yield s
    except Exception as e:
        s["error"] = f"{type(e).__name__}: {e}"
        raise
    finally:
        s["duration_ms"] = int(time.time() * 1000) - started
        spans.append(s)


def write_trace(run_dir: Path, ticket_idx: int, payload: dict) -> None:
    """Append one ticket's record to trace.jsonl."""
    path = run_dir / "trace.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ticket_idx": ticket_idx, **payload}, ensure_ascii=False) + "\n")


def show_run(run_id: str) -> None:
    """Print a previous run's summary.md to stdout, or its trace if no summary exists."""
    if run_id == "latest":
        run_dir = latest_run_dir()
        if run_dir is None:
            print("No runs found under code/runs/.")
            return
    else:
        run_dir = RUNS_DIR / run_id
        if not run_dir.exists():
            print(f"Run not found: {run_dir}")
            return

    summary = run_dir / "summary.md"
    if summary.exists():
        print(summary.read_text(encoding="utf-8"))
    else:
        trace = run_dir / "trace.jsonl"
        if trace.exists():
            print(f"(no summary yet; first 5 trace lines from {trace})")
            with trace.open(encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= 5:
                        break
                    print(line.rstrip())
        else:
            print(f"Run is empty: {run_dir}")
