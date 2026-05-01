"""Concurrent batch runner — reads CSV, dispatches tickets through the graph,
writes output.csv, emits a span trace and a run summary.

Concurrency is bounded by an asyncio.Semaphore. Default is 1 (Gemini free
tier); raise once billing is on.
"""

from __future__ import annotations

import asyncio
import csv
import os
import time
from collections import Counter
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from hro.config import (
    DEFAULT_CONCURRENCY,
    INPUT_CSV,
    LLM_BACKEND,
    OUTPUT_COLUMNS,
    OUTPUT_CSV,
    SAMPLE_CSV,
)
from hro.graph import build_graph
from hro.observability import new_run_dir, write_trace
from hro.schemas import Span, TicketInput, TriageOutput


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------


def run_batch(
    sample: bool = False,
    limit: int | None = None,
    concurrency: int = DEFAULT_CONCURRENCY,
    rebuild_index: bool = False,
) -> None:
    """Process tickets and write output.csv."""
    if rebuild_index:
        from hro.index.build import rebuild

        rebuild(force=True)

    input_path = SAMPLE_CSV if sample else INPUT_CSV
    if not input_path.exists():
        raise FileNotFoundError(f"input CSV missing: {input_path}")

    df = pd.read_csv(input_path, encoding="utf-8", keep_default_na=False)
    if limit:
        df = df.head(limit)

    tickets = _build_tickets(df)
    print(f"loaded {len(tickets)} tickets from {input_path.name}")
    _anth = bool(os.environ.get("ANTHROPIC_API_KEY"))
    _gem = bool(os.environ.get("GEMINI_API_KEY"))
    print(
        f"LLM backend={LLM_BACKEND!r} "
        f"(ANTHROPIC_API_KEY loaded={_anth}, GEMINI_API_KEY loaded={_gem})"
    )

    asyncio.run(_run_async(tickets, df, OUTPUT_CSV, concurrency))


# ---------------------------------------------------------------------------
# Internals.
# ---------------------------------------------------------------------------


def _build_tickets(df: pd.DataFrame) -> list[TicketInput]:
    tickets: list[TicketInput] = []
    for idx, row in df.iterrows():
        company_raw = str(row.get("Company", "")).strip()
        # Normalize "None " (with trailing space) to "None".
        if company_raw.lower() in ("", "none"):
            company = "None"
        elif company_raw.lower() == "hackerrank":
            company = "HackerRank"
        elif company_raw.lower() == "claude":
            company = "Claude"
        elif company_raw.lower() == "visa":
            company = "Visa"
        else:
            company = "None"
        tickets.append(
            TicketInput(
                idx=int(idx),
                issue=str(row.get("Issue", "")),
                subject=str(row.get("Subject", "")),
                company=company,
            )
        )
    return tickets


async def _run_async(
    tickets: list[TicketInput],
    df: pd.DataFrame,
    output_path: Path,
    concurrency: int,
) -> None:
    run_dir = new_run_dir()
    print(f"run dir: {run_dir}")

    graph = build_graph()

    sem = asyncio.Semaphore(concurrency)
    progress = tqdm(total=len(tickets), desc="tickets", ncols=80)

    results: dict[int, tuple[TriageOutput | None, list[Span], str | None, float]] = {}

    async def worker(ticket: TicketInput) -> None:
        t0 = time.time()
        async with sem:
            try:
                state = {"ticket": ticket, "spans": []}
                final = await graph.ainvoke(state)
                output = final.get("output")
                spans = final.get("spans", [])
                err = final.get("error")
            except Exception as e:
                output = None
                spans = []
                err = f"{type(e).__name__}: {e}"
        elapsed = time.time() - t0
        results[ticket.idx] = (output, spans, err, elapsed)
        # Trace as soon as the ticket finishes — durable record even on crash.
        write_trace(
            run_dir,
            ticket.idx,
            {
                "company_input": ticket.company,
                "subject": ticket.subject[:200],
                "issue_preview": ticket.issue[:300],
                "elapsed_s": round(elapsed, 2),
                "spans": spans,
                "output": output.model_dump() if output else None,
                "error": err,
            },
        )
        progress.update(1)

    await asyncio.gather(*(worker(t) for t in tickets))
    progress.close()

    _write_output_csv(df, results, output_path)
    _write_summary(run_dir, tickets, results)


def _write_output_csv(
    df: pd.DataFrame,
    results: dict[int, tuple[TriageOutput | None, list[Span], str | None, float]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(OUTPUT_COLUMNS)
        for idx, row in df.iterrows():
            res = results.get(int(idx))
            if res is None:
                output, err = None, "no_result"
            else:
                output, _, err, _ = res

            issue = str(row.get("Issue", ""))
            subject = str(row.get("Subject", ""))
            company = str(row.get("Company", ""))

            if output is not None:
                writer.writerow(
                    [
                        issue,
                        subject,
                        company,
                        output.response,
                        output.product_area,
                        output.status,
                        output.request_type,
                        output.justification,
                    ]
                )
            else:
                writer.writerow(
                    [
                        issue,
                        subject,
                        company,
                        "I'm routing this to a human agent — the pipeline could not produce a verified answer.",
                        "",
                        "escalated",
                        "invalid",
                        f"Pipeline error: {err}" if err else "Pipeline produced no output.",
                    ]
                )
    print(f"wrote {output_path} ({len(df)} rows)")


def _write_summary(
    run_dir: Path,
    tickets: list[TicketInput],
    results: dict[int, tuple[TriageOutput | None, list[Span], str | None, float]],
) -> None:
    n = len(tickets)
    statuses: Counter[str] = Counter()
    request_types: Counter[str] = Counter()
    product_areas: Counter[str] = Counter()
    cited_paths: Counter[str] = Counter()
    failed = 0
    elapsed_total = 0.0
    elapsed_per: list[float] = []

    for ticket in tickets:
        res = results.get(ticket.idx)
        if res is None:
            failed += 1
            continue
        output, spans, err, elapsed = res
        elapsed_per.append(elapsed)
        elapsed_total += elapsed
        if err or output is None:
            failed += 1
            continue
        statuses[output.status] += 1
        request_types[output.request_type] += 1
        if output.product_area:
            product_areas[output.product_area] += 1
        # Top-cited paths come from spans? Not directly; we need to inspect the
        # trace.jsonl for that. For the summary we'll keep this simple.

    elapsed_per.sort()
    p50 = elapsed_per[len(elapsed_per) // 2] if elapsed_per else 0.0
    p95 = elapsed_per[int(len(elapsed_per) * 0.95)] if len(elapsed_per) > 1 else (elapsed_per[0] if elapsed_per else 0.0)

    lines = [
        f"# Run summary",
        f"",
        f"- tickets:        {n}",
        f"- pipeline errors: {failed}",
        f"- elapsed total:  {elapsed_total:.1f}s",
        f"- elapsed p50:    {p50:.1f}s",
        f"- elapsed p95:    {p95:.1f}s",
        f"",
        f"## Status",
        *[f"- {k}: {v}" for k, v in statuses.most_common()],
        f"",
        f"## Request type",
        *[f"- {k}: {v}" for k, v in request_types.most_common()],
        f"",
        f"## Top product_areas",
        *[f"- {k}: {v}" for k, v in product_areas.most_common(15)],
        f"",
        f"_See {run_dir.name}/trace.jsonl for per-ticket spans and outputs._",
    ]
    summary = "\n".join(lines) + "\n"
    (run_dir / "summary.md").write_text(summary, encoding="utf-8")
    print()
    print(summary)
