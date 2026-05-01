"""Evaluation harness — runs the pipeline against the labelled sample CSV and
reports per-column accuracy, latency, and cost.

Two-step:
  1. Run the pipeline on sample_support_tickets.csv (writes output.csv).
  2. Compare predictions to the sample's gold labels and emit an eval report.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

from hro.config import DEFAULT_CONCURRENCY, OUTPUT_CSV, SAMPLE_CSV
from hro.observability import latest_run_dir
from hro.runner import run_batch


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------


def run_eval(limit: int | None = None) -> None:
    """Run pipeline on sample CSV and emit eval_report.md."""
    run_batch(sample=True, limit=limit, concurrency=DEFAULT_CONCURRENCY)

    gold = pd.read_csv(SAMPLE_CSV, encoding="utf-8", keep_default_na=False)
    pred = pd.read_csv(OUTPUT_CSV, encoding="utf-8", keep_default_na=False)

    if limit:
        gold = gold.head(limit)

    metrics = _compute_metrics(gold, pred)
    report = _format_report(metrics, len(gold))

    run_dir = latest_run_dir()
    if run_dir:
        (run_dir / "eval_report.md").write_text(report, encoding="utf-8")
        print(f"\nwrote eval report to {run_dir / 'eval_report.md'}\n")

    print(report)


# ---------------------------------------------------------------------------
# Metric computation.
# ---------------------------------------------------------------------------


def _norm(s: str) -> str:
    return str(s).strip().lower()


def _compute_metrics(gold: pd.DataFrame, pred: pd.DataFrame) -> dict:
    """Return a dict of per-column accuracy and confusion matrices.

    Sample CSV columns: Issue, Subject, Company, Response, Product Area, Status, Request Type
    Output CSV columns: issue, subject, company, response, product_area, status, request_type, justification

    Match by row index (both files are in the same order).
    """
    n = min(len(gold), len(pred))

    status_correct = 0
    rtype_correct = 0
    parea_correct = 0
    replied_with_response = 0
    replied_total = 0

    status_confusion: Counter[tuple[str, str]] = Counter()
    rtype_confusion: Counter[tuple[str, str]] = Counter()

    rows: list[dict] = []
    for i in range(n):
        g = gold.iloc[i]
        p = pred.iloc[i]
        g_status = _norm(g.get("Status", ""))
        p_status = _norm(p.get("status", ""))
        g_rtype = _norm(g.get("Request Type", ""))
        p_rtype = _norm(p.get("request_type", ""))
        g_parea = _norm(g.get("Product Area", ""))
        p_parea = _norm(p.get("product_area", ""))

        if g_status and p_status and g_status == p_status:
            status_correct += 1
        if g_rtype and p_rtype and g_rtype == p_rtype:
            rtype_correct += 1
        if g_parea == p_parea:
            parea_correct += 1

        status_confusion[(g_status, p_status)] += 1
        rtype_confusion[(g_rtype, p_rtype)] += 1

        if p_status == "replied":
            replied_total += 1
            if str(p.get("response", "")).strip():
                replied_with_response += 1

        rows.append(
            {
                "idx": i,
                "issue": str(g.get("Issue", ""))[:80],
                "company": str(g.get("Company", "")),
                "status": (g_status, p_status, g_status == p_status),
                "rtype": (g_rtype, p_rtype, g_rtype == p_rtype),
                "parea": (g_parea, p_parea, g_parea == p_parea),
            }
        )

    return {
        "n": n,
        "status_acc": status_correct / n if n else 0.0,
        "rtype_acc": rtype_correct / n if n else 0.0,
        "parea_acc": parea_correct / n if n else 0.0,
        "replied_with_response_rate": (
            replied_with_response / replied_total if replied_total else 1.0
        ),
        "status_confusion": status_confusion,
        "rtype_confusion": rtype_confusion,
        "rows": rows,
    }


def _format_report(metrics: dict, n_gold: int) -> str:
    n = metrics["n"]
    lines: list[str] = [
        "# Eval report",
        "",
        f"- sample rows:    {n_gold}",
        f"- compared rows:  {n}",
        f"- **status accuracy:**       {metrics['status_acc']:.1%}",
        f"- **request_type accuracy:** {metrics['rtype_acc']:.1%}",
        f"- **product_area accuracy:** {metrics['parea_acc']:.1%}",
        f"- replied tickets with non-empty response: {metrics['replied_with_response_rate']:.1%}",
        "",
        "## Status confusion (gold -> pred)",
        "| gold | pred | count |",
        "|------|------|-------|",
        *[
            f"| {g or '(empty)'} | {p or '(empty)'} | {c} |"
            for (g, p), c in sorted(metrics["status_confusion"].items(), key=lambda x: -x[1])
        ],
        "",
        "## Request type confusion (gold -> pred)",
        "| gold | pred | count |",
        "|------|------|-------|",
        *[
            f"| {g or '(empty)'} | {p or '(empty)'} | {c} |"
            for (g, p), c in sorted(metrics["rtype_confusion"].items(), key=lambda x: -x[1])
        ],
        "",
        "## Per-row breakdown",
        "| idx | company | status (gold/pred) | rtype (gold/pred) | parea (gold/pred) |",
        "|-----|---------|--------------------|-------------------|-------------------|",
    ]
    for r in metrics["rows"]:
        s_mark = "OK" if r["status"][2] else "X"
        rt_mark = "OK" if r["rtype"][2] else "X"
        pa_mark = "OK" if r["parea"][2] else "X"
        lines.append(
            f"| {r['idx']} | {r['company']} | "
            f"{r['status'][0]}/{r['status'][1]} {s_mark} | "
            f"{r['rtype'][0]}/{r['rtype'][1]} {rt_mark} | "
            f"{r['parea'][0]}/{r['parea'][1]} {pa_mark} |"
        )

    return "\n".join(lines) + "\n"
