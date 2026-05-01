"""HRO Triage Agent — terminal entry point.

Subcommands:
  run            Run the full pipeline against support_tickets.csv (default).
  eval           Run against sample_support_tickets.csv and report accuracy.
  rebuild-index  Force-rebuild the local corpus index.
  trace          Inspect a previous run's trace and summary.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `python code/main.py ...` without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import click


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx: click.Context) -> None:
    """Multi-domain support triage agent."""
    if ctx.invoked_subcommand is None:
        ctx.invoke(run)


@cli.command()
@click.option("--sample", is_flag=True, help="Run against sample_support_tickets.csv instead.")
@click.option("--limit", type=int, default=None, help="Process only the first N rows.")
@click.option("--concurrency", type=int, default=1, show_default=True, help="Parallel ticket workers (raise once Gemini billing is enabled).")
@click.option("--rebuild-index", "rebuild_index", is_flag=True, help="Force rebuild of the corpus index.")
def run(sample: bool, limit: int | None, concurrency: int, rebuild_index: bool) -> None:
    """Run the agent on support tickets and write output.csv."""
    from hro.runner import run_batch

    run_batch(sample=sample, limit=limit, concurrency=concurrency, rebuild_index=rebuild_index)


@cli.command(name="eval")
@click.option("--limit", type=int, default=None, help="Evaluate only the first N rows.")
def eval_cmd(limit: int | None) -> None:
    """Run against the labelled sample CSV and report per-column accuracy."""
    from hro.eval import run_eval

    run_eval(limit=limit)


@cli.command(name="rebuild-index")
def rebuild_index_cmd() -> None:
    """Rebuild the local corpus index from data/."""
    from hro.index.build import rebuild

    rebuild(force=True)


@cli.command()
@click.option("--run", "run_id", default="latest", show_default=True, help="Which run to inspect.")
def trace(run_id: str) -> None:
    """Inspect a previous run's trace and summary."""
    from hro.observability import show_run

    show_run(run_id)


if __name__ == "__main__":
    cli()
