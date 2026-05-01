"""Print a one-line-per-ticket trace summary for a given run dir."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main(run_dir_arg: str = "latest") -> None:
    if run_dir_arg == "latest":
        runs = sorted([p for p in Path("code/runs").iterdir() if p.is_dir()])
        run_dir = runs[-1]
    else:
        run_dir = Path(run_dir_arg)

    trace = run_dir / "trace.jsonl"
    print(f"# trace summary for {run_dir.name}\n")
    for line in trace.read_text(encoding="utf-8").splitlines():
        obj = json.loads(line)
        idx = obj["ticket_idx"]
        err = obj.get("error") or ""
        out = obj.get("output") or {}
        spans = obj.get("spans", [])
        parts: list[str] = []
        for s in spans:
            decision = s.get("decision") or ("ERR" if s.get("error") else "")
            parts.append(f"{s['node']}={decision}({s.get('duration_ms', 0)}ms)")
        spans_str = " > ".join(parts)
        print(
            f"#{idx} elapsed={obj.get('elapsed_s')}s status={out.get('status')} rtype={out.get('request_type')} parea={out.get('product_area')!r}"
        )
        print(f"   spans: {spans_str}")
        if err:
            print(f"   ERR: {err[:250]}")
        print()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "latest")
