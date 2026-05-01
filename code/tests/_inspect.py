"""One-off: inspect specific ticket outputs from the latest run trace."""

from __future__ import annotations

import json
import sys
from pathlib import Path

run_dir = sorted(p for p in Path("code/runs").iterdir() if p.is_dir())[-1]
trace = run_dir / "trace.jsonl"

want = {int(x) for x in sys.argv[1:]} if len(sys.argv) > 1 else None

for line in trace.read_text(encoding="utf-8").splitlines():
    obj = json.loads(line)
    idx = obj["ticket_idx"]
    if want and idx not in want:
        continue
    out = obj.get("output") or {}
    print(f"--- ticket {idx} ---")
    print(f"  issue: {obj.get('issue_preview','')[:120]}")
    print(f"  status: {out.get('status')}")
    print(f"  product_area: {out.get('product_area')}")
    print(f"  response:")
    print(f"    {out.get('response','')}")
    print(f"  justification:")
    print(f"    {out.get('justification','')}")
    print()
