"""One-off: count rows in input vs output CSVs and print issue previews."""

from __future__ import annotations

import pandas as pd

inp = pd.read_csv("support_tickets/support_tickets.csv", encoding="utf-8", keep_default_na=False)
out = pd.read_csv("support_tickets/output.csv", encoding="utf-8", keep_default_na=False)
print(f"input rows: {len(inp)}")
print(f"output rows: {len(out)}")
print()
print("input issues:")
for i, row in inp.iterrows():
    issue = row["Issue"][:80].replace("\n", " ")
    company = row["Company"]
    print(f"  {i:3d} [{company:11s}]: {issue!r}")
