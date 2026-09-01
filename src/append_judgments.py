"""Append Track B judgments. Usage: python src/append_judgments.py <<'ROWS'
P0026|left|slight|reason text
ROWS
"""
import sys, csv, pathlib, pandas as pd
rows = [l.split("|") for l in sys.stdin.read().strip().split("\n") if l.strip()]
assert all(len(r) == 4 for r in rows), "each line needs pair_id|winner|confidence|reason"
p = pathlib.Path("outputs/trackB_judgments.csv")
existing = set(pd.read_csv(p).pair_id) if p.stat().st_size > 40 else set()
new = [r for r in rows if r[0] not in existing]
with p.open("a", newline="", encoding="utf-8") as f:
    csv.writer(f).writerows(new)
d = pd.read_csv(p)
print(f"+{len(new)} appended ({len(rows)-len(new)} dupes skipped) | total {len(d)}/900 "
      f"({100*len(d)/900:.0f}%) | L{(d.winner=='left').sum()} R{(d.winner=='right').sum()} "
      f"T{(d.winner=='tie').sum()}")
