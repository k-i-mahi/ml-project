import sys, pandas as pd
from pathlib import Path
p = Path("outputs/trackB_judgments_v2.csv")
rows = [l.split("|") for l in sys.stdin.read().strip().split("\n") if l.strip()]
new = pd.DataFrame(rows, columns=["pair_id","winner","confidence","reason"])
existing = set(pd.read_csv(p).pair_id) if p.stat().st_size > 40 else set()
new = new[~new.pair_id.isin(existing)]
new.to_csv(p, mode="a", header=False, index=False)
d = pd.read_csv(p)
L,R,T = (d.winner=="left").sum(), (d.winner=="right").sum(), (d.winner=="tie").sum()
print(f"+{len(new)} appended ({len(rows)-len(new)} dupes skipped) | total {len(d)}/900 ({len(d)/9:.0f}%) | L{L} R{R} T{T}")
