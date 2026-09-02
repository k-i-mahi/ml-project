# -*- coding: utf-8 -*-
"""
Head-to-head comparison of two universities, dimension by dimension.

    python src/make_comparison.py                       # BUET vs KUET
    python src/make_comparison.py "dhaka" "buet"        # any two, matched on name

Writes report/figures/fig_<a>_<b>.png.

Built for the obvious viva question: BUET is the most prestigious engineering university in
the country, so why does the ranking put it below KUET? The answer is entirely in the data
and this figure shows it: the two are identical on academic information and navigation, and
BUET loses 14.6 of the 15.4 point gap on admission support and currency alone.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA, FIG = ROOT / "data", ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 130, "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.25})

# Two categorical hues from the palette the rest of the report already uses.
# validate_palette.js "#2b6cb0,#d69e2e" --mode light -> all checks pass; the amber's
# contrast WARN is relieved by the direct value labels on every mark.
C_A, C_B = "#d69e2e", "#2b6cb0"
INK, MUTED = "#1a202c", "#4a5568"

DIMS = [
    ("D1_academic_information", "D₁  Academic information", 28),
    ("D2_admission_support", "D₂  Admission support", 22),
    ("D3_currency_activity", "D₃  Currency and activity", 15),
    ("D4_navigation_findability", "D₄  Navigation and findability", 15),
    ("D5_usability_accessibility", "D₅  Usability and accessibility", 10),
    ("D6_technical_quality", "D₆  Technical and discoverability", 7),
    ("D7_institutional_transparency", "D₇  Identity and transparency", 3),
]

SHORT = {
    "Bangladesh University of Engineering and Technology": "BUET",
    "Khulna University of Engineering and Technology": "KUET",
    "Rajshahi University of Engineering and Technology": "RUET",
    "Chittagong University of Engineering and Technology": "CUET",
    "American International University-Bangladesh": "AIUB",
    "Bangladesh University of Professionals": "BUP",
}


def pick(frame, needle):
    hit = frame[frame.name.str.contains(needle, case=False, na=False)]
    if hit.empty:
        raise SystemExit(f"no university matching {needle!r}")
    return hit.iloc[0]


def label(row):
    return SHORT.get(row["name"], row["name"])


a_q = sys.argv[1] if len(sys.argv) > 1 else "Bangladesh University of Engineering"
b_q = sys.argv[2] if len(sys.argv) > 2 else "Khulna University of Eng"

dims = pd.read_csv(DATA / "dimension_scores.csv")
full = pd.read_csv(DATA / "university_website_scores.csv")
dd = pd.read_csv(DATA / "data_dictionary.csv")

A, B = pick(dims, a_q), pick(dims, b_q)
fa, fb = pick(full, a_q), pick(full, b_q)
na, nb = label(A), label(B)

country = full[full.country == A.country].sort_values("website_score", ascending=False)
country_rank = {r.uni_id: i for i, r in enumerate(country.itertuples(), 1)}
ra, rb = country_rank[A.uni_id], country_rank[B.uni_id]

pts_a = np.array([A[f"pts_{k[:2]}"] for k, _, _ in DIMS])
pts_b = np.array([B[f"pts_{k[:2]}"] for k, _, _ in DIMS])
gap = pts_b - pts_a
total_gap = B.website_score - A.website_score

fig = plt.figure(figsize=(13.6, 7.4))
grid = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.28)
ax = fig.add_subplot(grid[0, 0])

# ---------------------------------------------------------------- dumbbell: points earned
y = np.arange(len(DIMS))[::-1]
for i, (yy, pa, pb, (_, name, w)) in enumerate(zip(y, pts_a, pts_b, DIMS)):
    ax.plot([pa, pb], [yy, yy], color="#cbd5e0", lw=2.5, zorder=1,
            solid_capstyle="round")
    ax.plot([0, min(pa, pb)], [yy, yy], color="#edf2f7", lw=2.5, zorder=0)
    ax.scatter([pa], [yy], s=95, color=C_A, zorder=3,
               edgecolor="white", linewidth=1.6)
    ax.scatter([pb], [yy], s=95, color=C_B, zorder=3,
               edgecolor="white", linewidth=1.6)
    # direct labels on every mark: relief for the amber contrast WARN
    if abs(pa - pb) < 0.005:
        ax.text(pa + 0.8, yy, f"{pa:.2f}   identical", va="center", fontsize=8, color=MUTED)
    elif abs(pa - pb) < 1.3:
        ax.text(max(pa, pb) + 0.8, yy, f"{na} {pa:.2f}  ·  {nb} {pb:.2f}",
                va="center", fontsize=8, color=MUTED)
    else:
        ax.text(pa + (-0.7 if pa > pb else 0.7), yy, f"{pa:.2f}", va="center",
                ha="right" if pa > pb else "left", fontsize=8, color=MUTED)
        ax.text(pb + (0.7 if pb > pa else -0.7), yy, f"{pb:.2f}", va="center",
                ha="left" if pb > pa else "right", fontsize=8, color=MUTED)
    if gap[i] >= 0.5:
        ax.text((pa + pb) / 2, yy + 0.30, f"−{gap[i]:.2f}", ha="center", fontsize=8.5,
                color="#c53030", fontweight="bold")

ax.set_yticks(y)
ax.set_yticklabels([f"{n}\n" + " " * 4 + f"worth {w} pts" for _, n, w in DIMS], fontsize=8.5)
ax.set_xlabel("points earned out of the dimension's weight")
ax.set_xlim(-0.5, 31)
ax.set_ylim(-0.75, len(DIMS) - 0.25)
ax.set_title(f"Where the {abs(total_gap):.2f}-point gap comes from", fontsize=11,
             fontweight="bold", color=INK, pad=12)
ax.grid(axis="y", visible=False)
ax.scatter([], [], s=95, color=C_A, edgecolor="white", linewidth=1.6,
           label=f"{na}   {A.website_score:.2f}  (grade {A.grade}, national #{ra})")
ax.scatter([], [], s=95, color=C_B, edgecolor="white", linewidth=1.6,
           label=f"{nb}   {B.website_score:.2f}  (grade {B.grade}, national #{rb})")
ax.legend(loc="lower right", fontsize=8.5, frameon=True, framealpha=.95)

# ---------------------------------------------------------------- the attribute evidence
ax2 = fig.add_subplot(grid[0, 1])
ax2.axis("off")

feats = dd[dd.role == "feature"]
scored, excluded = [], []
for _, r in feats.iterrows():
    c = r["column"]
    if c not in full.columns or fa[c] == fb[c]:
        continue
    dim = str(r["dimension"])
    fmt = (lambda v: f"{v:.0f}") if float(v_ := fa[c]) == int(v_) else (lambda v: f"{v:.3g}")
    entry = (dim, c, fmt(fa[c]), fmt(fb[c]))
    (excluded if dim.startswith("not used") else scored).append(entry)

top = sorted(scored, key=lambda e: e[0])
lines = [(f"{na} {A.website_score:.2f}  ·  {nb} {B.website_score:.2f}"
          f"  ·  gap {abs(total_gap):.2f} points", "head", INK)]
lines.append((f"Neither site is gated. They are identical on "
              f"{sum(1 for g in gap if abs(g) < 0.005)} of the 7 dimensions.", "body", MUTED))
lines.append(("", "gap", INK))
lines.append((f"The {len(top)} scored attributes they differ on", "sub", INK))
last_dim = None
for dim, c, va, vb in top:
    if dim != last_dim:
        lines.append((f"  {dim}", "dim", MUTED))
        last_dim = dim
    lines.append((f"      {c:<24} {na} {va:>7}   {nb} {vb:>7}", "mono", INK))

lines.append(("", "gap", INK))
lines.append((f"The {len(excluded)} attributes where the difference is NOT scored", "sub", INK))
for dim, c, va, vb in excluded:
    lines.append((f"      {c:<24} {na} {va:>7}   {nb} {vb:>7}", "mono", MUTED))

yy = 0.985
for text, kind, color in lines:
    if kind == "gap":
        yy -= 0.022
        continue
    size, weight, family = 8.2, "normal", "DejaVu Sans"
    if kind == "head":
        size, weight = 10.5, "bold"
    elif kind == "sub":
        size, weight = 9.2, "bold"
    elif kind == "dim":
        size = 8.2
    elif kind == "mono":
        family = "DejaVu Sans Mono"
        size = 7.6
    ax2.text(0.0, yy, text, transform=ax2.transAxes, fontsize=size, color=color,
             fontweight=weight, family=family, va="top", ha="left")
    yy -= 0.048 if kind == "head" else (0.044 if kind == "sub" else 0.0315)

ax2.set_title("What is actually different on the two landing pages",
              fontsize=11, fontweight="bold", color=INK, pad=12, loc="left")

fig.suptitle(f"Why {na} ranks below {nb}", fontsize=13.5, fontweight="bold",
             color=INK, x=0.008, ha="left", y=0.975)
fig.text(0.008, 0.012,
         f"The score measures what a prospective applicant can find on the landing page, "
         f"not institutional reputation. {na} carries the prestige markers "
         f"(ranking badge, QS badge)\nbut those are among the 20 attributes deliberately "
         f"excluded from the label as prestige leakage — a website is not better because "
         f"the university behind it is famous.",
         fontsize=8.4, color=MUTED, ha="left", va="bottom")

fig.subplots_adjust(left=0.125, right=0.985, top=0.885, bottom=0.115)
out = FIG / f"fig_{na.lower()}_{nb.lower()}.png"
plt.savefig(out, bbox_inches="tight")
plt.close()

print(f"{na} {A.website_score:.2f} (grade {A.grade}, national #{ra})")
print(f"{nb} {B.website_score:.2f} (grade {B.grade}, national #{rb})")
print(f"gap {abs(total_gap):.2f} points")
for (key, _, _), g in zip(DIMS, gap):
    if abs(g) >= 0.005:
        print(f"  {key:<32} {g:>+7.2f}")
print(f"wrote {out}")
