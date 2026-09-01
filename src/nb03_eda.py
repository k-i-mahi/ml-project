# %% [markdown]
# # 03 — Exploratory Data Analysis
#
# Feature-type-appropriate analysis only. No decorative plots: every figure answers a question that
# changes a downstream decision. Outliers are **identified and classified**, never auto-removed.
#
# **Outputs:** `figures/*.png`, `outputs/eda_outliers.csv`, `outputs/eda_redundancy.csv`

# %%
import pathlib, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 60)

ROOT = pathlib.Path.cwd()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
OUT, FIG = ROOT / "outputs", ROOT / "figures"
FIG.mkdir(exist_ok=True)

df = pd.read_csv(OUT / "model_ready_dataset.csv")
G = pd.read_csv(OUT / "goodness_matrix.csv").drop(columns=["uni_id"])
META = json.load(open(OUT / "goodness_meta.json"))
BLOCK_MAP = json.load(open(OUT / "block_map.json"))
COL_BLOCK = {k: v for k, v in BLOCK_MAP["col_block"].items() if k in df.columns}
ATTR = list(COL_BLOCK)
BIN = [c for c in ATTR if set(pd.unique(df[c].dropna())) <= {0, 1}]

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 140, "font.size": 9,
                     "axes.grid": True, "grid.alpha": .25, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.bbox": "tight"})
C1, C2, C3 = "#2b6cb0", "#c05621", "#2f855a"
print(f"{df.shape[0]:,} universities x {df.shape[1]} columns | {len(BIN)} binary attributes")

def save(fig, name):
    fig.savefig(FIG / name)
    plt.close(fig)
    print(f"  saved figures/{name}")

# %% [markdown]
# ## 1. Missingness
#
# Only four columns carry meaningful missingness after cleaning. Each was assigned a semantic
# treatment in nb01/nb02 — none is filled with zero.

# %%
miss = (df.isna().mean() * 100).sort_values(ascending=False)
miss = miss[miss > 0]
print(miss.round(2).to_string())

fig, ax = plt.subplots(figsize=(7, 0.38 * len(miss) + 1.2))
ax.barh(miss.index[::-1], miss.values[::-1], color=C1)
for i, v in enumerate(miss.values[::-1]):
    ax.text(v + 0.6, i, f"{v:.1f}%", va="center", fontsize=8)
ax.set_xlabel("% missing")
ax.set_title("Missing values after cleaning\n(each treated semantically — never fillna(0))")
ax.set_xlim(0, max(miss.max() * 1.18, 5))
save(fig, "01_missingness.png")

# %% [markdown]
# ## 2. Binary attribute prevalence
#
# 55 of 67 attributes are presence flags. Prevalence determines how much a flag can possibly
# contribute: an attribute present on 99% of sites cannot separate them, no matter its weight.

# %%
prev = (df[BIN].mean() * 100).sort_values()
fig, ax = plt.subplots(figsize=(8, 0.20 * len(prev) + 1))
colors = [C2 if (v < 3 or v > 97) else C1 for v in prev.values]
ax.barh(prev.index, prev.values, color=colors, height=.75)
ax.axvline(50, color="grey", ls="--", lw=.8)
ax.set_xlabel("% of universities where the attribute is present")
ax.set_title("Binary attribute prevalence\n(orange = near-constant <3% or >97%: cannot discriminate)")
ax.tick_params(labelsize=6.5)
save(fig, "02_binary_prevalence.png")

print("Least common:"); print(prev.head(6).round(1).to_string())
print("\nMost common:"); print(prev.tail(6).round(1).to_string())
print(f"\nNear-constant (<3% or >97%): {int(((prev<3)|(prev>97)).sum())} attributes — these carry "
      "almost no ranking information regardless of the weight assigned to them.")

# %% [markdown]
# ## 3. Count attributes — zero inflation and censoring
#
# All four counts are zero-inflated, and three pile up on their extractor cap. Both facts break the
# assumption that "more is better", which is why nb02 applied non-linear curves.

# %%
counts = ["a03_nav_item_count", "a12_accred_count", "a15_stat_item_count", "a24_event_count",
          "a66_broken_links"]
caps = {"a03_nav_item_count": 20, "a15_stat_item_count": 10, "a24_event_count": 20}
fig, axes = plt.subplots(1, 5, figsize=(16, 2.9))
for ax, c in zip(axes, counts):
    s = df[c]
    top = s.quantile(.995) if c == "a66_broken_links" else s.max()
    ax.hist(s.clip(upper=top), bins=min(25, int(top) + 1), color=C1, edgecolor="white", lw=.4)
    ax.set_title(f"{c}\nzeros {100*(s==0).mean():.0f}%  skew {s.skew():.1f}", fontsize=8)
    if c in caps:
        ax.axvline(caps[c], color=C2, ls="--", lw=1.2)
        ax.text(caps[c], ax.get_ylim()[1]*.92, f" cap\n {100*(s==caps[c]).mean():.0f}%",
                color=C2, fontsize=7, va="top")
    ax.set_yscale("log"); ax.tick_params(labelsize=7)
fig.suptitle("Count attributes: zero-inflated and right-censored at the extractor cap (log y)", y=1.06)
save(fig, "03_count_distributions.png")

cs = pd.DataFrame({"pct_zero": [100*(df[c]==0).mean() for c in counts],
                   "skew": [df[c].skew() for c in counts],
                   "pct_at_cap": [100*(df[c]==caps[c]).mean() if c in caps else np.nan for c in counts],
                   "max": [df[c].max() for c in counts]}, index=counts)
print(cs.round(2).to_string())

# %% [markdown]
# ## 4. Continuous attributes
#
# `a53_contrast_ratio` and `a72_alt_text_pct` are the only genuinely continuous *website*
# measurements in the dataset. Both are strongly left-skewed toward the good end.

# %%
cont = ["a53_contrast_ratio", "a72_alt_text_pct", "load_time_s"]
fig, axes = plt.subplots(2, 3, figsize=(12, 5), height_ratios=[3, 1])
for j, c in enumerate(cont):
    s = df[c].dropna()
    axes[0, j].hist(s, bins=45, color=C1, edgecolor="white", lw=.3)
    axes[0, j].set_title(f"{c}\nmedian {s.median():.1f}   skew {s.skew():.2f}", fontsize=9)
    if c == "a53_contrast_ratio":
        for v, lab in [(4.5, "WCAG AA"), (7, "AAA")]:
            axes[0, j].axvline(v, color=C2, ls="--", lw=1)
            axes[0, j].text(v, axes[0, j].get_ylim()[1]*.95, f" {lab}", color=C2, fontsize=7, va="top")
    axes[1, j].boxplot(s, vert=False, widths=.6,
                       flierprops=dict(marker=".", markersize=3, alpha=.4))
    axes[1, j].set_yticks([])
fig.suptitle("Continuous attributes with distribution and outliers", y=1.0)
save(fig, "04_continuous_distributions.png")
print(df[cont].describe().T.round(2).to_string())
print(f"\na53 at the 21:1 ceiling: {100*(df.a53_contrast_ratio>=20.9).mean():.1f}% of sites — "
      "pure black on white. This is why the contrast curve plateaus at 7:1 rather than scaling to 21.")

# %% [markdown]
# ## 5. The confound, visualised
#
# This is the most important figure in the EDA. Each collector covered exactly one region, so the
# two explanations for this gradient — "these websites are slower" and "this collector had a slower
# connection" — cannot be separated by any statistical method.

# %%
order = df.groupby("region").load_time_s.median().sort_values().index.tolist()
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
data = [df.loc[df.region == r, "load_time_s"].values for r in order]
bp = axes[0].boxplot(data, vert=False, labels=[f"{r}\n({df[df.region==r].member.iloc[0]})" for r in order],
                     patch_artist=True, flierprops=dict(marker=".", markersize=3, alpha=.35))
for p in bp["boxes"]:
    p.set_facecolor(C1); p.set_alpha(.55)
axes[0].set_xlabel("load_time_s (raw)")
axes[0].set_title("RAW load time by region — each region = exactly one collector")
axes[0].tick_params(labelsize=7)

dz = [df.loc[df.region == r, "load_time_z_region"].values for r in order]
bp2 = axes[1].boxplot(dz, vert=False, labels=[""] * len(order), patch_artist=True,
                      flierprops=dict(marker=".", markersize=3, alpha=.35))
for p in bp2["boxes"]:
    p.set_facecolor(C3); p.set_alpha(.55)
axes[1].axvline(0, color="grey", ls="--", lw=.8)
axes[1].set_xlabel("load_time_z_region (region-standardised)")
axes[1].set_title("AFTER standardising — the between-region gradient is gone by construction")
save(fig, "05_loadtime_confound.png")

sm = df.groupby(["region", "member"]).load_time_s.agg(["median", "mean", "count"]).round(2)
print(sm.to_string())
print(f"\nMedian load time spans {df.groupby('region').load_time_s.median().max() - df.groupby('region').load_time_s.median().min():.2f}s "
      "across regions. UNRESOLVABLE: collector and region are perfectly confounded (audit F17).")

# %% [markdown]
# ## 6. Freshness
#
# 38% of sites show no dated notice at all, and 285 dated notices were impossible (after the crawl).
# Both facts are preserved as information rather than imputed away.

# %%
rec = df.notice_recency_days.dropna()
fig, axes = plt.subplots(1, 2, figsize=(12, 3.4))
axes[0].hist(rec.clip(upper=1095), bins=60, color=C1, edgecolor="white", lw=.3)
for v, lab in [(90, "90d"), (365, "1y")]:
    axes[0].axvline(v, color=C2, ls="--", lw=1)
    axes[0].text(v, axes[0].get_ylim()[1]*.95, f" {lab}", color=C2, fontsize=7, va="top")
axes[0].set_xlabel("days since most recent notice (clipped at 3y)")
axes[0].set_title("Notice recency where a date exists")

ev = df.notice_evidence.value_counts().sort_index()
labels = ["0 none", "1 undated\nboard", "2 dated", "3 dated\n<=90d"]
axes[1].bar(range(4), [ev.get(i, 0) for i in range(4)], color=[C2, "#d69e2e", C1, C3])
axes[1].set_xticks(range(4)); axes[1].set_xticklabels(labels, fontsize=8)
axes[1].set_title("notice_evidence — the reconciled freshness ordinal")
for i in range(4):
    axes[1].text(i, ev.get(i, 0) + 8, f"{ev.get(i,0)}\n{100*ev.get(i,0)/len(df):.0f}%",
                 ha="center", fontsize=7.5)
save(fig, "06_freshness.png")
print(f"No dated notice          : {int(df.a18_missing.sum())} ({100*df.a18_missing.mean():.1f}%)")
print(f"Impossible (future) dates: {int(df.notice_date_future.sum())} — censored to recency 0, flagged, kept")
print(f"Median recency where dated: {rec.median():.0f} days")

# %% [markdown]
# ## 7. Redundancy

# %%
corr = G.corr(method="spearman").abs()
np.fill_diagonal(corr.values, 0)
pairs = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool)).stack().sort_values(ascending=False)
red = pairs.head(15).reset_index()
red.columns = ["feature_a", "feature_b", "abs_spearman"]
red["flag"] = np.where(red.abs_spearman > .90, "REDUNDANT — keep one",
                       np.where(red.abs_spearman > .70, "related", "ok"))
print(red.round(3).to_string(index=False))
red.to_csv(OUT / "eda_redundancy.csv", index=False)
print(f"\nPairs above |rho|>0.90: {int((pairs>.90).sum())}   above 0.70: {int((pairs>.70).sum())}")

sel = [c for c in G.columns if G[c].std() > .05][:38]
cm = G[sel].corr(method="spearman")
fig, ax = plt.subplots(figsize=(11, 9.2))
im = ax.imshow(cm, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(sel))); ax.set_xticklabels(sel, rotation=90, fontsize=6)
ax.set_yticks(range(len(sel))); ax.set_yticklabels(sel, fontsize=6)
ax.set_title("Spearman correlation, informative features only\n"
             "(no dense block structure — the attributes are largely independent)")
fig.colorbar(im, shrink=.6)
ax.grid(False)
save(fig, "07_correlation_heatmap.png")

# %% [markdown]
# ## 8. Block scores by region
#
# Do the conceptual blocks separate universities, and does that separation look regional?

# %%
G_BLOCK = META["g_block"]
blocks = sorted(set(G_BLOCK.values()))
B = pd.DataFrame({b: G[[c for c in G.columns if G_BLOCK[c] == b]].mean(axis=1) for b in blocks})
B["region"] = df.region.values

fig, axes = plt.subplots(3, 4, figsize=(15, 8))
for ax, b in zip(axes.ravel(), blocks):
    d = [B.loc[B.region == r, b].values for r in order]
    bp = ax.boxplot(d, vert=True, labels=[r.split()[0][:7] for r in order], patch_artist=True,
                    flierprops=dict(marker=".", markersize=2, alpha=.3))
    for p in bp["boxes"]:
        p.set_facecolor(C1); p.set_alpha(.5)
    ax.set_title(f"{b}\nsd={B[b].std():.3f}", fontsize=8)
    ax.tick_params(labelsize=6, axis="x", rotation=45)
    ax.set_ylim(-.05, 1.05)
for ax in axes.ravel()[len(blocks):]:
    ax.axis("off")
fig.suptitle("Block goodness scores by region — narrow boxes mean the block cannot discriminate", y=1.0)
save(fig, "08_block_by_region.png")
print(B.drop(columns="region").describe().T[["mean", "std", "min", "max"]].round(3).to_string())

# %% [markdown]
# ## 9. Outliers — identified and classified, not removed

# %%
def flag_outliers(s, name, cls, note):
    q1, q3 = s.quantile([.25, .75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    m = (s < lo) | (s > hi)
    return dict(feature=name, n_outliers=int(m.sum()), pct=round(100 * m.mean(), 2),
                lower_fence=round(lo, 2), upper_fence=round(hi, 2),
                observed_max=round(s.max(), 2), classification=cls, action=note)

rows = [
    flag_outliers(df.a66_broken_links, "a66_broken_links", "MEASUREMENT ARTEFACT",
                  "One site reports 1,129 vs a second-highest of 59. Winsorised at p99 + log1p; row kept."),
    flag_outliers(df.load_time_s, "load_time_s", "INSTRUMENTATION",
                  "Confounded with collector; region-standardised. Extremes are network, not website."),
    flag_outliers(df.a53_contrast_ratio, "a53_contrast_ratio", "LEGITIMATE",
                  "Low ratios are genuinely inaccessible sites. Curve plateaus at 7:1; rows kept."),
    flag_outliers(df.a72_alt_text_pct, "a72_alt_text_pct", "LEGITIMATE",
                  "0% alt-text is a real accessibility failure, not an error. Rows kept."),
    flag_outliers(df.a24_event_count, "a24_event_count", "CENSORED",
                  "Pile-up at the cap of 20 means 'at least 20'. Inverted-U curve applied; rows kept."),
]
od = pd.DataFrame(rows)
print(od.to_string(index=False))
od.to_csv(OUT / "eda_outliers.csv", index=False)
print("\nNo rows were removed. Every extreme value is either a real website property, a known "
      "instrumentation artefact, or a documented censoring effect.")

# %% [markdown]
# ## 10. Equal-weight composite — a first look at the score landscape
#
# Shown only to check the score distribution is well spread and not degenerate. **This is not the
# ML target** — it is arithmetic on the features and is used purely as a baseline and as the
# stratifier for the Track B sample.

# %%
comp = B.drop(columns="region").mean(axis=1)
fig, axes = plt.subplots(1, 2, figsize=(12, 3.4))
axes[0].hist(comp * 100, bins=45, color=C1, edgecolor="white", lw=.3)
axes[0].set_xlabel("equal-block composite (0–100)")
axes[0].set_title(f"Composite distribution\nmean {comp.mean()*100:.1f}  sd {comp.std()*100:.1f}  "
                  f"skew {comp.skew():.2f}")
d = [comp[df.region.values == r].values * 100 for r in order]
bp = axes[1].boxplot(d, vert=False, labels=[r[:22] for r in order], patch_artist=True,
                     flierprops=dict(marker=".", markersize=3, alpha=.35))
for p in bp["boxes"]:
    p.set_facecolor(C3); p.set_alpha(.55)
axes[1].set_xlabel("equal-block composite (0–100)")
axes[1].set_title("By region")
axes[1].tick_params(labelsize=7)
save(fig, "09_composite_landscape.png")
print(f"Composite range {comp.min()*100:.1f}–{comp.max()*100:.1f}, sd {comp.std()*100:.1f} — "
      "well spread, so stratified sampling for Track B will get real coverage of bad/medium/good.")
print("\nBy region (mean):")
print((comp.groupby(df.region.values).mean() * 100).round(1).sort_values(ascending=False).to_string())

# %% [markdown]
# ## EDA findings that change downstream decisions
#
# 1. **Near-constant attributes cannot be fixed by weighting.** A flag present on 99% of sites
#    contributes nothing regardless of its nominal weight — this is why the variance audit precedes
#    the rubric.
# 2. **Every count is zero-inflated and three are capped.** Linear scaling of counts is invalid;
#    the non-linear curves are necessary, not stylistic.
# 3. **The load-time gradient is uninterpretable.** Region-standardisation is the only defensible
#    treatment short of re-crawling from a neutral host.
# 4. **Contrast and alt-text are the only rich continuous website measurements**, and both are
#    left-skewed toward good — so the accessibility block discriminates mainly at the bottom.
# 5. **No pair exceeds |rho| = 0.90** after cleaning, so no further redundancy pruning is required.
# 6. **The composite is well spread**, so the Track B stratified sample will cover the full range
#    rather than concentrating in the middle.

# %%
print("Figures written:")
for f in sorted(FIG.glob("*.png")):
    print(f"  figures/{f.name}  ({f.stat().st_size//1024} KB)")
