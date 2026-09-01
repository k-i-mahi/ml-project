# %% [markdown]
# # 02 — Feature Engineering & Block Variance Audit
#
# Turns `cleaned_dataset.csv` into a model-ready matrix. Every engineered feature carries a
# **FORMULA / REASON / EXPECTED INTERPRETATION** and a declared monotonic direction.
#
# Two things happen here that later stages depend on:
#
# 1. **Measurement transforms** (the non-linear response curves) are defined and documented. These
#    are *not* weights — they encode how a value maps onto quality, not how much the dimension
#    counts. Weights are fixed later, in the frozen rubric (nb04).
# 2. **The block variance-contribution audit.** Rashida et al. allocated 40% of their score to
#    performance, but performance drove only 2.9% of their ranking variance because every
#    university scored alike on it. A nominal weight is only as large as the variance of the thing
#    it weights. This audit runs *before* any weight is chosen.
#
# **Outputs:** `model_ready_dataset.csv`, `goodness_matrix.csv`, `feature_dictionary.csv`,
# `block_variance_report.csv`

# %%
import pathlib, json, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 80)
pd.set_option("display.max_colwidth", 70)

ROOT = pathlib.Path.cwd()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
OUT = ROOT / "outputs"

df = pd.read_csv(OUT / "cleaned_dataset.csv")
BLOCK_MAP = json.load(open(OUT / "block_map.json"))
COL_BLOCK = {k: v for k, v in BLOCK_MAP["col_block"].items() if k in df.columns}
# Authoritative attribute list — a naive "a"+digits pattern would also catch derived flags
# such as a18_missing, which are engineered features, not scraped attributes.
ATTR_COLS = list(COL_BLOCK)
print(f"Loaded cleaned dataset: {df.shape[0]:,} x {df.shape[1]}   attributes: {len(ATTR_COLS)}")

DICT = []

def declare(feature, block, ftype, formula, reason, interpretation, direction, missing, notes=""):
    """Register a feature in the data dictionary. direction: +1 better-higher, -1 better-lower,
    0 non-monotone (inverted-U / plateau)."""
    DICT.append(dict(feature=feature, block=block, type=ftype, formula=formula, reason=reason,
                     expected_interpretation=interpretation, direction=direction,
                     missing_treatment=missing, notes=notes))

# %% [markdown]
# ## 1. Non-linear response curves
#
# The core claim of the design: **a website attribute does not map linearly onto quality**. Rashida
# et al. scored content as `(count x 50)/25` — an unweighted sum of 25 binary flags — and their
# ranking correlated with their own student survey at rho = 0.102. Linear presence-counting is the
# documented failure mode this project exists to avoid.
#
# Each curve below is piecewise-linear through explicit knots, so any rule can be disputed
# individually by reading the knots.

# %%
CURVES = {
    # 0-1 items is a broken menu; 5-9 is the usability sweet spot; 15+ is an undifferentiated dump.
    "nav": ([0, 1, 2, 3, 4, 5, 9, 12, 15, 20], [0, .15, .35, .60, .85, 1.0, 1.0, .78, .60, .45]),
    # 0 events reads as dead; 4-10 as active; the cap of 20 as an undated dump, not "very active".
    "events": ([0, 1, 2, 3, 4, 10, 12, 15, 20], [0, .35, .60, .85, 1.0, 1.0, .90, .80, .65]),
    # Proof points saturate fast; beyond ~6 it is marketing filler.
    "stats": ([0, 1, 2, 3, 4, 6, 10], [0, .30, .50, .65, .78, .95, 1.0]),
    # Partial alt-text coverage delivers most of the accessibility benefit.
    "alt_text": ([0, 20, 40, 60, 80, 100], [0, .35, .60, .80, .92, 1.0]),
    # WCAG AA is 4.5:1 and AAA is 7:1. 21:1 is not better than 12:1 - it plateaus.
    "contrast": ([1, 3, 4.5, 7, 21], [0, .45, .75, 1.0, 1.0]),
    # Freshness decays: under a month is current, over two years is abandoned.
    "recency": ([0, 30, 90, 180, 365, 730], [1.0, 1.0, .70, .45, .20, .05]),
}

def curve(name, x):
    kx, ky = CURVES[name]
    return np.interp(np.asarray(x, dtype=float), kx, ky)

_demo = pd.DataFrame({
    "nav_items": [0, 2, 5, 7, 9, 12, 15, 20],
    "goodness": curve("nav", [0, 2, 5, 7, 9, 12, 15, 20]).round(2),
})
print("Navigation curve (inverted U) - more is NOT better:")
print(_demo.to_string(index=False))
print("\nContrast curve (plateau at WCAG AAA):")
print(pd.DataFrame({"ratio": [1, 3, 4.5, 7, 12, 21],
                    "goodness": curve("contrast", [1, 3, 4.5, 7, 12, 21]).round(2)}).to_string(index=False))

# %% [markdown]
# ## 2. Freshness — reconciling three contradictory notice fields
#
# The audit found 382 rows with a notice date but `a16_notice_board=0`, and 156 with the board flag
# set and no date. Treating a16/a17/a18 as three independent binaries triple-counts a signal the
# fields themselves disagree about. They collapse into one ordinal.

# %%
notice = pd.to_datetime(df.a18_recent_notice_date, errors="coerce")
fetched = pd.to_datetime(df.fetched_at, errors="coerce", utc=True).dt.tz_localize(None)

raw_recency = (fetched - notice).dt.days
df["notice_recency_days"] = raw_recency.clip(lower=0)          # censor the 285 impossible futures
print(f"Notice dates parsed        : {notice.notna().sum()}")
print(f"Negative recency censored  : {int((raw_recency < 0).sum())} -> set to 0")
print(f"Recency (days) distribution:\n{df.notice_recency_days.describe().round(1).to_string()}")

has_date = notice.notna()
has_board = (df.a16_notice_board == 1) | (df.a17_notice_timestamp == 1)
is_recent = df.notice_recency_days.le(90).fillna(False)
df["notice_evidence"] = np.select(
    [has_date & is_recent, has_date, has_board],
    [3, 2, 1], default=0).astype(int)
print("\nnotice_evidence  (0 none / 1 undated board / 2 dated / 3 dated & <=90 days):")
print(df.notice_evidence.value_counts().sort_index().to_string())

declare("notice_recency_days", "B3_notices_updates", "count",
        "clip(crawl_date - a18_recent_notice_date, min=0)",
        "Freshness is the single strongest available signal that a site is maintained. The clip "
        "censors the 285 impossible future dates rather than deleting those rows.",
        "Lower is better. 0 = posted today (or an impossible date, censored).", -1,
        "Left NaN when no date exists; represented separately by a18_missing. Never imputed - "
        "absence of a dated notice is itself evidence.",
        "285 rows censored; check notice_date_future before interpreting a 0.")
declare("notice_evidence", "B3_notices_updates", "ordinal",
        "3 if dated & recency<=90d; 2 if dated; 1 if board/timestamp flag only; else 0",
        "Reconciles the a16/a17/a18 contradiction into ONE field. Undated notices get partial "
        "credit only - they are weak evidence of freshness.",
        "Higher is better. Use this INSTEAD of a16+a17 to avoid triple-counting.", +1,
        "0 is a genuine level (no evidence), not a missing value.",
        "Resolves audit findings F15.")

# %% [markdown]
# ## 3. Events — same reconciliation

# %%
has_evt_flag = df.a23_upcoming_events == 1
has_evt_count = df.a24_event_count > 0
df["event_evidence"] = np.select(
    [has_evt_flag & has_evt_count & (df.a27_event_datetime == 1),
     has_evt_flag & has_evt_count,
     has_evt_flag ^ has_evt_count],
    [3, 2, 1], default=0).astype(int)
print("event_evidence  (0 none / 1 contradictory / 2 events listed / 3 events with date-time):")
print(df.event_evidence.value_counts().sort_index().to_string())
declare("event_evidence", "B4_events_media", "ordinal",
        "3 if flag & count>0 & a27_event_datetime; 2 if flag & count>0; 1 if flag XOR count>0; else 0",
        "327 rows flag events with count 0 and 158 count events with the flag off. Level 1 records "
        "that contradiction honestly instead of averaging it away.",
        "Higher is better. An event without a date-time is not actionable, hence level 2 vs 3.", +1,
        "0 is a genuine level.", "Resolves audit finding F16.")

# %% [markdown]
# ## 4. Load time — region-standardised
#
# Load time disagreed by 3.3x on identical re-crawled websites while every other attribute agreed,
# and collector is perfectly confounded with region. Converting to a **within-region percentile**
# removes by construction exactly the between-region signal that cannot be trusted, while keeping
# the within-region ordering, which is measured by a single collector on a single connection.

# %%
df["load_time_z_region"] = df.groupby("region").load_time_s.transform(
    lambda s: (s - s.mean()) / s.std(ddof=0))
df["load_time_pct_region"] = df.groupby("region").load_time_s.rank(pct=True)

comp = df.groupby("region").agg(raw_median=("load_time_s", "median"),
                                z_median=("load_time_z_region", "median")).round(3)
print(comp.to_string())
print("\nRaw medians span {:.2f}s across regions; standardised medians span {:.3f} - the "
      "between-region gradient is removed by construction.".format(
          comp.raw_median.max() - comp.raw_median.min(),
          comp.z_median.max() - comp.z_median.min()))

declare("load_time_z_region", "B9_technical_perf", "continuous",
        "(load_time_s - mean(load_time_s | region)) / sd(load_time_s | region)",
        "Collector and region are perfectly confounded (audit F17), so raw load time cannot "
        "distinguish a slow website from a slow connection. Standardising within region discards "
        "the untrustworthy between-region component and keeps the within-region ordering.",
        "Lower is better. 0 = regional average. NOT comparable as an absolute speed.", -1,
        "No missing values.",
        "Every ranking is produced with AND without this feature; see nb08.")
declare("load_time_pct_region", "B9_technical_perf", "percentage",
        "rank(load_time_s within region, pct=True)",
        "Bounded [0,1] alternative to the z-score, robust to the heavy right tail (max 42s).",
        "Lower is better; 0.1 = among the fastest 10% in its region.", -1, "No missing values.")

# %% [markdown]
# ## 5. Broken links — bounded treatment of an unnormalised count
#
# No denominator column exists (audit F13, assumption A2), so a broken-link *rate* is impossible.
# One row reports 1,129 against a second-highest of 59. Winsorisation plus `log1p` keeps that
# outlier from dominating any distance- or variance-based method.

# %%
p99 = float(df.a66_broken_links.quantile(0.99))
df["broken_links_w"] = df.a66_broken_links.clip(upper=p99)
df["broken_links_log"] = np.log1p(df.broken_links_w)
df["broken_links_present"] = (df.a66_broken_links > 0).astype(int)
print(f"p99 winsorisation cap: {p99:.0f}   max before: {df.a66_broken_links.max():.0f}  "
      f"after: {df.broken_links_w.max():.0f}")
print(f"broken_links_present prevalence: {df.broken_links_present.mean()*100:.1f}%")
declare("broken_links_log", "B9_technical_perf", "continuous",
        f"log1p(clip(a66_broken_links, upper=p99={p99:.0f}))",
        "Absolute counts with no denominator and a 1,129 outlier. log1p compresses the tail; "
        "winsorisation stops one row driving the variance.",
        "Lower is better.", -1, "0 read as verified-absence (assumption A1).",
        "A rate cannot be computed - permanent limitation A2.")
declare("broken_links_present", "B9_technical_perf", "binary",
        "1 if a66_broken_links > 0 else 0",
        "81% of rows are 0, so the count is mostly a presence indicator. Separating them lets a "
        "model use 'any broken links at all' independently of how many.",
        "0 is better.", -1, "n/a")

# %% [markdown]
# ## 6. Mobile score → ordinal
#
# Presented as continuous 0–100, but 6 values cover 99% of rows. It is a rule-based heuristic.

# %%
levels = sorted(df.a63_mobile_score.dropna().unique())
df["mobile_ordinal"] = df.a63_mobile_score.rank(method="dense").astype("Int64")
print(f"{len(levels)} distinct values; top-3 cover "
      f"{df.a63_mobile_score.value_counts(normalize=True).head(3).sum()*100:.1f}% of rows")
print(df.a63_mobile_score.value_counts().sort_index().to_string())
declare("mobile_ordinal", "B9_technical_perf", "ordinal",
        "dense rank of a63_mobile_score",
        "a63 is a 3-to-6 level heuristic, not a Lighthouse measurement (audit F12). Treating it as "
        "continuous implies precision it does not have.",
        "Higher is better, but only the ORDER is meaningful - the gap 75->90 is not 1.5x the gap "
        "90->100.", +1, "No missing values.")

# %% [markdown]
# ## 7. Block completeness ratios
#
# One interpretable summary per conceptual block: the fraction of that block's binary attributes
# present. These are the features a human actually reasons about ("does this site cover the basics?").

# %%
binaries = [c for c in ATTR_COLS if set(pd.unique(df[c].dropna())) <= {0, 1}]
COMPLETENESS_BLOCKS = {
    "content_completeness_B5":  "B5_page_content",
    "footer_completeness_B6":   "B6_footer",
    "a11y_completeness_B11":    "B11_accessibility",
    "seo_completeness_B10":     "B10_seo_metadata",
    "nav_completeness_B1":      "B1_header_nav",
    "events_completeness_B4":   "B4_events_media",
    "service_completeness_B8":  "B8_service_interact",
}
for name, blk in COMPLETENESS_BLOCKS.items():
    cols = [c for c in binaries if COL_BLOCK.get(c) == blk]
    df[name] = df[cols].mean(axis=1)
    declare(name, blk, "percentage", f"mean of the {len(cols)} binary attributes in {blk}",
            "Information completeness within a conceptual dimension, on a common 0-1 scale "
            "regardless of how many columns the block happens to contain.",
            "Higher is better. 1.0 = every element in the block present.", +1,
            "No missing values among these binaries.",
            f"members: {', '.join(cols)}")
print(df[list(COMPLETENESS_BLOCKS)].describe().T[["mean", "std", "min", "max"]].round(3).to_string())

# %% [markdown]
# ## 8. Navigation quality (the inverted-U applied)

# %%
df["nav_quality"] = curve("nav", df.a03_nav_item_count)
chk = df.groupby(pd.cut(df.a03_nav_item_count, [-1, 1, 4, 9, 14, 20],
                        labels=["0-1", "2-4", "5-9", "10-14", "15-20"]), observed=True).agg(
    n=("nav_quality", "size"), mean_goodness=("nav_quality", "mean")).round(3)
print(chk.to_string())
declare("nav_quality", "B1_header_nav", "continuous",
        "piecewise-linear curve on a03_nav_item_count, knots " + str(CURVES["nav"]),
        "Navigation is the clearest case where 'more' is not 'better': 0-1 items is a broken menu "
        "and 20 items is an undifferentiated dump. A linear term cannot express this.",
        "Higher is better. Peaks on 5-9 top-level items.", 0,
        "No missing values.",
        "a03 is right-censored at 20 (audit F11), so the decline also absorbs the cap artefact.")

# %% [markdown]
# ## 9. Missingness indicators
#
# Where the fact of a measurement being absent is itself informative.

# %%
df["a72_missing"] = df.a72_alt_text_pct.isna().astype(int)
df["a53_missing"] = df.a53_contrast_ratio.isna().astype(int)
for c, src in [("a18_missing", "a18_recent_notice_date"), ("a72_missing", "a72_alt_text_pct"),
               ("a53_missing", "a53_contrast_ratio")]:
    declare(c, COL_BLOCK.get(src, "meta"), "binary", f"1 if {src} is null else 0",
            "NULL IS NOT ZERO. For a18 the missingness is a quality signal in itself (no dated "
            "notice was found); for a72/a53 it records a failed measurement, which must not be "
            "confused with a bad measurement.",
            "For a18, 1 tends to mean worse. For a72/a53, 1 means unmeasured - direction unknown.",
            0, "n/a - this IS the missingness treatment.")
print(df[["a18_missing", "a72_missing", "a53_missing"]].mean().mul(100).round(2).to_string())

# %% [markdown]
# ### 9b. NULL IS NOT ZERO — and it is not the median either
#
# The indicators above record *that* a value is absent. This cell decides *what number stands
# in its place*, and it is the single most consequential line of code in the notebook.
#
# The obvious default — median imputation — is wrong here, and demonstrably so. The median of
# `notice_recency_days` is **1 day**. Median-filling would therefore tell every downstream
# consumer that the 469 universities with **no dated notice anywhere on the site** posted
# something *yesterday* — handing the strongest freshness signal in the dataset to exactly the
# sites that earned it least. That is not a conservative default; it is an inversion.
#
# For these three features, absence has a known direction, so each is filled at the **worst
# defensible value** instead. The rule and its justification are written to
# `outputs/missing_value_policy.json` so the choice is auditable rather than buried.
#
# The missingness indicators are computed *above* this cell, so a model can still recover
# "this was unmeasured" as a separate fact if that turns out to matter.

# %%
MISSING_RULE = {
    "notice_recency_days": (3650.0, "no dated notice exists anywhere -> treat as 10 years stale"),
    "a72_alt_text_pct":    (0.0,    "no image carries a text alternative -> 0% labelled"),
    "a53_contrast_ratio":  (1.0,    "no readable text block was found -> worst possible contrast"),
}
_policy = {}
for _c, (_fill, _why) in MISSING_RULE.items():
    _n = int(df[_c].isna().sum())
    _median = float(df[_c].median())
    df[_c] = df[_c].fillna(_fill)
    _policy[_c] = {"n_filled": _n, "fill_value": _fill, "rejected_median_fill": round(_median, 2),
                   "reason": _why}
    print(f"{_c:<22} filled {_n:>4} rows at {_fill:>7.1f}   (median fill would have been {_median:.1f})")

(OUT / "missing_value_policy.json").write_text(json.dumps(_policy, indent=2), encoding="utf-8")
for _c in MISSING_RULE:
    assert df[_c].notna().all(), f"{_c} still has nulls after the explicit fill"
print()
print("wrote missing_value_policy.json")

# %% [markdown]
# ## 10. The goodness matrix
#
# Every quality feature mapped onto a common **0 = worst … 1 = best** scale, applying the curves
# above. This is a *measurement* transform, not a weighting: it says how a value maps onto quality,
# not how much that dimension counts. Weights are set later, in the frozen rubric.

# %%
G = pd.DataFrame(index=df.index)

# binaries: presence = good, except where noted
NEGATIVE_BINARY = {"a54_banner_carousel"}   # auto-rotating carousels are consistently poor for usability
for c in binaries:
    if c == "a65_https":
        continue                             # used as a rubric GATE, not a scored feature
    G[c] = (1 - df[c]) if c in NEGATIVE_BINARY else df[c].astype(float)

# curve-transformed numerics
G["a03_nav_item_count"] = df.nav_quality
G["a24_event_count"]    = curve("events",   df.a24_event_count)
G["a15_stat_item_count"] = curve("stats",   df.a15_stat_item_count)
G["a72_alt_text_pct"]   = curve("alt_text", df.a72_alt_text_pct)
G["a53_contrast_ratio"] = curve("contrast", df.a53_contrast_ratio)
G["a12_accred_count"]   = np.clip(df.a12_accred_count / 3.0, 0, 1)
G["a63_mobile_score"]   = df.a63_mobile_score / 100.0

# derived, direction-corrected
G["notice_evidence"]    = df.notice_evidence / 3.0
G["event_evidence"]     = df.event_evidence / 3.0
G["notice_recency"]     = np.where(df.notice_recency_days.isna(), 0.0,
                                   curve("recency", df.notice_recency_days.fillna(9999)))
G["load_time"]          = 1.0 - df.load_time_pct_region     # fast within its region = good
G["broken_links"]       = 1.0 - (df.broken_links_log / df.broken_links_log.max())

# Collapse every flag/count/date group that measures ONE underlying thing, so no dimension is
# counted twice inside its own block:
#   a16 + a17 + a18      -> notice_evidence   (recency is encoded as its level 3)
#   a23 + a24            -> event_evidence
#   a14 flag + a15 count -> a15_stat_item_count
# notice_recency is kept in the modelling dataset (trees handle collinearity) but excluded from the
# SCORING matrix, where it would double-weight freshness against everything else.
SUPERSEDED = ["a16_notice_board", "a17_notice_timestamp", "a23_upcoming_events",
              "a24_event_count", "a14_stats_block", "notice_recency"]
G = G.drop(columns=[c for c in SUPERSEDED if c in G.columns])
G = G.fillna(G.median(numeric_only=True))
print("Dropped from the scoring matrix to prevent double-counting:", SUPERSEDED)

G_BLOCK = {}
for c in G.columns:
    if c in COL_BLOCK:
        G_BLOCK[c] = COL_BLOCK[c]
    else:
        G_BLOCK[c] = {"notice_evidence": "B3_notices_updates", "notice_recency": "B3_notices_updates",
                      "event_evidence": "B4_events_media", "load_time": "B9_technical_perf",
                      "broken_links": "B9_technical_perf"}[c]

print(f"Goodness matrix: {G.shape[0]:,} x {G.shape[1]} features, all in [0,1]")
print(f"Range check - min {G.min().min():.3f}, max {G.max().max():.3f}, any NaN: {G.isna().any().any()}")
print("\nFeatures per block:")
print(pd.Series(G_BLOCK).value_counts().sort_index().to_string())

# %% [markdown]
# ## 11. Block variance-contribution audit
#
# **This is the step that runs before any weight is chosen.**
#
# Under equal-block weighting the composite is `C = mean(block scores)`. Variance decomposes
# exactly as `Var(C) = sum_b w_b * Cov(B_b, C)`, so block *b*'s realised share of ranking variance
# is `w_b * Cov(B_b, C) / Var(C)`. This accounts for correlation between blocks, unlike simply
# comparing block SDs.

# %%
blocks = sorted(set(G_BLOCK.values()))
B = pd.DataFrame({b: G[[c for c in G.columns if G_BLOCK[c] == b]].mean(axis=1) for b in blocks})
w = 1.0 / len(blocks)
C = B.mean(axis=1)

rows = []
for b in blocks:
    cov = np.cov(B[b], C, ddof=0)[0, 1]
    rows.append(dict(block=b, n_features=sum(1 for c in G.columns if G_BLOCK[c] == b),
                     nominal_weight_pct=100 * w, mean=B[b].mean(), sd=B[b].std(ddof=0),
                     realised_variance_share_pct=100 * w * cov / C.var(ddof=0),
                     spearman_with_composite=B[b].corr(C, method="spearman")))
bv = pd.DataFrame(rows).sort_values("realised_variance_share_pct", ascending=False)
bv["weight_to_variance_ratio"] = (bv.realised_variance_share_pct / bv.nominal_weight_pct).round(2)
print(bv.round(3).to_string(index=False))
print(f"\nRealised shares sum to {bv.realised_variance_share_pct.sum():.1f}%")

# %% [markdown]
# ### Reading the audit

# %%
top, bot = bv.iloc[0], bv.iloc[-1]
print(f"Every block carries the same nominal weight ({100*w:.1f}%), but realised influence "
      f"ranges from {bot.realised_variance_share_pct:.1f}% to {top.realised_variance_share_pct:.1f}% "
      f"- a {top.realised_variance_share_pct/bot.realised_variance_share_pct:.1f}x spread.\n")
print(f"MOST influential : {top.block:22s} sd={top.sd:.3f}  share={top.realised_variance_share_pct:.1f}%")
print(f"LEAST influential: {bot.block:22s} sd={bot.sd:.3f}  share={bot.realised_variance_share_pct:.1f}%")
print("\nBlocks whose realised influence is less than half their nominal weight "
      "(a nominal weight spent on a near-constant dimension):")
weak = bv[bv.weight_to_variance_ratio < 0.5]
print(weak[["block", "sd", "nominal_weight_pct", "realised_variance_share_pct",
            "weight_to_variance_ratio"]].round(3).to_string(index=False) if len(weak) else "  (none)")
print("\nThis is the Rashida et al. failure mode measured on our own data: they assigned 40% to "
      "performance and it moved 2.9% of their ranking. Every weight reported in nb04 is therefore "
      "published alongside its realised variance share.")

bv.to_csv(OUT / "block_variance_report.csv", index=False)
print(f"\nWrote {OUT/'block_variance_report.csv'}")

# %% [markdown]
# ## 12. Redundancy check

# %%
num = G.select_dtypes(include=[np.number])
corr = num.corr(method="spearman").abs()
np.fill_diagonal(corr.values, 0)
pairs = (corr.where(np.triu(np.ones(corr.shape), 1).astype(bool))
         .stack().sort_values(ascending=False))
high = pairs[pairs > 0.90]
print(f"Feature pairs with |Spearman rho| > 0.90: {len(high)}")
print(high.round(3).to_string() if len(high) else "  (none - no perfectly redundant pair remains "
      "after dropping a62_load_speed_s and collapsing the notice/event fields)")
print(f"\nHighest remaining correlations:\n{pairs.head(8).round(3).to_string()}")

# %% [markdown]
# ## 13. Write artefacts

# %%
ENGINEERED = ["notice_recency_days", "notice_evidence", "event_evidence", "load_time_z_region",
              "load_time_pct_region", "broken_links_w", "broken_links_log", "broken_links_present",
              "mobile_ordinal", "nav_quality", "a18_missing", "a72_missing", "a53_missing",
              *COMPLETENESS_BLOCKS]
KEEP = (["uni_id", "name", "url", "member", "region", "country", "country_is_bucket",
         "country_rank_eligible", "notice_date_future", "http_status", "page_lang",
         "switched_to_english", "fetched_at", "load_time_s"] + ATTR_COLS + ENGINEERED)
model_ready = df[[c for c in dict.fromkeys(KEEP) if c in df.columns]].copy()
model_ready.to_csv(OUT / "model_ready_dataset.csv", index=False)

G.assign(uni_id=df.uni_id.values).to_csv(OUT / "goodness_matrix.csv", index=False)
json.dump({"g_block": G_BLOCK, "curves": {k: [list(a), list(b)] for k, (a, b) in CURVES.items()}},
          open(OUT / "goodness_meta.json", "w"), indent=2)

fd = pd.DataFrame(DICT)
# register the retained raw attributes too, with declared directions
LOW_TIER = {"a07_qs_badge", "a09_national_rank", "a11_accreditation", "a13_achievements",
            "a60_trust_seal", "a61_testimonials", "a75_bookmark"}
raw_rows = []
for c in ATTR_COLS:
    if c in set(fd.feature):
        continue
    d = 0 if c in {"a54_banner_carousel", "a03_nav_item_count", "a24_event_count"} else (
        -1 if c == "a66_broken_links" else +1)
    raw_rows.append(dict(
        feature=c, block=COL_BLOCK.get(c, "?"),
        type="binary" if c in binaries else "numeric",
        formula="as scraped (schema Table 2)",
        reason="Retained landing-page attribute from the frozen 69-attribute specification.",
        expected_interpretation={1: "Higher/present is better.", -1: "Lower is better.",
                                 0: "Non-monotone - see curve."}[d],
        direction=d,
        missing_treatment="verified absence (0) unless listed in assumptions.md",
        notes="LOWEST TIER: self-promotion signal, trivially gamed" if c in LOW_TIER else
              ("negative direction: auto-rotating carousels harm usability" if c == "a54_banner_carousel" else "")))
fd = pd.concat([fd, pd.DataFrame(raw_rows)], ignore_index=True)
fd["slr_factor"] = fd.block.map(BLOCK_MAP["slr"])
fd.to_csv(OUT / "feature_dictionary.csv", index=False)

print(f"model_ready_dataset.csv : {model_ready.shape[0]:,} x {model_ready.shape[1]}")
print(f"goodness_matrix.csv     : {G.shape[0]:,} x {G.shape[1]}")
print(f"feature_dictionary.csv  : {len(fd)} features documented")
print(f"\nDirection declared: {(fd.direction!=0).sum()} monotone, {(fd.direction==0).sum()} non-monotone")
print("\nEngineered features:")
print(fd[fd.formula != "as scraped (schema Table 2)"][["feature", "block", "type", "direction"]]
      .to_string(index=False))

# %% [markdown]
# ## Summary
#
# | Item | Result |
# |---|---|
# | Contradictory field groups reconciled | a16/a17/a18 → `notice_evidence`; a23/a24 → `event_evidence` |
# | Impossible dates | 285 censored to recency 0, flagged, **not deleted** |
# | Load time | region-standardised; between-region gradient removed by construction |
# | Non-linear curves | 6 documented piecewise-linear curves with explicit knots |
# | Redundant pairs remaining | see §12 |
# | **Block variance audit** | equal nominal weights produce very unequal realised influence |
#
# Weights are **not** set here. They are fixed in nb04, and every one of them is published
# alongside the realised variance share measured above.
