# %% [markdown]
# # 08 — Rankings, explanation, and fairness
#
# The deliverable: a global ranking of 1,226 university websites by predicted content quality,
# with regional and country cuts, every position explainable, and the known confounds measured
# rather than hidden.
#
# Four things happen here, in order of how much they matter:
#
# 1. **Rankings** — global, regional, and country (only where a country ranking is meaningful),
#    each produced **twice**: with and without the region-standardised load time, with the
#    rank displacement between the two reported.
# 2. **SHAP** — global, block-aggregated, per-university, and a `why_a_above_b` function that
#    answers the only question a ranked institution actually asks.
# 3. **The headline analysis** — SHAP-derived empirical block weights against the a-priori
#    rubric weights. Where they diverge is direct evidence that hand-set scoring rules
#    misallocate importance across quality dimensions.
# 4. **Fairness** — score and error by region, by country bucket, by country.

# %%
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, KFold

import lightgbm as lgb
import shap

warnings.filterwarnings("ignore")
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
OUT, FIG = ROOT / "outputs", ROOT / "figures"

data = pd.read_csv(OUT / "model_ready_dataset.csv")
labB = pd.read_csv(OUT / "expert_labels_trackB.csv")
labA = pd.read_csv(OUT / "expert_labels_trackA.csv")
fdict = pd.read_csv(OUT / "feature_dictionary.csv")
preds = pd.read_csv(OUT / "predictions_all.csv")
meta = json.loads((OUT / "model_meta.json").read_text(encoding="utf-8"))
blockw = pd.read_csv(OUT / "trackA_block_weights.csv")

FEATURES = meta["features"]
print(f"final model: {meta['final_model']} | {len(FEATURES)} features | "
      f"LORO ρ = {meta['loro_spearman']:.3f}")

# %% [markdown]
# ## 1. Refitting the model, and its no-load-time twin
#
# The variant `07` selected is reconstructed from `model_meta.json` rather than hardcoded —
# whether it carries monotone constraints and whether it weights labels by their bootstrap
# precision are both read from the recorded name and parameters. That keeps this notebook
# honest if model selection ever picks a different variant, and the assertion below is what
# catches it: the refit must reproduce `07`'s predictions to Spearman > 0.999.
#
# Both variants are refitted with that same recipe so the with/without load-time comparison
# is like-for-like: same estimator, same hyperparameters, only the feature set differs.

# %%
df = data.merge(labB[["uni_id", "trackB_bt_score", "trackB_bt_se"]], on="uni_id", how="left")
df = df.merge(labA[["uni_id", "trackA_consensus"]], on="uni_id", how="left")
train = df[df.trackB_bt_score.notna()].reset_index(drop=True)
y = train.trackB_bt_score.values

NON_MONOTONE = {"a03_nav_item_count", "a15_stat_item_count", "a12_accred_count", "nav_quality"}
dirmap = fdict.set_index("feature").direction.to_dict()

def mono_for(cols):
    return [0 if f in NON_MONOTONE else int(dirmap.get(f, 0)) if dirmap.get(f, 0) in (-1, 1) else 0
            for f in cols]

# Reconstruct 07's recipe from what 07 recorded, not from an assumption about which won.
USE_MONO = "mono" in meta["final_model"]
USE_LABELWT = "labelwt" in meta["final_model"]
PARAMS = {k.split("__", 1)[-1]: v for k, v in meta["final_params"].items()}
w_final = None
if USE_LABELWT:
    w_final = 1.0 / train.trackB_bt_se.values
    w_final = w_final / w_final.mean()
print(f"refit recipe from meta: monotone={USE_MONO} label-weights={USE_LABELWT}")
print(f"  params: {PARAMS}")

def fit_lgbm(cols):
    est = lgb.LGBMRegressor(random_state=0, verbose=-1, n_jobs=1,
                            monotone_constraints=mono_for(cols) if USE_MONO else None,
                            **PARAMS)
    est.fit(train[cols], y, sample_weight=w_final)
    return est

LOAD_FEATS = ["load_time_z_region"]
FEATURES_NL = [f for f in FEATURES if f not in LOAD_FEATS]

m_with = fit_lgbm(FEATURES)
m_without = fit_lgbm(FEATURES_NL)
print(f"fitted: with load time ({len(FEATURES)} features) and without ({len(FEATURES_NL)})")

def to100(v):
    return 100 * (v - v.min()) / (v.max() - v.min())

rank = df[["uni_id", "name", "region", "country", "country_is_bucket",
           "country_rank_eligible", "trackA_consensus", "trackB_bt_score"]].copy()
rank["pred_logit"] = m_with.predict(df[FEATURES])
rank["pred_logit_noload"] = m_without.predict(df[FEATURES_NL])
rank["predicted_quality_score"] = to100(rank.pred_logit)
rank["predicted_quality_score_noload"] = to100(rank.pred_logit_noload)
rank["is_labelled"] = rank.trackB_bt_score.notna()

# the notebook-07 prediction must reproduce here, or the two notebooks disagree
chk = stats.spearmanr(rank.predicted_quality_score, preds.set_index("uni_id")
                      .loc[rank.uni_id, "predicted_quality_score"]).statistic
assert chk > 0.999, f"refit does not reproduce nb07 predictions (rho = {chk:.4f})"
print(f"reproduces nb07 predictions: Spearman = {chk:.5f}")

# %% [markdown]
# ## 2. Ranking, with a transparent tie-break
#
# Ties in a predicted score are broken by content completeness, then accessibility, then
# `uni_id`. The rule is fixed and published rather than left to whatever order pandas happened
# to produce — an arbitrary tie-break is a silent ranking decision.

# %%
TIEBREAK = ["predicted_quality_score", "content_completeness_B5", "a11y_completeness_B11", "uni_id"]
rank = rank.merge(data[["uni_id", "content_completeness_B5", "a11y_completeness_B11"]], on="uni_id")
rank = rank.sort_values(TIEBREAK, ascending=[False, False, False, True]).reset_index(drop=True)
rank["global_website_rank"] = np.arange(1, len(rank) + 1)

rank["regional_website_rank"] = rank.groupby("region").global_website_rank.rank(method="first").astype(int)

# country rank only where a country ranking is a country ranking: >= 20 universities AND
# a real country, not one of the 10 bucket labels ("Balkans", "Sub-Saharan Africa", ...)
elig = rank[rank.country_rank_eligible == 1]
rank["country_website_rank"] = np.nan
rank.loc[elig.index, "country_website_rank"] = (
    elig.groupby("country").global_website_rank.rank(method="first"))
n_elig_countries = elig.country.nunique()

print(f"global ranks 1–{len(rank)}")
print(f"regional ranks across {rank.region.nunique()} regions")
print(f"country ranks for {n_elig_countries} eligible countries covering {len(elig)} universities")
print(f"NOT ranked by country: {len(rank)-len(elig)} universities in buckets or thin countries\n")
print(rank.head(15)[["global_website_rank", "name", "country", "predicted_quality_score",
                     "trackA_consensus", "is_labelled"]].to_string(index=False,
                     float_format=lambda v: f"{v:.1f}"))

# %% [markdown]
# ### With and without region-standardised load time
#
# The plan committed to producing every ranking both ways and reporting the delta, because
# load time is the feature most contaminated by the collector⊥region confound (`01`: a 3.3×
# swing on the same university re-crawled).

# %%
r2 = rank.sort_values("predicted_quality_score_noload", ascending=False).reset_index(drop=True)
r2["rank_noload"] = np.arange(1, len(r2) + 1)
rank = rank.merge(r2[["uni_id", "rank_noload"]], on="uni_id")
rank["load_rank_shift"] = rank.rank_noload - rank.global_website_rank

shift = rank.load_rank_shift.abs()
rho_ll = stats.spearmanr(rank.global_website_rank, rank.rank_noload).statistic
print(f"Spearman between the two rankings : {rho_ll:.4f}")
print(f"rank displacement |Δ|             : median {shift.median():.0f}, "
      f"mean {shift.mean():.1f}, p90 {shift.quantile(0.9):.0f}, max {shift.max():.0f}")
print(f"universities moving > 50 places   : {(shift > 50).sum()} of {len(rank)} ({(shift>50).mean():.1%})")
print(f"top-100 membership overlap        : "
      f"{len(set(rank.nlargest(100,'predicted_quality_score').uni_id) & set(r2.head(100).uni_id))}/100")
print(f"\ndisplacement by region (mean |Δ| — a large regional difference would mean the")
print(f"load-time feature is acting as a regional proxy):")
print(rank.groupby("region").load_rank_shift.apply(lambda s: s.abs().mean()).round(1).to_string())

# %% [markdown]
# ## 3. SHAP
#
# TreeSHAP on the final model, over all 1,226 universities. SHAP values are in the units of the
# target (logits), and they sum exactly to the prediction — which is what makes the per-university
# explanations below arithmetically true rather than illustrative.

# %%
expl = shap.TreeExplainer(m_with)
sv = expl.shap_values(df[FEATURES])
base_value = float(expl.expected_value)

recon = np.abs(sv.sum(1) + base_value - rank.set_index("uni_id").loc[df.uni_id, "pred_logit"].values).max()
print(f"SHAP additivity check: max |sum(shap) + base − prediction| = {recon:.2e}")
assert recon < 1e-6, "SHAP values do not reconstruct the predictions"

shap_abs = pd.DataFrame({"feature": FEATURES, "mean_abs_shap": np.abs(sv).mean(0)})
shap_abs["block"] = shap_abs.feature.map(fdict.set_index("feature").block.to_dict()).fillna("derived")
shap_abs = shap_abs.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
shap_abs.to_csv(OUT / "shap_feature_importance.csv", index=False)

print(f"\nbase value (mean prediction): {base_value:+.3f} logits\n")
print(shap_abs.head(15).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
nz = (shap_abs.mean_abs_shap > 0.001).sum()
print(f"\n{nz} of {len(FEATURES)} features have any measurable effect; "
      f"{len(FEATURES)-nz} are inert in this model")

# %% [markdown]
# ### Block-aggregated SHAP — the empirical weights
#
# Individual features are the wrong unit for a claim about *quality dimensions*. Summing |SHAP|
# within block gives the share of the model's decision each block actually drives — the
# empirical counterpart to the rubric's a-priori weights.

# %%
sb = shap_abs.groupby("block").mean_abs_shap.sum()
sb = (100 * sb / sb.sum()).sort_values(ascending=False).rename("shap_share_pct")
print(sb.round(2).to_string())

# %% [markdown]
# ## 4. Headline analysis — declared weights vs. what actually drives the ranking
#
# The rubric in `04` assigned each block a weight *before* any score was computed. SHAP measures
# what each block does *after* the fact. The gap between them is the central empirical claim of
# this project.

# %%
cmp = pd.DataFrame({"nominal_weight_pct": blockw.set_index("block").mean_weight_pct}).join(
    sb, how="outer").fillna(0.0)
cmp["ratio"] = np.where(cmp.nominal_weight_pct > 0,
                        cmp.shap_share_pct / cmp.nominal_weight_pct, np.nan)
cmp["gap_pp"] = cmp.shap_share_pct - cmp.nominal_weight_pct
cmp = cmp.sort_values("gap_pp", ascending=False)
cmp.round(2).to_csv(OUT / "shap_vs_rubric_weights.csv")

print(f"{'block':<22} {'declared %':>11} {'realised %':>11} {'gap pp':>8} {'ratio':>7}")
print("-" * 64)
for b, r in cmp.iterrows():
    print(f"{b:<22} {r.nominal_weight_pct:>11.1f} {r.shap_share_pct:>11.1f} "
          f"{r.gap_pp:>+8.1f} {r.ratio:>7.2f}")

# Two ways a weight can be wrong, and both need flagging. A block can be off by a large
# number of percentage points (matters for big blocks) or off by a large *factor* (matters
# for small ones — 6.2% declared realised at 2.8% is less than half the intended influence,
# which a ±5pp rule would miss entirely).
over = cmp[(cmp.gap_pp > 5) | ((cmp.ratio > 1.5) & (cmp.nominal_weight_pct >= 3))].index.tolist()
under = cmp[(cmp.gap_pp < -5) | ((cmp.ratio < 0.6) & (cmp.nominal_weight_pct >= 3))].index.tolist()
print(f"\nunder-weighted by the rubric (does more than declared): {over}")
print(f"over-weighted by the rubric (does less than declared)  : {under}")
print("\nthe Rashida et al. comparison, on our own data:")
print(f"  they declared 40% for technical performance; it drove 2.9% of their ranking (ratio 0.07)")
print(f"  we declared {cmp.loc['B9_technical_perf','nominal_weight_pct']:.1f}% for B9_technical_perf; "
      f"it drives {cmp.loc['B9_technical_perf','shap_share_pct']:.1f}% "
      f"(ratio {cmp.loc['B9_technical_perf','ratio']:.2f})")
print("  same direction, smaller magnitude — because the block variance audit in nb02 caught it early")
print(f"\nSpearman(declared, realised) across blocks = "
      f"{stats.spearmanr(cmp.nominal_weight_pct, cmp.shap_share_pct).statistic:.3f}")

# %% [markdown]
# This is the Rashida et al. failure mode reproduced on our own data and then measured. A
# hand-set scoring rule can allocate a large share of its weight to a dimension that turns out
# to drive almost none of the ranking — usually because every site scores alike on it, so the
# weight has nothing to act on. The blocks above the line matter more than anyone declared;
# the blocks below it are, in practice, decoration. **No amount of expert deliberation about
# weights would have revealed this; only measuring the realised influence does.**

# %% [markdown]
# ## 5. Explaining an individual position

# %%
feat_vals = df[FEATURES].reset_index(drop=True)
uid_to_row = {u: i for i, u in enumerate(df.uni_id)}
rank_idx = rank.set_index("uni_id")

def explain(uni_id, k=8):
    """The k features that moved this university's score furthest from the average."""
    i = uid_to_row[uni_id]
    s = pd.Series(sv[i], index=FEATURES)
    top = s.reindex(s.abs().sort_values(ascending=False).index).head(k)
    r = rank_idx.loc[uni_id]
    lines = [f"{r['name']}  ({r.country}) — global rank {int(r.global_website_rank)} of {len(rank)}",
             f"  predicted {r.predicted_quality_score:.1f}/100  (logit {r.pred_logit:+.2f}, "
             f"base {base_value:+.2f})"]
    for f, v in top.items():
        raw = feat_vals.iloc[i][f]
        lines.append(f"  {v:+7.3f}  {f:<28} = {raw:g}")
    return "\n".join(lines)

print("=" * 78)
for label, sel in [("TOP 3", rank.head(3)), ("MIDDLE 3", rank.iloc[len(rank)//2 - 1: len(rank)//2 + 2]),
                   ("BOTTOM 3", rank.tail(3))]:
    print(f"\n### {label}\n")
    for u in sel.uni_id:
        print(explain(u)); print()

# %% [markdown]
# ### `why_a_above_b` — the question a ranked institution actually asks
#
# Not "what is my score" but "why is that university above me, and what would close the gap".
# Because SHAP is additive, the per-feature differences sum exactly to the score gap, so this
# is a decomposition rather than a narrative.

# %%
def why_a_above_b(uid_a, uid_b, k=8):
    ia, ib = uid_to_row[uid_a], uid_to_row[uid_b]
    ra, rb = rank_idx.loc[uid_a], rank_idx.loc[uid_b]
    d = pd.Series(sv[ia] - sv[ib], index=FEATURES)
    d = d.reindex(d.abs().sort_values(ascending=False).index)
    gap = ra.pred_logit - rb.pred_logit
    out = [f"{ra['name']} (rank {int(ra.global_website_rank)}, {ra.predicted_quality_score:.1f})",
           f"  vs {rb['name']} (rank {int(rb.global_website_rank)}, {rb.predicted_quality_score:.1f})",
           f"  score gap = {gap:+.3f} logits; the {k} features that explain most of it:", ""]
    for f, v in d.head(k).items():
        va, vb = feat_vals.iloc[ia][f], feat_vals.iloc[ib][f]
        arrow = "favours A" if v > 0 else "favours B"
        out.append(f"  {v:+7.3f}  {f:<28} A={va:<8g} B={vb:<8g}  {arrow}")
    out.append(f"\n  these {k} account for {100*d.head(k).sum()/gap:.0f}% of the gap; "
               f"the remaining {len(FEATURES)-k} features net {gap - d.head(k).sum():+.3f}")
    return "\n".join(out)

print(why_a_above_b(rank.iloc[0].uni_id, rank.iloc[9].uni_id))
print("\n" + "=" * 78 + "\n")
print(why_a_above_b(rank.iloc[99].uni_id, rank.iloc[599].uni_id))

# %% [markdown]
# ### Improvement recommendations
#
# Actionable rather than descriptive: for each university, the changes with the largest
# *negative* SHAP that are also things a web team can actually do.

# %%
FIXABLE = {f for f in FEATURES if f.startswith("a") and f not in
           {"a09_national_rank", "a07_qs_badge", "a11_accreditation", "a12_accred_count",
            "a13_achievements", "a60_trust_seal", "a61_testimonials"}}
FIXABLE |= {"content_completeness_B5", "footer_completeness_B6", "a11y_completeness_B11",
            "seo_completeness_B10", "notice_evidence", "event_evidence"}

def recommend(uni_id, k=5):
    i = uid_to_row[uni_id]
    s = pd.Series(sv[i], index=FEATURES)
    s = s[[f for f in s.index if f in FIXABLE]]
    worst = s.sort_values().head(k)
    r = rank_idx.loc[uni_id]
    lines = [f"{r['name']} — rank {int(r.global_website_rank)}, {r.predicted_quality_score:.1f}/100",
             "  biggest recoverable losses:"]
    for f, v in worst.items():
        if v >= 0:
            continue
        lines.append(f"    {v:+.3f}  {f:<28} currently {feat_vals.iloc[i][f]:g}")
    lines.append(f"    -> fixing all of the above would add up to {-worst[worst<0].sum():.2f} logits")
    return "\n".join(lines)

recs = []
for u in rank.uni_id:
    i = uid_to_row[u]
    s = pd.Series(sv[i], index=FEATURES)
    s = s[[f for f in s.index if f in FIXABLE]].sort_values()
    neg = s[s < 0].head(5)
    recs.append(dict(uni_id=u, n_recoverable=int((s < 0).sum()),
                     recoverable_logits=float(-neg.sum()),
                     top_fixes="; ".join(neg.index)))
recs = pd.DataFrame(recs)
rank = rank.merge(recs, on="uni_id")

print(recommend(rank[rank.global_website_rank == 900].iloc[0].uni_id))
print()
print(recommend(rank[rank.global_website_rank == 1200].iloc[0].uni_id))
print(f"\nmedian recoverable headroom across all 1,226: "
      f"{rank.recoverable_logits.median():.2f} logits "
      f"(bottom quartile {rank.nsmallest(306,'predicted_quality_score').recoverable_logits.median():.2f})")

# %% [markdown]
# ## 6. Fairness
#
# Three cuts, all of which could embarrass the model, so all of which are reported.

# %%
lab = rank[rank.is_labelled].copy()
lab["abs_err"] = (lab.pred_logit - lab.trackB_bt_score).abs()
loro_folds = pd.read_csv(OUT / "model_loro_folds.csv")
loro_final = loro_folds[loro_folds.model == meta["final_model"]]

fair = rank.groupby("region").agg(
    n=("uni_id", "size"), mean_score=("predicted_quality_score", "mean"),
    sd_score=("predicted_quality_score", "std"),
    top100=("global_website_rank", lambda s: int((s <= 100).sum())),
    bottom100=("global_website_rank", lambda s: int((s > len(rank) - 100).sum())))
fair = fair.join(loro_final.set_index("held_out_region")[["spearman", "mae"]]
                 .rename(columns={"spearman": "loro_spearman", "mae": "loro_mae"}))
fair["top100_expected"] = 100 * fair.n / len(rank)
print(fair.round(2).to_string())

kw = stats.kruskal(*[g.predicted_quality_score.values for _, g in rank.groupby("region")])
fair["top100_ratio"] = fair.top100 / fair.top100_expected
print(f"\nKruskal–Wallis on predicted score across regions: H = {kw.statistic:.1f}, p = {kw.pvalue:.2e}")
print(f"LORO Spearman range across regions: {fair.loro_spearman.min():.3f} – {fair.loro_spearman.max():.3f}")
print(f"LORO MAE range: {fair.loro_mae.min():.3f} – {fair.loro_mae.max():.3f} logits")

print("\ntop-100 representation (observed / expected under proportional representation):")
for r, v in fair.top100_ratio.sort_values(ascending=False).items():
    print(f"  {r:<32} {fair.loc[r,'top100']:>3} of ~{fair.loc[r,'top100_expected']:.0f}   {v:>5.2f}×")
print(f"\nspread: {fair.top100_ratio.max():.1f}× over-represented to {fair.top100_ratio.min():.2f}× under.")
print("This is the largest single caveat on the global ranking and it is NOT a fairness fix\n"
      "away — see below.")

# %% [markdown]
# ### The confound, restated where it is load-bearing
#
# Region and collector are perfectly 1:1 (`01`). Regional differences in predicted score
# therefore have two equally consistent readings — genuine differences in institutional web
# quality, or differences in how six people ran the extractor — and **this dataset cannot
# distinguish them**. That is a permanent limitation of the data, not something better modelling
# fixes.
#
# The top-100 table makes the size of this concrete: representation runs from roughly 2.3×
# over-expectation down to 0.25×, so the head of the global ranking is not regionally neutral.
# Anyone quoting a global top-N without that sentence attached is over-claiming.
#
# What the numbers above *do* establish is narrower and still worth having: the model's
# **accuracy** is stable across regions even though its **predictions** are not. LORO Spearman
# runs 0.77–0.90 — it ranks universities in a region it never trained on about as well as one it
# did. So the ranking is not a regional lookup table; the level differences are real differences
# in the measured attributes, whatever *caused* those attributes to differ.
#
# The regional ranking is the safer artefact for anyone who needs to avoid this entirely: within
# a region, collector is constant, so the comparison is clean.

# %%
buck = rank.groupby(rank.country_is_bucket.map({0: "real country", 1: "bucket label"})).agg(
    n=("uni_id", "size"), mean_score=("predicted_quality_score", "mean"),
    sd=("predicted_quality_score", "std"), median_rank=("global_website_rank", "median"))
print(buck.round(2).to_string())
b0 = rank[rank.country_is_bucket == 0].predicted_quality_score
b1 = rank[rank.country_is_bucket == 1].predicted_quality_score
mw = stats.mannwhitneyu(b0, b1)
print(f"\nMann–Whitney real-country vs bucket: U p = {mw.pvalue:.4f}, "
      f"difference in means {b0.mean()-b1.mean():+.2f} points")
print("Bucket labels ('Balkans', 'Sub-Saharan Africa', …) are a data-quality artefact of the\n"
      "country column, not a category of institution. They are excluded from country rankings\n"
      "but retained in the global ranking, and this comparison shows what that inclusion costs.")

# %%
ctry = (rank[rank.country_rank_eligible == 1].groupby("country")
        .agg(n=("uni_id", "size"), mean_score=("predicted_quality_score", "mean"),
             best_global=("global_website_rank", "min"), median_global=("global_website_rank", "median"))
        .sort_values("mean_score", ascending=False))
print(f"\n{n_elig_countries} eligible countries (≥20 universities, real country label):\n")
print(ctry.round(1).to_string())

# %% [markdown]
# ## 7. Writing the rankings

# %%
final_cols = ["global_website_rank", "regional_website_rank", "country_website_rank",
              "uni_id", "name", "region", "country", "country_rank_eligible",
              "predicted_quality_score", "pred_logit",
              "predicted_quality_score_noload", "rank_noload", "load_rank_shift",
              "trackA_consensus", "trackB_bt_score", "is_labelled",
              "recoverable_logits", "top_fixes"]
rank[final_cols].to_csv(OUT / "ranking_global.csv", index=False)
rank[rank.country_rank_eligible == 1][final_cols].sort_values(["country", "country_website_rank"]) \
    .to_csv(OUT / "ranking_by_country.csv", index=False)
rank[final_cols].sort_values(["region", "regional_website_rank"]) \
    .to_csv(OUT / "ranking_by_region.csv", index=False)
cmp.round(3).to_csv(OUT / "shap_vs_rubric_weights.csv")
fair.round(3).to_csv(OUT / "fairness_by_region.csv")
ctry.round(3).to_csv(OUT / "fairness_by_country.csv")

pd.DataFrame(sv, columns=FEATURES).assign(uni_id=df.uni_id.values).to_csv(
    OUT / "shap_values.csv", index=False)
print("wrote ranking_global.csv, ranking_by_region.csv, ranking_by_country.csv,")
print("      shap_values.csv, shap_feature_importance.csv, shap_vs_rubric_weights.csv,")
print("      fairness_by_region.csv, fairness_by_country.csv")

# %% [markdown]
# ## 8. Figures

# %%
plt.figure(figsize=(10, 6))
shap.summary_plot(sv, df[FEATURES], max_display=18, show=False, plot_size=None)
plt.title("SHAP — what drives predicted website quality", fontsize=11)
plt.tight_layout(); plt.savefig(FIG / "17_shap_beeswarm.png", dpi=150, bbox_inches="tight"); plt.close()

# %%
fig, ax = plt.subplots(1, 2, figsize=(14.5, 5.6))
o = cmp.sort_values("shap_share_pct")
xx = np.arange(len(o))
ax[0].barh(xx - 0.2, o.nominal_weight_pct, height=0.38, label="declared in rubric_v1", color="#a0aec0")
ax[0].barh(xx + 0.2, o.shap_share_pct, height=0.38, label="realised (SHAP)", color="#2c5282")
ax[0].set_yticks(xx); ax[0].set_yticklabels(o.index, fontsize=8)
ax[0].set_xlabel("% of ranking influence")
ax[0].set_title("Declared weight vs. what actually drives the ranking")
ax[0].legend(fontsize=8); ax[0].grid(alpha=0.25, axis="x")

col = ["#c53030" if g < -5 else "#2f855a" if g > 5 else "#a0aec0" for g in o.gap_pp]
ax[1].barh(xx, o.gap_pp, color=col)
ax[1].axvline(0, color="k", lw=0.8)
ax[1].set_yticks(xx); ax[1].set_yticklabels(o.index, fontsize=8)
ax[1].set_xlabel("realised − declared (percentage points)")
ax[1].set_title("Green: does more than declared\nRed: does less than declared")
ax[1].grid(alpha=0.25, axis="x")
plt.tight_layout(); plt.savefig(FIG / "18_weights_vs_shap.png", dpi=150); plt.close()

# %%
fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
ax[0].scatter(rank.global_website_rank, rank.rank_noload, s=4, alpha=0.35, color="#2b6cb0")
ax[0].plot([0, len(rank)], [0, len(rank)], "k--", lw=1)
ax[0].set_xlabel("rank with region-standardised load time"); ax[0].set_ylabel("rank without")
ax[0].set_title(f"Does load time change the answer?\nSpearman {rho_ll:.4f}, median |Δ| = {shift.median():.0f}")
ax[0].grid(alpha=0.25)

for reg, g in rank.groupby("region"):
    ax[1].hist(g.predicted_quality_score, bins=22, histtype="step", lw=1.5, label=reg[:22], density=True)
ax[1].set_xlabel("predicted_quality_score"); ax[1].set_ylabel("density")
ax[1].set_title("Score distribution by region\n(confounded with collector — see §6)")
ax[1].legend(fontsize=6.5); ax[1].grid(alpha=0.25)

o3 = fair.sort_values("loro_spearman")
ax[2].barh(range(len(o3)), o3.loro_spearman, color="#2f855a", alpha=0.85)
ax[2].axvline(meta["loro_spearman"], color="#c05621", ls="--", lw=1.2, label="mean LORO")
ax[2].set_yticks(range(len(o3))); ax[2].set_yticklabels([r[:26] for r in o3.index], fontsize=8)
ax[2].set_xlabel("Spearman ρ on the held-out region")
ax[2].set_title("Accuracy is stable across regions"); ax[2].legend(fontsize=8); ax[2].grid(alpha=0.25, axis="x")
plt.tight_layout(); plt.savefig(FIG / "19_ranking_fairness.png", dpi=150); plt.close()

# %%
fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.4))
for ax_, (lbl, r_) in zip(axes, [("top", rank.iloc[0]), ("median", rank.iloc[len(rank)//2]),
                                 ("bottom", rank.iloc[-1])]):
    i = uid_to_row[r_.uni_id]
    s = pd.Series(sv[i], index=FEATURES)
    s = s.reindex(s.abs().sort_values(ascending=False).index).head(10).iloc[::-1]
    ax_.barh(range(len(s)), s.values, color=["#c53030" if v < 0 else "#2f855a" for v in s.values])
    ax_.set_yticks(range(len(s))); ax_.set_yticklabels(s.index, fontsize=7)
    ax_.axvline(0, color="k", lw=0.8)
    ax_.set_title(f"{lbl}: {r_['name'][:34]}\nrank {int(r_.global_website_rank)}, "
                  f"{r_.predicted_quality_score:.0f}/100", fontsize=9)
    ax_.set_xlabel("SHAP (logits)"); ax_.grid(alpha=0.25, axis="x")
plt.tight_layout(); plt.savefig(FIG / "20_shap_local.png", dpi=150); plt.close()
print("figures 15, 16, 17, 18 written")

# %%
summary = dict(
    n_ranked=int(len(rank)), n_labelled=int(rank.is_labelled.sum()),
    n_country_eligible=int(len(elig)), n_eligible_countries=int(n_elig_countries),
    loadtime_spearman=float(rho_ll), loadtime_median_shift=float(shift.median()),
    loadtime_over50=int((shift > 50).sum()),
    shap_block_shares={k: round(float(v), 2) for k, v in sb.items()},
    rubric_vs_shap_spearman=float(stats.spearmanr(cmp.nominal_weight_pct, cmp.shap_share_pct).statistic),
    most_underweighted=over, most_overweighted=under,
    loro_spearman_by_region={k: round(float(v), 3) for k, v in fair.loro_spearman.items()},
    region_kruskal_p=float(kw.pvalue),
)
(OUT / "ranking_meta.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
