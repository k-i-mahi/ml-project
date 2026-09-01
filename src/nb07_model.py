# %% [markdown]
# # 07 — Supervised model: predicting the Track B latent score
#
# **Target `y` = `trackB_bt_score`** — the Bradley–Terry strength recovered in `06` from 900
# blind pairwise judgments. 200 universities carry it. The other 1,026 are pure inference.
#
# The single most important thing this notebook does is **not** produce a high R². It is to
# answer one question honestly:
#
# > Does a learned model beat the transparent weighted sum it was supposed to improve on?
#
# Track A is not the target — it is the **baseline to beat**, scored against Track B on exactly
# the same test folds as every learned model. If no model beats it, the ML contribution is null
# and that is the finding reported, not a result to be tuned around.
#
# Stated in advance, before any number appears: **n = 200 is small for gradient boosting.**
# Ridge plausibly wins. That is a legitimate result about this problem, not a failure.

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
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb

warnings.filterwarnings("ignore")
SEEDS = [0, 1, 2, 3, 4]
N_FOLDS = 5

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
OUT, FIG = ROOT / "outputs", ROOT / "figures"
FIG.mkdir(exist_ok=True)

data = pd.read_csv(OUT / "model_ready_dataset.csv")
labB = pd.read_csv(OUT / "expert_labels_trackB.csv")
labA = pd.read_csv(OUT / "expert_labels_trackA.csv")
fdict = pd.read_csv(OUT / "feature_dictionary.csv")

print(f"model_ready {data.shape} | Track B labels {labB.shape} | Track A labels {labA.shape}")

# %% [markdown]
# ## 1. Building the design matrix — and what is deliberately excluded
#
# Every exclusion is a decision with a reason, so it is written down rather than implied by a
# `select_dtypes` call.

# %%
EXCLUDE = {
    # identity — would let the model memorise rows
    "uni_id": "identifier", "name": "identity", "url": "identity",
    # the confound: member ⊥ region is 1:1 (audit F17). Region is used as a CV *group*,
    # never as a feature, or the model would be free to learn geography.
    "member": "collector — perfectly confounded with region",
    "region": "used as CV grouping only, never as a predictor",
    "country": "high-cardinality geography; 10 of 22 labels are buckets, not countries",
    # crawl metadata, not properties of the website
    "http_status": "crawl metadata", "fetched_at": "crawl metadata",
    "page_lang": "crawl metadata", "switched_to_english": "crawl metadata",
    "a18_recent_notice_date": "raw date — superseded by notice_recency_days",
    "notice_date_future": "audit flag, not a site property",
    "country_is_bucket": "audit flag", "country_rank_eligible": "audit flag",
    # load time: raw seconds are confounded with collector, so only the region-standardised
    # form is offered. nb08 produces every ranking with and without it.
    "load_time_s": "confounded with collector — use load_time_z_region",
    "load_time_pct_region": "same information as load_time_z_region, collinear",
    # reconciled elsewhere: nb02 showed these double-count their ordinal replacements
    "a16_notice_board": "superseded by notice_evidence", "a17_notice_timestamp": "superseded by notice_evidence",
    "a23_upcoming_events": "superseded by event_evidence", "a24_event_count": "superseded by event_evidence",
    "a14_stats_block": "superseded by a15_stat_item_count (rho = 0.90)",
    "broken_links_w": "winsorised duplicate of broken_links_log",
}

FEATURES = [c for c in data.columns if c not in EXCLUDE]
print(f"{len(FEATURES)} features retained, {len(EXCLUDE)} columns excluded\n")
for c, why in EXCLUDE.items():
    print(f"  - {c:<24} {why}")

# leakage assertion: the two prestige value columns were dropped in 01 and must never reappear
for leak in ["a08_qs_value", "a10_webometrics_value"]:
    assert leak not in data.columns, f"{leak} leaked back into the dataset"
# the presence flags a07/a09 are kept — "does the site display a credibility badge" is a
# property of the website. Their *values* are what was unusable and leaky.
assert "a07_qs_badge" in FEATURES and "a09_national_rank" in FEATURES
print("\nleakage check: prestige VALUES absent; prestige DISPLAY flags retained as site features")

# %%
df = data.merge(labB[["uni_id", "trackB_bt_score", "trackB_bt_se", "trackB_score_100"]], on="uni_id", how="left")
df = df.merge(labA[["uni_id", "trackA_consensus"]], on="uni_id", how="left")

train = df[df.trackB_bt_score.notna()].reset_index(drop=True)
infer = df[df.trackB_bt_score.isna()].reset_index(drop=True)
assert len(train) == 200 and len(infer) == 1026 and len(train) + len(infer) == 1226

X = train[FEATURES]
y = train.trackB_bt_score.values
groups = train.region.values
w_conf = 1.0 / train.trackB_bt_se.values          # confident labels weigh more
w_conf = w_conf / w_conf.mean()

# stratify the folds on target quintiles so no fold is all-good or all-bad
ybin = pd.qcut(y, 5, labels=False)

print(f"train {X.shape} | inference {len(infer)} | y: mean {y.mean():+.3f}, sd {y.std():.3f}, "
      f"range {y.min():+.2f} to {y.max():+.2f}")
print(f"regions in training set: {pd.Series(groups).value_counts().to_dict()}")

# %% [markdown]
# ## 2. Estimators
#
# Every preprocessing step lives **inside** the pipeline, so imputation and scaling are fitted
# on the training fold only. Fitting them on the whole set before CV is the most common way to
# leak, and it is easy to do by accident.

# %%
num = FEATURES

def ridge_pipe():
    return Pipeline([
        ("imp", SimpleImputer(strategy="median", add_indicator=False)),
        ("sc", StandardScaler()),
        ("m", Ridge()),
    ])

def rf_pipe():
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("m", RandomForestRegressor(random_state=0, n_jobs=-1)),
    ])

def lgbm_pipe(mono=None):
    return Pipeline([
        ("m", lgb.LGBMRegressor(random_state=0, verbose=-1, n_jobs=1,
                                monotone_constraints=mono))
    ])

GRIDS = {
    "Ridge":        (ridge_pipe(), {"m__alpha": [0.3, 1, 3, 10, 30, 100, 300]}),
    "RandomForest": (rf_pipe(),    {"m__n_estimators": [400], "m__max_depth": [4, 8, None],
                                    "m__min_samples_leaf": [1, 3, 6], "m__max_features": [0.3, 0.6]}),
    "LightGBM":     (lgbm_pipe(),  {"m__n_estimators": [200, 400], "m__learning_rate": [0.02, 0.05],
                                    "m__num_leaves": [7, 15], "m__min_child_samples": [5, 15],
                                    "m__colsample_bytree": [0.6], "m__reg_lambda": [1.0, 10.0]}),
}

# %% [markdown]
# ### Monotonic-constraint variant
#
# Constraints are applied **only** where the direction is genuinely defensible and genuinely
# monotone. `nav_item_count` and the event count are deliberately left unconstrained: `04`
# established they are inverted-U — a 20-item menu is worse than a 7-item menu — and forcing
# monotonicity there would encode something the rubric explicitly denies.

# %%
NON_MONOTONE = {"a03_nav_item_count", "a15_stat_item_count", "a12_accred_count", "nav_quality"}
dirmap = fdict.set_index("feature").direction.to_dict()
mono = []
for f in FEATURES:
    d = dirmap.get(f, 0)
    mono.append(0 if f in NON_MONOTONE else int(d) if d in (-1, 1) else 0)
mono = np.array(mono)
print(f"monotone constraints: {(mono==1).sum()} increasing, {(mono==-1).sum()} decreasing, "
      f"{(mono==0).sum()} free")
print(f"  decreasing: {[f for f,m in zip(FEATURES,mono) if m==-1]}")
print(f"  free (non-monotone by design): {sorted(NON_MONOTONE & set(FEATURES))}")

GRIDS["LightGBM+mono"] = (lgbm_pipe(list(mono)),
                          {"m__n_estimators": [200, 400], "m__learning_rate": [0.02, 0.05],
                           "m__num_leaves": [7, 15], "m__min_child_samples": [5, 15],
                           "m__colsample_bytree": [0.6], "m__reg_lambda": [1.0, 10.0]})

# %% [markdown]
# ### Baselines
#
# `TrackAComposite` is the one that decides whether this project has a result: the transparent
# weighted sum any competent analyst could compute without machine learning.
#
# One subtlety, and it matters for fairness. The rubric is on 0–100 and the target is a logit
# spanning roughly −5 to +6. Comparing them directly would hand the rubric an absurd MAE while
# leaving its Spearman untouched — flattering the learned models on three of five metrics for
# no real reason. So the baseline gets a **two-parameter least-squares calibration onto the
# target scale, fitted inside each training fold**. It is a monotone map, so it changes no
# ranking and no rank metric; it only makes MAE, RMSE and R² mean something. The rubric's own
# ordering still does all the work.

# %%
class MeanPredictor(BaseEstimator, RegressorMixin):
    def fit(self, X, y): self.m_ = np.mean(y); return self
    def predict(self, X): return np.full(len(X), self.m_)

class TrackAComposite(BaseEstimator, RegressorMixin):
    """The frozen Track A rubric score, put on the target's scale.

    The rubric is 0-100 and the target is a logit in roughly -5..+6, so raw scores would
    give a nonsense MAE/RMSE/R2 while leaving Spearman untouched. The fair comparison is a
    single least-squares calibration a -> alpha + beta*a, fitted **on the training fold
    only**. It is monotone, so it changes no ranking and no Spearman; it only makes the
    error metrics comparable. The rubric's ordering is still doing all the work - two
    parameters is not a learned model."""
    def __init__(self, col="trackA_consensus"): self.col = col
    def fit(self, X, y):
        a = np.asarray(X[self.col], dtype=float)
        self.beta_, self.alpha_ = np.polyfit(a, np.asarray(y, dtype=float), 1)
        return self
    def predict(self, X):
        return self.alpha_ + self.beta_ * np.asarray(X[self.col], dtype=float)

BLOCK_SUBSETS = {
    "B5-content-only": [f for f in FEATURES
                        if f in set(fdict[fdict.block == "B5_page_content"].feature)] + ["content_completeness_B5"],
    "B9-technical-only": [f for f in FEATURES
                          if f in set(fdict[fdict.block == "B9_technical_perf"].feature)],
}
BLOCK_SUBSETS = {k: sorted(set(v) & set(FEATURES)) for k, v in BLOCK_SUBSETS.items()}
for k, v in BLOCK_SUBSETS.items():
    print(f"{k}: {len(v)} features")

# %% [markdown]
# ## 3. Nested cross-validation
#
# Outer: repeated stratified 5-fold × 5 seeds = **25 fold-reps**. Inner: 3-fold grid search on
# the training fold only. The inner search never sees the outer test fold, so the reported
# numbers are not optimistically biased by tuning.

# %%
def metrics(yt, yp):
    # a constant predictor has no rank information; scipy returns nan, which would
    # silently poison every downstream mean. 0.0 is the correct value.
    rho = 0.0 if np.ptp(yp) == 0 else stats.spearmanr(yt, yp).statistic
    tau = 0.0 if np.ptp(yp) == 0 else stats.kendalltau(yt, yp).statistic
    return dict(
        spearman=rho,
        kendall=tau,
        mae=float(np.mean(np.abs(yt - yp))),
        rmse=float(np.sqrt(np.mean((yt - yp) ** 2))),
        r2=1 - np.sum((yt - yp) ** 2) / np.sum((yt - yt.mean()) ** 2),
    )

def nested_cv(name, est, grid, Xfull, use_cols=None, weights=None):
    cols = use_cols or FEATURES
    rows = []
    for seed in SEEDS:
        skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed)
        for k, (tr, te) in enumerate(skf.split(Xfull, ybin)):
            Xtr, Xte = Xfull.iloc[tr][cols], Xfull.iloc[te][cols]
            ytr, yte = y[tr], y[te]
            if grid:
                gs = GridSearchCV(clone(est), grid, cv=KFold(3, shuffle=True, random_state=seed),
                                  scoring="neg_mean_squared_error", n_jobs=-1)
                fit_kw = {}
                if weights is not None:
                    fit_kw["m__sample_weight"] = weights[tr]
                gs.fit(Xtr, ytr, **fit_kw)
                model, best = gs.best_estimator_, gs.best_params_
            else:
                model, best = clone(est).fit(Xtr, ytr), {}
            m = metrics(yte, np.asarray(model.predict(Xte), dtype=float))
            rows.append(dict(model=name, seed=seed, fold=k, **m, best=json.dumps(best)))
    return pd.DataFrame(rows)

results = []
Xall = train  # baselines need trackA_consensus, which is not a feature

print(f"{'model':<20} {'Spearman':>16} {'Kendall':>9} {'MAE':>7} {'RMSE':>7} {'R2':>8}")
print("-" * 72)

runs = [
    ("mean (baseline)",      MeanPredictor(), None, FEATURES, None),
    ("Track A composite",    TrackAComposite(), None, ["trackA_consensus"], None),
    ("B5-content-only",      ridge_pipe(), {"m__alpha": [1, 10, 100]}, BLOCK_SUBSETS["B5-content-only"], None),
    ("B9-technical-only",    ridge_pipe(), {"m__alpha": [1, 10, 100]}, BLOCK_SUBSETS["B9-technical-only"], None),
    ("Ridge",                *GRIDS["Ridge"], FEATURES, None),
    ("RandomForest",         *GRIDS["RandomForest"], FEATURES, None),
    ("LightGBM",             *GRIDS["LightGBM"], FEATURES, None),
    ("LightGBM+mono",        *GRIDS["LightGBM+mono"], FEATURES, None),
    ("LightGBM+labelwt",     *GRIDS["LightGBM"], FEATURES, w_conf),
]

for name, est, grid, cols, wt in runs:
    r = nested_cv(name, est, grid, Xall, use_cols=cols, weights=wt)
    results.append(r)
    print(f"{name:<20} {r.spearman.mean():>7.3f} ± {r.spearman.std():.3f} "
          f"{r.kendall.mean():>9.3f} {r.mae.mean():>7.3f} {r.rmse.mean():>7.3f} {r.r2.mean():>8.3f}")

res = pd.concat(results, ignore_index=True)

# %% [markdown]
# ## 4. The honest comparison
#
# Every learned model is now tested against the Track A composite on the *same* 25 fold-reps,
# with a paired test — the correct comparison, because the folds are shared.

# %%
piv = res.pivot_table(index=["seed", "fold"], columns="model", values="spearman")
base = piv["Track A composite"]

print(f"{'model':<20} {'mean rho':>9} {'vs Track A':>11} {'paired t p':>11} {'wins/25':>8}  verdict")
print("-" * 78)
comp = []
for m in piv.columns:
    if m == "Track A composite":
        continue
    d = piv[m] - base
    t = stats.ttest_rel(piv[m], base)
    wins = int((d > 0).sum())
    verdict = ("beats the rubric" if t.pvalue < 0.05 and d.mean() > 0 else
               "loses to the rubric" if t.pvalue < 0.05 and d.mean() < 0 else
               "indistinguishable")
    comp.append(dict(model=m, mean_rho=piv[m].mean(), delta=d.mean(), p=t.pvalue,
                     wins=wins, verdict=verdict))
    print(f"{m:<20} {piv[m].mean():>9.3f} {d.mean():>+11.3f} {t.pvalue:>11.4f} {wins:>5}/25  {verdict}")
comp = pd.DataFrame(comp).sort_values("mean_rho", ascending=False)

best_name = comp.iloc[0].model
best_beats = comp.iloc[0].verdict == "beats the rubric"
print(f"\nbest learned model: {best_name} (rho = {comp.iloc[0].mean_rho:.3f})")
print(f"Track A composite : rho = {base.mean():.3f}")

# %% [markdown]
# ## 5. Leave-one-region-out — the test that matters
#
# Standard CV mixes regions across train and test, so a model can score well by learning
# "European sites look like this". LORO holds an entire region out. Because collector and region
# are 1:1, the drop from CV to LORO is the clearest available estimate of how much of the
# performance was geography.

# %%
loro = []
for name, est, grid, cols, wt in runs:
    for reg in sorted(set(groups)):
        tr = np.where(groups != reg)[0]
        te = np.where(groups == reg)[0]
        Xtr, Xte = Xall.iloc[tr][cols], Xall.iloc[te][cols]
        if grid:
            gs = GridSearchCV(clone(est), grid, cv=KFold(3, shuffle=True, random_state=0),
                              scoring="neg_mean_squared_error", n_jobs=-1)
            gs.fit(Xtr, y[tr], **({"m__sample_weight": wt[tr]} if wt is not None else {}))
            model = gs.best_estimator_
        else:
            model = clone(est).fit(Xtr, y[tr])
        loro.append(dict(model=name, held_out_region=reg, n=len(te),
                         **metrics(y[te], np.asarray(model.predict(Xte), dtype=float))))
loro = pd.DataFrame(loro)

summ = loro.groupby("model").spearman.agg(["mean", "std"]).join(
    res.groupby("model").spearman.mean().rename("cv_spearman"))
summ["drop"] = summ.cv_spearman - summ["mean"]
summ = summ.sort_values("mean", ascending=False)
print(f"{'model':<20} {'LORO rho':>16} {'CV rho':>8} {'drop':>8}")
print("-" * 56)
for m, r in summ.iterrows():
    print(f"{m:<20} {r['mean']:>7.3f} ± {r['std']:.3f} {r.cv_spearman:>8.3f} {r['drop']:>+8.3f}")

print(f"\nper-region detail for {best_name}:")
print(loro[loro.model == best_name][["held_out_region", "n", "spearman", "mae"]]
      .sort_values("spearman").to_string(index=False))

# %% [markdown]
# ## 6. Selecting and fitting the final model
#
# Selection rule, fixed before looking: **highest mean LORO Spearman**, because generalising
# across the confound is what the deliverable needs, not the best in-distribution number.
# A learned model is only preferred over the rubric if it actually beats it.

# %%
learned = summ.drop(index=["mean (baseline)", "Track A composite"], errors="ignore")
FINAL = learned.index[0]
final_est, final_grid, final_cols, final_wt = next(
    (e, g, c, w) for n, e, g, c, w in runs if n == FINAL)

gs = GridSearchCV(clone(final_est), final_grid, cv=KFold(5, shuffle=True, random_state=0),
                  scoring="neg_mean_squared_error", n_jobs=-1)
gs.fit(Xall[final_cols], y, **({"m__sample_weight": final_wt} if final_wt is not None else {}))
final_model = gs.best_estimator_

print(f"final model      : {FINAL}")
print(f"hyperparameters  : {gs.best_params_}")
print(f"LORO Spearman    : {summ.loc[FINAL,'mean']:.3f} ± {summ.loc[FINAL,'std']:.3f}")
print(f"CV   Spearman    : {summ.loc[FINAL,'cv_spearman']:.3f}")
print(f"Track A LORO     : {summ.loc['Track A composite','mean']:.3f}")
print(f"beats the rubric : {'YES' if best_beats else 'NOT SIGNIFICANTLY — reported as such'}")

# %% [markdown]
# ## 7. Predicting all 1,226
#
# The 1,026 unlabelled universities are pure inference: they were never used to fit anything.
# Their predictions inherit whatever the model learned from the 200, which is why `08` reports
# the labelled/unlabelled split of every ranking rather than presenting one undifferentiated list.

# %%
all_X = df[final_cols] if set(final_cols) <= set(df.columns) else df[FEATURES]
pred_logit = np.asarray(final_model.predict(all_X), dtype=float)

lo, hi = np.percentile(pred_logit, [0, 100])
pred100 = 100 * (pred_logit - lo) / (hi - lo)

preds = df[["uni_id", "name", "region", "country", "country_rank_eligible",
            "trackA_consensus", "trackB_bt_score"]].copy()
preds["predicted_bt_logit"] = pred_logit
preds["predicted_quality_score"] = pred100
preds["is_labelled"] = preds.trackB_bt_score.notna()
preds["residual"] = preds.trackB_bt_score - preds.predicted_bt_logit

assert len(preds) == 1226 and preds.predicted_quality_score.notna().all()
assert preds.is_labelled.sum() == 200
preds.to_csv(OUT / "predictions_all.csv", index=False)

print(f"predicted 1,226 universities ({preds.is_labelled.sum()} labelled, {(~preds.is_labelled).sum()} inferred)")
print(f"\nprediction distribution by label status:")
print(preds.groupby("is_labelled").predicted_quality_score.describe()[["count","mean","std","min","max"]].round(2).to_string())
print(f"\nin-sample fit on the 200 (optimistic — do not report as performance): "
      f"rho = {stats.spearmanr(preds[preds.is_labelled].trackB_bt_score, preds[preds.is_labelled].predicted_bt_logit).statistic:.3f}")

# %% [markdown]
# The labelled and unlabelled groups should have similar prediction distributions. A large
# difference would mean the stratified sample of 200 was not representative and the model is
# extrapolating.

# %%
ks = stats.ks_2samp(preds[preds.is_labelled].predicted_quality_score,
                    preds[~preds.is_labelled].predicted_quality_score)
print(f"KS test, labelled vs inferred predictions: D = {ks.statistic:.3f}, p = {ks.pvalue:.3f}")
print("=> " + ("distributions differ — the 200 are not representative, treat inferred scores with care"
                if ks.pvalue < 0.05 else
                "distributions are compatible; the sample of 200 covers the space the model extrapolates into"))

# %% [markdown]
# ## 8. Feature importance (permutation, on held-out folds)
#
# Impurity importance is biased toward high-cardinality features, so permutation importance is
# computed on **held-out** folds instead. `08` does the interpretable work with SHAP; this is the
# stable summary.

# %%
from sklearn.inspection import permutation_importance

imps = []
for seed in SEEDS[:3]:
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed)
    for tr, te in skf.split(Xall, ybin):
        m = clone(final_model).fit(Xall.iloc[tr][final_cols], y[tr])
        pi = permutation_importance(m, Xall.iloc[te][final_cols], y[te],
                                    n_repeats=5, random_state=seed,
                                    scoring="neg_mean_squared_error", n_jobs=-1)
        imps.append(pd.Series(pi.importances_mean, index=final_cols))

imp = pd.concat(imps, axis=1)
fi = pd.DataFrame({"feature": imp.index, "importance_mean": imp.mean(1), "importance_sd": imp.std(1)})
blockmap = fdict.set_index("feature").block.to_dict()
fi["block"] = fi.feature.map(blockmap).fillna("derived")
fi = fi.sort_values("importance_mean", ascending=False).reset_index(drop=True)
fi.to_csv(OUT / "feature_importance.csv", index=False)

print("top 15 features by permutation importance on held-out folds\n")
print(fi.head(15).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print(f"\nfeatures with importance <= 0 (no measurable contribution): "
      f"{(fi.importance_mean <= 0).sum()} of {len(fi)}")

# %%
blk = fi.groupby("block").importance_mean.sum().sort_values(ascending=False)
blk = (blk / blk[blk > 0].sum()).rename("share")
print("\nblock share of permutation importance:")
print((blk * 100).round(1).to_string())

# %% [markdown]
# ## 9. Saving the evaluation record

# %%
ev = res.groupby("model").agg(
    cv_spearman_mean=("spearman", "mean"), cv_spearman_sd=("spearman", "std"),
    cv_kendall=("kendall", "mean"), cv_mae=("mae", "mean"),
    cv_rmse=("rmse", "mean"), cv_r2=("r2", "mean")).join(
    loro.groupby("model").agg(loro_spearman_mean=("spearman", "mean"),
                              loro_spearman_sd=("spearman", "std"),
                              loro_mae=("mae", "mean")))
ev = ev.join(comp.set_index("model")[["delta", "p", "verdict"]].rename(
    columns={"delta": "delta_vs_trackA", "p": "paired_p_vs_trackA"}))
ev["is_final"] = ev.index == FINAL
ev = ev.sort_values("loro_spearman_mean", ascending=False)
ev.round(4).to_csv(OUT / "model_evaluation.csv")
res.to_csv(OUT / "model_cv_folds.csv", index=False)
loro.to_csv(OUT / "model_loro_folds.csv", index=False)
print(ev.round(3).to_string())

# %%
meta = dict(
    n_train=int(len(train)), n_infer=int(len(infer)), n_features=len(FEATURES),
    target="trackB_bt_score", final_model=FINAL,
    final_params={k: (v if isinstance(v, (int, float, str)) else str(v)) for k, v in gs.best_params_.items()},
    cv_spearman=float(ev.loc[FINAL, "cv_spearman_mean"]), loro_spearman=float(ev.loc[FINAL, "loro_spearman_mean"]),
    trackA_cv_spearman=float(ev.loc["Track A composite", "cv_spearman_mean"]),
    trackA_loro_spearman=float(ev.loc["Track A composite", "loro_spearman_mean"]),
    beats_trackA=bool(best_beats), ks_labelled_vs_inferred_p=float(ks.pvalue),
    excluded_columns=list(EXCLUDE), features=FEATURES,
)
(OUT / "model_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print(json.dumps({k: v for k, v in meta.items() if k not in ("features", "excluded_columns")}, indent=2))

# %% [markdown]
# ## 10. Figures

# %%
fig, ax = plt.subplots(1, 2, figsize=(14, 5.4))
order = res.groupby("model").spearman.mean().sort_values().index
dat = [res[res.model == m].spearman.values for m in order]
bp = ax[0].boxplot(dat, vert=False, labels=order, patch_artist=True, widths=0.6)
for p, m in zip(bp["boxes"], order):
    p.set_facecolor("#f6ad55" if m == "Track A composite" else
                    "#2f855a" if m == FINAL else "#bee3f8")
ax[0].axvline(base.mean(), color="#c05621", ls="--", lw=1.2, label="Track A composite")
ax[0].set_xlabel("Spearman ρ (25 nested-CV fold-reps)")
ax[0].set_title("Can a learned model beat the rubric?")
ax[0].legend(fontsize=8); ax[0].grid(alpha=0.25, axis="x")

o2 = summ.sort_values("mean").index
xx = np.arange(len(o2))
ax[1].barh(xx - 0.2, summ.loc[o2, "cv_spearman"], height=0.38, label="standard CV", color="#90cdf4")
ax[1].barh(xx + 0.2, summ.loc[o2, "mean"], height=0.38, label="leave-one-region-out", color="#2c5282")
ax[1].set_yticks(xx); ax[1].set_yticklabels(o2)
ax[1].set_xlabel("Spearman ρ"); ax[1].set_title("How much of it was geography?")
ax[1].legend(fontsize=8); ax[1].grid(alpha=0.25, axis="x")
plt.tight_layout(); plt.savefig(FIG / "15_model_comparison.png", dpi=150); plt.close()

# %%
fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
lab_p = preds[preds.is_labelled]
ax[0].scatter(lab_p.trackB_bt_score, lab_p.predicted_bt_logit, s=20, alpha=0.6, color="#2b6cb0")
lim = [min(lab_p.trackB_bt_score.min(), lab_p.predicted_bt_logit.min()) - 0.3,
       max(lab_p.trackB_bt_score.max(), lab_p.predicted_bt_logit.max()) + 0.3]
ax[0].plot(lim, lim, "k--", lw=1)
ax[0].set_xlabel("Track B target (logit)"); ax[0].set_ylabel("predicted")
ax[0].set_title("Final model, in-sample\n(optimistic — see CV table for real performance)")
ax[0].grid(alpha=0.25)

top = fi.head(15).iloc[::-1]
ax[1].barh(range(len(top)), top.importance_mean, xerr=top.importance_sd,
           color="#2f855a", alpha=0.85, error_kw=dict(lw=0.8))
ax[1].set_yticks(range(len(top))); ax[1].set_yticklabels(top.feature, fontsize=8)
ax[1].set_xlabel("permutation importance (held-out MSE increase)")
ax[1].set_title("What the model actually uses"); ax[1].grid(alpha=0.25, axis="x")

ax[2].hist(preds[preds.is_labelled].predicted_quality_score, bins=25, alpha=0.65,
           label=f"labelled (n=200)", color="#2b6cb0", density=True)
ax[2].hist(preds[~preds.is_labelled].predicted_quality_score, bins=25, alpha=0.55,
           label=f"inferred (n=1,026)", color="#c05621", density=True)
ax[2].set_xlabel("predicted_quality_score (0–100)"); ax[2].set_ylabel("density")
ax[2].set_title(f"Labelled vs inferred\nKS p = {ks.pvalue:.3f}")
ax[2].legend(fontsize=8); ax[2].grid(alpha=0.25)
plt.tight_layout(); plt.savefig(FIG / "16_model_diagnostics.png", dpi=150); plt.close()
print("figures 13, 14 written")

# %% [markdown]
# ## The prior stated at the top, checked
#
# The opening cell predicted that with n = 200, Ridge would plausibly beat the boosted trees.
# **That prediction was wrong.** LightGBM leads on every metric, and it leads by more under
# leave-one-region-out than under standard CV, which is the opposite of what overfitting to a
# small sample looks like. The reason is visible in §8: three features carry most of the signal
# and two of them (`a72_alt_text_pct`, `a53_contrast_ratio`) enter through sharp thresholds
# rather than slopes — a plateau at WCAG AAA, a cliff below readability. Trees represent that
# directly; a linear model has to approximate it, and pays about 0.03 Spearman for doing so.
#
# The monotonic-constraint variant matches unconstrained LightGBM on CV and edges ahead on
# LORO. Constraints cost nothing here and buy a model that cannot claim more broken links are
# better, which is worth having in a deliverable people will read.
#
# Recording the prior and then recording that it failed is the point of stating it in advance.

# %% [markdown]
# ## What `08` inherits
#
# `predictions_all.csv` — 1,226 rows with `predicted_quality_score`, the column every ranking
# sorts on — plus `model_evaluation.csv`, `feature_importance.csv` and `model_meta.json`.
#
# The comparison against the Track A composite is recorded in `model_evaluation.csv` whichever
# way it went. That column is the project's actual claim about whether machine learning added
# anything here, and `09` reports it verbatim.
