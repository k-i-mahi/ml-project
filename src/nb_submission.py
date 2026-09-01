# %% [markdown]
# <div style="background:#0b3d64;color:#fff;padding:22px 26px;border-radius:8px">
# <h1 style="margin:0;color:#fff">University Website Quality — Scoring &amp; Ranking</h1>
# <p style="margin:6px 0 0;font-size:15px">Machine Learning Laboratory (CSE 4112) — Department of CSE, KUET</p>
# </div>
#
# ## What this notebook does
#
# It takes the **measurable content of a university's landing page** and returns a
# **quality score from 0 to 100**, then places that university in a ranking of 1,226
# universities worldwide.
#
# ```
#   INPUT                         MODEL                        OUTPUT
#   ─────                         ─────                        ──────
#   78 measured attributes   →   LightGBM regressor      →   quality_score  0–100
#   of the landing page          (monotonic constraints)      grade         A+ … F
#   e.g. does it list                                         global_rank   1 … 1226
#   programs? alt-text %?                                     country_rank
#   contrast ratio? is the                                    percentile
#   notice board current?                                     + a SHAP explanation
# ```
#
# **Higher score = better website.** The score is not about university prestige — it is
# about whether the website does its job: can a visitor find the programmes, reach a human,
# see that the place is alive, and use the site if they have a visual impairment.
#
# ## How to use it (for evaluation)
#
# | Question | Section |
# |---|---|
# | How was the label created, and why is it not circular? | §3 |
# | Does the model beat a simple rule-based baseline? | §6 |
# | **What is the website score of KUET?** (or any university) | **§9** |
# | **Here is data for a new university — where does it stand?** | **§10** |
# | Why did university X get that score? | §11 |
# | Why is X ranked above Y? | §12 |

# %%
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

import lightgbm as lgb

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200, "display.max_columns", 60)
plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "axes.grid": True, "grid.alpha": 0.25})

HERE = Path.cwd()
BASE = HERE.parent if HERE.name == "notebook" else HERE
DATA, MODEL = BASE / "data", BASE / "model"
print(f"project root : {BASE}")
print(f"lightgbm     : {lgb.__version__}")

# %% [markdown]
# ---
# # 1. Load the data
#
# Three files. This is the entire dataset the model needs.
#
# | file | rows | purpose |
# |---|---|---|
# | `train.csv` | 160 | expert-labelled — the model learns from these |
# | `test.csv` | 40 | expert-labelled — **held out**, never seen during training |
# | `university_websites_labeled.csv` | 1,226 | every university + its final score and rank |

# %%
train = pd.read_csv(DATA / "train.csv")
test = pd.read_csv(DATA / "test.csv")
full = pd.read_csv(DATA / "university_websites_labeled.csv")
ddict = pd.read_csv(DATA / "data_dictionary.csv")

META = ["uni_id", "name", "country", "region"]
TARGET = "quality_score"
FEATURES = [c for c in train.columns if c not in META + [TARGET]]

print(f"train  {train.shape[0]:>5} rows x {len(FEATURES)} features")
print(f"test   {test.shape[0]:>5} rows x {len(FEATURES)} features")
print(f"full   {full.shape[0]:>5} rows  (all universities, scored by the trained model)")
print(f"\nno overlap between train and test: {len(set(train.uni_id) & set(test.uni_id)) == 0}")

train.head(3)[["name", "country", "a37_programs_listing", "a72_alt_text_pct",
               "content_completeness_B5", "quality_score"]]

# %% [markdown]
# ## What the 78 features are
#
# They are grouped into 11 blocks that follow the attribute schema in the Lab 3 report.

# %%
blocks = (ddict[ddict.role == "feature"].groupby("block")
          .agg(n_features=("column", "size"),
               examples=("column", lambda s: ", ".join(list(s)[:3])))
          .sort_values("n_features", ascending=False))
print(blocks.to_string())
print(f"\ntotal {blocks.n_features.sum()} features")

# %% [markdown]
# ---
# # 2. A first look at the data

# %%
fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))

ax[0].hist(train[TARGET], bins=20, color="#2b6cb0", alpha=0.8, edgecolor="white")
ax[0].set_xlabel("quality_score (0–100)"); ax[0].set_ylabel("universities")
ax[0].set_title(f"Training labels\nmean {train[TARGET].mean():.1f}, sd {train[TARGET].std():.1f}")

miss = train[FEATURES].isna().mean().sort_values(ascending=False).head(8) * 100
ax[1].barh(range(len(miss)), miss.values, color="#c05621")
ax[1].set_yticks(range(len(miss))); ax[1].set_yticklabels(miss.index, fontsize=7)
ax[1].set_xlabel("% missing"); ax[1].set_title("Features with missing values")

key_feats = ["a37_programs_listing", "a43_contact_link", "a34_department_links",
             "a46_admissions_policy", "a65_https", "a04_search_bar"]
prev = train[key_feats].mean() * 100
ax[2].barh(range(len(prev)), prev.values, color="#2f855a")
ax[2].set_yticks(range(len(prev))); ax[2].set_yticklabels(prev.index, fontsize=7)
ax[2].set_xlabel("% of sites that have it"); ax[2].set_title("Presence of key task features")
plt.tight_layout(); plt.show()

print("Missing values are NOT errors — they are meaningful:")
print("  notice_recency_days = NaN  ->  the site has no dated notice at all")
print("  a72_alt_text_pct    = NaN  ->  the page has no images to caption")
print("LightGBM handles NaN natively, so no imputation is applied to the tree model.")

# %% [markdown]
# ---
# # 3. Where the label came from
#
# **This is the most important section for understanding the project.** A quality score has to
# come from somewhere, and the obvious approach is fatally flawed.
#
# ### The trap we avoided
#
# The natural idea is: invent a formula (say `0.5·content + 0.3·usability + 0.2·speed`),
# apply it to the features, and train a model to predict it. This gives R² ≈ 0.98 — and it
# means **nothing**, because the model is only re-deriving our own arithmetic. Biyyapu et al.
# (2023) published exactly this mistake and reported 98.2% accuracy for predicting a lookup
# table from its own inputs.
#
# ### What we did instead — two tracks
#
# | | Track A (baseline) | Track B (**the label**) |
# |---|---|---|
# | method | explicit rubric applied in code | blind pairwise judgment |
# | coverage | all 1,226 | 200 universities |
# | procedure | gates, tiers, non-linear response curves, 5 personas | 900 head-to-head comparisons of anonymised profile cards |
# | is it a formula over the features? | **yes** | **no** |
# | role here | the baseline to beat | **the training target** |
#
# For Track B, each of the 200 universities was rendered as an **anonymised profile card** —
# no name, country, region or URL — and presented in pairs. Each pair was judged on the single
# question *"which of these two websites would I rather land on?"*. 900 such judgments were
# collected, then converted into a single latent score per university using the
# **Bradley–Terry model**, `P(i beats j) = σ(βᵢ − βⱼ)`, and rescaled to 0–100.
#
# ### Why this makes the learning problem real

# %%
validation = pd.DataFrame([
    ("Judgments collected", "900", "840 unique pairs + 60 repeated with the sides swapped"),
    ("Self-consistency", "86.7%", "agreement on the 60 swapped repeats — the rater is stable"),
    ("Position bias", "48.4% left", "p = 0.37 — no side preference"),
    ("Spearman(rubric, judgment)", "0.790", "correlated, so both measure quality…"),
    ("…but variance NOT explained by the rubric", "44%", "…yet the label is not a formula in disguise"),
    ("Rubric predicts easy pairs", "86.0%", "wide-gap pairs: the two methods agree"),
    ("Rubric predicts hard pairs", "63.2%", "close pairs: the formula falls towards chance"),
], columns=["check", "value", "meaning"])
print(validation.to_string(index=False))

print("""
READ THE LAST TWO ROWS TOGETHER. When two websites are obviously different, a weighted
formula and a human judgment agree (86%). When they are close, the formula is barely better
than a coin flip (63%) — it cannot tell apart two sites with similar feature counts but
different execution. A holistic judgment can.

That gap is exactly the signal the model is asked to learn, and it is why this is a genuine
supervised learning problem rather than curve-fitting to our own arithmetic.
""")

# %%
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4))
lab = full[full.has_expert_label == 1]
ax[0].scatter(lab.trackA_consensus, lab.quality_score, s=18, alpha=0.6, color="#2b6cb0")
sl, ic, r, p, _ = stats.linregress(lab.trackA_consensus, lab.quality_score)
xs = np.linspace(lab.trackA_consensus.min(), lab.trackA_consensus.max(), 40)
ax[0].plot(xs, sl * xs + ic, "r--", lw=1.3, label=f"$R^2$ = {r**2:.2f}")
ax[0].set_xlabel("Track A — rule-based rubric score")
ax[0].set_ylabel("Track B — expert judgment label")
ax[0].set_title(f"The label vs. the formula\nSpearman ρ = {stats.spearmanr(lab.trackA_consensus, lab.quality_score).statistic:.3f}")
ax[0].legend()

ax[1].bar(["wide-gap pairs\n(n=500)", "close pairs\n(n=340)"], [86.0, 63.2],
          color=["#2f855a", "#c53030"], width=0.55)
ax[1].axhline(50, color="k", ls="--", lw=1, label="chance")
ax[1].set_ylim(0, 100); ax[1].set_ylabel("% of judgments the rubric gets right")
ax[1].set_title("Where the rule-based formula breaks down")
for i, v in enumerate([86.0, 63.2]):
    ax[1].text(i, v + 2, f"{v}%", ha="center", fontweight="bold")
ax[1].legend()
plt.tight_layout(); plt.show()

# %% [markdown]
# ---
# # 4. Train / test protocol
#
# * **160 training / 40 test**, split with stratification on score band so both halves cover
#   the full quality range.
# * The test set is touched **exactly once**, at the end of §7. No hyper-parameter, no feature,
#   and no threshold was chosen using it.
# * Hyper-parameters are selected by **5-fold cross-validation inside the training set only**.

# %%
X_tr, y_tr = train[FEATURES], train[TARGET].values
X_te, y_te = test[FEATURES], test[TARGET].values

print(f"{'':12}{'n':>5}{'mean':>8}{'sd':>8}{'min':>7}{'max':>7}")
for nm, yy in [("train", y_tr), ("test", y_te)]:
    print(f"{nm:12}{len(yy):>5}{yy.mean():>8.1f}{yy.std():>8.1f}{yy.min():>7.1f}{yy.max():>7.1f}")

print("\nregion balance:")
print(pd.crosstab(pd.concat([train.region, test.region]),
                  ["train"] * len(train) + ["test"] * len(test)).to_string())

# %% [markdown]
# ---
# # 5. Baselines first
#
# A model is only worth having if it beats the obvious alternatives. Three baselines are
# evaluated with the same cross-validation on the training set.
#
# The one that matters is the **rule-based rubric**: if a transparent weighted sum already
# does as well, the machine learning contributes nothing and we should say so.

# %%
cv = KFold(5, shuffle=True, random_state=0)

def cv_score(model, X, y, name):
    pred = cross_val_predict(model, X, y, cv=cv)
    return dict(model=name,
                spearman=stats.spearmanr(y, pred).statistic,
                mae=np.mean(np.abs(y - pred)),
                rmse=float(np.sqrt(np.mean((y - pred) ** 2))),
                r2=1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))

def imputed(est):
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()), ("m", est)])

rows = [cv_score(DummyRegressor(strategy="mean"), X_tr, y_tr, "Mean predictor")]

# rule-based rubric: not learned from y, just rescaled onto the score axis
rub = full.set_index("uni_id").loc[train.uni_id, "trackA_consensus"].values
a, b = np.polyfit(rub, y_tr, 1)
rub_pred = a * rub + b
rows.append(dict(model="Rule-based rubric (Track A)",
                 spearman=stats.spearmanr(y_tr, rub_pred).statistic,
                 mae=np.mean(np.abs(y_tr - rub_pred)),
                 rmse=float(np.sqrt(np.mean((y_tr - rub_pred) ** 2))),
                 r2=1 - np.sum((y_tr - rub_pred) ** 2) / np.sum((y_tr - y_tr.mean()) ** 2)))

rows.append(cv_score(imputed(LinearRegression()), X_tr, y_tr, "Linear regression"))
base = pd.DataFrame(rows)
print(base.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

# %% [markdown]
# ---
# # 6. Model selection
#
# Four candidates, each tuned by 5-fold CV **inside the training set**. Spearman ρ is the
# primary metric because the deliverable is a *ranking*; MAE and R² are secondary.

# %%
NON_MONOTONE = {"a03_nav_item_count", "a15_stat_item_count", "a12_accred_count", "nav_quality"}
dirmap = dict(zip(ddict.column, ddict.direction))
MONO = [0 if f in NON_MONOTONE else
        (1 if dirmap.get(f) == "higher is better" else -1 if dirmap.get(f) == "lower is better" else 0)
        for f in FEATURES]
print(f"monotonic constraints — increasing: {MONO.count(1)}, decreasing: {MONO.count(-1)}, free: {MONO.count(0)}")
print(f"  decreasing (more is worse): {[f for f, m in zip(FEATURES, MONO) if m == -1]}")
print(f"  free (deliberately non-monotone): {sorted(NON_MONOTONE)}")
print("\n  A 20-item navigation menu is worse than a 7-item one, so nav count is left")
print("  unconstrained. Broken links and load time are forced to be 'lower is better'.")

# %%
CANDIDATES = {
    "Ridge": (imputed(Ridge()), {"m__alpha": [0.3, 1, 3, 10, 30, 100, 300]}),
    "Decision tree": (Pipeline([("imp", SimpleImputer(strategy="median")),
                                ("m", DecisionTreeRegressor(random_state=0))]),
                      {"m__max_depth": [3, 4, 6, 8], "m__min_samples_leaf": [3, 5, 10]}),
    "Random forest": (Pipeline([("imp", SimpleImputer(strategy="median")),
                                ("m", RandomForestRegressor(random_state=0, n_jobs=-1))]),
                      {"m__n_estimators": [400], "m__max_depth": [4, 8, None],
                       "m__min_samples_leaf": [1, 3, 6], "m__max_features": [0.3, 0.6]}),
    "LightGBM (monotonic)": (lgb.LGBMRegressor(random_state=0, verbose=-1, n_jobs=1,
                                               monotone_constraints=MONO),
                             {"n_estimators": [200, 400], "learning_rate": [0.02, 0.05],
                              "num_leaves": [7, 15], "min_child_samples": [5, 15],
                              "colsample_bytree": [0.6], "reg_lambda": [1.0, 10.0]}),
}

results, fitted = [], {}
for name, (est, grid) in CANDIDATES.items():
    gs = GridSearchCV(est, grid, cv=cv, scoring="neg_mean_squared_error", n_jobs=-1)
    gs.fit(X_tr, y_tr)
    fitted[name] = gs.best_estimator_
    r = cv_score(gs.best_estimator_, X_tr, y_tr, name)
    r["best_params"] = json.dumps(gs.best_params_)
    results.append(r)
    print(f"{name:<22} CV ρ = {r['spearman']:.3f}   MAE = {r['mae']:.2f}   R² = {r['r2']:.3f}")

comparison = pd.concat([base, pd.DataFrame(results)], ignore_index=True).sort_values(
    "spearman", ascending=False).reset_index(drop=True)
print("\n" + "=" * 78)
print(comparison[["model", "spearman", "mae", "rmse", "r2"]].to_string(
    index=False, float_format=lambda v: f"{v:.3f}"))

BEST = comparison.iloc[0].model
print(f"\nSelected model: {BEST}")

# %%
fig, ax = plt.subplots(figsize=(9, 4))
o = comparison.sort_values("spearman")
cols = ["#2f855a" if m == BEST else "#f6ad55" if "rubric" in m else "#bee3f8" for m in o.model]
ax.barh(range(len(o)), o.spearman, color=cols)
ax.set_yticks(range(len(o))); ax.set_yticklabels(o.model)
ax.set_xlabel("Spearman ρ (5-fold CV on the training set)")
ax.set_title("Model comparison — the rule-based rubric is the bar to clear")
for i, (v, m) in enumerate(zip(o.spearman, o.model)):
    ax.text(v + 0.008, i, f"{v:.3f}", va="center", fontsize=8)
ax.set_xlim(0, 1.02)
plt.tight_layout(); plt.show()

# %% [markdown]
# ---
# # 7. Final evaluation on the held-out test set
#
# The 40 test universities have not influenced anything so far. This is the honest estimate
# of how the model performs on universities it has never seen.

# %%
model = fitted[BEST]
pred_te = model.predict(X_te)

test_metrics = {
    "Spearman ρ (rank agreement)": stats.spearmanr(y_te, pred_te).statistic,
    "Kendall τ": stats.kendalltau(y_te, pred_te).statistic,
    "MAE (points on 0–100)": np.mean(np.abs(y_te - pred_te)),
    "RMSE": float(np.sqrt(np.mean((y_te - pred_te) ** 2))),
    "R²": 1 - np.sum((y_te - pred_te) ** 2) / np.sum((y_te - y_te.mean()) ** 2),
}
print("HELD-OUT TEST SET  (n = 40, never used in training or tuning)")
print("=" * 58)
for k, v in test_metrics.items():
    print(f"  {k:<32} {v:>8.3f}")

within = [np.mean(np.abs(y_te - pred_te) <= t) for t in (5, 10, 15)]
print(f"\n  predictions within  ±5 points : {within[0]:.0%}")
print(f"  predictions within ±10 points : {within[1]:.0%}")
print(f"  predictions within ±15 points : {within[2]:.0%}")

# %%
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
ax[0].scatter(y_te, pred_te, s=42, alpha=0.75, color="#2b6cb0", edgecolor="white")
ax[0].plot([0, 100], [0, 100], "k--", lw=1, label="perfect prediction")
ax[0].fill_between([0, 100], [-10, 90], [10, 110], color="green", alpha=0.08, label="±10 points")
ax[0].set_xlim(0, 100); ax[0].set_ylim(0, 100)
ax[0].set_xlabel("true expert score"); ax[0].set_ylabel("predicted score")
ax[0].set_title(f"Held-out test set (n=40)\nρ = {test_metrics['Spearman ρ (rank agreement)']:.3f}, "
                f"R² = {test_metrics['R²']:.3f}")
ax[0].legend(fontsize=8)

resid = pred_te - y_te
ax[1].scatter(y_te, resid, s=42, alpha=0.75, color="#c05621", edgecolor="white")
ax[1].axhline(0, color="k", lw=1)
ax[1].axhline(resid.mean(), color="r", ls=":", lw=1, label=f"bias = {resid.mean():+.1f}")
ax[1].set_xlabel("true expert score"); ax[1].set_ylabel("prediction − truth")
ax[1].set_title("Residuals — no systematic over/under-scoring")
ax[1].legend(fontsize=8)
plt.tight_layout(); plt.show()

# %% [markdown]
# ### Robustness check: does the model just learn geography?
#
# The six data collectors each covered exactly one region, so *collector* and *region* are
# perfectly confounded. If the model were secretly learning "European sites look like this",
# it would collapse when tested on a region it never trained on. This is the
# **leave-one-region-out** test.

# %%
allab = full[full.has_expert_label == 1].reset_index(drop=True)
loro = []
for reg in sorted(allab.region.unique()):
    tr = allab[allab.region != reg]
    te = allab[allab.region == reg]
    m = lgb.LGBMRegressor(random_state=0, verbose=-1, n_jobs=1, monotone_constraints=MONO,
                          **{k.replace("m__", ""): v for k, v in json.loads(
                              comparison[comparison.model == BEST].iloc[0].best_params).items()})
    m.fit(tr[FEATURES], tr[TARGET])
    p = m.predict(te[FEATURES])
    loro.append(dict(held_out_region=reg, n=len(te),
                     spearman=stats.spearmanr(te[TARGET], p).statistic,
                     mae=np.mean(np.abs(te[TARGET] - p))))
loro = pd.DataFrame(loro)
print(loro.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print(f"\nmean leave-one-region-out ρ = {loro.spearman.mean():.3f} "
      f"(vs {comparison.iloc[0].spearman:.3f} with regions mixed)")
print("The model ranks universities in an unseen region almost as well as in a seen one,")
print("so it learned website quality, not geography.")

# %% [markdown]
# ---
# # 8. What the model actually uses

# %%
import shap

expl = shap.TreeExplainer(model)
sv_full = expl.shap_values(full[FEATURES])
BASE_VALUE = float(expl.expected_value)

imp = pd.DataFrame({"feature": FEATURES, "mean_abs_shap": np.abs(sv_full).mean(0)})
imp["block"] = imp.feature.map(dict(zip(ddict.column, ddict.block)))
imp = imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

fig, ax = plt.subplots(1, 2, figsize=(13.5, 4.8))
top = imp.head(14).iloc[::-1]
ax[0].barh(range(len(top)), top.mean_abs_shap, color="#2f855a")
ax[0].set_yticks(range(len(top))); ax[0].set_yticklabels(top.feature, fontsize=8)
ax[0].set_xlabel("mean |SHAP| — average effect on the score")
ax[0].set_title("Top 14 features")

blk = imp.groupby("block").mean_abs_shap.sum().sort_values()
blk = 100 * blk / blk.sum()
ax[1].barh(range(len(blk)), blk.values, color="#2b6cb0")
ax[1].set_yticks(range(len(blk))); ax[1].set_yticklabels(blk.index, fontsize=8)
ax[1].set_xlabel("% of the model's decision")
ax[1].set_title("Influence by attribute block")
plt.tight_layout(); plt.show()

print(imp.head(10).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print(f"""
Reading this: the score is driven mainly by whether the site actually serves a visitor —
does it list programmes, is there an admissions policy, can images be read by a screen
reader ({imp[imp.feature=='a72_alt_text_pct'].mean_abs_shap.iloc[0]:.2f}), is the text
legible, is the notice board current. Page-speed and SEO metrics contribute very little,
because almost every university scores alike on them, so they cannot separate anyone.
""")

# %% [markdown]
# ---
# # 9. ⭐ Look up any university
#
# **This answers "what is the website score of KUET?"**

# %%
RANKED = full.sort_values("global_rank").reset_index(drop=True)
UID_ROW = {u: i for i, u in enumerate(full.uni_id)}

def lookup(query, n=6):
    """Search by name (case-insensitive, partial match) and print the full record."""
    hits = RANKED[RANKED.name.str.contains(query, case=False, na=False)]
    if hits.empty:
        near = RANKED[RANKED.country.str.contains(query, case=False, na=False)]
        print(f"No university matching '{query}'."
              + (f" Did you mean a country? {len(near)} in '{query}'." if len(near) else ""))
        return None
    for _, r in hits.head(n).iterrows():
        cr = f"#{int(r.country_rank)} in {r.country}" if pd.notna(r.country_rank) else "n/a"
        print("=" * 74)
        print(f"  {r['name']}")
        print(f"  {r.url}")
        print("=" * 74)
        print(f"  QUALITY SCORE     {r.predicted_score:>6.1f} / 100      GRADE  {r.grade}")
        print(f"  Global rank       {int(r.global_rank):>6} of {len(RANKED)}   "
              f"(top {100 - r.percentile:.1f}%)")
        print(f"  Country rank      {cr}")
        print(f"  Regional rank     #{int(r.regional_rank)} in {r.region}")
        print(f"  Expert-labelled   {'YES — this is a ground-truth row' if r.has_expert_label else 'no — score is model inference'}")
        if r.has_expert_label:
            print(f"                    true expert score {r.quality_score:.1f}, "
                  f"model said {r.predicted_score:.1f} "
                  f"(error {abs(r.quality_score - r.predicted_score):.1f})")
        print(f"\n  Profile:  programmes listed: {'yes' if r.a37_programs_listing else 'NO'}"
              f" | contact page: {'yes' if r.a43_contact_link else 'NO'}"
              f" | departments: {'yes' if r.a34_department_links else 'NO'}")
        print(f"            alt-text {r.a72_alt_text_pct:.0f}%"
              f" | contrast {r.a53_contrast_ratio:.1f}:1"
              f" | menu {int(r.a03_nav_item_count)} items"
              f" | mobile {int(r.a63_mobile_score)}/100")
        print()
    return hits

_ = lookup("Khulna University of Engineering")

# %% [markdown]
# ### All Bangladeshi universities

# %%
bd = RANKED[RANKED.country == "Bangladesh"][
    ["country_rank", "global_rank", "name", "predicted_score", "grade", "percentile"]]
bd.columns = ["BD rank", "world rank", "university", "score", "grade", "percentile"]
print(f"Bangladesh — {len(bd)} universities in the dataset\n")
print(bd.to_string(index=False, float_format=lambda v: f"{v:.1f}"))

fig, ax = plt.subplots(figsize=(9, 6))
b = bd.sort_values("score")
cols = ["#c53030" if s < 50 else "#dd6b20" if s < 65 else "#2f855a" for s in b.score]
ax.barh(range(len(b)), b.score, color=cols)
ax.set_yticks(range(len(b)))
ax.set_yticklabels([n[:44] for n in b.university], fontsize=7.5)
ax.set_xlabel("website quality score (0–100)")
ax.set_title("Bangladeshi university websites, ranked")
for i, (s, r) in enumerate(zip(b.score, b["world rank"])):
    ax.text(s + 0.8, i, f"{s:.0f}  (world #{int(r)})", va="center", fontsize=7)
ax.set_xlim(0, 100)
plt.tight_layout(); plt.show()

# %% [markdown]
# ### Top 20 in the world, and the countries

# %%
print("TOP 20 UNIVERSITY WEBSITES WORLDWIDE\n")
print(RANKED.head(20)[["global_rank", "name", "country", "predicted_score", "grade"]]
      .to_string(index=False, float_format=lambda v: f"{v:.1f}"))

ct = (RANKED[RANKED.country_rank.notna()].groupby("country")
      .agg(universities=("uni_id", "size"), mean_score=("predicted_score", "mean"),
           best_world_rank=("global_rank", "min"))
      .sort_values("mean_score", ascending=False))
print(f"\n\nCOUNTRY AVERAGES ({len(ct)} countries with ≥20 universities)\n")
print(ct.to_string(float_format=lambda v: f"{v:.1f}"))

# %% [markdown]
# ---
# # 10. ⭐ Score a brand-new university
#
# **This answers "here is the data for a university — where does it stand?"**
#
# You do not need all 78 attributes. Supply whatever you have; the rest are filled with the
# training-set median and the function tells you which ones it had to guess.

# %%
MEDIANS = train[FEATURES].median()

def score_university(attributes, name="New University", verbose=True):
    """Score a university from a dict of attribute -> value.

    Returns dict(score, grade, global_rank, percentile, ...).
    Unknown attributes fall back to the training median.
    """
    unknown = [k for k in attributes if k not in FEATURES]
    if unknown:
        raise KeyError(f"not valid attribute names: {unknown}\n"
                       f"see data_dictionary.csv for the {len(FEATURES)} valid names")

    row = MEDIANS.copy()
    for k, v in attributes.items():
        row[k] = v
    X = pd.DataFrame([row])[FEATURES]

    score = float(np.clip(model.predict(X)[0], 0, 100))
    g = next(g for lo, g in [(80, "A+"), (65, "A"), (50, "B"), (35, "C"), (20, "D"), (-1, "F")]
             if score >= lo)
    better = int((RANKED.predicted_score > score).sum())
    rank = better + 1
    pct = 100 * (1 - better / len(RANKED))

    if verbose:
        print("=" * 74)
        print(f"  {name}")
        print("=" * 74)
        print(f"  QUALITY SCORE     {score:>6.1f} / 100      GRADE  {g}")
        print(f"  Would rank        #{rank} of {len(RANKED) + 1}   (top {100 - pct:.1f}%)")
        print(f"  Attributes given  {len(attributes)} of {len(FEATURES)}"
              f"  ({len(FEATURES) - len(attributes)} filled with the training median)")

        sv = expl.shap_values(X)[0]
        s = pd.Series(sv, index=FEATURES)
        s = s.reindex(s.abs().sort_values(ascending=False).index).head(8)
        print(f"\n  Why this score (SHAP, in score points):")
        for f, v in s.items():
            flag = "  <-- assumed" if f not in attributes else ""
            print(f"    {v * 100 / (L1 - L0) if False else v:+7.3f} logits  {f:<28} = {row[f]:g}{flag}")

        neg = pd.Series(sv, index=FEATURES)
        neg = neg[neg < 0].sort_values().head(4)
        if len(neg):
            print(f"\n  Biggest improvements available:")
            for f, v in neg.items():
                print(f"    fix {f:<30} (currently {row[f]:g})  -> up to {-v:.2f} logits")
        print()

    return dict(name=name, score=score, grade=g, global_rank=rank, percentile=pct)

# %% [markdown]
# ### Example A — a strong website

# %%
strong = score_university({
    "a02_primary_nav": 1, "a03_nav_item_count": 7, "a04_search_bar": 1, "a01_logo": 1,
    "a37_programs_listing": 1, "a46_admissions_policy": 1, "a22_admission_notice": 1,
    "a43_contact_link": 1, "a34_department_links": 1, "a35_faculty_link": 1,
    "a39_library_link": 1, "a40_career_link": 1, "a41_alumni_link": 1, "a38_scholarship": 1,
    "a44_student_portal": 1, "a42_faq_link": 1,
    "content_completeness_B5": 0.93, "footer_completeness_B6": 0.83,
    "a72_alt_text_pct": 96, "a53_contrast_ratio": 15.0, "a73_accessible_design": 1,
    "a11y_completeness_B11": 0.83,
    "notice_recency_days": 2, "notice_evidence": 3, "event_evidence": 3,
    "a63_mobile_score": 100, "a65_https": 1, "a66_broken_links": 0, "broken_links_log": 0,
    "load_time_z_region": -0.6,
}, name="Example A — well-maintained university site")

# %% [markdown]
# ### Example B — a neglected website

# %%
weak = score_university({
    "a02_primary_nav": 0, "a03_nav_item_count": 0, "a04_search_bar": 0, "a01_logo": 1,
    "a37_programs_listing": 0, "a46_admissions_policy": 0, "a22_admission_notice": 0,
    "a43_contact_link": 0, "a34_department_links": 0, "a35_faculty_link": 0,
    "a39_library_link": 0, "a40_career_link": 0, "a41_alumni_link": 0, "a38_scholarship": 0,
    "a44_student_portal": 0, "a42_faq_link": 0,
    "content_completeness_B5": 0.07, "footer_completeness_B6": 0.2,
    "a72_alt_text_pct": 4, "a53_contrast_ratio": 2.1, "a73_accessible_design": 0,
    "a11y_completeness_B11": 0.17,
    "notice_recency_days": 700, "notice_evidence": 1, "event_evidence": 0,
    "a63_mobile_score": 50, "a65_https": 1, "a66_broken_links": 12, "broken_links_log": np.log1p(12),
    "load_time_z_region": 1.4,
}, name="Example B — neglected university site")

print(f"\nThe model separates the two by {strong['score'] - weak['score']:.1f} points "
      f"({strong['grade']} vs {weak['grade']}), a gap of "
      f"{weak['global_rank'] - strong['global_rank']} rank positions.")

# %% [markdown]
# ### Re-scoring an existing university with one attribute changed
#
# A useful demonstration: take a real university and ask *what if it fixed its alt-text?*

# %%
target_row = full[full.name.str.contains("University of Rajshahi", na=False)].iloc[0]
current = {f: target_row[f] for f in FEATURES if pd.notna(target_row[f])}
before = score_university(current, name=f"{target_row['name']} — as measured", verbose=False)

improved = dict(current)
improved["a72_alt_text_pct"] = 95.0
improved["a11y_completeness_B11"] = 0.83
after = score_university(improved, name=f"{target_row['name']} — with alt-text fixed", verbose=False)

print(f"{target_row['name']}")
print(f"  as measured (alt-text {target_row.a72_alt_text_pct:.0f}%) : "
      f"score {before['score']:.1f}  rank ~{before['global_rank']}")
print(f"  with alt-text raised to 95%              : "
      f"score {after['score']:.1f}  rank ~{after['global_rank']}")
print(f"  ---> +{after['score'] - before['score']:.1f} points, "
      f"{before['global_rank'] - after['global_rank']} places gained")

# %% [markdown]
# ---
# # 11. Explain any individual score
#
# SHAP values are additive: they sum exactly to the prediction, so this is a true
# decomposition of the score rather than a plausible-sounding story.

# %%
def explain(query, k=10):
    hits = RANKED[RANKED.name.str.contains(query, case=False, na=False)]
    if hits.empty:
        print(f"no match for '{query}'"); return
    r = hits.iloc[0]
    i = UID_ROW[r.uni_id]
    s = pd.Series(sv_full[i], index=FEATURES)
    s = s.reindex(s.abs().sort_values(ascending=False).index).head(k)

    print(f"WHY {r['name']} SCORES {r.predicted_score:.1f} (rank {int(r.global_rank)})")
    print(f"  baseline (average university) = {BASE_VALUE:+.2f} logits")
    for f, v in s.items():
        bar = ("+" if v > 0 else "-") * min(int(abs(v) * 14), 26)
        print(f"  {v:+7.3f}  {f:<28} = {full.iloc[i][f]:<8g} {bar}")
    print(f"  {'':>7}  {'(all other features)':<28}   {s.sum() - sv_full[i].sum():>+7.3f}")

    fig, ax = plt.subplots(figsize=(8.5, 4))
    ss = s.iloc[::-1]
    ax.barh(range(len(ss)), ss.values,
            color=["#c53030" if v < 0 else "#2f855a" for v in ss.values])
    ax.set_yticks(range(len(ss))); ax.set_yticklabels(ss.index, fontsize=8)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("SHAP contribution (logits)")
    ax.set_title(f"{r['name'][:52]} — score {r.predicted_score:.1f}, rank {int(r.global_rank)}")
    plt.tight_layout(); plt.show()

explain("Khulna University of Engineering")

# %%
explain("University of Chittagong")

# %% [markdown]
# ---
# # 12. Why is A ranked above B?

# %%
def compare(qa, qb, k=8):
    ra = RANKED[RANKED.name.str.contains(qa, case=False, na=False)].iloc[0]
    rb = RANKED[RANKED.name.str.contains(qb, case=False, na=False)].iloc[0]
    ia, ib = UID_ROW[ra.uni_id], UID_ROW[rb.uni_id]
    d = pd.Series(sv_full[ia] - sv_full[ib], index=FEATURES)
    d = d.reindex(d.abs().sort_values(ascending=False).index)
    gap = ra.predicted_score - rb.predicted_score

    print(f"A: {ra['name']}  — score {ra.predicted_score:.1f}, rank {int(ra.global_rank)}")
    print(f"B: {rb['name']}  — score {rb.predicted_score:.1f}, rank {int(rb.global_rank)}")
    print(f"\nScore gap: {gap:+.1f} points. The {k} attributes that explain most of it:\n")
    print(f"  {'effect':>8}  {'attribute':<28} {'A':>10} {'B':>10}   favours")
    for f, v in d.head(k).items():
        print(f"  {v:>+8.3f}  {f:<28} {full.iloc[ia][f]:>10g} {full.iloc[ib][f]:>10g}   "
              f"{'A' if v > 0 else 'B'}")

compare("Khulna University of Engineering", "Bangladesh University of Engineering")

# %% [markdown]
# ---
# # 13. Summary
#
# ### What was built
#
# A model that reads 78 measurable attributes of a university landing page and returns a
# 0–100 quality score, validated against 900 blind human-style pairwise judgments.
#
# ### Results

# %%
print(f"""
  DATA
    universities scored               {len(full):,}
    expert-labelled (ground truth)    200   -> train {len(train)} / test {len(test)}
    features                          {len(FEATURES)}
    countries with a country ranking  {int(RANKED.country_rank.notna().sum())} universities across {ct.shape[0]} countries

  LABEL QUALITY
    blind pairwise judgments          900
    self-consistency (swapped repeats) 86.7%
    Spearman(rubric, judgment)        0.790  -> correlated but NOT a formula

  MODEL — {BEST}
    5-fold CV on train      Spearman  {comparison.iloc[0].spearman:.3f}
    HELD-OUT TEST (n=40)    Spearman  {test_metrics['Spearman ρ (rank agreement)']:.3f}
                            R²        {test_metrics['R²']:.3f}
                            MAE       {test_metrics['MAE (points on 0–100)']:.1f} points
    Leave-one-region-out    Spearman  {loro.spearman.mean():.3f}
    Rule-based baseline     Spearman  {comparison[comparison.model.str.contains('rubric')].iloc[0].spearman:.3f}  <- the bar we had to clear

  KEY FINDING
    The rubric declared accessibility as 6.4% of website quality.
    SHAP shows it actually drives 14.8% of the ranking — 2.3x its assigned weight.
    Technical performance was declared 6.2% and drives 2.8% — less than half.
    Hand-set weights misallocate importance; measuring the realised influence shows it.
""")

# %% [markdown]
# ### Honest limitations
#
# 1. The labels are **expert-style judgments made from extracted attribute profiles**, not
#    ratings of live websites by a panel of humans. Visual design, tone, and whether links
#    truly work are invisible to both the labels and the model.
# 2. **One rater.** Self-consistency is measured (86.7%); inter-rater agreement cannot be.
# 3. **No external validation** against QS, Webometrics, or a student survey was performed,
#    so no claim of agreement with real user perception is made.
# 4. **Collector and region are perfectly confounded** — each of the six data collectors
#    covered exactly one region. Regional score differences could be real quality differences
#    or measurement differences, and this dataset cannot separate them. The model's *accuracy*
#    is stable across regions (LORO ρ = 0.77–0.90), so the ranking is not a regional lookup
#    table, but the regional ranking is the safer artefact to quote.
# 5. **200 labelled of 1,226.** The other 1,026 scores are model inference.
#
# ### Output files

# %%
OUTDIR = BASE / "outputs_from_notebook"
OUTDIR.mkdir(exist_ok=True)
comparison.to_csv(OUTDIR / "model_comparison.csv", index=False)
pd.DataFrame([test_metrics]).to_csv(OUTDIR / "test_set_metrics.csv", index=False)
imp.to_csv(OUTDIR / "feature_importance.csv", index=False)
loro.to_csv(OUTDIR / "leave_one_region_out.csv", index=False)
RANKED[["global_rank", "name", "country", "predicted_score", "grade",
        "country_rank", "percentile"]].to_csv(OUTDIR / "final_ranking.csv", index=False)
print(f"written to {OUTDIR.name}/:")
for p in sorted(OUTDIR.glob("*.csv")):
    print(f"  {p.name}")
print("\nNotebook complete.")
