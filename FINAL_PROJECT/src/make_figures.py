"""Regenerate the figures used in report/report.tex as standalone PNGs."""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (ExtraTreesRegressor, GradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
import lightgbm as lgb

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 200, "font.size": 8, "axes.grid": True, "grid.alpha": .25})

ROOT = Path(__file__).resolve().parent.parent
DATA, FIG = ROOT / "data", ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

full = pd.read_csv(DATA / "university_website_scores.csv")
train = pd.read_csv(DATA / "train.csv")
test = pd.read_csv(DATA / "test.csv")
ddict = pd.read_csv(DATA / "data_dictionary.csv")

META = ["uni_id", "name", "url", "country", "region", "rank", "regional_rank", "country_rank"]
T = "website_score"
F = [c for c in train.columns if c not in META + [T, "grade"]]
X_tr, y_tr, X_te, y_te = train[F], train[T].values, test[F], test[T].values

# ======================================================================= fig 1: the target
gated = (full.a02_primary_nav == 0) | (full.a65_https == 0)
fig, ax = plt.subplots(1, 3, figsize=(7.5, 2.9))
ax[0].hist([full[T][~gated], full[T][gated]], bins=35, stacked=True,
           color=["#2b6cb0", "#c53030"], edgecolor="white",
           label=["no gate applied", "gate applied"])
ax[0].set_xlabel("website score"); ax[0].set_ylabel("universities")
ax[0].set_title("Distribution of the score"); ax[0].legend(fontsize=7.5)

order = ["A+", "A", "B", "C", "D", "F"]
gc = full.grade.value_counts().reindex(order)
ax[1].bar(order, gc.values,
          color=["#22543d", "#2f855a", "#68a678", "#d69e2e", "#dd6b20", "#c53030"])
for i, v in enumerate(gc.values):
    ax[1].text(i, v + 5, str(v), ha="center", fontsize=7.5)
ax[1].set_ylabel("universities"); ax[1].set_title("Grade distribution")

reg = full.groupby("region")[T].median().sort_values()
ax[2].barh(range(len(reg)), reg.values, color="#4a5568")
ax[2].set_yticks(range(len(reg)))
ax[2].set_yticklabels([r[:24] for r in reg.index], fontsize=7.0)
ax[2].set_xlabel("median score"); ax[2].set_title("Median score by region")
plt.tight_layout(); plt.savefig(FIG / "fig_target.png"); plt.close()

# ======================================================================= fig 2: the curves
fig, ax = plt.subplots(1, 4, figsize=(7.5, 2.2))
n = np.arange(0, 21)
nav = np.where(n <= 0, 0, np.where(n <= 2, .3, np.where(n <= 4, .75,
      np.where(n <= 9, 1., np.where(n <= 12, .9, np.where(n <= 15, .75, .55))))))
ax[0].step(n, nav, where="mid", color="#2b6cb0", lw=2)
ax[0].fill_between(n, nav, step="mid", alpha=.15, color="#2b6cb0")
ax[0].set_title("Menu size\n(inverted U)", fontsize=8.5); ax[0].set_xlabel("menu items")
ax[0].set_ylabel("sub-score")

d = np.arange(0, 400, 2)
rec = np.where(d <= 7, 1., np.where(d <= 30, .9, np.where(d <= 90, .7,
      np.where(d <= 180, .45, np.where(d <= 365, .2, 0.)))))
ax[1].step(d, rec, where="post", color="#2f855a", lw=2)
ax[1].fill_between(d, rec, step="post", alpha=.15, color="#2f855a")
ax[1].set_title("Notice age\n(staleness decay)", fontsize=8.5); ax[1].set_xlabel("days")

r = np.linspace(1, 21, 200)
con = np.where(r < 3, (r - 1) / 2 * .3, np.where(r < 4.5, .3 + (r - 3) / 1.5 * .3,
      np.where(r < 7, .6 + (r - 4.5) / 2.5 * .4, 1.)))
ax[2].plot(r, con, color="#b7791f", lw=2); ax[2].axvline(7, color="crimson", ls=":", lw=1.2)
ax[2].text(7.5, .3, "WCAG AAA", fontsize=7.0, color="crimson")
ax[2].set_title("Contrast\n(plateau at 7:1)", fontsize=8.5); ax[2].set_xlabel("contrast ratio")

b = np.arange(0, 51)
bro = np.where(b <= 0, 1., np.where(b <= 2, .85, np.where(b <= 5, .65,
      np.where(b <= 15, .4, np.where(b <= 40, .15, 0.)))))
ax[3].step(b, bro, where="post", color="#c53030", lw=2)
ax[3].fill_between(b, bro, step="post", alpha=.15, color="#c53030")
ax[3].set_title("Broken links\n(step penalty)", fontsize=8.5); ax[3].set_xlabel("broken links")
for a in ax:
    a.set_ylim(-.05, 1.08)
plt.tight_layout(); plt.savefig(FIG / "fig_curves.png"); plt.close()

# ======================================================================= models
def scaled(m):
    return Pipeline([("scale", StandardScaler()), ("model", m)])


MODELS = {
    "Mean baseline":     DummyRegressor(strategy="mean"),
    "Linear Regression": scaled(LinearRegression()),
    "Ridge":             scaled(Ridge(alpha=1.0)),
    "Lasso":             scaled(Lasso(alpha=0.1)),
    "k-NN (k=5)":        scaled(KNeighborsRegressor(n_neighbors=5)),
    "SVR (RBF kernel)":  scaled(SVR(C=100, gamma="scale")),
    "Neural Net (MLP)":  TransformedTargetRegressor(
                             scaled(MLPRegressor(hidden_layer_sizes=(128, 64), alpha=1e-2,
                                                 max_iter=4000, early_stopping=True,
                                                 n_iter_no_change=40, random_state=0)),
                             transformer=StandardScaler()),
    "Decision Tree":     DecisionTreeRegressor(random_state=0),
    "Extra Trees":       ExtraTreesRegressor(n_estimators=400, random_state=0, n_jobs=-1),
    "Random Forest":     RandomForestRegressor(n_estimators=400, random_state=0, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(random_state=0),
    "LightGBM":          lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05,
                                           num_leaves=31, random_state=0, verbose=-1),
}
FAMILY = {"Mean baseline": "baseline", "Linear Regression": "linear", "Ridge": "linear",
          "Lasso": "linear", "k-NN (k=5)": "instance", "SVR (RBF kernel)": "kernel",
          "Neural Net (MLP)": "neural", "Decision Tree": "tree", "Extra Trees": "ensemble",
          "Random Forest": "ensemble", "Gradient Boosting": "ensemble", "LightGBM": "ensemble"}

rows, preds = [], {}
cv = KFold(5, shuffle=True, random_state=0)
for name, m in MODELS.items():
    m.fit(X_tr, y_tr)
    p = m.predict(X_te)
    preds[name] = p
    s = cross_val_score(m, X_tr, y_tr, cv=cv, scoring="r2", n_jobs=-1)
    rows.append(dict(model=name, family=FAMILY[name], R2=r2_score(y_te, p),
                     MAE=mean_absolute_error(y_te, p),
                     RMSE=float(np.sqrt(np.mean((y_te - p) ** 2))),
                     Spearman=stats.spearmanr(y_te, p).statistic if p.std() > 0 else 0.0,
                     cv_mean=s.mean(), cv_sd=s.std()))
comp = pd.DataFrame(rows).sort_values("R2", ascending=False).reset_index(drop=True)
comp.to_csv(ROOT / "results" / "model_comparison_report.csv", index=False)

# ======================================================================= fig 3: leaderboard
cmap = {"baseline": "#a0aec0", "linear": "#e53e3e", "instance": "#dd6b20",
        "kernel": "#d69e2e", "neural": "#805ad5", "tree": "#38a169", "ensemble": "#2b6cb0"}
d = comp.iloc[::-1]
cols = [cmap[f] for f in d.family]
fig, ax = plt.subplots(1, 3, figsize=(7.5, 3.6))
for a, col, lab, xmax in [(ax[0], "R2", "R²  (higher is better)", 1.05),
                          (ax[1], "MAE", "MAE in score points  (lower is better)", None),
                          (ax[2], "Spearman", "Spearman ρ  (rank agreement)", 1.05)]:
    a.barh(range(len(d)), d[col], color=cols)
    a.set_yticks(range(len(d)))
    a.set_yticklabels(d.model if col == "R2" else [], fontsize=8.0)
    a.set_xlabel(lab, fontsize=8.5)
    if xmax:
        a.set_xlim(0, xmax)
    for i, v in enumerate(d[col]):
        a.text(max(v, 0) + (.012 if xmax else .2), i, f"{v:.3f}" if xmax else f"{v:.2f}",
               va="center", fontsize=7.0)
h = [plt.Rectangle((0, 0), 1, 1, color=c) for c in cmap.values()]
fig.legend(h, cmap.keys(), fontsize=7.5, ncol=7, loc="lower center",
           bbox_to_anchor=(0.5, -0.02), frameon=False)
ax[0].set_title("Variance explained", fontsize=9.5)
ax[1].set_title("Average error", fontsize=9.5)
ax[2].set_title("Rank agreement", fontsize=9.5)
plt.tight_layout(rect=[0, 0.06, 1, 1]); plt.savefig(FIG / "fig_comparison.png"); plt.close()

# ======================================================================= fig 4: CV
d2 = comp.sort_values("cv_mean").reset_index(drop=True)
fig, ax = plt.subplots(figsize=(5.0, 2.9))
ax.barh(range(len(d2)), d2.cv_mean, xerr=d2.cv_sd, color="#2b6cb0",
        error_kw=dict(ecolor="#1a365d", capsize=3, lw=1))
ax.set_yticks(range(len(d2))); ax.set_yticklabels(d2.model, fontsize=8.0)
ax.set_xlabel("R² — 5-fold cross-validation on the training set (mean ± sd)")
ax.set_xlim(0, 1.05)
plt.tight_layout(); plt.savefig(FIG / "fig_cv.png"); plt.close()

# ======================================================================= tuned model
from sklearn.model_selection import GridSearchCV
grid = {"n_estimators": [300, 600, 900], "learning_rate": [0.03, 0.05, 0.1],
        "num_leaves": [15, 31, 63], "min_child_samples": [5, 10, 20]}
gs = GridSearchCV(lgb.LGBMRegressor(random_state=0, verbose=-1, importance_type="gain"), grid, cv=cv,
                  scoring="r2", n_jobs=-1).fit(X_tr, y_tr)
best = gs.best_estimator_
yp = best.predict(X_te)

# ======================================================================= fig 5: predictions
fig, ax = plt.subplots(1, 3, figsize=(7.5, 2.9))
lims = [min(y_te.min(), yp.min()) - 3, max(y_te.max(), yp.max()) + 3]
ax[0].scatter(y_te, yp, s=15, alpha=.55, color="#2b6cb0", edgecolor="none")
ax[0].plot(lims, lims, "r--", lw=1.2)
ax[0].set_xlabel("actual score"); ax[0].set_ylabel("predicted score")
ax[0].set_title(f"Predicted vs actual (R² = {r2_score(y_te, yp):.3f})", fontsize=9.5)

res = yp - y_te
ax[1].scatter(yp, res, s=15, alpha=.55, color="#2f855a", edgecolor="none")
ax[1].axhline(0, color="crimson", ls="--", lw=1.2)
ax[1].set_xlabel("predicted score"); ax[1].set_ylabel("residual")
ax[1].set_title("Residuals", fontsize=9.5)

ax[2].hist(res, bins=30, color="#4a5568", edgecolor="white")
ax[2].axvline(0, color="crimson", ls="--", lw=1.2)
ax[2].set_xlabel("residual")
ax[2].set_title(f"Error distribution (MAE {np.abs(res).mean():.2f})", fontsize=9.5)
plt.tight_layout(); plt.savefig(FIG / "fig_predictions.png"); plt.close()

# ======================================================================= fig 6: the gates
gate_te = ((test.a02_primary_nav == 0) | (test.a65_https == 0)).values
names = ["Linear Regression", "Ridge", "Lasso", "SVR (RBF kernel)", "Neural Net (MLP)",
         "Decision Tree", "Random Forest", "Gradient Boosting", "LightGBM"]
gaps = pd.DataFrame([dict(model=n,
                          gated=np.abs(y_te - preds[n])[gate_te].mean(),
                          ungated=np.abs(y_te - preds[n])[~gate_te].mean()) for n in names])
gaps["ratio"] = gaps.gated / gaps.ungated
gaps.to_csv(ROOT / "results" / "error_by_gate_report.csv", index=False)

fig, ax = plt.subplots(1, 2, figsize=(7.5, 3.2))
x = np.arange(len(gaps)); w = .38
ax[0].bar(x - w/2, gaps.ungated, w, label="no gate applied", color="#2b6cb0")
ax[0].bar(x + w/2, gaps.gated, w, label="gate applied", color="#c53030")
ax[0].set_xticks(x); ax[0].set_xticklabels(gaps.model, rotation=35, ha="right", fontsize=7.5)
ax[0].set_ylabel("mean absolute error"); ax[0].legend(fontsize=8.0)
ax[0].set_title("Error by whether a gate applied", fontsize=9.0)

lin = preds["Linear Regression"]
ax[1].scatter(y_te[~gate_te], lin[~gate_te], s=15, alpha=.5, color="#2b6cb0", label="no gate")
ax[1].scatter(y_te[gate_te], lin[gate_te], s=26, alpha=.85, color="#c53030", label="gate applied")
ax[1].plot(lims, lims, "k--", lw=1)
ax[1].set_xlabel("actual score"); ax[1].set_ylabel("Linear Regression prediction")
ax[1].set_title("Linear regression over-predicts\ncapped sites", fontsize=9.0)
ax[1].legend(fontsize=8.0)
plt.tight_layout(); plt.savefig(FIG / "fig_gates.png"); plt.close()

# ======================================================================= fig 7: importance
imp = pd.DataFrame({"feature": F, "importance": best.feature_importances_})
imp["importance"] = imp.importance / imp.importance.sum() * 100
imp = imp.sort_values("importance", ascending=False).reset_index(drop=True)
DIM = dict(zip(ddict.column, ddict.dimension))
imp["dimension"] = imp.feature.map(DIM).fillna("—")
imp.to_csv(ROOT / "results" / "feature_importance_report.csv", index=False)

pal = {d: c for d, c in zip(sorted(ddict.dimension.dropna().unique()),
                            ["#2b6cb0", "#2f855a", "#d69e2e", "#c53030", "#805ad5",
                             "#dd6b20", "#38a169", "#a0aec0"])}
top = imp.head(18).iloc[::-1]
bydim = imp.groupby("dimension").importance.sum().sort_values()
fig, ax = plt.subplots(1, 2, figsize=(7.5, 3.8))
ax[0].barh(range(len(top)), top.importance,
           color=[pal.get(d, "#a0aec0") for d in top.dimension])
ax[0].set_yticks(range(len(top))); ax[0].set_yticklabels(top.feature, fontsize=7.0)
ax[0].set_xlabel("share of model importance (%)"); ax[0].set_title("Top 18 features", fontsize=9.5)
ax[1].barh(range(len(bydim)), bydim.values,
           color=[pal.get(d, "#a0aec0") for d in bydim.index])
ax[1].set_yticks(range(len(bydim)))
ax[1].set_yticklabels([d[:36] for d in bydim.index], fontsize=8.0)
ax[1].set_xlabel("total importance (%)")
ax[1].set_title("Importance by dimension", fontsize=9.5)
for i, v in enumerate(bydim.values):
    ax[1].text(v + .4, i, f"{v:.1f}%", va="center", fontsize=7.5)
plt.tight_layout(); plt.savefig(FIG / "fig_importance.png"); plt.close()

# ======================================================================= report numbers
metrics = dict(
    r2=float(r2_score(y_te, yp)), mae=float(mean_absolute_error(y_te, yp)),
    rmse=float(np.sqrt(np.mean((y_te - yp) ** 2))),
    spearman=float(stats.spearmanr(y_te, yp).statistic),
    kendall=float(stats.kendalltau(y_te, yp).statistic),
    within1=float((np.abs(y_te - yp) <= 1).mean()),
    within2=float((np.abs(y_te - yp) <= 2).mean()),
    within5=float((np.abs(y_te - yp) <= 5).mean()),
    best_params=gs.best_params_, best_cv=float(gs.best_score_),
    n_grid=len(gs.cv_results_["params"]),
    n_gated_test=int(gate_te.sum()),
    cv_test_agreement=float(stats.spearmanr(comp.cv_mean, comp.R2).statistic),
)
(ROOT / "results" / "report_numbers.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

# ======================================================================================
# ADDITIONAL ANALYSES
# ======================================================================================
RES = ROOT / "results"
cat = pd.read_csv(DATA / "feature_catalog.csv")
dims = pd.read_csv(DATA / "dimension_scores.csv")
DIMCOLS = [c for c in dims.columns if c.startswith("D") and c[1].isdigit()]
DIMSHORT = [c[:2] for c in DIMCOLS]
W = dict(zip(DIMSHORT, [28, 22, 15, 15, 10, 7, 3]))

# ============================================================ fig 8: Bangladesh case study
bd_mask = (test.country.str.strip().str.casefold() == "bangladesh").values
bd = test[bd_mask].copy()
bd["predicted"] = yp[bd_mask]
bd["actual"] = bd[T].values
bd["error"] = bd.predicted - bd.actual
bd["abs_error"] = bd.error.abs()
bd["rank"] = bd.uni_id.map(dict(zip(full.uni_id, full["rank"])))
bd["country_rank"] = bd.uni_id.map(dict(zip(full.uni_id, full.country_rank)))

SHORT = {
    "Bangladesh University of Engineering and Technology": "BUET",
    "Khulna University of Engineering and Technology": "KUET",
    "Rajshahi University of Engineering and Technology": "RUET",
    "Chittagong University of Engineering and Technology": "CUET",
    "American International University-Bangladesh": "AIUB",
    "Bangladesh University of Professionals": "BUP",
    "Patuakhali Science and Technology University": "PSTU",
    "Daffodil International University": "Daffodil Int. Univ.",
    "Bangladesh University of Textiles": "BUTEX",
    "Islamic University of Technology": "IUT",
    "Bangladesh Agricultural University": "BAU",
    "North South University": "North South Univ.",
    "Jahangirnagar University": "Jahangirnagar Univ.",
    "Bangladesh University of Engineering & Technology": "BUET",
}
bd["short"] = [SHORT.get(n, n.replace("University", "Univ.")) for n in bd.name]
bd = bd.sort_values("actual", ascending=False).reset_index(drop=True)
bd["predicted_rank_in_bd"] = bd.predicted.rank(ascending=False, method="min").astype(int)
bd["actual_rank_in_bd"] = np.arange(1, len(bd) + 1)
bd_out = bd[["name", "short", "url", "actual", "predicted", "error", "grade", "rank",
             "country_rank", "actual_rank_in_bd", "predicted_rank_in_bd"]]
bd_out.to_csv(RES / "bangladesh_predictions_report.csv", index=False)

rest_mae = float(np.abs(y_te - yp)[~bd_mask].mean())
bd_mae = float(bd.abs_error.mean())
bd_stats = dict(
    n=int(len(bd)), mae=bd_mae, rest_mae=rest_mae,
    rmse=float(np.sqrt((bd.error ** 2).mean())),
    r2=float(r2_score(bd.actual, bd.predicted)),
    spearman=float(stats.spearmanr(bd.actual, bd.predicted).statistic),
    max_abs_error=float(bd.abs_error.max()),
    within2=float((bd.abs_error <= 2).mean()),
    mean_score=float(bd.actual.mean()), world_mean=float(full[T].mean()),
    best=bd.iloc[0]["name"], best_score=float(bd.iloc[0].actual),
    n_top200=int((bd["rank"] <= 200).sum()),
)

fig, ax = plt.subplots(1, 3, figsize=(7.5, 3.2))

ax[0].scatter(bd.actual, bd.predicted, s=46, color="#006a4e", edgecolor="white", zorder=3)
lo, hi = bd.actual.min() - 6, bd.actual.max() + 6
ax[0].plot([lo, hi], [lo, hi], "r--", lw=1.1, zorder=1)
# Point labels are omitted here: the centre panel names every university,
# and at this figure size 22 annotations would overlap illegibly.
ax[0].set_xlabel("actual score"); ax[0].set_ylabel("predicted score")
ax[0].set_title(f"{len(bd)} unseen universities\n"
                f"R² = {bd_stats['r2']:.3f}   MAE = {bd_mae:.2f}", fontsize=9.0)

o = bd.sort_values("error")
cols = ["#c53030" if e > 0 else "#2b6cb0" for e in o.error]
ax[1].barh(range(len(o)), o.error, color=cols)
ax[1].axvline(0, color="black", lw=.8)
ax[1].set_yticks(range(len(o))); ax[1].set_yticklabels(o["short"], fontsize=6.5)
ax[1].set_xlabel("prediction error (predicted − actual, points)")
ax[1].set_title("Signed error", fontsize=9.0)

bd_ids = set(bd.uni_id)
d_bd = dims[dims.uni_id.isin(bd_ids)][DIMCOLS].mean().values
d_world = dims[DIMCOLS].mean().values
d_top = dims.nlargest(100, "website_score")[DIMCOLS].mean().values
x = np.arange(7); w_ = .27
ax[2].bar(x - w_, d_bd, w_, label="Bangladesh (22)", color="#006a4e")
ax[2].bar(x, d_world, w_, label="all 1,225", color="#4a5568")
ax[2].bar(x + w_, d_top, w_, label="global top 100", color="#d69e2e")
ax[2].set_xticks(x); ax[2].set_xticklabels(DIMSHORT, fontsize=8.5)
ax[2].set_ylabel("mean sub-score (0–1)"); ax[2].set_ylim(0, 1.05)
ax[2].legend(fontsize=6.5, loc="lower left", framealpha=0.9)
ax[2].set_title("Sub-score profile", fontsize=9.0)
plt.tight_layout(); plt.savefig(FIG / "fig_bangladesh.png"); plt.close()

prof = pd.DataFrame({"dimension": DIMCOLS, "bangladesh": d_bd.round(3),
                     "world": d_world.round(3), "top100": d_top.round(3),
                     "weight": [W[d[:2]] for d in DIMCOLS]})
prof["gap_vs_world_pts"] = ((prof.bangladesh - prof.world) * prof.weight).round(2)
prof["gap_vs_top100_pts"] = ((prof.bangladesh - prof.top100) * prof.weight).round(2)
prof.to_csv(RES / "bangladesh_dimension_profile_report.csv", index=False)

# ============================================================ error by country / region
err = pd.DataFrame({"country": test.country.values, "region": test.region.values,
                    "abs_error": np.abs(y_te - yp)})
by_country = (err.groupby("country").abs_error.agg(["size", "mean"])
              .rename(columns={"size": "n", "mean": "mae"}).sort_values("mae"))
by_country[by_country.n >= 5].round(3).to_csv(RES / "error_by_country.csv")
err.groupby("region").abs_error.agg(["size", "mean"]).round(3).to_csv(RES / "error_by_region.csv")

# ============================================================ fig 9: feature catalogue
fig, ax = plt.subplots(1, 3, figsize=(7.5, 5.9))

grp = cat.assign(scored=cat.scored_by != "--").groupby("source_group").scored.agg(["sum", "size"])
grp["unscored"] = grp["size"] - grp["sum"]
grp = grp.sort_values("size")
ax[0].barh(range(len(grp)), grp["sum"], color="#2b6cb0", label="enters the score")
ax[0].barh(range(len(grp)), grp.unscored, left=grp["sum"], color="#cbd5e0",
           label="observed, not scored")
ax[0].set_yticks(range(len(grp))); ax[0].set_yticklabels(grp.index, fontsize=7.5)
ax[0].set_xlabel("number of features")
ax[0].legend(fontsize=6.5, loc="lower right", framealpha=0.9)
ax[0].set_title(f"{len(cat)} features by page region", fontsize=9.0)

mt = cat.measurement_type.value_counts().sort_values()
ax[1].barh(range(len(mt)), mt.values, color="#805ad5")
ax[1].set_yticks(range(len(mt))); ax[1].set_yticklabels(mt.index, fontsize=7.5)
for i, v in enumerate(mt.values):
    ax[1].text(v + .5, i, str(v), va="center", fontsize=7.5)
ax[1].set_xlabel("number of features")
ax[1].set_title("By measurement type", fontsize=9.0)

sc = cat[cat.scored_by != "--"].sort_values("corr_with_score")
palD = {"D1": "#2b6cb0", "D2": "#2f855a", "D3": "#d69e2e", "D4": "#c53030",
        "D5": "#805ad5", "D6": "#dd6b20", "D7": "#38a169"}
ax[2].barh(range(len(sc)), sc.corr_with_score, color=[palD[d] for d in sc.scored_by])
ax[2].set_yticks(range(len(sc)))
ax[2].set_yticklabels([f"{f[:26]}" for f in sc.feature], fontsize=5.6)
ax[2].axvline(0, color="black", lw=.7)
ax[2].set_xlabel("Pearson r with score")
h = [plt.Rectangle((0, 0), 1, 1, color=palD[k]) for k in palD]
ax[2].legend(h, palD.keys(), fontsize=6.0, ncol=2, loc="lower right", framealpha=0.95)
ax[2].set_title(f"{len(sc)} scored features", fontsize=9.0)
plt.tight_layout(); plt.savefig(FIG / "fig_features.png"); plt.close()

# ============================================================ fig 10: weight sensitivity
base_rank = dims.website_score.rank(ascending=False, method="min").values
cap_v = dims.cap.values
Dmat = dims[DIMCOLS].values
w0 = np.array([W[d[:2]] for d in DIMCOLS], float)


def score_with(w):
    return np.minimum(Dmat @ w, cap_v)


rng = np.random.default_rng(0)
levels = [0.10, 0.20, 0.30, 0.50]
sens_rows = []
for lv in levels:
    rho, top10, top50 = [], [], []
    for _ in range(300):
        w = w0 * (1 + rng.uniform(-lv, lv, 7))
        w = w / w.sum() * 100
        sc_ = score_with(w)
        rho.append(stats.spearmanr(sc_, dims.website_score).statistic)
        r_ = pd.Series(sc_).rank(ascending=False, method="min").values
        top10.append(len(set(np.where(r_ <= 10)[0]) & set(np.where(base_rank <= 10)[0])) / 10)
        top50.append(len(set(np.where(r_ <= 50)[0]) & set(np.where(base_rank <= 50)[0])) / 50)
    sens_rows.append(dict(perturbation=f"±{lv:.0%}", spearman_mean=np.mean(rho),
                          spearman_min=np.min(rho), top10_overlap=np.mean(top10),
                          top50_overlap=np.mean(top50)))
sens = pd.DataFrame(sens_rows)

oat_rows = []
for i, d in enumerate(DIMSHORT):
    for mult, lab in [(0.0, "removed"), (0.5, "halved"), (2.0, "doubled")]:
        w = w0.copy(); w[i] = w0[i] * mult
        w = w / w.sum() * 100
        sc_ = score_with(w)
        r_ = pd.Series(sc_).rank(ascending=False, method="min").values
        oat_rows.append(dict(dimension=d, change=lab, weight_from=w0[i], weight_to=round(w[i], 1),
                             spearman=stats.spearmanr(sc_, dims.website_score).statistic,
                             top10_overlap=len(set(np.where(r_ <= 10)[0])
                                               & set(np.where(base_rank <= 10)[0])) / 10,
                             median_rank_shift=float(np.median(np.abs(r_ - base_rank)))))
oat = pd.DataFrame(oat_rows)
sens.round(4).to_csv(RES / "weight_sensitivity_report.csv", index=False)
oat.round(4).to_csv(RES / "weight_sensitivity_oat_report.csv", index=False)

fig, ax = plt.subplots(1, 3, figsize=(7.5, 2.9))
ax[0].errorbar(range(len(sens)), sens.spearman_mean,
               yerr=[sens.spearman_mean - sens.spearman_min, np.zeros(len(sens))],
               fmt="o-", color="#2b6cb0", capsize=4)
ax[0].set_xticks(range(len(sens))); ax[0].set_xticklabels(sens.perturbation)
ax[0].set_ylabel("Spearman ρ vs published")
ax[0].set_xlabel("perturbation on every weight")
ax[0].set_ylim(0.9, 1.005)
ax[0].set_title("Stability under weight\nperturbation", fontsize=9.0)

ax[1].plot(range(len(sens)), sens.top10_overlap * 100, "o-", color="#c53030", label="top 10")
ax[1].plot(range(len(sens)), sens.top50_overlap * 100, "s-", color="#2f855a", label="top 50")
ax[1].set_xticks(range(len(sens))); ax[1].set_xticklabels(sens.perturbation)
ax[1].set_ylabel("% of the league table retained"); ax[1].set_ylim(0, 105)
ax[1].legend(fontsize=8.0); ax[1].set_xlabel("perturbation")
ax[1].set_title("Top of the table retained", fontsize=9.0)

rem = oat[oat.change == "removed"].set_index("dimension").loc[DIMSHORT]
ax[2].bar(range(7), rem.spearman, color=[palD[d] for d in DIMSHORT])
ax[2].set_xticks(range(7)); ax[2].set_xticklabels(DIMSHORT)
ax[2].set_ylim(0.7, 1.035); ax[2].set_ylabel("Spearman ρ vs published")
for i, v in enumerate(rem.spearman):
    ax[2].text(i, v + .005, f"{v:.3f}", ha="center", fontsize=6.0)
ax[2].set_title("After deleting one dimension", fontsize=9.0)
plt.tight_layout(); plt.savefig(FIG / "fig_sensitivity.png"); plt.close()

# ============================================================ fig 11: block ablation
GRP_OF = dict(zip(cat.feature, cat.source_group))
DIM_OF_F = dict(zip(cat.feature, cat.scored_by))
abl_rows = []
full_lgb = lgb.LGBMRegressor(**gs.best_params_, random_state=0, verbose=-1,
                             importance_type="gain").fit(X_tr, y_tr)
base_r2 = r2_score(y_te, full_lgb.predict(X_te))
base_mae = mean_absolute_error(y_te, full_lgb.predict(X_te))
for dim in ["D1", "D2", "D3", "D4", "D5", "D6", "D7"]:
    keep = [c for c in F if DIM_OF_F.get(c) != dim]
    m = lgb.LGBMRegressor(**gs.best_params_, random_state=0, verbose=-1).fit(X_tr[keep], y_tr)
    p_ = m.predict(X_te[keep])
    abl_rows.append(dict(block=dim, kind="scoring dimension", n_removed=len(F) - len(keep),
                         R2=r2_score(y_te, p_), MAE=mean_absolute_error(y_te, p_)))
for grp_ in sorted(cat.source_group.unique()):
    keep = [c for c in F if GRP_OF.get(c) != grp_]
    m = lgb.LGBMRegressor(**gs.best_params_, random_state=0, verbose=-1).fit(X_tr[keep], y_tr)
    p_ = m.predict(X_te[keep])
    abl_rows.append(dict(block=grp_, kind="source group", n_removed=len(F) - len(keep),
                         R2=r2_score(y_te, p_), MAE=mean_absolute_error(y_te, p_)))
abl = pd.DataFrame(abl_rows)
abl["dR2"] = abl.R2 - base_r2
abl["dMAE"] = abl.MAE - base_mae
abl.round(4).to_csv(RES / "block_ablation_report.csv", index=False)

fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.4))
a1 = abl[abl.kind == "scoring dimension"].sort_values("dMAE")
ax[0].barh(range(len(a1)), a1.dMAE, color=[palD[d] for d in a1.block])
ax[0].set_yticks(range(len(a1))); ax[0].set_yticklabels(a1.block, fontsize=8.5)
ax[0].set_xlabel("increase in test MAE (points)")
ax[0].set_title(f"By scoring dimension (full MAE {base_mae:.2f})", fontsize=9.0)
for i, v in enumerate(a1.dMAE):
    ax[0].text(v, i, f" +{v:.2f}", va="center", fontsize=7.0)

a2 = abl[abl.kind == "source group"].sort_values("dMAE")
ax[1].barh(range(len(a2)), a2.dMAE, color="#4a5568")
ax[1].set_yticks(range(len(a2))); ax[1].set_yticklabels(a2.block, fontsize=7.5)
ax[1].set_xlabel("increase in test MAE (points)")
ax[1].set_title("By page region", fontsize=9.0)
plt.tight_layout(); plt.savefig(FIG / "fig_ablation.png"); plt.close()

# Region names are long; these short forms keep the tick labels readable without clipping.
REGION_SHORT = {
    "South & Central Asia + Oceania": "S & C Asia + Oceania",
    "Western & Northern Europe": "W & N Europe",
    "Eastern & Southern Europe": "E & S Europe",
    "Asia (East & Southeast)": "E & SE Asia",
    "Latin America & Africa": "Latin America & Africa",
    "North America": "North America",
}

# ============================================================ fig 12: EDA / structure
fig, ax = plt.subplots(1, 3, figsize=(7.5, 3.2))

M = dims[DIMCOLS + ["website_score"]].corr().values
im = ax[0].imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
lab = DIMSHORT + ["score"]
ax[0].set_xticks(range(len(lab))); ax[0].set_xticklabels(lab, fontsize=6.5, rotation=45)
ax[0].set_yticks(range(len(lab))); ax[0].set_yticklabels(lab, fontsize=6.5)
for i in range(len(lab)):
    for j in range(len(lab)):
        ax[0].text(j, i, f"{M[i, j]:.2f}".replace("0.", "."), ha="center",
                   va="center", fontsize=5.4,
                   color="white" if abs(M[i, j]) > .55 else "black")
ax[0].set_title("Correlation of sub-scores", fontsize=9.0)
ax[0].grid(False)
fig.colorbar(im, ax=ax[0], fraction=.046).ax.tick_params(labelsize=6)

contrib = pd.DataFrame({d[:2]: dims[f"pts_{d[:2]}"] for d in DIMCOLS})
band = pd.cut(dims.website_score, [-1, 35, 50, 65, 75, 85, 101],
              labels=["F", "D", "C", "B", "A", "A+"])
stack = contrib.groupby(band, observed=False).mean()
bottom = np.zeros(len(stack))
for d in DIMSHORT:
    ax[1].bar(stack.index.astype(str), stack[d], bottom=bottom, label=d, color=palD[d])
    bottom += stack[d].values
ax[1].set_ylabel("mean points contributed"); ax[1].set_xlabel("grade band")
ax[1].legend(fontsize=6.0, ncol=2, loc="upper left", framealpha=0.95)
ax[1].set_title("Points by grade band", fontsize=9.0)

reg_order = full.groupby("region")[T].median().sort_values().index
ax[2].boxplot([full[full.region == r][T].values for r in reg_order], vert=False,
              patch_artist=True, widths=.6,
              boxprops=dict(facecolor="#bee3f8", edgecolor="#2b6cb0"),
              medianprops=dict(color="#c53030", lw=1.4), flierprops=dict(ms=2, alpha=.4))
ax[2].set_yticklabels([REGION_SHORT.get(r, r[:20]) for r in reg_order], fontsize=6.5)
ax[2].set_xlabel("website score")
ax[2].axvline(full[T].mean(), color="#4a5568", ls="--", lw=1,
              label=f"global mean {full[T].mean():.1f}")
ax[2].legend(fontsize=6.5, loc="lower right", framealpha=0.95)
ax[2].set_title("Score by region", fontsize=9.0)
plt.tight_layout(); plt.savefig(FIG / "fig_eda.png"); plt.close()

# ============================================================ extra report numbers
metrics.update(
    n_train=int(len(train)), n_test=int(len(test)),
    bd=bd_stats,
    base_mae=float(base_mae), base_r2=float(base_r2),
    sens_top10_30pct=float(sens.loc[sens.perturbation == "±30%", "top10_overlap"].iloc[0]),
    sens_rho_30pct=float(sens.loc[sens.perturbation == "±30%", "spearman_mean"].iloc[0]),
    sens_rho_50pct=float(sens.loc[sens.perturbation == "±50%", "spearman_mean"].iloc[0]),
)
(RES / "report_numbers.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

print("\nBangladesh holdout (22 universities, none in training):")
for k in ("r2", "mae", "rest_mae", "spearman", "max_abs_error"):
    print(f"  {k:<14}{bd_stats[k]:.4f}")
print("\nweight sensitivity:\n" + sens.round(3).to_string(index=False))
print("\nablation (worst 5 by MAE increase):\n" +
      abl.sort_values("dMAE", ascending=False).head(5).round(3).to_string(index=False))

print("figures written to report/figures/:")
for p in sorted(FIG.glob("*.png")):
    print(f"  {p.name}")
print("\ntuned model on the held-out test set:")
for k in ("r2", "mae", "rmse", "spearman", "kendall"):
    print(f"  {k:<10}{metrics[k]:.4f}")
print(f"\ngate error ratio, Linear Regression: {gaps[gaps.model=='Linear Regression'].ratio.iloc[0]:.2f}x")
print(f"gate error ratio, LightGBM         : {gaps[gaps.model=='LightGBM'].ratio.iloc[0]:.2f}x")
