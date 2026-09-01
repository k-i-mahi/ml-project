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
plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.grid": True, "grid.alpha": .25})

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
fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.6))
ax[0].hist([full[T][~gated], full[T][gated]], bins=35, stacked=True,
           color=["#2b6cb0", "#c53030"], edgecolor="white",
           label=["no gate applied", "gate applied"])
ax[0].set_xlabel("website score"); ax[0].set_ylabel("universities")
ax[0].set_title("Distribution of the score"); ax[0].legend(fontsize=7)

order = ["A+", "A", "B", "C", "D", "F"]
gc = full.grade.value_counts().reindex(order)
ax[1].bar(order, gc.values,
          color=["#22543d", "#2f855a", "#68a678", "#d69e2e", "#dd6b20", "#c53030"])
for i, v in enumerate(gc.values):
    ax[1].text(i, v + 5, str(v), ha="center", fontsize=7)
ax[1].set_ylabel("universities"); ax[1].set_title("Grade distribution")

reg = full.groupby("region")[T].median().sort_values()
ax[2].barh(range(len(reg)), reg.values, color="#4a5568")
ax[2].set_yticks(range(len(reg)))
ax[2].set_yticklabels([r[:24] for r in reg.index], fontsize=6.5)
ax[2].set_xlabel("median score"); ax[2].set_title("Median score by region")
plt.tight_layout(); plt.savefig(FIG / "fig_target.png"); plt.close()

# ======================================================================= fig 2: the curves
fig, ax = plt.subplots(1, 4, figsize=(13.5, 2.9))
n = np.arange(0, 21)
nav = np.where(n <= 0, 0, np.where(n <= 2, .3, np.where(n <= 4, .75,
      np.where(n <= 9, 1., np.where(n <= 12, .9, np.where(n <= 15, .75, .55))))))
ax[0].step(n, nav, where="mid", color="#2b6cb0", lw=2)
ax[0].fill_between(n, nav, step="mid", alpha=.15, color="#2b6cb0")
ax[0].set_title("Menu size: inverted U", fontsize=8.5); ax[0].set_xlabel("menu items")
ax[0].set_ylabel("sub-score")

d = np.arange(0, 400, 2)
rec = np.where(d <= 7, 1., np.where(d <= 30, .9, np.where(d <= 90, .7,
      np.where(d <= 180, .45, np.where(d <= 365, .2, 0.)))))
ax[1].step(d, rec, where="post", color="#2f855a", lw=2)
ax[1].fill_between(d, rec, step="post", alpha=.15, color="#2f855a")
ax[1].set_title("Notice age: staleness decay", fontsize=8.5); ax[1].set_xlabel("days")

r = np.linspace(1, 21, 200)
con = np.where(r < 3, (r - 1) / 2 * .3, np.where(r < 4.5, .3 + (r - 3) / 1.5 * .3,
      np.where(r < 7, .6 + (r - 4.5) / 2.5 * .4, 1.)))
ax[2].plot(r, con, color="#b7791f", lw=2); ax[2].axvline(7, color="crimson", ls=":", lw=1.2)
ax[2].text(7.5, .3, "WCAG AAA", fontsize=6.5, color="crimson")
ax[2].set_title("Contrast: plateau at 7:1", fontsize=8.5); ax[2].set_xlabel("contrast ratio")

b = np.arange(0, 51)
bro = np.where(b <= 0, 1., np.where(b <= 2, .85, np.where(b <= 5, .65,
      np.where(b <= 15, .4, np.where(b <= 40, .15, 0.)))))
ax[3].step(b, bro, where="post", color="#c53030", lw=2)
ax[3].fill_between(b, bro, step="post", alpha=.15, color="#c53030")
ax[3].set_title("Broken links: step penalty", fontsize=8.5); ax[3].set_xlabel("broken links")
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
fig, ax = plt.subplots(1, 3, figsize=(14, 4.4))
for a, col, lab, xmax in [(ax[0], "R2", "R²  (higher is better)", 1.05),
                          (ax[1], "MAE", "MAE in score points  (lower is better)", None),
                          (ax[2], "Spearman", "Spearman ρ  (rank agreement)", 1.05)]:
    a.barh(range(len(d)), d[col], color=cols)
    a.set_yticks(range(len(d)))
    a.set_yticklabels(d.model if col == "R2" else [], fontsize=7.5)
    a.set_xlabel(lab, fontsize=8)
    if xmax:
        a.set_xlim(0, xmax)
    for i, v in enumerate(d[col]):
        a.text(max(v, 0) + (.012 if xmax else .2), i, f"{v:.3f}" if xmax else f"{v:.2f}",
               va="center", fontsize=6.5)
h = [plt.Rectangle((0, 0), 1, 1, color=c) for c in cmap.values()]
ax[0].legend(h, cmap.keys(), fontsize=6.5, loc="lower right", title="family", title_fontsize=7)
ax[0].set_title("Variance explained", fontsize=9)
ax[1].set_title("Average error", fontsize=9)
ax[2].set_title("Rank agreement", fontsize=9)
plt.tight_layout(); plt.savefig(FIG / "fig_comparison.png"); plt.close()

# ======================================================================= fig 4: CV
d2 = comp.sort_values("cv_mean").reset_index(drop=True)
fig, ax = plt.subplots(figsize=(8, 3.8))
ax.barh(range(len(d2)), d2.cv_mean, xerr=d2.cv_sd, color="#2b6cb0",
        error_kw=dict(ecolor="#1a365d", capsize=3, lw=1))
ax.set_yticks(range(len(d2))); ax.set_yticklabels(d2.model, fontsize=7.5)
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
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
lims = [min(y_te.min(), yp.min()) - 3, max(y_te.max(), yp.max()) + 3]
ax[0].scatter(y_te, yp, s=15, alpha=.55, color="#2b6cb0", edgecolor="none")
ax[0].plot(lims, lims, "r--", lw=1.2)
ax[0].set_xlabel("actual score"); ax[0].set_ylabel("predicted score")
ax[0].set_title(f"Predicted vs actual (R² = {r2_score(y_te, yp):.3f})", fontsize=9)

res = yp - y_te
ax[1].scatter(yp, res, s=15, alpha=.55, color="#2f855a", edgecolor="none")
ax[1].axhline(0, color="crimson", ls="--", lw=1.2)
ax[1].set_xlabel("predicted score"); ax[1].set_ylabel("residual")
ax[1].set_title("Residuals", fontsize=9)

ax[2].hist(res, bins=30, color="#4a5568", edgecolor="white")
ax[2].axvline(0, color="crimson", ls="--", lw=1.2)
ax[2].set_xlabel("residual")
ax[2].set_title(f"Error distribution (MAE {np.abs(res).mean():.2f})", fontsize=9)
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

fig, ax = plt.subplots(1, 2, figsize=(13.5, 4.2))
x = np.arange(len(gaps)); w = .38
ax[0].bar(x - w/2, gaps.ungated, w, label="no gate applied", color="#2b6cb0")
ax[0].bar(x + w/2, gaps.gated, w, label="gate applied", color="#c53030")
ax[0].set_xticks(x); ax[0].set_xticklabels(gaps.model, rotation=35, ha="right", fontsize=7)
ax[0].set_ylabel("mean absolute error"); ax[0].legend(fontsize=7.5)
ax[0].set_title("Linear models fail on the gated universities", fontsize=9)

lin = preds["Linear Regression"]
ax[1].scatter(y_te[~gate_te], lin[~gate_te], s=15, alpha=.5, color="#2b6cb0", label="no gate")
ax[1].scatter(y_te[gate_te], lin[gate_te], s=26, alpha=.85, color="#c53030", label="gate applied")
ax[1].plot(lims, lims, "k--", lw=1)
ax[1].set_xlabel("actual score"); ax[1].set_ylabel("Linear Regression prediction")
ax[1].set_title("Linear Regression over-predicts capped sites", fontsize=9)
ax[1].legend(fontsize=7.5)
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
fig, ax = plt.subplots(1, 2, figsize=(13.5, 5))
ax[0].barh(range(len(top)), top.importance,
           color=[pal.get(d, "#a0aec0") for d in top.dimension])
ax[0].set_yticks(range(len(top))); ax[0].set_yticklabels(top.feature, fontsize=6.5)
ax[0].set_xlabel("share of model importance (%)"); ax[0].set_title("Top 18 features", fontsize=9)
ax[1].barh(range(len(bydim)), bydim.values,
           color=[pal.get(d, "#a0aec0") for d in bydim.index])
ax[1].set_yticks(range(len(bydim)))
ax[1].set_yticklabels([d[:36] for d in bydim.index], fontsize=7.5)
ax[1].set_xlabel("total importance (%)")
ax[1].set_title("Importance by dimension", fontsize=9)
for i, v in enumerate(bydim.values):
    ax[1].text(v + .4, i, f"{v:.1f}%", va="center", fontsize=7)
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

print("figures written to report/figures/:")
for p in sorted(FIG.glob("*.png")):
    print(f"  {p.name}")
print("\ntuned model on the held-out test set:")
for k in ("r2", "mae", "rmse", "spearman", "kendall"):
    print(f"  {k:<10}{metrics[k]:.4f}")
print(f"\ngate error ratio, Linear Regression: {gaps[gaps.model=='Linear Regression'].ratio.iloc[0]:.2f}x")
print(f"gate error ratio, LightGBM         : {gaps[gaps.model=='LightGBM'].ratio.iloc[0]:.2f}x")
