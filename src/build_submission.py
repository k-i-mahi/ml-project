"""Build the SUBMISSION folder: 3 CSVs, ARFF files, data dictionary, and the deployed model.

Run from ML_PROJECT/:  python src/build_submission.py
"""
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
import lightgbm as lgb

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
SUB = ROOT / "SUBMISSION"
RNG_SEED = 42

for d in ["data", "data/weka", "notebook", "model", "report", "report/figures"]:
    (SUB / d).mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------- load
data = pd.read_csv(OUT / "model_ready_dataset.csv")
labB = pd.read_csv(OUT / "expert_labels_trackB.csv")
labA = pd.read_csv(OUT / "expert_labels_trackA.csv")
fdict = pd.read_csv(OUT / "feature_dictionary.csv")
meta = json.loads((OUT / "model_meta.json").read_text(encoding="utf-8"))
FEATURES = meta["features"]

# ------------------------------------------------- the single 0-100 scale
# Anchored on the EXPERT-LABEL range, not on the prediction range, so that
# label and prediction are the same quantity measured on the same ruler.
L0, L1 = float(labB.trackB_bt_score.min()), float(labB.trackB_bt_score.max())

def to_score(logit):
    return np.clip(100.0 * (np.asarray(logit, float) - L0) / (L1 - L0), 0.0, 100.0)

GRADES = [(80, "A+ Excellent"), (65, "A  Good"), (50, "B  Adequate"),
          (35, "C  Weak"), (20, "D  Poor"), (-1, "F  Critical")]

def grade_of(s):
    for lo, g in GRADES:
        if s >= lo:
            return g.split()[0]
    return "F"

df = data.merge(labB[["uni_id", "trackB_bt_score", "trackB_bt_se"]], on="uni_id", how="left")
df = df.merge(labA[["uni_id", "trackA_consensus"]], on="uni_id", how="left")
df["quality_score"] = to_score(df.trackB_bt_score)          # ground truth, 200 rows
df["has_expert_label"] = df.trackB_bt_score.notna().astype(int)

lab = df[df.has_expert_label == 1].reset_index(drop=True)
assert len(lab) == 200

# --------------------------------------------- stratified train/test split
band = pd.cut(lab.quality_score, [-0.1, 25, 45, 65, 100.1], labels=False)
tr_idx, te_idx = train_test_split(np.arange(len(lab)), test_size=0.20,
                                  random_state=RNG_SEED, stratify=band)
train_df, test_df = lab.iloc[np.sort(tr_idx)], lab.iloc[np.sort(te_idx)]
print(f"split: train {len(train_df)} / test {len(test_df)}")
print(f"  train score mean {train_df.quality_score.mean():.1f} sd {train_df.quality_score.std():.1f}")
print(f"  test  score mean {test_df.quality_score.mean():.1f} sd {test_df.quality_score.std():.1f}")

# ------------------------------------------------------------- model spec
NON_MONOTONE = {"a03_nav_item_count", "a15_stat_item_count", "a12_accred_count", "nav_quality"}
dirmap = fdict.set_index("feature").direction.to_dict()
MONO = [0 if f in NON_MONOTONE else int(dirmap.get(f, 0)) if dirmap.get(f, 0) in (-1, 1) else 0
        for f in FEATURES]
GRID = {"n_estimators": [200, 400], "learning_rate": [0.02, 0.05], "num_leaves": [7, 15],
        "min_child_samples": [5, 15], "colsample_bytree": [0.6], "reg_lambda": [1.0, 10.0]}

def fit(X, y):
    est = lgb.LGBMRegressor(random_state=0, verbose=-1, n_jobs=1, monotone_constraints=MONO)
    gs = GridSearchCV(est, GRID, cv=KFold(5, shuffle=True, random_state=0),
                      scoring="neg_mean_squared_error", n_jobs=-1)
    gs.fit(X, y)
    return gs.best_estimator_, gs.best_params_

# Model 1: train split only -> gives the HONEST held-out test metrics
m_tr, p_tr = fit(train_df[FEATURES], train_df.trackB_bt_score.values)
pred_te = to_score(m_tr.predict(test_df[FEATURES]))
true_te = test_df.quality_score.values
test_metrics = dict(
    spearman=float(stats.spearmanr(true_te, pred_te).statistic),
    kendall=float(stats.kendalltau(true_te, pred_te).statistic),
    mae=float(np.mean(np.abs(true_te - pred_te))),
    rmse=float(np.sqrt(np.mean((true_te - pred_te) ** 2))),
    r2=float(1 - np.sum((true_te - pred_te) ** 2) / np.sum((true_te - true_te.mean()) ** 2)),
)
print(f"\nheld-out test (n={len(test_df)}): " +
      " ".join(f"{k}={v:.3f}" for k, v in test_metrics.items()))

# Model 2: refit on all 200 -> the DEPLOYED model that scores all 1,226
m_final, p_final = fit(lab[FEATURES], lab.trackB_bt_score.values)
print(f"final model params: {p_final}")

df["predicted_score"] = to_score(m_final.predict(df[FEATURES]))
df["grade"] = df.predicted_score.apply(grade_of)

# ------------------------------------------------------------ ranking
TIE = ["predicted_score", "content_completeness_B5", "a11y_completeness_B11", "uni_id"]
df = df.sort_values(TIE, ascending=[False, False, False, True]).reset_index(drop=True)
df["global_rank"] = np.arange(1, len(df) + 1)
df["regional_rank"] = df.groupby("region").global_rank.rank(method="first").astype(int)
df["country_rank"] = np.nan
el = df[df.country_rank_eligible == 1]
df.loc[el.index, "country_rank"] = el.groupby("country").global_rank.rank(method="first")
df["percentile"] = (100 * (1 - (df.global_rank - 1) / len(df))).round(1)

print(f"\ngrade distribution:\n{df.grade.value_counts().reindex(['A+','A','B','C','D','F']).to_string()}")
kuet = df[df.name.str.contains("Khulna University of Engineering", na=False)].iloc[0]
print(f"\nKUET -> rank {kuet.global_rank} global, {int(kuet.country_rank)} in Bangladesh, "
      f"score {kuet.predicted_score:.1f}, grade {kuet.grade}")

# --------------------------------------------------------- CSV 1: labeled
ID = ["uni_id", "name", "url", "country", "region"]
SCORE = ["predicted_score", "grade", "global_rank", "regional_rank", "country_rank",
         "percentile", "quality_score", "has_expert_label", "trackA_consensus"]
full = df[ID + SCORE + FEATURES].copy()
full.to_csv(SUB / "data/university_websites_labeled.csv", index=False)

# ------------------------------------------------ CSV 2 & 3: train / test
def ml_frame(d):
    return d[["uni_id", "name", "country", "region"] + FEATURES + ["quality_score"]].copy()

train_out, test_out = ml_frame(train_df), ml_frame(test_df)
train_out.to_csv(SUB / "data/train.csv", index=False)
test_out.to_csv(SUB / "data/test.csv", index=False)
print(f"\nwrote 3 CSVs: labeled {full.shape}, train {train_out.shape}, test {test_out.shape}")

# ------------------------------------------------------- data dictionary
rows = []
for c in ID:
    rows.append(dict(column=c, role="identifier", type="text", block="-",
                     description={"uni_id": "unique row id", "name": "university name",
                                  "url": "landing page crawled", "country": "country label (10 of 22 are regional buckets)",
                                  "region": "one of 6 collection regions"}[c], direction="", missing=""))
for c, desc in [("predicted_score", "MODEL OUTPUT 0-100. The website quality score. Higher = better."),
                ("grade", "A+ >=80, A >=65, B >=50, C >=35, D >=20, F <20"),
                ("global_rank", "1 = best of 1,226"),
                ("regional_rank", "rank within its region"),
                ("country_rank", "rank within country; only for 19 eligible countries"),
                ("percentile", "100 = top of the ranking"),
                ("quality_score", "GROUND TRUTH 0-100 from expert judgment. Populated for 200 rows only."),
                ("has_expert_label", "1 if this row carries the ground-truth label"),
                ("trackA_consensus", "rule-based rubric baseline score 0-100 (NOT the target)")]:
    rows.append(dict(column=c, role="label/output", type="numeric" if c != "grade" else "nominal",
                     block="-", description=desc, direction="", missing=""))
fd = fdict.set_index("feature")
for c in FEATURES:
    r = fd.loc[c] if c in fd.index else None
    rows.append(dict(
        column=c, role="feature",
        type=("binary" if set(pd.unique(data[c].dropna())) <= {0, 1} else "numeric"),
        block=(r.block if r is not None else "derived"),
        description=(r.reason if r is not None else ""),
        direction=({1: "higher is better", -1: "lower is better", 0: "non-monotone"}
                   .get(int(r.direction), "") if r is not None else ""),
        missing=(str(r.missing_treatment)[:120] if r is not None else "")))
dd = pd.DataFrame(rows)
dd.to_csv(SUB / "data/data_dictionary.csv", index=False)
print(f"wrote data_dictionary.csv ({len(dd)} rows)")

# ------------------------------------------------------------------ ARFF
def arff(path, frame, relation, target, target_type):
    lines = [f"% University Website Quality — {relation}",
             f"% {len(frame)} instances, {len(FEATURES)} predictive attributes",
             f"% Generated for Weka. Target attribute: {target}", "",
             f"@relation {relation}", ""]
    for c in FEATURES:
        lines.append(f"@attribute {c} numeric")
    if target_type == "numeric":
        lines.append(f"@attribute {target} numeric")
    else:
        lines.append(f"@attribute {target} {{{','.join(target_type)}}}")
    lines += ["", "@data"]
    cols = FEATURES + [target]
    for _, r in frame[cols].iterrows():
        vals = []
        for c in cols:
            v = r[c]
            if pd.isna(v):
                vals.append("?")
            elif isinstance(v, str):
                vals.append(v)
            else:
                vals.append(f"{v:g}")
        lines.append(",".join(vals))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(frame)

W = SUB / "data/weka"
GRADE_LEVELS = ["A+", "A", "B", "C", "D", "F"]
train_g = train_out.assign(grade=train_out.quality_score.apply(grade_of))
test_g = test_out.assign(grade=test_out.quality_score.apply(grade_of))

n1 = arff(W / "train.arff", train_out, "university_website_quality_train", "quality_score", "numeric")
n2 = arff(W / "test.arff", test_out, "university_website_quality_test", "quality_score", "numeric")
n3 = arff(W / "train_classification.arff", train_g, "university_website_grade_train", "grade", GRADE_LEVELS)
n4 = arff(W / "test_classification.arff", test_g, "university_website_grade_test", "grade", GRADE_LEVELS)
n5 = arff(W / "labeled_200_regression.arff", ml_frame(lab), "university_website_quality_all200",
          "quality_score", "numeric")
print(f"wrote ARFF: train {n1}, test {n2}, train_cls {n3}, test_cls {n4}, all200 {n5}")
print(f"  grade balance (train): {train_g.grade.value_counts().to_dict()}")

# ----------------------------------------------------------------- model
joblib.dump({"model": m_final, "features": FEATURES, "L0": L0, "L1": L1,
             "params": p_final, "monotone": MONO},
            SUB / "model/final_model.joblib")
joblib.dump({"model": m_tr, "features": FEATURES, "L0": L0, "L1": L1},
            SUB / "model/model_trained_on_train_split.joblib")

card = dict(
    name="University Website Quality Scorer", version="1.0",
    algorithm="LightGBM gradient-boosted trees with monotonic constraints",
    hyperparameters={k: v for k, v in p_final.items()},
    n_features=len(FEATURES), n_train_total=200, n_train_split=len(train_df), n_test=len(test_df),
    target="expert quality score 0-100 (Bradley-Terry latent strength, rescaled)",
    score_anchor={"logit_min": L0, "logit_max": L1,
                  "formula": "score = clip(100*(logit-logit_min)/(logit_max-logit_min), 0, 100)"},
    held_out_test=test_metrics,
    cross_validation={"spearman_mean": 0.881, "spearman_sd": 0.026, "protocol": "nested 5-fold x 5 seeds"},
    leave_one_region_out={"spearman_mean": 0.844, "spearman_sd": 0.056},
    baseline_rule_based={"spearman_cv": 0.791, "spearman_loro": 0.708},
    monotone_constraints={"increasing": int(sum(1 for m in MONO if m == 1)),
                          "decreasing": int(sum(1 for m in MONO if m == -1)),
                          "free": int(sum(1 for m in MONO if m == 0))},
    deployment="scores all 1,226 universities; 1,026 of them are pure inference",
)
(SUB / "model/model_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")

# ------------------------------------------------------- copy report figs
FIGMAP = {
    "05_loadtime_confound.png": "fig_confound.png",
    "12_trackA_vs_trackB.png": "fig_validation.png",
    "13_trackB_diagnostics.png": "fig_trackb_diag.png",
    "15_model_comparison.png": "fig_model_comparison.png",
    "17_shap_beeswarm.png": "fig_shap.png",
    "18_weights_vs_shap.png": "fig_weights.png",
    "19_ranking_fairness.png": "fig_fairness.png",
    "20_shap_local.png": "fig_local.png",
}
for src, dst in FIGMAP.items():
    s = ROOT / "figures" / src
    if s.exists():
        shutil.copy(s, SUB / "report/figures" / dst)
print(f"copied {len(FIGMAP)} figures to report/figures/")

# --------------------------------------------------- machine-readable summary
summary = dict(
    n_universities=int(len(df)), n_expert_labeled=200, n_train=len(train_df), n_test=len(test_df),
    n_features=len(FEATURES), score_range=[float(df.predicted_score.min()), float(df.predicted_score.max())],
    grade_counts={k: int(v) for k, v in df.grade.value_counts().items()},
    test_metrics=test_metrics,
    kuet=dict(name=str(kuet["name"]), score=float(kuet.predicted_score),
              global_rank=int(kuet.global_rank), country_rank=int(kuet.country_rank),
              grade=str(kuet.grade), percentile=float(kuet.percentile)),
    n_countries_ranked=int(el.country.nunique()),
)
(SUB / "data/dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print("\n" + json.dumps(summary, indent=2)[:900])
print("\nSUBMISSION data assets built.")
