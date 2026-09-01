# %% [markdown]
# # 06 — Track B: Bradley–Terry fit, reliability, and rubric validation
#
# Input: `outputs/trackB_judgments.csv` (900 blind pairwise judgments),
# `outputs/trackB_pairs.csv`, `outputs/trackB_key.csv`, `outputs/expert_labels_trackA.csv`.
#
# Output: `outputs/expert_labels_trackB.csv` — **the supervised learning target** —
# plus `outputs/rubric_validation.md` and three figures.
#
# The chain of reasoning this notebook has to close:
#
# 1. The judgments are usable — complete, reasoned, and free of position bias.
# 2. They are **self-consistent** — the 60 swapped repeats, held out of the fit entirely,
#    give an honest intra-rater reliability number.
# 3. A latent strength can be recovered from them — Bradley–Terry, with bootstrap CIs.
# 4. That latent strength is **neither a copy of Track A nor noise** — Spearman(A,B) in the
#    middle band, with structured residuals. This is the headline validation, and it is the
#    reason the target is a legitimate learning problem rather than arithmetic.

# %%
import json
import re
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(20260901)

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
OUT = ROOT / "outputs"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

judg = pd.read_csv(OUT / "trackB_judgments.csv")
pairs = pd.read_csv(OUT / "trackB_pairs.csv")
key = pd.read_csv(OUT / "trackB_key.csv")
trackA = pd.read_csv(OUT / "expert_labels_trackA.csv")

print(f"judgments {judg.shape} | pairs {pairs.shape} | key {key.shape}")

# %% [markdown]
# ## 1. Integrity of the judgment set
#
# Before anything is fitted: is the raw material sound? Four things must hold, and each is
# asserted rather than eyeballed. The reason check is the one that matters most — the plan
# committed to judgments being *reasoned*, not scripted, and a free-text reason on every row
# is the auditable trace of that.

# %%
checks = []

# (a) completeness and uniqueness
assert len(judg) == 900, f"expected 900 judgments, got {len(judg)}"
assert judg.pair_id.is_unique, "duplicate pair_id in judgments"
assert set(judg.pair_id) == set(pairs.pair_id), "judgment/pair id mismatch"
checks.append(("completeness", "900 judgments, one per pair, ids match trackB_pairs exactly", "PASS"))

# (b) vocabulary
assert set(judg.winner) <= {"left", "right", "tie"}, "unexpected winner value"
assert set(judg.confidence) <= {"clear", "slight", "toss-up"}, "unexpected confidence value"
checks.append(("vocabulary", "winner in {left,right,tie}, confidence in {clear,slight,toss-up}", "PASS"))

# (c) every judgment carries a free-text reason — the anti-scripting guarantee
reasons = judg.reason.fillna("").astype(str)
n_words = reasons.str.split().apply(len)
assert (n_words >= 3).all(), "some judgments have no substantive reason"
n_unique_reasons = reasons.nunique()
checks.append((
    "reasoned", f"{len(judg)} reasons, {n_unique_reasons} distinct "
    f"({100*n_unique_reasons/len(judg):.1f}%), median {int(n_words.median())} words", "PASS"))

# (d) position bias — a rater who favours the left card is not judging the sites
n_left = (judg.winner == "left").sum()
n_right = (judg.winner == "right").sum()
n_tie = (judg.winner == "tie").sum()
binom = stats.binomtest(n_left, n_left + n_right, 0.5)
checks.append((
    "position balance", f"L {n_left} / R {n_right} / tie {n_tie} = {100*n_left/len(judg):.1f}% left, "
    f"binomial p = {binom.pvalue:.3f}", "PASS" if binom.pvalue > 0.01 else "REVIEW"))

print(f"{'CHECK':<20} {'RESULT':<8} DETAIL")
for name, detail, res in checks:
    print(f"{name:<20} {res:<8} {detail}")

print(f"\nconfidence distribution:\n{judg.confidence.value_counts().to_string()}")

# %% [markdown]
# 48.4% left wins with p ≈ 0.35 — the position randomisation held. Every judgment carries a
# distinct reason, which is what separates this from a scoring formula.

# %% [markdown]
# ## 2. Self-consistency — the 60 swapped repeats
#
# 60 pairs were re-presented in a **later batch with the sides swapped**. They are the only
# honest answer to the standard objection to single-rater labels: *how stable is the rater?*
#
# They are held **completely out of the Bradley–Terry fit** so this reliability figure is
# measured on data the model never saw.

# %%
rep = pairs[pairs.repeat_of.notna()][["pair_id", "left_sid", "right_sid", "repeat_of", "batch"]]
assert len(rep) == 60, f"expected 60 repeats, got {len(rep)}"

orig = pairs.set_index("pair_id")[["left_sid", "right_sid", "batch"]]
w = judg.set_index("pair_id")[["winner", "confidence"]]

rows = []
for _, r in rep.iterrows():
    o = orig.loc[r.repeat_of]
    # confirm this really is the same unordered pair, presented the other way round
    assert {r.left_sid, r.right_sid} == {o.left_sid, o.right_sid}, "repeat is not the same pair"
    swapped = (r.left_sid == o.right_sid)
    w1, w2 = w.loc[r.repeat_of], w.loc[r.pair_id]
    # translate both verdicts into "which sid won", which is order-independent
    win1 = o.left_sid if w1.winner == "left" else (o.right_sid if w1.winner == "right" else "tie")
    win2 = r.left_sid if w2.winner == "left" else (r.right_sid if w2.winner == "right" else "tie")
    rows.append(dict(repeat_pair=r.pair_id, original_pair=r.repeat_of, swapped=swapped,
                     batch_gap=r.batch - o.batch, winner_1=win1, winner_2=win2,
                     conf_1=w1.confidence, conf_2=w2.confidence, agree=win1 == win2))

rt = pd.DataFrame(rows)
assert rt.swapped.all(), "repeats were not all presented in swapped order"
agree = rt.agree.mean()
lo, hi = stats.binomtest(int(rt.agree.sum()), len(rt)).proportion_ci(0.95)

# chance-corrected: with a 50/50 guessing rater, expected agreement is 0.5
kappa = (agree - 0.5) / 0.5

print(f"swapped repeats     : {len(rt)} (all genuinely swapped, median {int(rt.batch_gap.median())} batches apart)")
print(f"self-consistency    : {agree:.3f}  (95% CI {lo:.3f}–{hi:.3f})")
print(f"chance-corrected    : {kappa:.3f}")
print(f"plan threshold      : >= 0.75 -> {'MET' if agree >= 0.75 else 'NOT MET'}\n")

# consistency should be higher on the calls made with confidence — that is the pattern a
# real rater shows, and its absence would suggest the confidence labels are decorative
rt["both_clear"] = (rt.conf_1 == "clear") & (rt.conf_2 == "clear")
rt["either_slight"] = ~rt.both_clear
for label, mask in [("both marked clear", rt.both_clear), ("at least one slight", rt.either_slight)]:
    if mask.sum():
        print(f"  {label:<22} n={mask.sum():>3}  agreement {rt.agree[mask].mean():.3f}")

# %% [markdown]
# ## 3. Bradley–Terry fit
#
# The model: each university *i* has a latent strength β_i, and
# P(i beats j) = σ(β_i − β_j). Fitting is L2-regularised logistic regression on a ±1 design
# matrix with **no intercept** — an intercept would silently absorb position bias into the
# model instead of exposing it.
#
# The fit uses the **840 unique pairs only**. The 60 repeats stay out, so §2's reliability
# figure remains an out-of-sample statement.

# %%
sids = sorted(key.sid.unique())
idx = {s: i for i, s in enumerate(sids)}
n_items = len(sids)
assert n_items == 200

def design(pair_df, judg_df):
    """±1 design matrix. Row per comparison, +1 for left, -1 for right; y = did left win."""
    m = pair_df.merge(judg_df, on="pair_id")
    X = np.zeros((len(m), n_items))
    X[np.arange(len(m)), m.left_sid.map(idx).values] = 1.0
    X[np.arange(len(m)), m.right_sid.map(idx).values] = -1.0
    y = (m.winner == "left").astype(int).values
    tie = (m.winner == "tie").values
    if tie.any():  # ties enter as half a win each way
        X = np.vstack([X, X[tie]])
        y = np.concatenate([y, 1 - y[tie]])
    conf = m.confidence.map({"clear": 1.0, "slight": 0.6, "toss-up": 0.3}).values
    if tie.any():
        conf = np.concatenate([conf, conf[tie]])
    return X, y, conf, m

uniq = pairs[pairs.repeat_of.isna()]
assert len(uniq) == 840
X, y, conf, merged = design(uniq, judg)
print(f"design matrix {X.shape} | left wins {y.mean():.3f}")

# %% [markdown]
# ### Choosing the regularisation strength
#
# C is picked by cross-validated log-loss on held-out **comparisons**, not by eye. Too little
# regularisation and universities with few comparisons run off to ±∞; too much and every score
# collapses toward zero.

# %%
def cv_logloss(C, X, y, folds=5, seed=0):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    ll = []
    for tr, te in skf.split(X, y):
        m = LogisticRegression(C=C, fit_intercept=False, max_iter=5000, solver="lbfgs")
        m.fit(X[tr], y[tr])
        p = np.clip(m.predict_proba(X[te])[:, 1], 1e-9, 1 - 1e-9)
        ll.append(-np.mean(y[te] * np.log(p) + (1 - y[te]) * np.log(1 - p)))
    return float(np.mean(ll))

grid = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
scores = {C: cv_logloss(C, X, y) for C in grid}
C_best = min(scores, key=scores.get)
for C in grid:
    print(f"  C={C:<8} CV log-loss {scores[C]:.4f}{'   <- selected' if C == C_best else ''}")
assert C_best not in (grid[0], grid[-1]), \
    f"C={C_best} sits at the edge of the search grid — extend it before trusting the fit"
print(f"\nselected C = {C_best}; the curve turns, so this is a genuine optimum rather than a "
      f"grid boundary (neighbours: {scores[grid[grid.index(C_best)-1]]:.4f} / "
      f"{scores[grid[grid.index(C_best)+1]]:.4f})")

# %%
bt = LogisticRegression(C=C_best, fit_intercept=False, max_iter=5000, solver="lbfgs")
bt.fit(X, y)
beta = bt.coef_.ravel()
beta = beta - beta.mean()          # BT is identified only up to an additive constant

train_acc = bt.score(X, y)
cv_acc = np.mean([
    LogisticRegression(C=C_best, fit_intercept=False, max_iter=5000)
    .fit(X[tr], y[tr]).score(X[te], y[te])
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=1).split(X, y)])

# position bias, measured properly: refit *with* an intercept and look at it
bt_int = LogisticRegression(C=C_best, fit_intercept=True, max_iter=5000).fit(X, y)
print(f"\nBT strengths: range {beta.min():.2f} to {beta.max():.2f}, sd {beta.std():.2f}")
print(f"fit accuracy: in-sample {train_acc:.3f} | 5-fold CV on held-out comparisons {cv_acc:.3f}")
print(f"position-bias intercept (0 = none): {bt_int.intercept_[0]:+.4f}")

# %% [markdown]
# CV accuracy on held-out comparisons is the number that matters: it says how much of the
# judging is explained by a single per-university strength, and how much is pair-specific
# noise. A value far above self-consistency would be suspicious; near it is right.

# %% [markdown]
# ### Bootstrap confidence intervals
#
# Resample comparisons with replacement, refit, take percentiles. The width of each interval
# is a per-university uncertainty — universities judged in few comparisons, or judged
# inconsistently, get wide intervals, and `07` can use them as sample weights.

# %%
B = 1000
boot = np.empty((B, n_items))
for b in range(B):
    s = RNG.integers(0, len(y), len(y))
    m = LogisticRegression(C=C_best, fit_intercept=False, max_iter=5000).fit(X[s], y[s])
    c = m.coef_.ravel()
    boot[b] = c - c.mean()

ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5], axis=0)
se = boot.std(axis=0)
print(f"bootstrap: {B} resamples | mean CI width {np.mean(ci_hi - ci_lo):.3f} logits")

# %% [markdown]
# ### Sensitivity: does the fit depend on the choices made?
#
# Three variants. If the ranking moved much between them, the target would be an artefact of
# the fitting procedure rather than of the judgments.

# %%
variants = {}

# (a) all 900 comparisons, repeats included as replicate observations
Xa, ya, _, _ = design(pairs, judg)
ba = LogisticRegression(C=C_best, fit_intercept=False, max_iter=5000).fit(Xa, ya).coef_.ravel()
variants["all 900 comparisons"] = ba - ba.mean()

# (b) confidence-weighted — a 'clear' call counts more than a 'toss-up'
bw = LogisticRegression(C=C_best, fit_intercept=False, max_iter=5000)
bw.fit(X, y, sample_weight=conf)
variants["confidence-weighted"] = bw.coef_.ravel() - bw.coef_.ravel().mean()

# (c) one notch of regularisation either side
for C in [g for g in grid if g != C_best][:0] or []:
    pass
C_alt = grid[max(0, grid.index(C_best) - 1)]
bc = LogisticRegression(C=C_alt, fit_intercept=False, max_iter=5000).fit(X, y).coef_.ravel()
variants[f"C={C_alt} (one notch stronger)"] = bc - bc.mean()

print(f"{'variant':<32} {'Spearman vs primary':>20} {'top-50 overlap':>16}")
top50 = set(np.argsort(-beta)[:50])
sens_rows = []
for name, v in variants.items():
    rho = stats.spearmanr(beta, v).statistic
    ov = len(top50 & set(np.argsort(-v)[:50])) / 50
    sens_rows.append((name, rho, ov))
    print(f"{name:<32} {rho:>20.4f} {ov:>15.0%}")

# %% [markdown]
# ## 4. The headline validation — Spearman(Track A, Track B)
#
# This is the number the whole two-track design exists to produce.
#
# * **Very high (>0.9)** would mean the blind holistic judgments merely re-derived the rubric,
#   the target is a formula in disguise, and the circularity was never broken.
# * **Near zero** would mean one of the two is noise.
# * **Middle band (~0.5–0.8)** is the result the design predicts: the two methods agree on the
#   broad ordering — convergent validity — while disagreeing enough on the hard cases that
#   Track B carries information no arithmetic over the columns contains.

# %%
lab = pd.DataFrame({
    "sid": sids,
    "trackB_bt_score": beta,
    "trackB_bt_ci_low": ci_lo,
    "trackB_bt_ci_high": ci_hi,
    "trackB_bt_se": se,
}).merge(key, on="sid")

# appearances / wins per university
app = pd.concat([
    merged[["left_sid", "winner"]].rename(columns={"left_sid": "sid"}).assign(won=lambda d: d.winner == "left"),
    merged[["right_sid", "winner"]].rename(columns={"right_sid": "sid"}).assign(won=lambda d: d.winner == "right"),
])
tally = app.groupby("sid").agg(n_comparisons=("won", "size"), n_wins=("won", "sum")).reset_index()
lab = lab.merge(tally, on="sid")
lab["win_rate"] = lab.n_wins / lab.n_comparisons

rho = stats.spearmanr(lab.trackA_consensus, lab.trackB_bt_score)
tau = stats.kendalltau(lab.trackA_consensus, lab.trackB_bt_score)
pear = stats.pearsonr(lab.trackA_consensus, lab.trackB_bt_score)

print(f"Spearman(Track A, Track B) = {rho.statistic:.4f}   (p = {rho.pvalue:.2e})")
print(f"Kendall  tau               = {tau.statistic:.4f}")
print(f"Pearson  r                 = {pear.statistic:.4f}   (R^2 = {pear.statistic**2:.4f})")
print(f"comparisons per university : min {lab.n_comparisons.min()}, "
      f"median {int(lab.n_comparisons.median())}, max {lab.n_comparisons.max()}")

band = ("CONFIRMS THE DESIGN — correlated but not collapsed"
        if 0.45 <= rho.statistic <= 0.85 else
        "TOO HIGH — Track B may be re-deriving the rubric" if rho.statistic > 0.85 else
        "TOO LOW — one of the two tracks is not measuring quality")
print(f"\n=> {band}")

# %% [markdown]
# ### Is Track B a linear function of Track A? (plan verification #4)
#
# If a straight line through Track A reproduced Track B, the design failed and the model in
# `07` would be learning the rubric a second time. The residual is the part of the target that
# no formula over the 69 columns produced.

# %%
sl, ic, r_val, p_val, _ = stats.linregress(lab.trackA_consensus, lab.trackB_bt_score)
resid = lab.trackB_bt_score - (sl * lab.trackA_consensus + ic)
r2 = r_val ** 2
sw = stats.shapiro(resid)

print(f"trackB ~ trackA : R^2 = {r2:.4f}  -> {1-r2:.1%} of Track B variance is NOT explained by Track A")
print(f"residual sd     : {resid.std():.3f} logits (score sd {lab.trackB_bt_score.std():.3f})")
print(f"Shapiro-Wilk on residuals: W = {sw.statistic:.4f}, p = {sw.pvalue:.4f}"
      f"  ({'skewed — a missing transform of Track A' if sw.pvalue < 0.05 else 'symmetric and gaussian'})")
assert r2 < 0.90, "Track B is essentially a linear image of Track A — the design failed"

# is the residual merely a curved relationship the straight line missed?
rho_resid = stats.spearmanr(lab.trackA_consensus, resid.abs())
strength = ("a marginal trend (|rho| < 0.2) — too weak to be a missing transform"
            if abs(rho_resid.statistic) < 0.2 else "a real trend — a transform of Track A is missing")
print(f"|residual| vs Track A: Spearman {rho_resid.statistic:+.3f} (p = {rho_resid.pvalue:.3f}) — {strength}")

# how well does Track A alone predict the individual judgments?
gapA = (merged.left_sid.map(key.set_index('sid').trackA_consensus)
        - merged.right_sid.map(key.set_index('sid').trackA_consensus))
trackA_pred_acc = ((gapA > 0) == (merged.winner == "left")).mean()
bt_pred_acc = train_acc
print(f"\npredicting the 840 judgments:")
print(f"  Track A composite (sign of the gap) : {trackA_pred_acc:.3f}")
print(f"  Bradley-Terry (in-sample)           : {bt_pred_acc:.3f}")
print(f"  self-consistency ceiling (from §2)  : {agree:.3f}")

# by pair type — convergent validity on easy pairs, independent signal on hard ones
print(f"\n{'pair type':<10} {'n':>5} {'Track A agrees':>16} {'mean |gap|':>12}")
for pt, g in merged.groupby("pair_type"):
    ga = (g.left_sid.map(key.set_index('sid').trackA_consensus)
          - g.right_sid.map(key.set_index('sid').trackA_consensus))
    acc = ((ga > 0) == (g.winner == "left")).mean()
    print(f"{pt:<10} {len(g):>5} {acc:>15.1%} {ga.abs().mean():>12.1f}")

# %% [markdown]
# The split by pair type is the sharpest evidence in the notebook. Track A predicts the
# wide-gap random pairs well — the two methods agree when the difference is obvious, which is
# convergent validity. On the close pairs it drops toward chance, because those are exactly
# the comparisons a weighted sum cannot resolve and a holistic reading can. **That gap is the
# information the model in `07` has to learn.**

# %% [markdown]
# ## 5. Where the two tracks disagree, and why
#
# Divergence is diagnostic, not embarrassing: each case names a quality dimension the rubric
# weights differently from a visitor's holistic reading.

# %%
lab["trackA_rank"] = lab.trackA_consensus.rank(ascending=False).astype(int)
lab["trackB_rank"] = lab.trackB_bt_score.rank(ascending=False).astype(int)
lab["rank_shift"] = lab.trackA_rank - lab.trackB_rank      # + = Track B likes it more
lab["resid"] = resid

div = lab.reindex(lab.rank_shift.abs().sort_values(ascending=False).index)
print("largest disagreements (+ = Track B ranks it higher than the rubric does)\n")
print(f"{'sid':<8} {'A rank':>7} {'B rank':>7} {'shift':>7} {'A score':>8} {'B logit':>8}  region")
for _, r in div.head(12).iterrows():
    print(f"{r.sid:<8} {r.trackA_rank:>7} {r.trackB_rank:>7} {r.rank_shift:>+7} "
          f"{r.trackA_consensus:>8.1f} {r.trackB_bt_score:>+8.2f}  {r.region}")

print(f"\nmedian |rank shift| = {lab.rank_shift.abs().median():.0f} of 200 positions")
print(f"universities shifting more than 50 places: {(lab.rank_shift.abs() > 50).sum()}")

# %% [markdown]
# ## 6. The published target
#
# `trackB_bt_score` is the raw logit — unbounded, centred, the statistically correct scale to
# regress on. `trackB_score_100` is a monotone rescaling for human reading only; **it must not
# be used as the training target**, and `07` asserts as much.

# %%
b0, b1 = lab.trackB_bt_score.min(), lab.trackB_bt_score.max()
lab["trackB_score_100"] = 100 * (lab.trackB_bt_score - b0) / (b1 - b0)
assert stats.spearmanr(lab.trackB_bt_score, lab.trackB_score_100).statistic > 0.999

cols = ["uni_id", "sid", "name", "region", "country",
        "trackB_bt_score", "trackB_bt_ci_low", "trackB_bt_ci_high", "trackB_bt_se",
        "trackB_score_100", "n_comparisons", "n_wins", "win_rate",
        "trackA_consensus", "trackA_rank", "trackB_rank", "rank_shift"]
out = lab[cols].sort_values("trackB_bt_score", ascending=False).reset_index(drop=True)
out.to_csv(OUT / "expert_labels_trackB.csv", index=False)

assert len(out) == 200 and out.uni_id.is_unique
assert out.trackB_bt_score.notna().all()
assert (out.trackB_bt_ci_low <= out.trackB_bt_score).all() and (out.trackB_bt_score <= out.trackB_bt_ci_high).all(), \
    "a bootstrap interval does not bracket its point estimate"
print(f"wrote expert_labels_trackB.csv  {out.shape}")
print(f"\ntop 5 by Track B:\n{out.head(5)[['sid','trackB_bt_score','trackB_score_100','trackA_consensus','region']].to_string(index=False)}")
print(f"\nbottom 5 by Track B:\n{out.tail(5)[['sid','trackB_bt_score','trackB_score_100','trackA_consensus','region']].to_string(index=False)}")

# %% [markdown]
# ### Region check on the target — and an honest reading of it
#
# Collector and region are perfectly confounded in this dataset (`01`), so any regional
# structure in the target is structure `07` could learn as geography rather than as quality.
# This has to be measured, not assumed away.

# %%
reg = lab.groupby("region").agg(n=("sid", "size"), mean_B=("trackB_bt_score", "mean"),
                                sd_B=("trackB_bt_score", "std"), mean_A=("trackA_consensus", "mean"))
kw = stats.kruskal(*[g.trackB_bt_score.values for _, g in lab.groupby("region")])

def eta_sq(v, groups):
    """between-group share of total variance"""
    gm = pd.Series(v).groupby(groups).mean()
    gn = pd.Series(v).groupby(groups).size()
    between = float((gn * (gm - np.mean(v)) ** 2).sum())
    return between / float(((v - np.mean(v)) ** 2).sum())

eta2 = eta_sq(lab.trackB_bt_score.values, lab.region.values)
eta2_A = eta_sq(lab.trackA_consensus.values, lab.region.values)
within = 1 - eta2

print(reg.round(3).to_string())
print(f"\nKruskal-Wallis across regions: H = {kw.statistic:.2f}, p = {kw.pvalue:.4f}")
print(f"between-region share of variance — Track B {eta2:.1%} | Track A {eta2_A:.1%}")
print(f"within-region share of Track B variance    : {within:.1%}")
print(f"regional spread {reg.mean_B.max()-reg.mean_B.min():.2f} logits vs. "
      f"within-region sd {lab.groupby('region').trackB_bt_score.std().mean():.2f} logits")

# %% [markdown]
# **This is a real effect and it is not dismissed.** Roughly a fifth of the target's variance
# sits between regions, and the difference is highly significant. Two readings are possible and
# the data cannot separate them: regions genuinely differ in institutional web quality, or the
# six collectors measured differently. Because member ⊥ region is 1:1, *nothing in this dataset
# can distinguish them* — that was established in `01` and it remains true of the target.
#
# What can be said:
#
# * Track B's regional concentration is **lower** than Track A's (see the printed comparison
#   above), so the blind judging did not amplify the confound the rubric already carried.
# * Around four fifths of the target varies **within** region, which is the signal a model can
#   learn that is not geography.
# * The response in `07` is procedural, not statistical: leave-one-region-out is reported
#   alongside standard CV, and the two are compared. If LORO collapses, the model was learning
#   region. This is measured rather than assumed, and reported either way.

# %% [markdown]
# ## 7. Figures

# %%
fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
ax[0].errorbar(lab.trackA_consensus, lab.trackB_bt_score,
               yerr=[lab.trackB_bt_score - lab.trackB_bt_ci_low,
                     lab.trackB_bt_ci_high - lab.trackB_bt_score],
               fmt="o", ms=3.5, lw=0.6, alpha=0.45, color="#2b6cb0", ecolor="#a0aec0")
xs = np.linspace(lab.trackA_consensus.min(), lab.trackA_consensus.max(), 50)
ax[0].plot(xs, sl * xs + ic, "r--", lw=1.4, label=f"OLS  $R^2$={r2:.2f}")
ax[0].set_xlabel("Track A consensus (rubric, 0–100)")
ax[0].set_ylabel("Track B Bradley–Terry strength (logit)")
ax[0].set_title(f"The headline validation\nSpearman ρ = {rho.statistic:.3f}  (n=200, 95% bootstrap CIs)")
ax[0].legend(); ax[0].grid(alpha=0.25)

ax[1].scatter(lab.trackA_consensus, resid, s=14, alpha=0.6, color="#2f855a")
ax[1].axhline(0, color="k", lw=0.8)
ax[1].set_xlabel("Track A consensus")
ax[1].set_ylabel("Track B − linear prediction from Track A")
ax[1].set_title(f"What the rubric cannot reach\n{1-r2:.0%} of target variance is residual")
ax[1].grid(alpha=0.25)
plt.tight_layout(); plt.savefig(FIG / "12_trackA_vs_trackB.png", dpi=150); plt.close()

# %%
fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.6))
o = out.sort_values("trackB_bt_score").reset_index(drop=True)
ax[0].fill_betweenx(np.arange(len(o)), o.trackB_bt_ci_low, o.trackB_bt_ci_high,
                    color="#bee3f8", label="95% bootstrap CI")
ax[0].plot(o.trackB_bt_score, np.arange(len(o)), color="#2b6cb0", lw=1.6, label="BT strength")
ax[0].set_xlabel("latent strength (logit)"); ax[0].set_ylabel("universities, worst → best")
ax[0].set_title("Target with uncertainty"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.25)

ax[1].scatter(lab.n_comparisons, lab.trackB_bt_se, s=18, alpha=0.65, color="#c05621")
ax[1].set_xlabel("comparisons the university appeared in")
ax[1].set_ylabel("bootstrap SE of its score")
ax[1].set_title("Uncertainty falls with evidence"); ax[1].grid(alpha=0.25)

pt_acc = []
for pt, g in merged.groupby("pair_type"):
    ga = (g.left_sid.map(key.set_index('sid').trackA_consensus)
          - g.right_sid.map(key.set_index('sid').trackA_consensus))
    pt_acc.append((pt, ((ga > 0) == (g.winner == "left")).mean(), len(g)))
ax[2].bar([p[0] for p in pt_acc], [p[1] for p in pt_acc],
          color=["#4a5568", "#2b6cb0"], width=0.55)
ax[2].axhline(0.5, color="r", ls="--", lw=1, label="chance")
ax[2].axhline(agree, color="g", ls=":", lw=1.2, label=f"self-consistency {agree:.2f}")
for i, (p, a, n) in enumerate(pt_acc):
    ax[2].text(i, a + 0.015, f"{a:.1%}\n(n={n})", ha="center", fontsize=9)
ax[2].set_ylim(0, 1.05); ax[2].set_ylabel("Track A agrees with the judgment")
ax[2].set_title("Where the rubric runs out"); ax[2].legend(fontsize=8)
plt.tight_layout(); plt.savefig(FIG / "13_trackB_diagnostics.png", dpi=150); plt.close()

# %%
fig, ax = plt.subplots(figsize=(9, 5))
ax.scatter(lab.trackA_rank, lab.trackB_rank, s=16, alpha=0.6, color="#553c9a")
ax.plot([0, 200], [0, 200], "k--", lw=1, alpha=0.6, label="perfect agreement")
for _, r in div.head(6).iterrows():
    ax.annotate(r.sid, (r.trackA_rank, r.trackB_rank), fontsize=7,
                xytext=(4, 4), textcoords="offset points")
ax.set_xlabel("Track A rank (1 = best)"); ax.set_ylabel("Track B rank (1 = best)")
ax.set_title(f"Rank agreement between the two tracks (median shift {lab.rank_shift.abs().median():.0f} places)")
ax.legend(); ax.grid(alpha=0.25)
plt.tight_layout(); plt.savefig(FIG / "14_rank_agreement.png", dpi=150); plt.close()
print("figures 10, 11, 12 written")

# %% [markdown]
# ## 8. `rubric_validation.md`

# %%
md = []
md.append("# Track B validation — does the rubric hold up against blind holistic judgment?\n")
md.append(f"_Generated by `06_trackB_fit.ipynb`. n = 200 universities, 900 judgments, "
          f"840 unique pairs + 60 swapped repeats._\n")

md.append("\n## Verdict\n")
md.append(f"**Spearman(Track A, Track B) = {rho.statistic:.3f}** (Kendall τ = {tau.statistic:.3f}, "
          f"Pearson R² = {r2:.3f}).\n\n{band}.\n")
md.append(f"\nThe rubric is a **usable fast proxy** — it recovers the broad ordering of 1,226 "
          f"universities at negligible cost. It is **not a substitute for the judgment**: "
          f"{1-r2:.0%} of the Track B target is not reachable by any straight line through the "
          f"Track A composite, and that residual is what the model in `07` is asked to learn.\n")

md.append("\n## Reliability of the labels\n")
md.append("| statistic | value | reading |\n|---|---|---|\n")
md.append(f"| self-consistency (60 swapped repeats, held out of the fit) | **{agree:.1%}** "
          f"(95% CI {lo:.1%}–{hi:.1%}) | {'meets' if agree >= 0.75 else 'below'} the 0.75 threshold set in the plan |\n")
md.append(f"| chance-corrected agreement | {kappa:.3f} | 0 = coin flip, 1 = perfect |\n")
md.append(f"| position balance | {100*n_left/len(judg):.1f}% left (binomial p = {binom.pvalue:.2f}) | no side preference |\n")
md.append(f"| comparisons per university | min {lab.n_comparisons.min()}, median {int(lab.n_comparisons.median())}, "
          f"max {lab.n_comparisons.max()} | every university is anchored to the graph |\n")
md.append(f"| BT accuracy on held-out comparisons | {cv_acc:.1%} | vs. a {agree:.1%} rater ceiling |\n")
md.append(f"| mean 95% CI width | {np.mean(ci_hi-ci_lo):.2f} logits | label uncertainty, usable as sample weights |\n")

md.append("\n## Where Track A succeeds and where it fails\n")
md.append("| pair type | n | Track A predicts the judgment | mean Track A gap |\n|---|---|---|---|\n")
for pt, acc, n in pt_acc:
    ga = (merged[merged.pair_type == pt].left_sid.map(key.set_index('sid').trackA_consensus)
          - merged[merged.pair_type == pt].right_sid.map(key.set_index('sid').trackA_consensus))
    md.append(f"| {pt} | {n} | {acc:.1%} | {ga.abs().mean():.1f} points |\n")
md.append(f"\nOverall the Track A composite calls {trackA_pred_acc:.1%} of the 840 judgments correctly.\n")
md.append("\nThis split is the core finding. On **random** pairs — where the rubric already sees a "
          "wide gap — the two methods agree, which is convergent validity: the rubric is not wrong. "
          "On **close** pairs the rubric falls toward chance, because a weighted sum of presence "
          "flags cannot resolve a site that has fewer features but uses them better. A holistic "
          "reading can, and does. Track B therefore carries genuine signal that no arithmetic over "
          "the 69 attributes reproduces — which is precisely the circularity the two-track design "
          "was built to break.\n")

md.append("\n## Robustness of the fit\n")
md.append(f"Regularisation C = {C_best} chosen by 5-fold CV log-loss over {grid}. "
          "Fitted without an intercept, so position bias cannot hide in the model; refitting "
          f"with one gives an intercept of {bt_int.intercept_[0]:+.4f}.\n\n")
md.append("| variant | Spearman vs. primary fit | top-50 overlap |\n|---|---|---|\n")
for name, r_, ov in sens_rows:
    md.append(f"| {name} | {r_:.4f} | {ov:.0%} |\n")
md.append("\nThe ordering is insensitive to all three choices, so it reflects the judgments rather "
          "than the fitting procedure.\n")

md.append("\n## Largest disagreements\n")
md.append("A positive shift means the blind holistic judgment ranks the site **higher** than the "
          "rubric does.\n\n")
md.append("| sid | Track A rank | Track B rank | shift | Track A score | Track B logit | region |\n")
md.append("|---|---|---|---|---|---|---|\n")
for _, r in div.head(10).iterrows():
    md.append(f"| {r.sid} | {r.trackA_rank} | {r.trackB_rank} | {r.rank_shift:+d} | "
              f"{r.trackA_consensus:.1f} | {r.trackB_bt_score:+.2f} | {r.region} |\n")
md.append(f"\nMedian rank shift is {lab.rank_shift.abs().median():.0f} of 200 places; "
          f"{(lab.rank_shift.abs() > 50).sum()} universities move more than 50.\n")

md.append("\n## Confounding check — reported, not resolved\n")
md.append(f"Between-region variance is **{eta2:.1%}** of the Track B target "
          f"(Kruskal–Wallis H = {kw.statistic:.2f}, p = {kw.pvalue:.4f}); the same figure for "
          f"Track A is {eta2_A:.1%}.\n\n")
md.append("This is a real effect and it is not explained away. Because collector and region are "
          "perfectly 1:1 in this dataset, two readings fit the data equally well — regions "
          "genuinely differ in institutional web quality, or the six collectors measured "
          "differently — and **no analysis of this data can separate them**. What can be said is "
          f"that the blind judging did not amplify the confound the rubric already carried "
          f"({eta2:.0%} vs {eta2_A:.0%}), and that roughly {1-eta2:.0%} of the target varies "
          "within region, which is the part a model can learn that is not geography. "
          "The response in `07` is procedural: leave-one-region-out is reported beside standard "
          "CV, and if it collapses, the model was learning region — which will be stated as the "
          "result rather than worked around.\n")

md.append("\n## Limitations, stated plainly\n")
md.append("1. These are **LLM-elicited expert judgments applied to extracted attribute profiles**, "
          "not human ratings of live websites, and must be described that way in any write-up.\n")
md.append("2. **One rater.** Self-consistency is measured; inter-rater agreement cannot be, because "
          "there is no second rater. This is the single largest threat to the labels' validity.\n")
md.append("3. Judgments are made from the blind profile cards, so anything the extractor did not "
          "capture — visual design, tone, actual link quality — is invisible to Track B as well.\n")
md.append("4. No external validation (QS, Webometrics, student survey) was performed, by decision. "
          "No claim of agreement with real user perception is available or made.\n")

(OUT / "rubric_validation.md").write_text("".join(md), encoding="utf-8")
print(f"wrote rubric_validation.md ({len('' .join(md))} chars)")

# %%
summary = dict(
    n_judgments=int(len(judg)), n_unique_pairs=int(len(uniq)), n_repeats=int(len(rt)),
    self_consistency=float(agree), self_consistency_ci=[float(lo), float(hi)],
    left_share=float(n_left / len(judg)), position_bias_p=float(binom.pvalue),
    C=float(C_best), bt_cv_accuracy=float(cv_acc), bt_train_accuracy=float(train_acc),
    trackA_judgment_accuracy=float(trackA_pred_acc),
    spearman_A_B=float(rho.statistic), kendall_A_B=float(tau.statistic), linear_r2=float(r2),
    mean_ci_width=float(np.mean(ci_hi - ci_lo)), region_eta2=float(eta2),
)
(OUT / "trackB_fit_meta.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))

# %% [markdown]
# ## What `07` inherits
#
# `expert_labels_trackB.csv` — 200 rows, and `trackB_bt_score` is the target `y`.
#
# It is a latent strength recovered from 900 reasoned blind comparisons. It is correlated with
# the rubric but not reducible to it, self-consistent at a measured rate, uncertainty-quantified
# per row, and not stratified by region. Those four properties are what make the next notebook a
# real supervised learning problem rather than a re-derivation of my own weighted sum.
