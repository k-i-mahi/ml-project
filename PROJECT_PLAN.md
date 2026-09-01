# University Website Quality — Labeling & Full ML Project Plan

> ## ✅ Status: executed in full — 2026-09-01
>
> All nine notebooks run end to end with zero error outputs; all 8 verification conditions
> below PASS. See `DELIVERABLES.md` for the finished artefact set.
>
> | what the plan predicted | what actually happened |
> |---|---|
> | Track B judging, ~900 pairs | **900/900 complete**, 48.4% left wins (no position bias) |
> | Self-consistency ≥ 0.75 | **0.867** (95% CI 0.754–0.941) on 60 held-out swapped repeats |
> | Spearman(A,B) ≈ 0.6–0.8 validates the rubric | **0.790** — in band; 44% of the target is not a linear image of Track A |
> | Track A may match the model — report it either way | **model wins**: LightGBM+mono LORO ρ **0.844** vs rubric **0.708**, 25/25 folds, p < 0.0001 |
> | "n=200 is small; Ridge plausibly wins" | **wrong** — LightGBM led on every metric, and led by more under LORO. Recorded in `07` §prior |
> | SHAP vs rubric weights is the headline | **accessibility 6.4% declared → 14.8% realised (2.3×)**; technical performance 6.2% → 2.8% (0.45×) |
>
> Two findings the plan did not anticipate: the technical block **alone** is nearly useless
> (LORO ρ = 0.069), and top-100 regional representation spans 2.3× to 0.25×, which is the
> largest standing caveat on the global ranking.

## Context

The goal is a **global** university-website-quality ranking learned from measurable landing-page
attributes — not from prestige. The dataset (`all_universities.csv.xls`, 1,230 × 85) is complete
and the Lab 3 report supplies the authoritative attribute schema (Table 2/3, 69 attributes in
11 blocks B1–B11).

**The blocking problem: there is no label column.** The report's planned gold label
(`label_expert_score`: 180 sites × 3 trained human raters) is not achievable — the team can't do
it. Without a label, the only remaining move is to build a composite score from the features and
train a model to predict it, which yields a guaranteed-high R² that measures nothing. Biyyapu et
al. (2023) published exactly that mistake (98.2% accuracy predicting a lookup table from its own
inputs); this project's contribution depends on not repeating it.

**Resolution: a two-track LLM labeling design.** Track A is an explicit, auditable rubric applied
in code to all 1,230. Track B is holistic blind pairwise judgment over 200 universities, which is
*not* a closed-form function of the features and therefore constitutes a genuine learning target.
Spearman(A, B) becomes a real empirical result rather than an artefact.

Per user decisions: **Track B full scope** (200 universities, ~900 pairs); **load time
region-standardized** with both variants reported; **no external validation** (no QS/Webometrics
or student-survey correlation); **Jupyter notebooks**, executed locally with outputs saved; the
Lab 3 report is reference material for the schema only, not a constraint on design.

---

## What was verified in the data

All claims carried over from `ok.md` were re-checked against the CSV and hold:

| Finding | Verified value |
|---|---|
| Shape | 1,230 × 85; 69 `aXX_*` attributes, mapping **exactly** onto report blocks B1–B11 |
| Prestige columns unusable | `a10_webometrics_value` 99.35% null, `a08_qs_value` 94.96% null |
| Exact duplicate column | `load_time_s` ≡ `a62_load_speed_s` in all 1,230 rows |
| Constant columns | `ok` (all True), `extractor_version`, `render_error` (100% null) |
| **Member ⊥ region is 1:1** | Each M1–M6 maps to exactly one region — collector and geography are inseparable |
| Load-time regional gradient | Median 8.07s (M2/W-Europe) → 12.94s (M6/LatAm-Africa) |
| Duplicate domains | 4 pairs: `unam.mx`, `tec.mx`, `ipn.mx`, `aun.edu.eg` (8 rows) |
| Impossible dates | 285 of 759 notice dates are after the crawl date, max 2027-12-31 |
| Notice inconsistency | 382 rows: date present but `a16_notice_board=0`; 156 rows: board but no date |
| Event inconsistency | 327 rows: `a23=1, a24=0`; 158 rows: `a23=0, a24>0` |
| Broken links | 81.1% zeros, no denominator column, one row at 1,129 (next highest 59) |
| Mobile score is ordinal | 6 levels cover 1,216/1,230 rows: 75 (645), 90 (310), 100 (222), 50, 20, 0 |
| Country mixes levels | 10 bucket labels ("Sub-Saharan Africa", "Balkans", …) covering 170 rows |
| Country ranking eligibility | **19** real countries with ≥20 universities (22 minus 3 buckets) |
| Attribute gaps resolved | a19/a52/a55/a56/a64/a68 are **intentional** exclusions (report §4.6: 75 − 6 = 69), not scraper loss |

Environment confirmed: pandas 2.3.3, numpy 2.4.4, scikit-learn 1.8.0, lightgbm 4.6.0, shap 0.51.0.

---

## The target column (what the ML model predicts)

```
FEATURES (X)                    TARGET (y)                  MODEL OUTPUT
~75 engineered columns    →    trackB_bt_score        →    predicted_quality_score
from the 69 attributes         (only 200 rows have it)     (all 1,226 rows)
                                                                  ↓
                                                    global / regional / country rank
```

| Column | Rows populated | Range | Role |
|---|---|---|---|
| `trackA_consensus` | all 1,226 | 0–100 | **Baseline, NOT the target.** Mean of the 5 persona rubric scores |
| **`trackB_bt_score`** | **200** | **logit, ~−3…+3** | **← `y`. The supervised learning target** |
| `trackB_score_100` | 200 | 0–100 | Same value rescaled, for human reading only |
| `trackB_bt_ci_low/high` | 200 | logit | Bootstrap CI — label uncertainty, usable as sample weights |
| `predicted_quality_score` | all 1,226 | → 0–100 | **Model output.** What the final ranking sorts on |

`trackB_bt_score` is the Bradley–Terry latent strength recovered from ~900 blind pairwise
judgments over 200 universities. It comes from *judgments*, not from arithmetic over the columns,
which is the entire reason it is a legitimate target.

**Why `trackA_consensus` is not the target.** It is a deterministic formula over the same 69
features. A model predicting it would report R² ≈ 0.98 while proving only that the learner can
re-derive my own weighted sum — the published error in Biyyapu et al. (98.2% accuracy predicting a
lookup table from its own inputs). Track A is therefore the **baseline the model must beat**,
scored against Track B alongside every learned model.

**Train / predict.** Fit on the 200 rows carrying `trackB_bt_score` (nested CV +
leave-one-region-out). Training uses the raw BT logit — the statistically correct scale, unbounded
and centred; the 0–100 version is presentation only. Then predict `predicted_quality_score` for
all 1,226; the remaining 1,026 are pure inference and are never used in fitting.

---

## Design decisions

**Blocks.** Keep the report's B1–B11 as the operational partition (finer-grained, already
documented, verified to cover all 69). Add a documented crosswalk B1–B11 → Saleh et al. six SLR
factors (information quality, specific content, usability, web appearance, service interaction,
functionality) for literature framing. Both, not either.

**Prestige.** `a08_qs_value` / `a10_webometrics_value` are dropped — unusable *and* leaky.
`a07_qs_badge` and `a09_national_rank` are **kept as website features** (does the site display a
credibility signal) but sit in the lowest rubric tier with near-zero weight. This distinction is
stated explicitly, not left implicit.

**Load time.** Region z-score (`load_time_z_region`), which removes exactly the between-region
signal that is confounded with collector. Every ranking is produced with and without it and the
delta reported. The 3.3× disagreement on re-crawled UNAM is cited as the justification.

**Weights before variance.** Rashida et al. allocated 40% to performance but it drove 2.9% of
their ranking variance, because every university scored alike on it. A **block variance-contribution
audit runs before any weight is fixed**, and every reported weight is shown as *nominal weight*
alongside *realised variance share*.

---

## Notebooks

Location: `ML_PROJECT/notebooks/`, artefacts to `ML_PROJECT/outputs/`, figures to
`ML_PROJECT/figures/`. Each notebook runs standalone from the previous notebook's artefacts.

This plan is also written to `ML_PROJECT/PROJECT_PLAN.md` as the first step, so it lives with the
project rather than only in the planning directory.

### 01_audit.ipynb → `audit_report.csv`, `cleaned_dataset.csv`, `assumptions.md`

Full forensic audit in EXPECTED / ACTUAL / PROBLEM / FIX format, one row per finding. Nothing is
silently corrected.

- **Drop** `ok`, `extractor_version`, `render_error` (constant/empty); `a62_load_speed_s`
  (exact duplicate); `a08_qs_value`, `a10_webometrics_value` (unusable + prestige leakage).
- **Resolve 4 duplicate domains** → 1,226 rows. The two crawls agree on 68/69 attributes, so keep
  one row, set `load_time_s` to the median of the pair, and fix the mislabeled `country` on
  UNAM/IPN (recorded as United States).
- **Censor 285 future notice dates** to the crawl date and add `notice_date_future` — censored,
  not deleted.
- **Add `country_is_bucket`** (10 labels, 170 rows) and `country_rank_eligible` (19 countries).
- **Flag near-constant** features (`a75_bookmark` 1.0%, `a59_feedback_form` 2.0%, `a65_https`
  98.9%, `a30_image_gallery` 97.1%) — flagged, not dropped; trees are unharmed by low variance.
- **Zero-vs-null semantics** decided per feature and written to `assumptions.md` with the
  reasoning and the risk if wrong. The load-bearing one: `a66_broken_links = 0` is read as
  *"checked, none found"* because `ok` is True and `render_error` is null for all 1,230 rows —
  there is no evidence of checker failure anywhere in the crawl metadata. The absence of a
  denominator column is recorded as a permanent limitation.

### 02_features.ipynb → `model_ready_dataset.csv`, `feature_dictionary.csv`, `block_variance_report.csv`

Every engineered feature carries FORMULA / REASON / EXPECTED INTERPRETATION in
`feature_dictionary.csv`, plus its block and declared monotonic direction.

- `notice_recency_days` = crawl − `a18`, future-censored; `a18_missing` indicator (38.3%, and the
  missingness is itself informative — no dated notices found).
- `notice_evidence` (ordinal 0–3) and `event_evidence` (ordinal), which *reconcile* the
  a16/a17/a18 and a23/a24 inconsistencies instead of leaving 382 + 327 contradictory rows.
- Block completeness ratios: `content_completeness_B5` (15 binaries), `footer_completeness_B6`,
  `a11y_completeness_B11`, `seo_completeness_B10`.
- `nav_quality` — inverted-U transform of `a03_nav_item_count`.
- `load_time_z_region`; `log1p(a66_broken_links)` winsorized at p99 (isolates the 1,129 outlier);
  `broken_links_present` binary.
- `a63_mobile_score` recoded as a 6-level **ordinal**, not continuous.
- Missingness indicators for `a72`, `a53`.
- **Block variance-contribution audit** — for each block, its share of total ranking variance
  under equal-block weighting. This runs *before* the rubric fixes any weight.

### 03_eda.ipynb → figures + written interpretations

Feature-type-appropriate only, no decorative plots: binary prevalence bars; count distributions
with zero-inflation and skew; continuous histograms/box plots; missingness chart; Spearman matrix
with |ρ| > 0.90 pairs flagged; region-wise distributions; the load-time-by-collector plot that
demonstrates the confound. Outliers are identified and classified (legitimate / measurement error
/ scraper error / impossible) — **never auto-removed**.

### 04_rubric_trackA.ipynb → `rubric_v1.md`, `expert_labels_trackA.csv`

`rubric_v1.md` is written and frozen **before** any score is computed, so the rules can be
disputed independently of their results.

**Gates** (cap the maximum achievable score regardless of everything else): no `a02_primary_nav`
→ cap 45; no `a65_https` → cap 60. Few, and each justified.

**Tiers** (Tier 1 heaviest → Tier 6 near-zero): T1 can-I-do-the-task (programs, admissions,
contact, search, departments, faculty) · T2 is-this-alive (dated notices, news, events with
datetimes) · T3 depth of service · T4 accessibility & technical · T5 polish · T6 self-promotion
(QS badge, trust seals, testimonials, bookmark) at near-zero weight.

**Non-linear response curves** — the part a correlation matrix cannot produce:

| Feature | Rule |
|---|---|
| `a03_nav_item_count` | Inverted U — 0–2 broken, 5–9 ideal, 15+ cluttered |
| `a24_event_count` | 0 dead, 3–10 active, 20 (the cap) reads as an undated dump |
| `a15_stat_item_count` | Hard diminishing returns after ~6 |
| `a72_alt_text_pct` | Concave — steep 0→60%, gentle 60→100% |
| `a53_contrast_ratio` | Plateau at 7:1 (WCAG AAA); 21:1 is not better than 12:1 |
| `a16` without `a17` | Half credit — undated notices are weak freshness evidence |

**Five personas**, each a distinct defensible weighting over the same gates and curves:
prospective domestic student · prospective international student · accessibility/usability
specialist · researcher/academic visitor · literature-aligned (SLR emphasis, appearance ~10%).

Outputs per university: five persona scores, `trackA_consensus` (mean), `trackA_sd`
(disagreement = per-university uncertainty, genuinely useful for finding edge cases), and
**ICC(2,k)** across personas as a reportable reliability statistic. Top-50 stability across the
five personas is a publishable robustness result.

### 05_trackB_generate.ipynb → `trackB_profiles.md`, `trackB_pairs.csv`

- **Sample 200** universities, stratified by region (≈33/region) and within region across the full
  `trackA_consensus` range, so coverage isn't concentrated in the middle.
- **Render blind profile cards**: natural-language, grouped by *user need* rather than by block,
  with **no name, country, region, URL, collector, or Track A score**. Blinding is what guarantees
  no prestige and no rubric-anchoring leaks into the labels. Stable non-ordinal IDs (`S-4471`).
- **900 pairs**: 500 random (graph connectivity, wide gaps) + 340 close pairs matched within ±5
  Track A points (the hard, informative cases) + **60 repeat pairs re-presented in swapped order**
  in a later batch. Left/right position randomized throughout. ~9 comparisons per item.

### ⟶ Judging stage (conversational, not scripted)

I read profile batches and emit judgments — winner, confidence (clear / slight / toss-up), and a
one-line reason — appended incrementally to `trackB_judgments.csv`. **This must not be a script**;
a formula here would reintroduce exactly the circularity the design exists to break.

**This is the long pole.** ~900 judgments at ~25 per batch ≈ 36 batches, spanning many turns.
The CSV is append-only and checkpointed so the work is resumable across sessions.

### 06_trackB_fit.ipynb → `expert_labels_trackB.csv`, `rubric_validation.md`

- Fit **Bradley–Terry** via L2-regularized logistic regression on the ±1 pair design matrix (ties
  → half-win each). Bootstrap confidence intervals on the latent scores.
- **Self-consistency** from the 60 swapped repeats — an intra-rater reliability figure, which is
  the standard reviewer objection to single-rater labels and is answered here with a number.
- **Spearman(Track A, Track B)** — the headline validation. ~0.6–0.8 validates the rubric as a
  fast proxy; low means the rubric is missing something and the divergent cases show what. Either
  outcome is a result. `rubric_validation.md` diagnoses where they disagree and why.

### 07_model.ipynb → `model_evaluation.csv`, `feature_importance.csv`

- **Target: the Track B latent score** on the 200 labeled universities. Track A is a *baseline to
  beat*, never the target.
- Models: Ridge (interpretable floor) · RandomForest · LightGBM · LGBMRanker/LambdaMART with
  region as query group.
- Nested CV: outer repeated stratified 5-fold × 5 seeds, inner 3-fold tuning, **all** imputation /
  encoding / scaling / selection fitted inside the training fold. Mean ± SD across 25 fold-reps.
- **Leave-one-region-out** — the real test of whether "quality" was learned or geography was.
- Baselines it must beat: mean predictor; **the Track A composite scored directly against Track B**
  (the one that matters — if a transparent weighted sum matches the model, the ML contribution is
  null and that gets reported); B5-content-only; B9-technical-only.
- Metrics: **Spearman ρ primary** (the deliverable is a rank list), Kendall τ, MAE, RMSE, R²
  secondary.
- Monotonic constraints tested as a variant where defensible (`a72`↑, broken links↓, load time↓,
  content completeness↑) — not forced where the relationship is genuinely non-monotone.
- Then predict all 1,226.

**Stated up front:** n=200 is small for gradient boosting. Ridge plausibly wins. That is a
legitimate finding to report honestly, not a failure to paper over.

### 08_ranking_explain.ipynb → rankings, SHAP, fairness

- `global_website_rank`, `regional_website_rank`, `country_website_rank` (**19 eligible countries
  only** — for the 10 bucket labels it is not a country ranking and is not produced).
  Transparent tie-break: predicted score → `content_completeness_B5` → `a11y_completeness_B11` →
  `uni_id`.
- Every ranking produced twice: with and without `load_time_z_region`; the rank-displacement
  distribution between them is reported.
- SHAP: global beeswarm, **block-aggregated SHAP**, local waterfalls for top/middle/bottom-3,
  a `why_a_above_b(uni_a, uni_b)` function, and per-university improvement recommendations.
- **Headline analysis:** SHAP-derived empirical block weights vs the a-priori rubric block weights.
  Where they diverge is direct evidence that hand-set scoring rules misallocate importance across
  quality dimensions — the central claim.
- Fairness: predicted score and LORO error by region, by `country_is_bucket`, by country.
  The member⊥region confound is reported as **unresolvable with this data**, not glossed.

### 09_deliverables.ipynb → assembled outputs

Assembles the 28-item deliverable list from `basic.md`, the final model-ready schema, and a
**limitations** section stating plainly: (1) labels are LLM-elicited expert judgment applied to
extracted features, **not** human ratings of live websites, and must be described that way;
(2) no external human validation was performed, so no claim of agreement with real user perception
is available; (3) collector and region are perfectly confounded; (4) `a66_broken_links` has no
denominator. Also emits a short `report_corrections.md` noting the Lab 3 report deltas (global vs
Bangladeshi scope, 1,200 vs 1,230, stale §7/§8, L3 replaced) as a byproduct — not a rewrite.

Related work retains the Rashida et al. re-analysis (their automated tool vs their own student
survey: ρ = 0.102, n=20) as the empirical argument that naive equal-weight presence-counting fails
to predict human judgment — which is precisely what the gated, tiered, non-linear rubric avoids.
It is cited as literature, not used as our validation.

---

## Verification

1. **Audit reproduces the table above.** `01_audit.ipynb` re-derives every verified finding
   (1,230→1,226 rows, 285 future dates, 4 duplicate domains, 19 eligible countries, the 1:1
   member⊥region map). Any mismatch means the audit logic is wrong, not the data.
2. **Rubric frozen before scoring.** `rubric_v1.md` is committed before `04` computes a score;
   confirm via git that the rubric file predates `expert_labels_trackA.csv`.
3. **Blinding holds.** Grep `trackB_profiles.md` for every university name, country, region, URL
   token, and for `trackA` — zero hits required before judging starts.
4. **Judging is not scripted.** `trackB_judgments.csv` must contain a free-text reason per row;
   confirm judgments are not reproducible by any linear function of Track A (regress B on A and
   check residual structure — perfect fit means the design failed).
5. **Self-consistency.** The 60 swapped repeats yield agreement ≥ ~0.75; below that the profile
   cards are ambiguous and need revision before Track B is used as a target.
6. **No leakage.** Assert `a08_qs_value` / `a10_webometrics_value` are absent from the design
   matrix; assert preprocessing is fitted inside CV folds by checking a deliberately corrupted
   validation fold does not change training-fold statistics.
7. **The honest comparison runs.** `model_evaluation.csv` must contain the Track A composite
   scored against Track B alongside every learned model. If no model beats it, that is the
   reported result.
8. **End-to-end.** Re-run `01` → `09` from the raw CSV in order; all artefacts regenerate and the
   global ranking is reproducible from a fixed seed.
