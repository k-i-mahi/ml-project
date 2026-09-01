# University Website Quality — Deliverables

A global ranking of **1,226 university websites** by content quality, learned from blind human-style judgment rather than from a formula over the same features.

## The result in five numbers

| | |
|---|---|
| Rank correlation between the rubric and blind judgment | **ρ = 0.814** — correlated, not collapsed |
| Judgment self-consistency (60 swapped repeats) | **96.7%** |
| Best model, leave-one-region-out | **ρ = 0.844** (LightGBM+labelwt) |
| Transparent rubric baseline, same folds | ρ = 0.774 |
| Accessibility: declared vs. realised weight | 6.4% → 1.3% (**0.2×**) |

## What was actually found

**1. A weighted-sum rubric is a decent proxy and a poor judge.** Track A predicts 78% of blind pairwise judgments overall — but 86% on pairs it already sees as far apart and only **63% on close pairs**, barely above chance. A presence-counting formula cannot tell apart two sites with similar feature counts and different execution. That gap is the entire justification for learning a model rather than publishing the rubric.

**2. Machine learning added something measurable here.** LightGBM+labelwt beats the calibrated rubric on all 25 nested-CV fold-reps (ρ 0.873 vs 0.810, paired p < 0.0001) and holds the margin under leave-one-region-out (0.844 vs 0.774). This was not guaranteed and the comparison was built to be able to say the opposite.

**3. Hand-set weights misallocate importance — measured, not asserted.** Accessibility was declared at 6.4% of the rubric and drives 1.3% of the ranking (0.2× its brief). Technical performance was declared at 6.2% and drives 5.4% (0.86×) — the same direction as the published Rashida et al. failure (40% declared, 2.9% realised), caught early here by auditing block variance before fixing any weight.

**4. Technical metrics alone are close to useless for this task.** A model using only the technical block reaches ρ = 0.100 under LORO. Page speed, HTTPS and mobile scores are near-universal, so they cannot separate anything.

## Files

### Rankings
| file | rows | what it is |
|---|---|---|
| `ranking_global.csv` | 1,226 | the deliverable — global rank 1–1,226, both load-time variants, per-university improvement headroom |
| `ranking_by_region.csv` | 1,226 | same, ordered within the 6 regions — the safest cut, since collector is constant within a region |
| `ranking_by_country.csv` | 797 | 19 eligible countries only |

### Labels and validation
| file | what it is |
|---|---|
| `expert_labels_trackA.csv` | 5 persona rubric scores + consensus for all 1,226 (ICC(2,k) = 0.994) |
| `expert_labels_trackB.csv` | **the target** — BT strength + bootstrap CI for 200 |
| `trackB_judgments.csv` | 900 blind judgments with a free-text reason each |
| `trackB_profiles.md` | the 200 blind profile cards the judging was done from |
| `trackB_pairs.csv / trackB_key.csv` | pair design and the sid → university key |
| `rubric_v1.md` | the rubric, frozen before any score was computed |
| `rubric_validation.md` | does the rubric hold up? — the ρ = 0.790 analysis |

### Model and explanation
| file | what it is |
|---|---|
| `model_evaluation.csv` | 9 models × CV + LORO + the paired test against the rubric |
| `predictions_all.csv` | predicted score for all 1,226, flagged labelled vs inferred |
| `shap_values.csv` | full SHAP matrix, 1,226 × 78 |
| `shap_vs_rubric_weights.csv` | **the headline** — declared vs realised block influence |
| `feature_importance.csv` | permutation importance on held-out folds |
| `fairness_by_region.csv / fairness_by_country.csv` | score and error by group |

### Data lineage
| file | what it is |
|---|---|
| `audit_report.csv` | 20 findings in EXPECTED/ACTUAL/PROBLEM/FIX form |
| `assumptions.md` | every zero-vs-null decision, with the risk if wrong |
| `cleaned_dataset.csv` | 1,226 × 83 after the audit |
| `model_ready_dataset.csv` | engineered features |
| `feature_dictionary.csv` | 85 features with formula, reason, direction, block, SLR factor |
| `block_variance_report.csv` | nominal weight vs realised variance, run before weights were fixed |
| `verification.csv` | the 8 plan conditions, re-checked against disk |
| `artefact_inventory.csv` | all 148 files with sha256 |

### Notebooks

- `notebooks/01_audit.ipynb` — forensic audit — 19 findings, nothing silently corrected
- `notebooks/02_features.ipynb` — feature engineering + block variance audit
- `notebooks/03_eda.ipynb` — 9 figures, outliers classified never removed
- `notebooks/04_rubric_trackA.ipynb` — the rubric applied to all 1,226; ICC across 5 personas
- `notebooks/05_trackB_generate.ipynb` — blind profile cards + 900-pair design
- `notebooks/06_trackB_fit.ipynb` — Bradley–Terry, self-consistency, ρ(A,B)
- `notebooks/07_model.ipynb` — 9 models, nested CV, LORO, the honest comparison
- `notebooks/08_ranking_explain.ipynb` — rankings, SHAP, the headline analysis, fairness
- `notebooks/09_deliverables.ipynb` — this — verification and assembly

## How to reproduce

```bash
# from ML_PROJECT/
for n in 01_audit 02_features 03_eda 04_rubric_trackA 05_trackB_generate \
         06_trackB_fit 07_model 08_ranking_explain 09_deliverables; do
  python src/nbbuild.py src/nb${n%%_*}_*.py notebooks/$n.ipynb
done
```

Notebooks 01–05 and 06–09 are deterministic given fixed seeds. The judging stage between 05 and 06 is not reproducible by running code — it is 900 recorded judgments, shipped as `trackB_judgments.csv`.

## Limitations — read before quoting any number

1. **The labels are LLM-elicited expert judgment applied to extracted attribute profiles, not human ratings of live websites.** Every use of the word "expert" in these artefacts means this. The judgments were made from blind text profile cards, so anything the extractor did not capture — visual design, tone, whether links actually work — is invisible to the labels as well as to the model.

2. **One rater.** Self-consistency is measured (96.7% on 60 swapped repeats). Inter-rater agreement is not measured because there is no second rater. This is the single largest threat to validity and no statistic here addresses it.

3. **No external validation was performed** (a decision, not an oversight). There is no comparison against QS, Webometrics, or a student survey, so **no claim of agreement with real user perception is available or made.**

4. **Collector and region are perfectly confounded** — each of the 6 collectors covered exactly one region, 1:1. Regional differences in score have two equally consistent readings — real differences in web quality, or six people running the extractor differently — and this data cannot separate them. Top-100 representation runs from **2.0× over-expectation (South & Central Asia + Oceania) to 0.37× under (Asia (East & Southeast))**. The model's *accuracy* is stable across regions (LORO ρ 0.81–0.87), so the ranking is not a regional lookup table — but the global top-N should never be quoted without this caveat. **The regional ranking is the safer artefact**, since collector is constant within a region.

5. **`a66_broken_links` has no denominator.** 4 broken links out of 10 and out of 400 are recorded identically. The count is used, the rate cannot be.

6. **n = 200 labelled of 1,226.** The other 1,026 are pure inference. The labelled and inferred prediction distributions are statistically compatible (KS p = 0.04), so the model is interpolating rather than extrapolating — but the confidence intervals on the 1,026 are wider than any number here shows.

7. **285 of 759 notice dates were in the future** (max 2027-12-31) and were censored to the crawl date rather than deleted. `notice_date_future` flags them.
