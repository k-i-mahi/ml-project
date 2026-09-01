# University Website Quality Ranking

**Machine Learning Laboratory — CSE 4112**
Department of Computer Science and Engineering, Khulna University of Engineering & Technology

---

## What this is, in one paragraph

We score university websites from **0 to 100** — higher is better — using 78 measurable
attributes of the landing page, and rank 1,226 universities globally, by region, and by
country. The score is produced by a LightGBM model trained on 200 universities that carry a
human-style expert label, and validated on 40 of them that the model never saw.

**The hard part was not the model, it was the label.** A dataset of website attributes has no
"quality" column. The tempting shortcut — invent a formula over the features, then train a
model to predict it — gives R² ≈ 0.98 and proves nothing. Everything in `report/report.pdf`
§3–§5 is about how we avoided that.

---

## Start here (5 minutes)

| I want to… | Open |
|---|---|
| understand the whole project | `report/report.pdf` — 12 pages |
| see the code, run the demos | `notebook/University_Website_Quality_Ranking.ipynb` |
| look up one university's score | notebook §9 — `lookup("kuet")` |
| score a university from its attributes | notebook §10 — `score_university({...})` |
| run it in Weka | `WEKA_GUIDE.md` |
| answer viva questions | `VIVA_NOTES.md` |

---

## Folder contents

```
SUBMISSION/
├── README.md                  <- you are here
├── WEKA_GUIDE.md              <- click-by-click Weka instructions + expected numbers
├── VIVA_NOTES.md              <- likely questions and how to answer them
│
├── data/
│   ├── university_websites_labeled.csv   1,226 rows — every university, scored and ranked
│   ├── train.csv                           160 rows — expert-labelled, used for fitting
│   ├── test.csv                             40 rows — expert-labelled, HELD OUT
│   ├── data_dictionary.csv               what each of the 78 features means
│   ├── dataset_summary.json              headline numbers, machine-readable
│   ├── weka/                             the same splits as ARFF (5 files)
│   └── label_provenance/                 the raw evidence behind the label
│       ├── trackB_judgments_v2.csv         900 judgments — the label we used
│       ├── trackB_judgments.csv            900 judgments — the first pass, superseded
│       ├── trackB_pairs.csv                which universities were compared
│       ├── trackB_key.csv                  anonymous card ID -> real university
│       ├── trackB_profiles_v2.md           the 200 anonymised cards, as judged
│       ├── trackB_fit_meta.json            Bradley-Terry fit statistics
│       ├── ranking_meta.json               SHAP block shares, LORO by region
│       ├── trackA_block_weights.csv        the rubric's declared weights
│       └── missing_value_policy.json       why nulls were filled the way they were
│
├── notebook/
│   └── University_Website_Quality_Ranking.ipynb   the full pipeline, already executed
│
├── outputs_from_notebook/      <- CSVs the notebook writes when you re-run it
│   ├── final_ranking.csv
│   ├── model_comparison.csv
│   ├── test_set_metrics.csv
│   ├── feature_importance.csv
│   └── leave_one_region_out.csv
│
├── model/
│   ├── final_model.joblib                  trained on all 200 — used for the ranking
│   ├── model_trained_on_train_split.joblib  trained on 160 — used for the honest test
│   └── model_card.json                     features, hyper-parameters, metrics
│
└── report/
    ├── report.pdf                          the report
    ├── report.tex                          its source
    └── figures/                            8 figures used in the report
```

---

## The three CSV files

| file | rows | what it is |
|---|---|---|
| `train.csv` | 160 | Expert-labelled. The model learns from these. |
| `test.csv` | 40 | Expert-labelled. **Held out** — touched exactly once, at the end. |
| `university_websites_labeled.csv` | 1,226 | Every university with its final score, grade and three ranks. 200 have a ground-truth label (`has_expert_label = 1`); the other 1,026 are model inference. |

Columns in `university_websites_labeled.csv`:

`uni_id`, `name`, `url`, `country`, `region`, **`predicted_score`** (0–100), **`grade`**
(A+/A/B/C/D/F), **`global_rank`**, `regional_rank`, `country_rank`, `percentile`,
`quality_score` (the expert label, where one exists), `has_expert_label`, `trackA_consensus`
(the rubric baseline), then the 78 features.

---

## Headline results

| | |
|---|---|
| Universities scored | 1,226 |
| Expert-labelled | 200 → 160 train / 40 test |
| Features | 78 |
| **Held-out test Spearman ρ** | **0.922** |
| Held-out test R² | 0.818 |
| Held-out test MAE | 9.0 points on 0–100 |
| Rule-based rubric baseline | ρ = 0.810 ← the bar the model had to clear |
| Leave-one-region-out ρ | 0.844 |
| Label self-consistency | 96.7% on 60 swapped repeat pairs |

**KUET:** score **85.9**, grade **A+**, **global rank 31 of 1226**, **1st in Bangladesh**.

---

## How the label was made (the part worth reading)

There is no ground-truth "website quality" column anywhere, so we built one in two tracks:

- **Track A — the rubric.** An explicit, frozen scoring rubric with gates, six weighted tiers
  and non-linear response curves, averaged over five personas. It is a *formula over the
  features*, so it is used as the **baseline**, never as the target.
- **Track B — blind pairwise judgment.** 200 universities rendered as anonymised profile
  cards (no name, country, region or URL), presented in 900 pairs, each judged on which site a
  prospective applicant would trust more. Fitted with the Bradley–Terry model to a latent
  score. This is the **target**.

Because Track B is not a function of the features, `Spearman(A, B) = 0.814` is a real result.
And the rubric predicts *wide-gap* pairs at 88% but *close* pairs at only 62% — that 26-point
gap is the signal the model actually learns.

### We labelled it twice, and we say so

The first labelling pass was completed, measured, and then **discarded as a target**. Two
things were wrong with it:

1. `notice_recency_days` had been median-imputed. The median is **1 day**, so the 469 sites
   with *no dated notice at all* were described to the judge as "posted yesterday".
2. The profile cards listed **alt-text coverage** as a headline number, and 37.6% of the
   judge's stated reasons ended up citing it — an attribute no visitor can perceive.

Both were fixed and **all 900 pairs were re-judged** on the same universities, so the two
passes are directly comparable:

| reason cites | pass 1 | pass 2 |
|---|---|---|
| alt-text / screen-reader | 37.6% | **0.0%** |
| admission-task content | 38.1% | **73.1%** |
| freshness | 7.9% | **26.3%** |
| navigation | 35.1% | 41.7% |

**16.4% of winners changed.** Not 2% (the rewrite would have done nothing) and not 45% (the
judgments would be noise). Both passes ship in `data/label_provenance/` so you can check this.

---

## Honest limitations

Read these before quoting a number.

1. **The dataset cannot see design.** Every feature is a flag, a count or a measurement.
   Whether a page is cluttered, dated or unpleasant is not in the 78 columns, so it cannot be
   in the score. A site that ticks every content box ranks highly even if a human would find
   it ugly. This is the biggest gap between this ranking and a real impression.
2. **2.0% of rows are extraction failures and they sit at the bottom.** 25 universities have
   no navigation, no programmes, no departments and no contact recorded — almost certainly
   JavaScript-rendered pages the crawler could not read, not empty websites. 21 of them are in
   the bottom 100. **The top of the ranking is unaffected; do not quote the bottom without
   this caveat.**
3. **The labels are expert-style judgments made from extracted attribute profiles**, not
   ratings of live websites by a human panel.
4. **One rater.** Self-consistency is measured (96.7%); inter-rater agreement cannot be.
5. **No external validation** against QS, Webometrics or a student survey.
6. **Collector and region are perfectly confounded** — each of six collectors covered exactly
   one region. Regional score differences cannot be separated from measurement differences.
   Model *accuracy* is stable across regions (LORO ρ = 0.81–0.87), so it is not a regional
   lookup table.
7. **200 labelled of 1,226.** The rest are inference, flagged by `has_expert_label`.

---

## Reproducing

The notebook is already executed — every output you see was produced by the code above it.
To re-run it:

```
cd SUBMISSION/notebook
jupyter lab University_Website_Quality_Ranking.ipynb
```

It reads only from `../data/` and `../model/` and needs nothing outside this folder.

**Environment:** Python 3.13 · pandas 2.3.3 · numpy 2.4.4 · scikit-learn 1.8.0 ·
lightgbm 4.6.0 · shap 0.51.0

The nine research notebooks that produced these artefacts (`01_audit` … `09_deliverables`)
live in the parent project directory and run in order from the raw CSV with a fixed seed.
