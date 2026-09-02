# University Website Quality — Scoring and Ranking

**Machine Learning Laboratory — CSE 4112**
Department of Computer Science and Engineering, Khulna University of Engineering & Technology

---

## In one paragraph

We collected 1,230 university landing pages worldwide, each described by 69 automatically
measurable attributes, and cleaned them into a dataset of **1,225 universities × 71 features**.
We defined a **seven-dimension scoring model** that converts those attributes into a website
quality score from **0 to 100** — higher is better — then split the data **80/20**, with **all
22 Bangladeshi universities forced into the test set**, and trained **twelve regression
algorithms** to predict that score. The best model, a tuned LightGBM, reaches **R² = 0.993**
with an average error of **1.15 points** on 244 universities it never saw during training —
and **0.65 points** on the 22 Bangladeshi universities, an entire country held out of training
by design.

---

## Start here

| I want to… | Open |
|---|---|
| understand the whole project | `report/report.pdf` — 36 pages |
| see the code and run the demos | `notebook/University_Website_Quality.ipynb` — 16 sections |
| see the Bangladesh case study | report §9, or notebook §9 |
| look up how a feature is classified | `data/feature_catalog.csv`, or report Table 3 |
| check the arithmetic of any score | `data/dimension_scores.csv` |
| look up one university | notebook §14 — `lookup("kuet")` |
| score a university from its attributes | notebook §15 — `score_university({...})` |
| run it in Weka | `WEKA_GUIDE.md` |
| prepare for the presentation | `VIVA_QA.md` |

---

## Folder contents

```
FINAL_PROJECT/
├── README.md                   <- you are here
├── WEKA_GUIDE.md               <- click-by-click Weka steps + expected numbers
├── VIVA_QA.md                  <- likely questions and how to answer them
│
├── data/
│   ├── university_website_scores.csv   1,225 universities: features + score + grade + 3 ranks
│   ├── train.csv                         981 rows (80%, no Bangladeshi university)
│   ├── test.csv                          244 rows (20%, includes all 22 Bangladeshi)
│   ├── test_bangladesh.csv                22 rows — the case study on its own
│   ├── feature_catalog.csv             every feature on three classification axes
│   ├── dimension_scores.csv            every sub-score, contribution, cap and final score
│   ├── split_stratification.csv        band drift caused by the country holdout
│   ├── data_dictionary.csv             every column, its dimension, its range
│   ├── dataset_summary.json            headline numbers, machine-readable
│   ├── missing_value_policy.json       how missing values were filled and why
│   └── weka/                           the same splits as ARFF (5 files)
│
├── notebook/
│   └── University_Website_Quality.ipynb    the full analysis, already executed
│
├── src/
│   ├── build_dataset.py        raw file -> scored dataset -> 80/20 split -> catalogue -> ARFF
│   ├── attribute_reference.py  the human-readable spec of all 71 attributes
│   ├── notebook_src.py         the notebook as a plain .py script
│   ├── make_figures.py         regenerates every figure in the report
│   ├── make_tables.py          regenerates every data-driven table in the report
│   └── nbbuild.py              builds and executes the notebook from notebook_src.py
│
├── model/
│   ├── final_model.joblib      the tuned LightGBM model
│   └── model_card.json         hyper-parameters, features, metrics, holdout
│
├── results/
│   ├── model_comparison.csv          the twelve-algorithm leaderboard
│   ├── cross_validation.csv          5-fold CV results
│   ├── feature_importance.csv        gain-based importance
│   ├── error_by_gate.csv             the error analysis behind the main finding
│   ├── bangladesh_predictions.csv    all 22, predicted vs actual
│   ├── bangladesh_dimension_profile.csv   BD vs world vs global top 100
│   ├── weight_sensitivity.csv        how much the ranking depends on our weights
│   ├── block_ablation.csv            which blocks of features the model needs
│   ├── error_by_country.csv          where the model is strong and weak geographically
│   └── test_metrics.csv              final held-out metrics
│       (`*_report.csv` variants are written by make_figures.py for the report;
│        the unsuffixed files are written by the notebook. Same seeds, same numbers.)
│
└── report/
    ├── report.pdf              the report — 36 pages
    ├── report.tex              its source
    ├── tables/                 22 auto-generated .tex fragments + numeric macros
    └── figures/                12 figures
```

---

## The three CSV files

| file | rows | what it is |
|---|---|---|
| `train.csv` | 981 | 80% — the model learns from these. **Contains no Bangladeshi university.** |
| `test.csv` | 244 | 20% — **held out**, used exactly once at the end |
| `university_website_scores.csv` | 1,225 | every university with its score, grade, and global / regional / country rank |

---

## How every feature is classified

Each of the 71 features is classified on **three independent axes** in
`data/feature_catalog.csv`:

| axis | question it answers | values |
|---|---|---|
| **source group** | where on the page did the extractor find it? | 12 blocks — B1 header & navigation (6), B2 rankings & recognition (7), B3 notices & updates (6), B4 events & media (9), B5 page content (15), B6 footer (5), B7 visual design (3), B8 service & interaction (4), B9 technical performance (6), B10 SEO & metadata (3), B11 accessibility (4), B12 measurement quality (3) |
| **scoring role** | which dimension consumes it, with what weight, through what transform? | D1…D7, or "not scored" |
| **measurement type** | what kind of number is it? | binary flag (57), count (6), indicator (3), ratio, percentage, index, z-score, log-count |

**51 features enter the score. 20 do not — each for a recorded reason:**

| reason | n | examples |
|---|---|---|
| prestige leakage | 2 | `a07_qs_badge`, `a09_national_rank` |
| quality not judgeable from presence | 7 | `a30_image_gallery`, `a54_banner_carousel` |
| no applicant information need | 3 | `a14_stats_block`, `a61_testimonials` |
| depth count of a scored presence | 2 | `a12_accred_count`, `a15_stat_item_count` |
| too rare to carry weight | 2 | `a59_feedback_form` (1.9%), `a75_bookmark` (1.0%) |
| redundant with a scored attribute | 1 | `broken_links_log` |
| measurement metadata | 3 | the three `*_was_missing` indicators |

All 71 are still supplied to every model as input; only the *target* omits these 20.
Report Table 3 lists all 71 with data type, value domain and justification.

---

## The scoring model

Seven dimensions, each answering part of *"what does a prospective student need from this
website?"*, combined as a weighted sum subject to two gates:

```
Score = min( 28·D1 + 22·D2 + 15·D3 + 15·D4 + 10·D5 + 7·D6 + 3·D7 ,  cap )
```

Each dimension is itself a weighted sum with inner weights summing to 1, so attribute *j* of
dimension *k* is worth exactly `w_k · λ_kj` points at its maximum. All 51 of those weights are
published in report Table 12 and in `feature_catalog.csv`; they total exactly 100.00 points.

| | dimension | weight | asks | strongest term |
|---|---|---|---|---|
| D1 | Academic information | 28 | What can I study, and who teaches it? | programmes listing (8.40 pts) |
| D2 | Admission support | 22 | How do I apply, what will it cost, who do I ask? | admissions policy (5.72 pts) |
| D3 | Currency and activity | 15 | Is this institution alive and current? | notice recency (4.80 pts) |
| D4 | Navigation and findability | 15 | Can I find what I need? | primary nav / menu breadth (3.75 each) |
| D5 | Usability and accessibility | 10 | Can I use it, on any device? | mobile score (3.20 pts) |
| D6 | Technical quality and discoverability | 7 | Does it work, and can it be found? | HTTPS (1.82 pts) |
| D7 | Institutional identity and transparency | 3 | Is it clear who they are? | vision & mission (0.60 pts) |

**Gates** — hard caps that apply regardless of everything else:

- no primary navigation → **capped at 45** (182 universities)
- no HTTPS → **capped at 60** (14 universities)

A gate applies to **186 of 1,225 universities (15.2%)**, destroying 2,652 points in total —
an average of 18.8 points each.

**Six non-linear response curves** are used inside the dimensions: menu size (an inverted U —
5–9 items is ideal, 20 is a wall of links), notice age (staleness decay), contrast (a plateau
at WCAG AAA 7:1), broken links (a step penalty), events (a count scaled by whether the events
are dated), and alt-text (concave √). All six are written out as equations in report §5.5.

---

## Results

### The model comparison

| # | model | family | R² | MAE | RMSE | Spearman |
|---|---|---|---|---|---|---|
| 1 | **LightGBM** | ensemble | **0.990** | **1.31** | **1.80** | **0.989** |
| 2 | Gradient Boosting | ensemble | 0.988 | 1.54 | 1.97 | 0.988 |
| 3 | Neural Net (MLP) | neural | 0.976 | 1.56 | 2.74 | 0.987 |
| 4 | Random Forest | ensemble | 0.974 | 2.10 | 2.86 | 0.971 |
| 5 | Extra Trees | ensemble | 0.947 | 2.99 | 4.12 | 0.946 |
| 6 | Lasso | linear | 0.945 | 2.86 | 4.18 | 0.984 |
| 7 | SVR (RBF) | kernel | 0.942 | 2.40 | 4.31 | 0.973 |
| 8 | Linear Regression | linear | 0.933 | 2.86 | 4.61 | 0.983 |
| 9 | Ridge | linear | 0.930 | 2.91 | 4.72 | 0.982 |
| 10 | Decision Tree | tree | 0.929 | 3.38 | 4.75 | 0.939 |
| 11 | k-NN (k=5) | instance | 0.768 | 5.37 | 8.60 | 0.868 |
| 12 | Mean baseline | baseline | 0.000 | 14.97 | 17.86 | — |

### The tuned model on the held-out test set

| | |
|---|---|
| R² | **0.9925** |
| MAE | **1.15** points on 0–100 |
| RMSE | 1.54 |
| Spearman ρ | **0.9918** |
| within ±2 points | 84.8% |

### Bangladesh — an entire country held out of training

All 22 Bangladeshi universities were excluded from the training set by construction. The model
was fitted on 981 universities, none of them Bangladeshi.

| | |
|---|---|
| MAE | **0.65** points (against 1.20 on the rest of the test set) |
| R² | **0.9963** |
| Spearman ρ | **0.9898** |
| largest single error | 1.40 points |
| national positions predicted exactly | 10 of 22 |

**The national ranking** (every position an out-of-sample prediction):

| # | university | world | score | grade |
|---|---|---|---|---|
| 1 | North South University | 44 | 86.82 | A+ |
| 2 | RUET | 107 | 83.75 | A |
| 3 | BRAC University | 139 | 82.35 | A |
| 4 | **KUET** | **141** | **82.18** | **A** |
| 5 | CUET | 173 | 81.40 | A |
| 6 | Bangladesh Univ. of Professionals | 186 | 81.03 | A |
| 7 | AIUB | 201 | 80.81 | A |
| 8 | Patuakhali S&T University | 273 | 78.57 | A |
| … | | | | |
| 18 | BUET | 698 | 66.80 | B |

**What the profile says.** Bangladeshi sites average 71.8 against a world mean of 63.9 and are
strong on academic content (D1 = 0.943 vs 0.785 worldwide). The gap to the global top 100 is
12.1 points, and **7.76 of it is in just two dimensions** — admission support (−4.97) and
navigation (−2.79). D4 is the only dimension where Bangladeshi sites fall *below* the world
average. Those are cheap fixes with a large scoring consequence.

**On prestige.** BUET ranks 18th of 22 on website quality. Institutional prestige and website
quality are close to unrelated in this dataset — which is exactly why the QS and Webometrics
columns were dropped in cleaning and the ranking badges excluded from the score.

### The finding worth presenting

All twelve algorithms saw identical data, yet R² ranged from 0.768 to 0.990. **The separation
is concentrated entirely on the 35 test universities to which a gating rule applies:**

| model | MAE on gated | MAE on ungated | ratio |
|---|---|---|---|
| Linear Regression | **8.88** | 1.85 | 4.80× |
| Ridge | 9.05 | 1.88 | 4.81× |
| Lasso | 8.12 | 1.98 | 4.11× |
| SVR (RBF) | 5.73 | 1.84 | 3.12× |
| Random Forest | 0.55 | 2.36 | 0.23× |
| **LightGBM** | **0.77** | 1.40 | 0.55× |

A linear model computes a weighted sum and cannot express *"whatever else is true, stop here"*.
A decision tree can, because a split is exactly that statement. Gain-based importance confirms
it: the single attribute that triggers the navigation gate accounts for **24.9% of all loss
reduction** in the tuned model — the model discovered the gate on its own.

**Ablation reaches the same conclusion from the other side.** Deleting D4's six navigation
features costs +4.89 points of test MAE; deleting D1's six academic features costs +1.83,
despite D1 carrying almost twice the declared weight.

### How much of the ranking is our weights?

Perturb all seven weights randomly, renormalise, recompute the whole ranking — 300 draws per
level:

| perturbation | Spearman ρ | worst of 300 | top 10 kept | top 50 kept |
|---|---|---|---|---|
| ±10% | 0.9995 | 0.9987 | 95% | 98% |
| ±20% | 0.9986 | 0.9964 | 92% | 95% |
| ±30% | 0.9969 | 0.9901 | 90% | 92% |
| ±50% | 0.9924 | 0.9778 | 86% | 86% |

The league table is a property of the websites far more than of our weighting judgement.

---

## Limitations

1. **The label is a function of the features.** A model can in principle reconstruct it
   exactly, so the accuracy figures measure *recoverability*, not quality prediction. The
   report states this in full (§5.10) and narrows its research question accordingly.
2. **The dataset cannot see visual design.** Every attribute is a flag, a count or a
   measurement. Clutter, layout and whether a page looks modern are not in the 71 columns —
   which is also why seven media and visual attributes are excluded from the score.
3. **About 2% of rows are extraction failures** — JavaScript-rendered pages the crawler could
   not read. Because the navigation gate fires on exactly this failure mode, an extraction
   failure and a genuinely unnavigable site are indistinguishable. Do not quote the bottom of
   the ranking without this caveat.
4. **The score reflects the landing page**, not the whole site and not the institution.
5. **The weights are a documented judgement** — but the sensitivity of the ranking to them is
   measured, not assumed (see above).
6. **The exclusion of 20 attributes is itself a judgement**, most arguably for
   `a28_contests`.
7. **Region and data collector are perfectly confounded** — each of six collectors covered one
   region. Load time is region-standardised for this reason.
8. **The crawl is a snapshot**; freshness attributes would differ on another day.

---

## Reproducing

```bash
python src/build_dataset.py      # raw file -> data/  (dataset, splits, catalogue, ARFF)
python src/make_figures.py       # -> report/figures/ + results/*_report.csv
python src/make_tables.py        # -> report/tables/  (every table in the report)
pdflatex report/report.tex       # x3, for the table of contents and cross-references
jupyter lab notebook/University_Website_Quality.ipynb
```

The notebook is already executed — every output shown was produced by the code above it.
No number in the report is typed by hand: the tables and the numeric macros under
`report/tables/` are generated by `make_tables.py` and `\input` by `report.tex`, so the
document cannot drift from the code.

All seeds are fixed (42 for the split, 0 for the models and the sensitivity sampler).

**Environment:** Python 3.13 · pandas 2.3.3 · numpy 2.4.4 · scikit-learn 1.8.0 · LightGBM 4.6.0
· matplotlib 3.10.8
