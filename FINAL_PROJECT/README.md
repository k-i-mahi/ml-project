# University Website Quality — Scoring and Ranking

**Machine Learning Laboratory — CSE 4112**
Department of Computer Science and Engineering, Khulna University of Engineering & Technology

---

## In one paragraph

We collected 1,230 university landing pages worldwide, each described by 69 automatically
measurable attributes, and cleaned them into a dataset of **1,225 universities × 71 features**.
We defined a **seven-dimension scoring model** that converts those attributes into a website
quality score from **0 to 100** — higher is better — then split the data **80/20** and trained
**twelve regression algorithms** to predict that score. The best model, a tuned LightGBM,
reaches **R² = 0.993** with an average error of **1.11 points** on 246 universities it never
saw during training.

---

## Start here

| I want to… | Open |
|---|---|
| understand the whole project | `report/report.pdf` — 14 pages |
| see the code and run the demos | `notebook/University_Website_Quality.ipynb` |
| look up one university | notebook §12 — `lookup("kuet")` |
| score a university from its attributes | notebook §13 — `score_university({...})` |
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
│   ├── train.csv                         979 rows (80%)
│   ├── test.csv                          246 rows (20%)
│   ├── data_dictionary.csv             every column, its dimension, its range
│   ├── dataset_summary.json            headline numbers, machine-readable
│   ├── missing_value_policy.json       how missing values were filled and why
│   └── weka/                           the same splits as ARFF (5 files)
│
├── notebook/
│   └── University_Website_Quality.ipynb    the full analysis, already executed
│
├── src/
│   ├── build_dataset.py        raw file -> scored dataset -> 80/20 split -> ARFF
│   ├── notebook_src.py         the notebook as a plain .py script
│   ├── make_figures.py         regenerates every figure in the report
│   └── nbbuild.py              builds and executes the notebook from notebook_src.py
│
├── model/
│   ├── final_model.joblib      the tuned LightGBM model
│   └── model_card.json         hyper-parameters, features, metrics
│
├── results/
│   ├── model_comparison.csv    the twelve-algorithm leaderboard
│   ├── cross_validation.csv    5-fold CV results
│   ├── feature_importance.csv  gain-based importance
│   ├── error_by_gate.csv       the error analysis behind the main finding
│   └── test_metrics.csv        final held-out metrics
│
└── report/
    ├── report.pdf              the report
    ├── report.tex              its source
    └── figures/                7 figures
```

---

## The three CSV files

| file | rows | what it is |
|---|---|---|
| `train.csv` | 979 | 80% — the model learns from these |
| `test.csv` | 246 | 20% — **held out**, used exactly once at the end |
| `university_website_scores.csv` | 1,225 | every university with its score, grade, and global / regional / country rank |

**Columns:** `uni_id`, `name`, `url`, `country`, `region`, `rank`, `regional_rank`,
`country_rank`, then the **71 features**, then **`website_score`** (the target, 0–100) and
**`grade`** (A+/A/B/C/D/F).

---

## The scoring model

Seven dimensions, each answering part of *"what does a prospective student need from this
website?"*, combined as a weighted sum subject to two gates:

```
Score = min( 28·D1 + 22·D2 + 15·D3 + 15·D4 + 10·D5 + 7·D6 + 3·D7 ,  cap )
```

| | dimension | weight | asks |
|---|---|---|---|
| D1 | Academic information | 28 | What can I study, and who teaches it? |
| D2 | Admission support | 22 | How do I apply, and what will it cost? |
| D3 | Currency and activity | 15 | Is this institution alive and current? |
| D4 | Navigation and findability | 15 | Can I find what I need? |
| D5 | Usability and accessibility | 10 | Can I use it, on any device? |
| D6 | Technical quality | 7 | Does the site actually work? |
| D7 | Institutional transparency | 3 | Is it clear who they are? |

**Gates** — hard caps that apply regardless of everything else:

- no primary navigation → **capped at 45**
- no HTTPS → **capped at 60**

A gate applies to **186 of 1,225 universities (15.2%)**.

**Four non-linear response curves** are used inside the dimensions, because these attributes
do not reward proportionally: menu size (an inverted U — 5–9 items is ideal, 20 is a wall of
links), notice age (staleness decay), contrast (a plateau at WCAG AAA 7:1), and broken links
(a step penalty).

---

## Results

### The model comparison

| # | model | family | R² | MAE | RMSE | Spearman |
|---|---|---|---|---|---|---|
| 1 | **LightGBM** | ensemble | **0.992** | **1.26** | **1.71** | **0.992** |
| 2 | Gradient Boosting | ensemble | 0.989 | 1.51 | 1.94 | 0.990 |
| 3 | Neural Net (MLP) | neural | 0.985 | 1.72 | 2.27 | 0.984 |
| 4 | Random Forest | ensemble | 0.971 | 2.35 | 3.22 | 0.969 |
| 5 | SVR (RBF) | kernel | 0.961 | 2.48 | 3.72 | 0.979 |
| 6 | Linear Regression | linear | 0.959 | 2.56 | 3.80 | 0.984 |
| 7 | Ridge | linear | 0.959 | 2.60 | 3.80 | 0.984 |
| 8 | Lasso | linear | 0.956 | 2.82 | 3.93 | 0.983 |
| 9 | Extra Trees | ensemble | 0.945 | 3.13 | 4.41 | 0.939 |
| 10 | Decision Tree | tree | 0.915 | 3.78 | 5.47 | 0.926 |
| 11 | k-NN (k=5) | instance | 0.786 | 5.97 | 8.67 | 0.875 |
| 12 | Mean baseline | baseline | 0.000 | 15.73 | 18.76 | — |

### The tuned model on the held-out test set

| | |
|---|---|
| R² | **0.9931** |
| MAE | **1.11** points on 0–100 |
| RMSE | 1.56 |
| Spearman ρ | **0.9931** |
| within ±2 points | 83.3% |

### The finding worth presenting

All twelve algorithms saw identical data, yet R² ranged from 0.786 to 0.992. **The separation
is concentrated entirely on the 36 test universities to which a gating rule applies:**

| model | MAE on gated | MAE on ungated | ratio |
|---|---|---|---|
| Linear Regression | **6.41** | 1.91 | 3.37× |
| Ridge | 6.37 | 1.96 | 3.25× |
| SVR (RBF) | 4.49 | 2.14 | 2.10× |
| Random Forest | 1.05 | 2.57 | 0.41× |
| **LightGBM** | **1.07** | 1.29 | 0.83× |

A linear model computes a weighted sum and cannot express *"whatever else is true, stop here"*.
A decision tree can, because a split is exactly that statement. Gain-based importance confirms
it: the single attribute that triggers the navigation gate accounts for **26.3% of all loss
reduction** in the tuned model — the model discovered the gate on its own.

---

## Rankings

**Global top 5:** University of New England (AU) 95.4 · University of Porto (PT) 95.0 ·
Eotvos Lorand University (HU) 95.0 · University of Strasbourg (FR) 94.3 ·
Southern Cross University (AU) 93.3

**Bangladesh top 5:** North South University (global 45, 87.4) · RUET (79, 85.5) ·
CUET (114, 83.8) · **KUET (115, 83.8)** · AIUB (136, 83.1)

Country rankings are produced for the **18 countries** with at least 20 universities.

---

## Limitations

1. **The dataset cannot see visual design.** Every attribute is a flag, a count or a
   measurement. Clutter, layout and whether a page looks modern are not in the 71 columns, so
   they cannot be in the score. This is the biggest gap between this ranking and a human
   impression.
2. **About 2% of rows are extraction failures** — JavaScript-rendered pages the crawler could
   not read. They land at the bottom. The top of the ranking is unaffected; do not quote the
   bottom without this caveat.
3. **The score reflects the landing page**, not the whole site and not the institution.
4. **The weights are a documented judgement, not a measured truth.** A different reasonable
   analyst would pick somewhat different weights and get a somewhat different ranking.
5. **Region and data collector are perfectly confounded** — each of six collectors covered one
   region. Load time is region-standardised for this reason.
6. **The crawl is a snapshot**; freshness attributes would differ on another day.

---

## Reproducing

```bash
python src/build_dataset.py      # raw file -> data/  (dataset, splits, ARFF)
python src/make_figures.py       # -> report/figures/
jupyter lab notebook/University_Website_Quality.ipynb
```

The notebook is already executed — every output shown was produced by the code above it.
All seeds are fixed (42 for the split, 0 for the models).

**Environment:** Python 3.13 · pandas 2.3.3 · numpy 2.4.4 · scikit-learn 1.8.0 · LightGBM 4.6.0
