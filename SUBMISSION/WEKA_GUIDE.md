# Running this project in Weka

**Weka Explorer, step by step, with the numbers you should actually see.**

Everything here uses the five ARFF files in `data/weka/`. They are the *same* rows as the
CSVs — nothing was re-split, re-sampled, or cleaned differently, so any result you get in
Weka is directly comparable to the notebook.

---

## 0. What is in `data/weka/`

| file | rows | target attribute | use it for |
|---|---|---|---|
| `train.arff` | 160 | `quality_score` (numeric) | training a regression model |
| `test.arff` | 40 | `quality_score` (numeric) | **held-out** evaluation of that model |
| `train_classification.arff` | 160 | `grade` (A+/A/B/C/D/F) | training a classifier |
| `test_classification.arff` | 40 | `grade` | **held-out** evaluation of that classifier |
| `labeled_200_regression.arff` | 200 | `quality_score` | all labelled data, for 10-fold CV in one file |

Each file has **78 predictive attributes** plus the target. Identifier columns (`uni_id`,
`name`, `country`, `region`) are deliberately **not** in the ARFF files — a model must not be
able to memorise a university by name, and `region` is confounded with who collected the data.

> **Regression or classification?**
> The project's real task is *ranking*, so regression on `quality_score` is the honest one.
> The `grade` files exist because a classification demo is easier to read in the Explorer and
> because the confusion matrix makes a nice talking point. Use regression for your headline
> result and classification as the supporting demo.

---

## 1. Load the data

1. Start Weka → **Explorer**.
2. **Preprocess** tab → **Open file…** → `data/weka/train.arff`.
3. You should see:

```
Relation:  university_website_quality_train
Instances: 160
Attributes: 79
```

4. Click the target attribute `quality_score` in the attribute list. The histogram at the
   bottom right should be roughly bell-shaped and span about **0 to 92**.

If the instance count is not 160, you opened the wrong file.

---

## 2. The baseline you must beat (do this first)

Never report a model score without a baseline next to it.

1. **Classify** tab → **Choose** → `rules` → **ZeroR**.
2. **Test options** → *Supplied test set* → **Set…** → `data/weka/test.arff` → Close.
3. **Start**.

**Expect roughly:**

```
Correlation coefficient          0
Mean absolute error             ~22.8
Root mean squared error         ~26.8
```

ZeroR predicts the training mean for everybody. Correlation is **0** by construction. Any
model that does not clearly beat MAE ≈ 22.8 has learned nothing.

---

## 3. Regression — the main experiment

Keep *Supplied test set* = `test.arff` for all three runs below, so the comparison is fair.

### 3a. Linear regression

**Choose** → `functions` → **LinearRegression** → **Start**.

**Expect roughly:**

```
Correlation coefficient          ~0.82
Mean absolute error              ~12.1
Root mean squared error          ~15.4
```

Scroll up in the output: Weka prints the fitted equation. The attributes it keeps with the
largest positive coefficients should be recognisable — things like `a46_admissions_policy`,
`content_completeness_B5`, `a34_department_links`. That is a good sanity check that the label
means what we claim it means.

### 3b. A single tree

**Choose** → `trees` → **REPTree** → **Start**.
(If your Weka build has `M5P` under `trees`, run that too — it fits a linear model in each
leaf and usually does slightly better than REPTree here.)

**Expect roughly:**

```
Correlation coefficient          ~0.71
Mean absolute error              ~15.7
Root mean squared error          ~19.6
```

Worse than linear regression. One tree on 160 rows overfits. This is a *useful* result to
report, not a failure — it motivates the ensemble.

### 3c. Random forest — the best Weka model

**Choose** → `trees` → **RandomForest**.
Click the name to open options and set **numIterations = 100** (older versions call this
`numTrees`). **Start**.

**Expect roughly:**

```
Correlation coefficient          ~0.88
Mean absolute error              ~11.3
Root mean squared error          ~14.2
```

### Summary table to put in your report

| scheme | correlation | MAE | RMSE |
|---|---|---|---|
| ZeroR (baseline) | 0.00 | 22.8 | 26.8 |
| LinearRegression | 0.82 | 12.1 | 15.4 |
| REPTree | 0.71 | 15.7 | 19.6 |
| **RandomForest (100)** | **0.88** | **11.3** | **14.2** |
| *LightGBM (the notebook's model)* | *0.92* | *9.0* | *11.4* |

The last row is from the Python notebook and is **not** reproducible in Weka — Weka has no
gradient-boosting scheme with monotone constraints. Quote it as the project's result and the
Weka rows as an independent cross-check that the label is learnable by ordinary methods too.

---

## 4. Cross-validation instead of a fixed test set

If your teacher asks for cross-validation rather than a held-out file:

1. Open `data/weka/labeled_200_regression.arff` (all 200 labelled rows).
2. **Test options** → *Cross-validation*, **Folds = 10**.
3. Run the same three schemes.

**Expect roughly** (numbers move a little with the fold seed):

| scheme | correlation | MAE | RMSE |
|---|---|---|---|
| ZeroR | ~0 | 21.5 | 25.0 |
| LinearRegression | 0.77 | 13.7 | 17.1 |
| REPTree | 0.73 | 13.6 | 17.4 |
| RandomForest | 0.82 | 11.5 | 14.2 |

Cross-validated numbers are slightly **worse** than the held-out numbers in §3. That is normal
and worth saying out loud: each CV fold trains on less data, and the 40-row test set is small
enough to be a bit lucky.

---

## 5. Classification — the grade demo

1. Open `data/weka/train_classification.arff`.
2. **Classify** → *Supplied test set* → `test_classification.arff`.

### 5a. J48 decision tree

**Choose** → `trees` → **J48** → **Start**.

**Expect roughly:**

```
Correctly Classified Instances     ~13   (~32 %)
Kappa statistic                    ~0.18
```

### 5b. Random forest

**Choose** → `trees` → **RandomForest** → **Start**.

**Expect roughly:**

```
Correctly Classified Instances     ~17   (~42 %)
Kappa statistic                    ~0.30
```

### Do not panic at 42%

Six classes, 160 training rows, and grade boundaries cut through a *continuous* score. A site
scoring 64.9 is graded A and one scoring 65.1 is graded A+ — no model can see a difference
that isn't there.

Read the **confusion matrix** Weka prints. Almost every error is one grade away from the
truth, not across the table:

| tolerance | J48 | RandomForest |
|---|---|---|
| exact grade | 32% | 42% |
| **within one grade** | **82%** | **82%** |

ZeroR gets 20%. So both models are well above chance, and the residual error is boundary
noise rather than confusion between good and bad websites. **Say this in your report** — it
is exactly the kind of reading a marker is looking for, and it is the honest reason the
project treats the task as regression rather than classification.

---

## 6. Attribute selection (optional, one extra slide)

1. **Select attributes** tab.
2. *Attribute Evaluator*: **CfsSubsetEval**. *Search Method*: **BestFirst**.
3. *Attribute Selection Mode*: **Full training set** → **Start**.

You should see a short subset selected out of the 78 — dominated by page-content and
notice/freshness attributes. Compare that list against the SHAP importance chart in the
notebook (§8). They should broadly agree, which is a nice independent confirmation: two
completely different methods, in two different tools, point at the same handful of
attributes.

---

## 7. Common problems

| symptom | cause | fix |
|---|---|---|
| `Cannot handle numeric class` | you chose a classifier (J48) on a regression file | use `*_classification.arff`, or pick a numeric scheme |
| `Train and test set are not compatible` | mixed a regression file with a classification file | both files must be the same family |
| `OutOfMemoryError` | default heap is small | launch as `java -Xmx2g -jar weka.jar` |
| RandomForest is very slow | `numIterations` too high | 100 is plenty for 160 rows |
| results differ slightly from this guide | different Weka version / random seed | small differences are expected; the *ordering* of the schemes should hold |

---

## 8. What to say about the Weka results

Three sentences that will serve you well in a viva:

1. *"ZeroR sets the floor at MAE 22.8; RandomForest reaches 11.3 on data it has never seen,
   so roughly half the error is explained by the website attributes."*
2. *"A single tree does worse than linear regression, which tells us the signal is spread
   across many weak attributes rather than concentrated in a few splits — that is why an
   ensemble helps."*
3. *"Weka independently reproduces the ranking with correlation 0.88 using an off-the-shelf
   random forest, which shows the label is learnable and not an artefact of one particular
   model."*
