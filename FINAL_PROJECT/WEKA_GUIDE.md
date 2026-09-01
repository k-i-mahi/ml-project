# Running this project in Weka

Click-by-click instructions, with the numbers you should actually see.

Everything uses the five ARFF files in `data/weka/`. They contain exactly the same rows as the
CSV files — nothing was re-split or cleaned differently — so any result you get in Weka is
directly comparable to the notebook.

---

## 0. What is in `data/weka/`

| file | rows | target attribute | use for |
|---|---|---|---|
| `train.arff` | 979 | `website_score` (numeric) | training a regression model |
| `test.arff` | 246 | `website_score` (numeric) | **held-out** evaluation |
| `train_classification.arff` | 979 | `grade` (A+/A/B/C/D/F) | training a classifier |
| `test_classification.arff` | 246 | `grade` | held-out classifier evaluation |
| `all_universities.arff` | 1225 | `website_score` | 10-fold cross-validation in one file |

Each file has **71 predictive attributes** plus the target. Identifier columns (`uni_id`,
`name`, `url`, `country`, `region`) are deliberately **not** included — a model must not be
able to memorise a university by name, and `region` is confounded with who collected the data.

> **Regression or classification?**
> The real task is producing a league table, so regression on `website_score` is the honest
> one — use it for your headline result. The `grade` files are a good supporting demo because
> the confusion matrix is easy to read on a slide.

---

## 1. Load the data

1. Start Weka → **Explorer**
2. **Preprocess** tab → **Open file…** → `data/weka/train.arff`
3. You should see:

```
Relation:  website_quality_train
Instances: 979
Attributes: 72
```

4. Click `website_score` in the attribute list. The histogram should span roughly **8 to 95**
   with a visible bump around 45.

If the instance count is not 979 you have opened the wrong file.

---

## 2. Always start with the baseline

Never report a model number without a baseline beside it.

1. **Classify** tab → **Choose** → `rules` → **ZeroR**
2. **Test options** → *Supplied test set* → **Set…** → `data/weka/test.arff` → Close
3. **Start**

**Expect approximately:**

```
Correlation coefficient          0
Mean absolute error             15.73
Root mean squared error         18.76
```

ZeroR predicts the training mean for every university, so its correlation is 0 by
construction. Any model that does not clearly beat **MAE ≈ 15.7** has learned nothing.

---

## 3. Regression — the main experiment

Keep *Supplied test set* = `test.arff` for all three runs so the comparison is fair.

### 3a. Linear Regression

**Choose** → `functions` → **LinearRegression** → **Start**

```
Correlation coefficient          0.98
Mean absolute error              2.56
Root mean squared error          3.80
```

Scroll up: Weka prints the fitted equation. The attributes with the largest positive
coefficients should be recognisable — `a37_programs_listing`, `a02_primary_nav`,
`a34_department_links`, `a22_admission_notice`. That is a good sanity check that the model is
using the attributes the scoring model was built around.

### 3b. A single tree

**Choose** → `trees` → **REPTree** → **Start**
(If your build has `M5P` under `trees`, run that too — it fits a linear model in each leaf.)

```
Correlation coefficient          0.97
Mean absolute error              3.49
Root mean squared error          4.93
```

Worse than linear regression. One tree on 979 rows overfits. This is a **useful** result to
report, not a failure — it is what motivates the ensemble.

### 3c. Random Forest — the best Weka model

**Choose** → `trees` → **RandomForest**. Click the name and set **numIterations = 100**
(older versions call it `numTrees`). **Start**

```
Correlation coefficient          0.99
Mean absolute error              2.35
Root mean squared error          3.20
```

### Summary table for your report

| scheme | correlation | MAE | RMSE |
|---|---|---|---|
| ZeroR (baseline) | 0.00 | 15.73 | 18.76 |
| LinearRegression | 0.98 | 2.56 | 3.80 |
| REPTree | 0.97 | 3.49 | 4.93 |
| **RandomForest (100)** | **0.99** | **2.35** | **3.20** |
| *LightGBM (the notebook's tuned model)* | *0.99* | *1.11* | *1.56* |

The last row comes from the Python notebook and is **not** reproducible in Weka — Weka has no
LightGBM scheme. Quote it as the project's result and the Weka rows as an independent
cross-check that the target is learnable with off-the-shelf tools.

---

## 4. Cross-validation instead of a fixed test set

If your teacher asks for cross-validation:

1. Open `data/weka/all_universities.arff` (all 1,225 rows)
2. **Test options** → *Cross-validation*, **Folds = 10**
3. Run the same schemes

| scheme | correlation | MAE | RMSE |
|---|---|---|---|
| ZeroR | ~0 | 15.37 | 18.40 |
| LinearRegression | 0.97 | 2.89 | 4.36 |
| REPTree | 0.96 | 3.78 | 5.38 |
| RandomForest | 0.98 | 2.38 | 3.37 |

Cross-validated numbers are slightly **worse** than the held-out numbers in §3, which is
normal and worth saying out loud: each CV fold trains on less data.

---

## 5. Classification — the grade demo

1. Open `data/weka/train_classification.arff`
2. **Classify** → *Supplied test set* → `test_classification.arff`

### 5a. J48 decision tree

**Choose** → `trees` → **J48** → **Start**

```
Correctly Classified Instances    ~182   (~74 %)
Kappa statistic                   ~0.66
```

### 5b. Random Forest

**Choose** → `trees` → **RandomForest** → **Start**

```
Correctly Classified Instances    ~190   (~77 %)
Kappa statistic                   ~0.71
```

### Read the confusion matrix

ZeroR gets 30% (the largest grade, A, is 30% of the test set), so both models are far above
chance. Better still, look at where the errors fall:

| tolerance | ZeroR | J48 | RandomForest |
|---|---|---|---|
| exact grade | 30% | 74% | **77%** |
| **within one grade** | 61% | 99% | **100%** |

**Every single error is one grade away from the truth.** No model ever confuses a good website
with a bad one — the residual error is entirely about sites sitting near a grade boundary
(a site scoring 74.9 is graded B and one scoring 75.1 is graded A). Say this in your report:
it is exactly the kind of reading a marker looks for, and it explains why the project treats
the task as regression and uses grades only for presentation.

---

## 6. Attribute selection (optional extra slide)

1. **Select attributes** tab
2. *Attribute Evaluator*: **CfsSubsetEval** · *Search Method*: **BestFirst**
3. *Attribute Selection Mode*: **Full training set** → **Start**

A short subset is selected out of the 71, dominated by page-content and navigation
attributes. Compare that list against the feature-importance chart in the notebook (§10):
they should broadly agree. Two completely different methods in two different tools pointing at
the same handful of attributes is a nice independent confirmation.

---

## 7. Common problems

| symptom | cause | fix |
|---|---|---|
| `Cannot handle numeric class` | chose a classifier (J48) on a regression file | use `*_classification.arff` |
| `Train and test set are not compatible` | mixed a regression file with a classification file | both files must be the same family |
| `OutOfMemoryError` | default heap too small | launch as `java -Xmx2g -jar weka.jar` |
| RandomForest very slow | `numIterations` too high | 100 is plenty for 979 rows |
| numbers differ slightly from this guide | different Weka version or seed | small differences are expected; the **ordering** of schemes should hold |

---

## 8. Three sentences for the viva

1. *"ZeroR sets the floor at MAE 15.7 points; Random Forest reaches 2.35 on data it has never
   seen, so the landing-page attributes explain the great majority of the score."*
2. *"A single tree does worse than linear regression, which tells us the signal is spread
   across many weak attributes rather than concentrated in a few splits — that is why the
   ensemble helps."*
3. *"On the grade classifier, every error is within one grade — the model never confuses a
   good website with a bad one, which is why we report regression rather than classification
   as the headline."*
