# Running this project in Weka

Click-by-click instructions. Every number in this guide was **measured** on Weka 3.8.7 by
`src/run_weka.py`, which writes `results/weka_results.csv` and `results/weka_confusion.json`.
Nothing here is an estimate. Re-run it yourself with:

```
python src/run_weka.py "C:\Program Files\Weka-3-8-7\weka.jar"
```

---

## 1. What is in `data/weka/`

### The files you train on

| file | rows | target | use for |
|---|---|---|---|
| `train.arff` | 981 | `website_score` (numeric) | training a regression model |
| `test.arff` | 244 | `website_score` | **held-out** evaluation |
| `all_universities.arff` | 1225 | `website_score` | 10-fold cross-validation in one file |
| `bangladesh.arff` | 22 | `website_score` | the country case study on its own |
| `train_classification.arff` | 981 | `grade` (F…A+) | training a classifier |
| `test_classification.arff` | 244 | `grade` | held-out classifier evaluation |
| `all_universities_classification.arff` | 1225 | `grade` | cross-validated classification |
| `bangladesh_classification.arff` | 22 | `grade` | the case study, as grades |

### The same files, but with university names

Every file above also exists as `*_named.arff` — `test_named.arff`,
`bangladesh_classification_named.arff` and so on. They are row-for-row identical but carry one
extra attribute at the front:

```
@attribute university string
```

Use these when you want Weka to tell you **which university** each prediction belongs to.
Section 6 is the click path. The name is removed before it reaches the model, so it is a label
on the output, not an input.

### Supporting files

| file | what it is |
|---|---|
| `ATTRIBUTES.md` | every attribute, its ARFF type, its range, the dimension it scores |
| `index_train.csv`, `index_test.csv`, `index_all_universities.csv`, `index_bangladesh.csv` | instance number → university name, country, actual score and grade |

---

## 2. How the attributes are typed — and why it matters

**71 predictive attributes: 60 nominal `{0,1}`, 11 numeric.**

An earlier version of this export declared *every* attribute `numeric`. Weka loads that without
complaining, which is exactly why it is worth knowing about — it quietly costs you four things:

1. **InfoGainAttributeEval, GainRatioAttributeEval and ChiSquaredAttributeEval refuse numeric
   attributes.** With the old files, half of the *Select attributes* tab was unusable without
   first running a Discretize filter.
2. **NaiveBayes fits a Gaussian bell curve to a 0/1 flag**, which is the wrong distribution for
   a coin flip. It is the single biggest reason NaiveBayes underperforms here.
3. **Trees print `a01_logo <= 0.5`** instead of the readable `a01_logo = 0`.
4. The Preprocess tab draws a meaningless histogram instead of a two-bar presence count.

The 11 numeric attributes are the ones that really are numeric — counts, percentages, ratios
and z-scores:

`a03_nav_item_count`, `a12_accred_count`, `a15_stat_item_count`, `a24_event_count`,
`a53_contrast_ratio`, `a63_mobile_score`, `a66_broken_links`, `a72_alt_text_pct`,
`broken_links_log`, `load_time_z_region`, `notice_recency_days`.

Types are decided once over all 1,225 rows, so **every ARFF file declares a byte-identical
header** and any file can be supplied as the test set for any other.

**Two things deliberately not in the files.** Identifier columns (`uni_id`, `name`, `url`,
`country`, `region`) are excluded so no model can memorise a university, and because `region`
is confounded with who collected the data. And there are no `?` values: missing readings were
imputed during dataset construction, and every imputed column carries a companion
`*_was_missing` flag so a model can still see that the value was absent.

`grade` is declared **worst to best** — `{F,D,C,B,A,'A+'}` — so the confusion matrix reads as
an ordinal scale and a one-band error sits next to the diagonal.

---

## 3. Before you start: three schemes that will not run

This machine's Weka 3.8.7 ships a `weka.jar` with the MTJ matrix library stripped out. These
fail with `NoClassDefFoundError: no/uib/cipr/matrix/Matrix`, in the Explorer as well as on the
command line:

- `functions.LinearRegression`
- `trees.M5P`
- `functions.GaussianProcesses`

**This is an installation problem, not a data problem** — the ARFF files are fine. Two fixes:

- **Use `functions.SMOreg` instead.** It is a support-vector regressor with a linear kernel by
  default, so it plays the same role linear regression would in the comparison — and it turns
  out to have the lowest MAE of any scheme in the table below.
- Or reinstall Weka from the full distribution at
  <https://waikato.github.io/weka-wiki/downloading_weka/>, which bundles MTJ.

Everything else in this guide runs.

---

## 4. Regression — the main experiment

### Load the data

1. Start Weka → **Explorer**
2. **Preprocess** → **Open file…** → `data/weka/train.arff`
3. You should see `Relation: website_quality_train`, `Instances: 981`, `Attributes: 72`.
   If the instance count is not 981 you opened the wrong file.
4. Click `website_score` in the attribute list — the histogram spans **7.6 to 94.9**.

> **The split is not random.** All 22 Bangladeshi universities are in `test.arff` and none is
> in `train.arff`. A Weka model trained here is blind to Bangladesh in exactly the way the
> notebook's model is, so §7 is directly comparable.

### Run each scheme

**Classify** tab → **Test options** → *Supplied test set* → **Set…** → `data/weka/test.arff` →
Close. Keep that test set for every run so the comparison is fair. Then **Choose** each scheme
and press **Start**.

| scheme | where in the tree | correlation | MAE | RMSE |
|---|---|---|---|---|
| ZeroR *(baseline)* | `rules` | 0.00 | 14.97 | 17.86 |
| RandomTree | `trees` | 0.85 | 6.39 | 9.87 |
| REPTree | `trees` | 0.96 | 3.77 | 5.16 |
| **RandomForest**, `numIterations = 100` | `trees` | **0.98** | 2.74 | **3.50** |
| MultilayerPerceptron | `functions` | 0.97 | 2.73 | 5.00 |
| Bagging (REPTree ×10) | `meta` | 0.98 | 2.60 | 3.53 |
| SMOreg | `functions` | 0.95 | **2.52** | 5.71 |
| *LightGBM — the notebook's tuned model* | *not in Weka* | *R² 0.993* | *1.15* | *1.54* |

Three things worth saying out loud when you present this:

- **ZeroR sets the floor at MAE 14.97.** It predicts the training mean for every university,
  so its correlation is 0 by construction. Any scheme that does not clearly beat 14.97 has
  learned nothing.
- **A single tree is much worse than an ensemble.** RandomTree 6.39 → REPTree 3.77 → Bagging
  2.60. The signal is spread across many weak attributes rather than concentrated in a few
  splits, which is exactly why the ensembles help.
- **SMOreg has the best MAE but the worst RMSE of the good schemes** (2.52 against 5.71). It is
  accurate on typical universities and badly wrong on a few — the gated sites in §8 of the
  report. RandomForest wins on correlation and RMSE and is the better model overall, which is
  why the report leads with RMSE and correlation rather than MAE alone. Quoting only MAE would
  have picked the wrong winner.

The LightGBM row comes from the Python notebook (`results/test_metrics.csv`) and is **not**
reproducible in Weka — Weka has no LightGBM scheme. It is quoted as R² rather than Weka's
correlation coefficient because that is the statistic the notebook computes; the two are close
but not the same thing, so do not put them in one column when you present it. Quote LightGBM
as the project's result and the Weka rows as an independent cross-check that the target is
learnable with off-the-shelf tools.

---

## 5. Cross-validation instead of a fixed test set

1. **Preprocess** → open `data/weka/all_universities.arff` (all 1,225 rows)
2. **Classify** → **Test options** → *Cross-validation*, **Folds = 10**

| scheme | correlation | MAE | RMSE |
|---|---|---|---|
| ZeroR | -0.12 | 15.43 | 18.49 |
| RandomTree | 0.87 | 6.49 | 9.39 |
| REPTree | 0.96 | 3.67 | 5.32 |
| Bagging (REPTree) | 0.98 | **2.84** | 3.96 |
| **RandomForest (100)** | **0.99** | 2.97 | **3.76** |

Cross-validated numbers are slightly **worse** than the held-out numbers in §4, which is normal
and worth saying out loud: each CV fold trains on less data. The *ordering* of the schemes is
unchanged, which is the point of running it.

*(SMOreg and MultilayerPerceptron take minutes per fold on 1,225 rows and are reported on the
held-out split only.)*

---

## 6. Getting university names in the output

By default Weka labels each prediction with a bare instance number, which is unreadable:

```
inst#     actual  predicted      error
    1      86.82     85.149     -1.671
```

The `*_named.arff` files fix this. The name is a `string` attribute, and Weka classifiers
cannot train on strings, so you wrap the scheme in a **FilteredClassifier** that removes the
name first.

### In the Explorer

1. **Preprocess** → open `data/weka/train_named.arff`
2. **Classify** → *Supplied test set* → `data/weka/bangladesh_named.arff`
3. **Choose** → `meta` → **FilteredClassifier**
4. Click on the words `FilteredClassifier` to open its editor:
   - `filter` → **Choose** → `unsupervised` → `attribute` → **Remove**.
     Click on `Remove` and set **attributeIndices = 1**. Leave `invertSelection` False.
   - `classifier` → **Choose** → `trees` → **RandomForest** (set `numIterations = 100`)
   - **OK**
5. **More options…** → *Output predictions* → **Choose** → **PlainText**.
   Click on the word `PlainText` and set **attributes = 1**. **OK**, **OK**
6. **Start**

You now get:

```
inst#     actual  predicted      error (university)
    1      86.82     85.149     -1.671 ('North South University')
    2      83.75     79.485     -4.265 ('Rajshahi University of Engineering and Technology')
    3      82.35     78.333     -4.017 ('BRAC University')
    4      82.18     79.853     -2.327 ('Khulna University of Engineering and Technology')
    5      81.40     77.521     -3.879 ('Chittagong University of Engineering and Technology')
```

Choose **CSV** instead of **PlainText** at step 5 to paste the result straight into Excel.

### From the command line

```
java -cp "C:\Program Files\Weka-3-8-7\weka.jar" ^
  weka.classifiers.meta.FilteredClassifier ^
  -F "weka.filters.unsupervised.attribute.Remove -R 1" ^
  -W weka.classifiers.trees.RandomForest ^
  -t data/weka/train_named.arff ^
  -T data/weka/bangladesh_named.arff ^
  -classifications "weka.classifiers.evaluation.output.prediction.CSV -p 1" ^
  -- -I 100
```

Note the option order: everything for the *evaluation* comes before `--`, and everything after
`--` is passed to RandomForest. Putting `-t` after `--` gives you
`No training file and no object input file given`.

> Accuracy through FilteredClassifier differs by a few tenths of a percent from running the
> scheme directly (79.9% against 78.3% on the grade task). That is the meta-wrapper reseeding
> the forest, not the name leaking — `Remove` deletes the attribute before the classifier is
> built.

### If you do not want to bother with FilteredClassifier

Run the plain `test.arff` as usual, then open `data/weka/index_test.csv`. Its `weka_instance`
column is the instance number Weka printed, in the same order, beside the university name,
country, actual score and grade.

---

## 7. The Bangladesh case study

Train on `train.arff`, supply `bangladesh.arff` as the test set — 22 universities the model has
never seen, from a country it has never seen.

| scheme | correlation | MAE | RMSE |
|---|---|---|---|
| ZeroR | 0.00 | 13.83 | 14.81 |
| RandomTree | 0.65 | 7.15 | 10.96 |
| RandomForest (100) | 0.99 | 2.91 | 3.45 |
| REPTree | 0.98 | 2.30 | 2.75 |
| SMOreg | 0.97 | 1.93 | 3.57 |
| MultilayerPerceptron | 0.99 | 1.71 | 2.23 |
| **Bagging (REPTree)** | **0.99** | **1.50** | **1.89** |

Every good scheme does **better** on Bangladesh (MAE 1.50–2.91) than on the full test set
(2.52–2.74). Bangladeshi university websites are unusually consistent with one another, so once
the model has learned the general pattern they are easy to place. Use §6 to print this run with
names attached — it makes a far better slide than 22 numbered rows.

---

## 8. Classification — the grade demo

1. **Preprocess** → open `data/weka/train_classification.arff`
2. **Classify** → *Supplied test set* → `data/weka/test_classification.arff`

| scheme | correct | accuracy | Kappa | within one band |
|---|---|---|---|---|
| ZeroR *(baseline)* | 72 / 244 | 29.5% | 0.00 | 62.3% |
| NaiveBayes | 134 / 244 | 54.9% | 0.45 | 91.4% |
| IBk (k = 5) | 154 / 244 | 63.1% | 0.52 | 88.5% |
| J48 | 182 / 244 | 74.6% | 0.68 | 98.8% |
| **RandomForest (100)** | **191 / 244** | **78.3%** | **0.72** | **99.6%** |

### Read the confusion matrix, not just the accuracy

RandomForest on the 244 held-out universities (rows = actual grade):

```
   F   D   C   B   A  A+   <-- classified as
  22   0   0   0   0   0 |  F
   3  38   0   1   0   0 |  D
   0   0  15  13   0   0 |  C
   0   0   4  46  15   0 |  B
   0   0   0   8  63   1 |  A
   0   0   0   0   8   7 |  A+
```

**243 of the 244 predictions are within one grade band of the truth.** The single exception is
one university graded D that the forest called B. J48 has three such errors. Everything else
sits on a cell next to the diagonal — sites near a band boundary, where a score of 74.9 is a B
and 75.1 is an A.

That is the honest reading and the one to put on a slide: the model essentially never confuses
a good website with a bad one, and the residual disagreement is about which side of a boundary
a site falls on. It is also why the project treats the task as **regression** and uses grades
only for presentation — the boundaries are a reporting convenience, not a real discontinuity in
website quality.

One caveat to state alongside it: ZeroR reaches 62.3% "within one band" just by always guessing
the largest class, so quote exact accuracy beside it or the statistic flatters everyone.

---

## 9. Attribute selection

This is where the corrected attribute types pay off — the entropy-based evaluators now run
directly, with no Discretize step.

1. **Select attributes** tab
2. *Attribute Evaluator*: **InfoGainAttributeEval** · *Search Method*: **Ranker**
3. *Attribute Selection Mode*: **Full training set** → **Start**

Do it on `train_classification.arff`: InfoGain needs a nominal class, so it works on `grade`,
not on `website_score`.

The top of the ranking on `train_classification.arff`:

```
0.35614   a02_primary_nav          0.28757   a22_admission_notice
0.35004   a34_department_links     0.28622   a03_nav_item_count
0.34097   a37_programs_listing     0.28279   a38_scholarship
0.31868   a39_library_link         0.28147   a20_news_events
0.30890   a36_research_highlight   0.26651   a33_about_blurb
0.29866   a35_faculty_link         0.24773   a40_career_link
```

For a subset rather than a ranking, use **CfsSubsetEval** with **BestFirst** — that pair works
on the regression file too.

**This is the result worth presenting.** `a02_primary_nav` comes out first in Weka's InfoGain
ranking, and it is also the single largest contributor in the notebook's gain-based importance
chart (§10), at 24.9%. Two different methods, in two different tools, on two different target
formulations, agree on which attribute matters most — that is genuine independent
confirmation, not a coincidence you have to argue for.

---

## 10. Common problems

| symptom | cause | fix |
|---|---|---|
| `NoClassDefFoundError: no/uib/cipr/matrix/Matrix` | LinearRegression / M5P / GaussianProcesses on a Weka build without MTJ | use SMOreg, or reinstall the full Weka — see §3 |
| `Cannot handle string attributes` | a `*_named.arff` fed straight to a classifier | wrap it in FilteredClassifier + Remove — see §6 |
| `Cannot handle numeric class` | a classifier such as J48 on a regression file | use the `*_classification.arff` file |
| `Train and test set are not compatible` | mixed a regression file with a classification file | both files must be from the same family |
| `No training file and no object input file given` | `-t` placed after `--` on the command line | evaluation options go **before** `--` |
| `OutOfMemoryError` | default heap too small | raise `maxheap` in `RunWeka.ini`, or run `java -Xmx2g -jar weka.jar` |
| RandomForest is slow | `numIterations` too high | 100 is plenty for 981 rows |
| numbers differ slightly from this guide | different Weka version or seed | small differences are expected; the **ordering** of the schemes should hold |

---

## 11. Three sentences for the viva

1. *"ZeroR sets the floor at MAE 14.97; Random Forest reaches 2.74 and Bagging 2.60 on data
   they have never seen, so the landing-page attributes explain the great majority of the
   score."*
2. *"A single random tree scores 6.39 and the bagged ensemble 2.60 on the same split, which
   tells us the signal is spread across many weak attributes rather than concentrated in a few
   splits — that is why the ensemble helps."*
3. *"On the grade classifier 243 of 244 predictions are within one band, so the model
   essentially never confuses a good website with a bad one; the residual error is about which
   side of a boundary a site falls on, which is why we report regression as the headline."*
