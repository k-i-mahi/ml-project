# Presentation and viva preparation

Questions you are likely to get, and answers you can actually defend.
Every number here comes from the notebook or the report — nothing is invented for this file.

---

## ⭐ THE QUESTION TO PREPARE FOR

### "If you built the score from the features, isn't the model just re-learning your own formula?"

**This is the sharpest question available, and you should welcome it.** Have this ready:

> "Yes — and that is exactly what we set out to test. We defined the scoring model, so a
> perfect learner *should* be able to recover it. The research question is not *can* it be
> recovered, but **which algorithms can and which cannot, and why**.
>
> That turned out to be a real question with a real answer. R² ranged from 0.79 to 0.99 across
> twelve algorithms on identical data. The gap is not noise — it is concentrated entirely on
> the 36 test universities where a gating rule caps the score. Linear regression is off by
> 6.41 points on those and only 1.91 elsewhere; LightGBM is at 1.07 on both. A weighted sum
> mathematically cannot represent a hard cap; a tree split can.
>
> So the experiment measures something genuine: which model families can recover a scoring
> function containing thresholds and saturation from raw attributes alone."

**Then add the honest caveat, unprompted — it makes you look stronger, not weaker:**

> "What this does *not* prove is that our scoring model matches what a real student would
> think. Validating that would need a survey, and we say so in the limitations."

**If they push harder — "so the high R² is meaningless?"**

> "The R² on its own is not the interesting number, no. The interesting numbers are the
> *differences* between algorithms, and the error analysis that explains them. That is why
> section 7 of the report is the error breakdown rather than the leaderboard."

---

## A. The demo they will ask for

### 1. "What is the website score of KUET?"

Notebook **§12**:

```python
lookup("kuet")
```

```
  Khulna University of Engineering and Technology
  https://www.kuet.ac.bd
  WEBSITE SCORE       83.8 / 100      GRADE  A
  Global rank          115 of 1225
  Country rank      #4 in Bangladesh
  Regional rank     #40 in South & Central Asia + Oceania

  Programmes listed   yes      Departments page  yes
  Admissions policy   yes      Scholarships      yes
  Contact page        yes      Search box        NO
  Menu items          6         Last notice       30 days ago
  Mobile score        75        Contrast          13.3:1
  HTTPS               yes      Broken links      0
```

`lookup()` accepts a partial name, a domain (`"kuet"`), a country, or a rank number.

**If asked why KUET is 4th and not 1st in Bangladesh:** it has no search box, its notice board
was 30 days old at crawl time, and its mobile score is 75 against 90 for the sites above it.
Those are the specific attributes — you can point at them.

### 2. "Here is a university's data — where does it stand?"

Notebook **§13**. Give it whatever attributes you know:

```python
score_university({
    "a37_programs_listing": 1, "a34_department_links": 1, "a46_admissions_policy": 1,
    "a02_primary_nav": 1, "a03_nav_item_count": 7, "a04_search_bar": 1,
    "notice_recency_days": 3, "a63_mobile_score": 90, "a65_https": 1,
}, name="Example University")
```

Returns the predicted score, the grade, where it would rank among the 1,225, and which
attributes moved the score most. Anything you omit is filled with the training median and
reported as assumed — nothing is silently invented.

### 3. "Compare two universities"

```python
compare("kuet", "ruet")
```

Prints the attribute-by-attribute differences that account for the score gap.

---

## B. About the data

**"How many rows, and where did the data come from?"**
1,230 university landing pages crawled worldwide, 69 attributes each, automatically extracted.
After cleaning: **1,225 universities × 71 features**. Every cleaning step is in
`src/build_dataset.py` and logged in report Table 1.

**"What did you have to clean?"**
Five things: three constant/empty columns dropped; one exact duplicate column
(`a62_load_speed_s` was identical to `load_time_s`); the QS and Webometrics *value* columns
dropped (95–99% missing, and they describe the institution rather than the website); five
universities that had been crawled twice through different entry URLs merged; and 284 notice
dates that fell *after* the crawl date censored rather than deleted.

**"How did you handle missing values?"** ⭐ *good answer available*
Not with the median — that would have been actively wrong. The median of
`notice_recency_days` is **1 day**, so median-filling would have recorded the **468
universities with no dated notice anywhere** as having posted *yesterday*, handing the
strongest freshness signal to the sites that earned it least. Where absence has a known
direction we fill at the worst defensible value (3650 days = ten years stale) and add a
`*_was_missing` flag computed *before* the fill, so "never measured" stays recoverable.

**"Why 71 features and not 69?"**
We dropped some raw columns and added four engineered ones: `notice_recency_days`,
`load_time_z_region` (load time standardised within region), `broken_links_log`, and the three
missingness indicators.

---

## C. About the scoring model

**"How did you decide the weights?"**
By asking what a prospective student actually needs. Content they came for — what can I study,
how do I apply — carries half the score (D1 + D2 = 50). The machinery that delivers it
(currency, navigation, usability, technical) carries 47. Self-presentation carries 3. The
weights were fixed before any score was computed.

**"Why the gates?"**
Because some failures are not trade-offs. A site with no main menu cannot be navigated no
matter how much content is buried in it, and a site without HTTPS should not be trusted with
an application form. A weighted sum would let a rich site "buy back" those failures with other
content, which is wrong. 186 universities (15.2%) hit a gate.

**"Why the non-linear curves?"**
Because those four attributes genuinely do not reward proportionally:
- **menu size** — 20 items is a wall of links, not better navigation than 7. An inverted U.
- **notice age** — a post from last week and one from last year are not 51 weeks apart in
  value. "Recent" and "abandoned" are plateaus with a cliff between them.
- **contrast** — WCAG AAA is 7:1; beyond that more contrast is not a better reading
  experience, so the curve plateaus.
- **broken links** — one is an oversight, forty is neglect.

**"Isn't the weighting subjective?"**
Yes, and we say so in limitation 4. It is a documented judgement, not a measured truth. Every
weight, gate and curve is stated in full so anyone can disagree with a specific number rather
than with the method. What we did *not* do is tune the weights to produce a ranking we liked.

---

## D. About the modelling

**"Why 80/20?"**
Standard practice and it leaves 246 test universities — enough for stable metrics. Stratified
on score band so both halves span the full range, seed fixed at 42 so it is reproducible.

**"How do you know it isn't overfitting?"**
Three things. The test set was used **once**, at the very end — no algorithm, hyper-parameter
or threshold was chosen with it. Model selection and the 81-combination grid search both ran
on 5-fold cross-validation **inside the training set**. And the CV ordering agrees with the
test ordering at Spearman ρ = 0.888, so the comparison is not a fluke of one split.

**"Why LightGBM?"**
It won on every metric — R² 0.992, MAE 1.26, Spearman 0.992 — and it was the most stable in
cross-validation (0.986 ± 0.001, the smallest spread of any model). After tuning: R² 0.993,
MAE 1.11.

**"Why is k-NN so much worse?"**
0.786 against 0.992. In a 71-dimensional, mostly binary space, Euclidean distance is a poor
measure of similarity — everything is roughly equidistant from everything else. It is a
textbook demonstration of the curse of dimensionality, and a good thing to have in the
comparison for exactly that reason.

**"Did any model give you trouble?"** ⭐ *shows real engineering*
The MLP. Trained on the raw 0–100 target, one fold in five diverged and cross-validated R²
collapsed to −0.03 ± 1.88. Neural networks need the *target* scaled, not just the inputs. We
wrapped it in a `TransformedTargetRegressor` and added L2 regularisation, which stabilised it
at 0.943 ± 0.035. The tree models needed no equivalent care — part of why they are the
practical choice at this data size.

**"Which features matter most?"**
By **gain** (how much each split reduced the loss): `a37_programs_listing` 29%,
`a02_primary_nav` 26%, `a35_faculty_link` 7%, `a34_department_links` 5%. Aggregated by
dimension: academic information 49.8%, navigation 29.3%, admission support 10.2%.

**Note this if asked about importance:** we deliberately used *gain* rather than the default
*split-count* importance. Split-count flatters continuous attributes, which get split many
times, over binary ones, which are split once. Under split-count, load time looked like the
single most important feature; under gain it is under 1%.

---

## E. Questions with an edge

**"Your D5, D6 and D7 dimensions carry 20 points but drive under 1% of the model. Isn't that a design flaw?"**
It is a real finding, and worth stating as one: **a declared weight is not a realised
influence**. A dimension can only differentiate universities to the extent that they actually
differ on it, and 98.9% of sites use HTTPS while most have adequate contrast. Those dimensions
still *belong* in the scoring model — a site that failed them should be penalised — but they
separate almost nobody in this population. The same effect is documented in the
website-evaluation literature, where a scheme allocating 40% to performance was later found to
drive under 3% of its ranking variance.

**"Why is Beijing Normal University near the bottom? That's a top university."**
Because its landing page returned almost nothing to the crawler — no navigation, no
programmes, no departments, no contact link. It is a JavaScript-rendered site the extractor
could not read. About 2% of rows look like this and they cluster at the bottom. It is a
scraping limitation, not a judgement about the institution, and it is limitation 2 in the
report. **The top of the ranking is unaffected.**

**"You are ranking websites, not universities."**
Correct, and deliberate. We dropped the QS and Webometrics value columns precisely so
institutional prestige could not leak into a score that is supposed to be about the website.
We kept the badge-*display* flags, because "this site shows a ranking badge" is a fact about
the page — and they turn out to drive almost nothing.

**"What would you do differently?"**
Four things: add screenshot-derived visual features and a small panel of human ratings to
close the design blind spot; use a headless browser to fix the JavaScript extraction failures;
run a sensitivity analysis over the dimension weights to see how much of the ranking is a
property of the websites versus a property of our choices; and validate against a student
survey.

---

## F. Thirty-second summary if you are put on the spot

> "There is no quality label in the raw data, so we built a seven-dimension scoring model
> around what a prospective student needs — what can I study, how do I apply, is the site
> alive, can I navigate it, can I use it — with two gating rules and four non-linear response
> curves. We applied it to 1,225 universities, split 80/20, and compared twelve regression
> algorithms. LightGBM won with R² 0.993 and an average error of 1.1 points on data it never
> saw. The interesting part is *why* the algorithms differed: the gap is concentrated entirely
> on the sites where a gate caps the score, because a linear model cannot represent a
> threshold and a tree can."

---

## G. Slide-ready numbers

| | |
|---|---|
| Universities | 1,225 |
| Features | 71 |
| Train / test | 979 / 246 (80/20, stratified, seed 42) |
| Algorithms compared | 12, across 6 families |
| Score range | 8.3 – 95.4, mean 64.4, sd 18.4 |
| Universities gated | 186 (15.2%) |
| **Best model** | **LightGBM (tuned)** |
| **Test R²** | **0.9931** |
| **Test MAE** | **1.11 points** |
| **Test Spearman ρ** | **0.9931** |
| Baseline MAE | 15.73 |
| Linear Regression MAE | 2.56 |
| Grid search | 81 combinations, 5-fold CV |
| Gated MAE: linear vs LightGBM | 6.41 vs 1.07 |
| Top feature by gain | `a37_programs_listing` (29%) |
| Gate attribute by gain | `a02_primary_nav` (26%) |
