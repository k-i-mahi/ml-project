# Viva / demo notes

Questions a teacher is likely to ask, and short answers you can actually defend.
Every number here is in the notebook or the report — nothing is invented for this file.

---

## A. The demo your teacher will probably ask for

### 1. "What is the website score of KUET?"

Open the notebook, go to **§9**, run:

```python
lookup("kuet")
```

Output:

```
  Khulna University of Engineering and Technology
  https://www.kuet.ac.bd
  QUALITY SCORE       85.9 / 100      GRADE  A+
  Global rank           31 of 1226   (top 2.4%)
  Country rank      #1 in Bangladesh
  Regional rank     #13 in South & Central Asia + Oceania
  Expert-labelled   no — score is model inference

  Profile:  programmes listed: yes | contact page: yes | departments: yes
            notice board 30d old | contrast 13.3:1 | menu 6 items | mobile 75/100
```

`lookup()` accepts a partial name (`"kuet"`, `"Otago"`), a country, or a rank.

### 2. "Here is a university's data — where does it stand?"

**§10**. Hand the function whatever attributes you know; the rest are filled with the
training median and marked as assumed:

```python
score_university({
    "a37_programs_listing": 1, "a46_admissions_policy": 1, "a34_department_links": 1,
    "a43_contact_link": 1,     "a38_scholarship": 1,       "notice_recency_days": 2,
    "a03_nav_item_count": 7,   "a53_contrast_ratio": 15.0, "a63_mobile_score": 90,
}, name="Example University")
```

It returns the score, the grade, where it *would* rank among the 1,226, a SHAP breakdown of
**why** it got that score, and the biggest improvements available.

### 3. "Why is A ranked above B?"

**§12**: `compare("kuet", "ruet")` — prints the attribute-by-attribute differences that
account for the score gap.

### 4. "Explain one score."

**§11**: `explain("kuet")` — a SHAP waterfall for that single university.

---

## B. Questions about the label

### "Where did the target variable come from? The dataset has no score column."

That is exactly the problem we had to solve. We built the label in two separate tracks:

- **Track A** is an explicit rubric — gates, six weighted tiers, non-linear curves, five
  personas. It is a *formula over the features*, so we use it only as the **baseline**.
- **Track B** is the target: 200 universities rendered as anonymised profile cards (no name,
  country, region or URL), compared in **900 blind pairs**, then fitted with the
  **Bradley–Terry** model `P(i beats j) = σ(βᵢ − βⱼ)` to recover one latent score each.

### "Why not just make a formula and predict it?" ⭐ *the key question*

Because the model would then only be re-deriving our own arithmetic. It would report R² ≈ 0.98
and mean nothing. **Biyyapu et al. (2023) published exactly that** and reported 98.2% accuracy
for predicting a lookup table from its own inputs.

The evidence that we avoided it: Track A and Track B correlate at ρ = 0.814, so both measure
quality — **but the rubric predicts wide-gap pairs at 88% and close pairs at only 62%.**
If the label were the formula in disguise, both numbers would be ~100%.

### "You labelled it yourself — isn't that biased?"

Three defences, all measured rather than asserted:

- **Blinding.** The judge never saw a name, country, region or URL — only an attribute card
  with a random ID like `S-4471`. Prestige could not leak in.
- **Self-consistency 96.7%.** 60 pairs were re-presented later with the sides swapped and
  held out of the fit. The judge agreed with itself on 58 of 60.
- **No position bias.** 48.2% of wins went to the left card; binomial p = 0.30.

We also state plainly that it is **one rater**, so inter-rater agreement cannot be measured.
That is limitation #4 in the report.

### "You changed the label halfway through. Isn't that cheating?"

No — and it is the most defensible thing in the project, because we published both passes.

The first pass looked healthy by its own statistics: 86.7% self-consistent, no position bias,
correlated with the rubric at 0.790. But reading the judge's *stated reasons* showed
**37.6% of them cited alt-text coverage** — an attribute no visitor can perceive. It was on
the card only because it was a convenient number. Separately, `notice_recency_days` had been
median-imputed, and since the median is 1 day, the 469 sites with **no dated notice at all**
were described as having "posted yesterday".

We fixed both, re-framed the question around what an applicant needs, and **re-judged the same
900 pairs**. Both files are in `data/label_provenance/`.

The crucial point: **the test set was never used to make that decision.** The relabel was
driven by reading reasons, not by chasing a test score.

### "How do you know the relabel improved things rather than just changing them?"

Four independent signals:

| | pass 1 | pass 2 |
|---|---|---|
| reasons citing alt-text | 37.6% | 0.0% |
| reasons citing admission content | 38.1% | 73.1% |
| self-consistency | 86.7% | **96.7%** |
| held-out test ρ | 0.885 | **0.922** |

And 16.4% of winners flipped — not 2% (nothing changed) and not 45% (noise). The obvious pairs
held; the close ones moved.

---

## C. Questions about the model

### "Which model, and why?"

**LightGBM with label weights** wᵢ ∝ 1/SE(βᵢ), so universities whose latent score was estimated
confidently count for more.

It is *not* the best on plain cross-validation — ordinary LightGBM is (ρ = 0.881 vs 0.873) —
but it is the best under **leave-one-region-out** (0.844 vs 0.835), which is the harder test.

### "How do you know it isn't overfitting?"

- The 40-row test set was used **exactly once**, at the end. No feature, hyper-parameter or
  threshold was chosen with it.
- Model selection used nested CV: outer repeated 5-fold × 5 seeds, inner 3-fold tuning, with
  every preprocessing step fitted *inside* the training fold.
- **Leave-one-region-out** ρ = 0.844: hold out an entire region, train on the other five.

### "Does it beat a simple baseline?"

| model | CV ρ | LORO ρ |
|---|---|---|
| **LightGBM + label weights** | **0.873** | **0.844** |
| Ridge regression | 0.855 | 0.830 |
| Track A rubric (baseline) | 0.810 | 0.774 |
| B5 content block only | 0.752 | 0.693 |
| B9 technical block only | 0.199 | 0.100 |
| Mean predictor | 0.000 | 0.000 |

All learned models beat the rubric at p < 0.001 on a paired test over 25 fold-repetitions.
Weka's RandomForest independently reaches correlation 0.88 on the same split.

### "Is it just learning which region a university is in?"

That was our worry too, because **collector and region are perfectly confounded** — each of
the six data collectors covered exactly one region.

Two things we did: `region`, `country` and `member` are excluded from the design matrix
entirely; and load time is region-z-scored, with every ranking produced twice, with and
without it (the two orderings correlate at ρ = 0.992).

The test: leave-one-region-out accuracy is **stable at 0.81–0.87 across all six regions**. If
it were a geography lookup, holding out a region would collapse it.

### "What actually drives the score?"

Top features by permutation importance: `content_completeness_B5`, `a34_department_links`,
`a03_nav_item_count`, `a46_admissions_policy`, `a53_contrast_ratio`, `notice_recency_days`,
`a38_scholarship`, `a37_programs_listing`.

In plain terms: **can a visitor find the programmes, the departments, the admissions rules and
the deadlines, can they navigate, and is the page current.** Page speed and SEO metadata
contribute very little, because nearly every university scores alike on them.

---

## D. Questions with a sharp edge

### "KUET's website isn't that good — why is it 31st?" ⭐ *be ready for this*

Because on the attributes this dataset actually measures, KUET's landing page ticks nearly
every box: programmes listed, departments, faculty, library and research pages, a notice board
posted within the last month, admission notices, scholarships, a contact route, HTTPS, no
broken links, contrast 13.3:1.

**What the dataset cannot see is design** — clutter, layout, whether the page looks modern.
There is no screenshot, no visual feature, nothing about aesthetics in the 78 columns. So a
site that satisfies every content requirement will rank highly even if a human would find it
dated.

We report that openly as limitation #1 rather than hand-tuning the score to match intuition.
Closing the gap would need screenshots and human raters, not more modelling.

### "Why is Harbin Institute of Technology near the bottom? That's a top university."

Because its landing page returned essentially nothing to the crawler — no navigation, no
programmes, no departments, no contact link. It is a JavaScript-rendered site the scraper
could not read.

**25 rows (2.0%) look like this, and 21 of them are in the bottom 100.** It is a scraping
limitation, not a judgment about the institution — and it is why the report says the bottom of
the ranking should not be quoted without that caveat. The top is unaffected.

### "This ranks websites, not universities — is that useful?"

That is the correct reading, and it is deliberate. We deliberately **dropped** the QS and
Webometrics *value* columns — they were 95–99% null and would have leaked prestige into a
label that is supposed to be about the website.

We *kept* the badge-display flags (does the site show a QS badge) as website features, because
"this site advertises a badge" is a fact about the page. They sit in the rubric's lowest tier
and SHAP confirms they drive only 2.4% of the ranking.

### "Why is classification accuracy only 42% in Weka?"

Six classes, 160 training rows, and the grades are cut from a *continuous* score — a site at
64.9 is A and one at 65.1 is A+. No model can see a difference that isn't there.

Look at the confusion matrix: **82% of predictions are within one grade** (ZeroR gets 20%).
The error is boundary noise, not confusion between good and bad websites. That is exactly why
we treat the task as regression and use the grades only for presentation.

### "What would you do next?"

1. **Screenshots + human raters** — close limitation #1, the design blind spot.
2. **A rendering crawler** (headless browser) — fix the 2% of extraction failures.
3. **A second rater** on a subset, to report inter-rater agreement alongside self-consistency.
4. **External validation** against a student survey, which we explicitly did not do.

---

## E. Thirty-second summary if you are put on the spot

> "There was no quality label in the data, so we built one two ways: a transparent rubric as
> a baseline, and 900 blind pairwise judgments fitted with Bradley–Terry as the target. Keeping
> them separate is what stops the project being circular — the rubric agrees with the label on
> obvious pairs at 88% but on close pairs only 62%, and that gap is what the model learns.
> LightGBM reaches Spearman 0.92 on a test set we touched once, against 0.81 for the rubric.
> We also labelled it twice: the first pass was measuring alt-text coverage, which no visitor
> can see, so we fixed the cards and re-judged all 900 pairs. Both passes ship with the
> submission."
