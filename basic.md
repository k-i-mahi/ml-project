You are a senior Data Scientist, Machine Learning Engineer, Statistical Analyst, and Data Quality Engineer with 20+ years of experience in tabular machine learning, feature engineering, data preprocessing, website-quality assessment, ranking systems, learning-to-rank, explainable AI, and experimental design.

I am giving you THREE main inputs:

1. The complete university website dataset that we scraped.
2. The final optimized attribute/schema document containing every attribute, datatype, range, meaning, justification, and reference.
3. Our project/lab report explaining the research objective, previous methodology, data-collection procedure, proposed ML pipeline, and ranking objective.

You MUST read and understand all three files before doing the analysis.

============================================================
PROJECT GOAL
============================================================

We are developing a Machine Learning framework that evaluates the QUALITY OF UNIVERSITY WEBSITES and ultimately produces a ranking of universities according to their website quality.

The basic idea is:

University website
    -> scraped measurable website-quality attributes
    -> preprocessing and feature engineering
    -> ML model / ranking model
    -> predicted website-quality score
    -> worldwide website-quality ranking
    -> regional website-quality ranking
    -> country-level website-quality ranking
    -> explain WHY each university received that position

Universities whose websites have objectively better values for meaningful website-quality attributes should naturally receive higher predicted quality scores and therefore appear higher in the final ranking.

By "better", I mean website-quality characteristics supported by the supplied attribute definitions: information completeness, freshness, usability, navigation, accessibility, technical performance, SEO, interaction, reliability, etc.

DO NOT artificially favour a university because it is famous, highly ranked academically, public/private, located in a developed country, in a large city, has many foreign students, or has a good QS/Webometrics rank.

The ranking preference must come from WEBSITE QUALITY.

Institutional/context variables may be used for stratification, exploratory analysis, fairness/bias testing, or secondary experiments, but they must NOT silently dominate the primary website-quality ranking.

============================================================
CRITICAL METHODOLOGICAL RULES
============================================================

Before modelling, determine which columns belong to these groups:

A. Website-quality predictor features
B. University/context attributes
C. Identification fields
D. Crawl/extraction metadata
E. External ranking/benchmark variables
F. Target/label variables
G. Model outputs

Keep these groups conceptually separate.

Examples:

Identification fields such as:
uni_id, university name, URL, member/collector identifier
should NOT be ML predictors.

External prestige variables such as:
QS world rank,
QS regional rank,
Webometrics world rank,
Webometrics country rank
must NOT be used in the PRIMARY website-quality model because they may create prestige leakage.

Use them later only for:
- external comparison,
- construct-validity analysis,
- correlation with our generated ranking,
- secondary controlled experiments.

Similarly, variables such as:
- city,
- urban/rural category,
- university ownership/control,
- total students,
- faculty count,
- student-faculty ratio,
- international-student percentage,
- geographical region

should normally be treated as CONTEXT variables rather than direct website-quality predictors.

They can be used for:
- subgroup analysis,
- fairness/bias analysis,
- stratification,
- regional generalization tests,
- secondary models,

but do not let them create an unfair advantage in the primary quality ranking.

============================================================
VERY IMPORTANT: NULL IS NOT ZERO
============================================================

Analyse missing values semantically.

Do NOT blindly use:

fillna(0)

A zero can mean:
"we verified that the attribute is absent."

A NULL can mean:
"the attribute could not be extracted",
"measurement failed",
"not applicable",
"not enough information",
or "value was unavailable".

Therefore determine the correct missing-value treatment FEATURE BY FEATURE using the schema document.

Consider appropriate methods such as:

- retain zero for verified absence
- median imputation for appropriate numeric measurements
- most-frequent imputation for appropriate categorical features
- explicit "Unknown" category
- missingness indicator columns
- model-native missing handling
- dropping features only when justified

Explain every major missing-data decision.

============================================================
PHASE 1 — DATASET AUDIT
============================================================

Start with a complete forensic audit of the supplied dataset.

Report:

- number of rows
- number of columns
- duplicate university IDs
- duplicate URLs/domains
- duplicated universities
- datatypes
- unique-value counts
- constant columns
- near-constant columns
- impossible values
- out-of-range values according to the schema
- malformed categorical values
- malformed numeric fields
- suspicious zeros
- missing values per feature:
    count
    percentage
- HTTP/crawl anomalies if present
- inconsistent fields
- duplicated information between columns
- potential data leakage
- possible scraper/extractor errors

Compare the actual dataset against the supplied final attribute definition document.

For every important mismatch, explicitly report:

EXPECTED SCHEMA
ACTUAL DATA
PROBLEM
RECOMMENDED FIX

Do not silently correct data.

============================================================
PHASE 2 — CLEANING
============================================================

Create a defensible cleaning strategy.

Handle:

- duplicated records
- duplicated universities
- invalid ranges
- inconsistent Boolean/count combinations
- impossible numerical measurements
- malformed categorical values
- sparse columns
- constant/near-constant features
- redundant columns
- crawl metadata that should not be predictors
- feature leakage
- incorrect datatype assignments
- missing values

Do NOT remove a feature only because it contains some missing data.

For each removed feature explain exactly WHY it is removed.

Examples of legitimate reasons:

- almost completely missing
- no variance
- redundant with another stronger measurement
- scraper unreliable
- target leakage
- institutional/prestige leakage
- not actually measuring website quality
- conceptually invalid

Produce a CLEAN DATASET after this stage.

============================================================
PHASE 3 — FEATURE TYPE SEPARATION
============================================================

Separate features into at least:

1. Binary variables
2. Count variables
3. Continuous variables
4. Percentage variables
5. Categorical variables
6. Date/time variables
7. Context variables
8. External benchmark variables

Do not treat every variable identically.

For example:

Binary features:
analyse prevalence and class imbalance.

Count features:
analyse distributions, skewness, zero inflation, and outliers.

Continuous features:
analyse distribution, outliers, scale, and transformations.

Date features:
convert to meaningful derived features where appropriate.

For example:

notice_recency_days =
crawl_date - latest_notice_date

Prefer meaningful derived variables over raw dates when they are more suitable for ML.

============================================================
PHASE 4 — EXPLORATORY DATA ANALYSIS
============================================================

Perform meaningful EDA.

Do not generate plots just for decoration.

Use suitable visualisations based on feature type.

For continuous/count variables consider:

- histogram
- KDE/distribution plot
- box plot
- violin plot when useful

For binary variables consider:

- prevalence bar chart
- percentage present/absent

For categorical variables:

- category counts
- distributions across regions/groups where relevant

Also produce:

- missing-value chart
- correlation heatmap for suitable numeric features
- Spearman correlation matrix
- feature redundancy analysis
- target-vs-feature plots if an independent target exists
- region-wise distributions where useful
- important outlier visualisations

For boxplots, identify which observations are statistical outliers but DO NOT automatically remove them.

Determine whether they are:
- legitimate extreme universities,
- measurement errors,
- scraper errors,
- or impossible values.

============================================================
PHASE 5 — FEATURE ENGINEERING
============================================================

Engineer meaningful features only when they improve the representation of website quality.

Consider, where supported by the available dataset:

- notice recency
- news recency
- content completeness measures
- event completeness
- broken-link percentage rather than only raw broken-link count
- metadata completeness
- footer/service completeness
- accessibility completeness
- compression coverage
- standardized technical metrics
- log1p transformation for heavily skewed count variables
- normalized rates when university websites have substantially different numbers of links/items/resources

Avoid arbitrary feature engineering.

Every engineered feature must include:

FORMULA
REASON
EXPECTED INTERPRETATION

============================================================
PHASE 6 — FEATURE QUALITY AND FEATURE SELECTION
============================================================

I want you to determine WHICH FEATURES ARE ACTUALLY USEFUL.

Do NOT decide this from one correlation coefficient alone.

Evaluate feature usefulness using multiple criteria:

- missingness
- variance
- prevalence
- distribution quality
- redundancy
- Pearson correlation where appropriate
- Spearman correlation
- mutual information
- univariate predictive association with an independent target, if available
- permutation importance
- tree-based importance
- SHAP importance
- feature importance stability across cross-validation folds
- block-wise ablation
- leave-one-feature/block-out experiments when useful

For highly correlated features, prefer the feature that is:

- conceptually stronger
- more directly measurable
- less missing
- more reliable
- more interpretable

Flag pairs with very high correlation, for example |rho| > 0.90, and recommend whether to keep both.

Do not perform feature selection using the full dataset before validation in a way that leaks test information.

Any supervised feature-selection process must occur INSIDE the training/CV pipeline.

At the end, give me:

FINAL RECOMMENDED WEBSITE FEATURES
REMOVED FEATURES
SECONDARY/CONTEXT FEATURES
BENCHMARK-ONLY FEATURES

with reasons for every decision.

============================================================
PHASE 7 — FIRST DETERMINE WHETHER A TRUE ML TARGET EXISTS
============================================================

This is extremely important.

Inspect the dataset/report and determine whether we have an INDEPENDENT ground-truth target such as:

label_expert_score

or some other independently collected human website-quality rating.

CASE A — IF AN INDEPENDENT EXPERT/HUMAN QUALITY LABEL EXISTS:

Treat the problem primarily as supervised ranking/regression.

Use the expert score as the gold target.

The desired output is a continuous website-quality score, because we ultimately need a ranking.

Consider models such as:

- Ridge/ElasticNet baseline
- Random Forest
- Gradient Boosting
- XGBoost / LightGBM / CatBoost where available
- suitable neural model only if justified

Compare multiple models rather than assuming one model is best.

Primary ranking-oriented evaluation should include:

- MAE
- RMSE
- R²
- Spearman rank correlation
- Kendall rank correlation if useful

If ordinal quality grades exist, optionally also test:

- ordinal classification
- macro F1
- quadratic weighted kappa
- confusion matrix

CASE B — IF THERE IS NO INDEPENDENT TARGET:

DO NOT pretend we have supervised ground truth.

Do NOT train a model on a score constructed entirely from the same features and then claim high predictive accuracy as evidence of model success.

Instead:

1. Clearly explain that the dataset is currently unlabeled for genuine supervised quality learning.
2. Build a TRANSPARENT rule-based / composite website-quality score only as an initial baseline.
3. Normalize attributes appropriately.
4. Define directionality:

Examples:
higher alt-text coverage = better
higher mobile usability = better
lower page-load time = better
lower broken-link percentage = better
more complete information = generally better
more recent content = generally better

5. Avoid arbitrary weights.
6. Propose defensible weighting approaches, for example:
   - equal weights by conceptual block
   - expert-defined weights
   - entropy/data-driven weighting as a secondary experiment
   - PCA only as exploratory dimensionality reduction, NOT automatically as a quality definition

7. Clearly label this ranking as:
RULE-BASED BASELINE
not independently validated ML ground truth.

8. Design the expert-rating procedure needed to create a real gold label.

If the report already specifies an expert-rated subset, follow the report rather than inventing another methodology.

============================================================
PHASE 8 — QUALITY-ORIENTED RANKING DESIGN
============================================================

We want the ranking to have an appropriate QUALITY-ORIENTED INDUCTIVE BIAS.

This means features representing better website quality should influence the result in the logically correct direction.

Examples:

lower broken-link rate should be better
lower loading delay should be better
higher accessibility coverage should be better
better content completeness should be better
greater freshness should be better
better mobile usability should be better

Analyse the expected monotonic direction of important attributes.

Do not manipulate individual universities.

Do not manually boost famous universities.

The ranking rule/model must be university-agnostic and feature-based.

If the selected ML framework supports monotonic constraints and they are scientifically defensible, investigate whether they improve the robustness and interpretability of the model.

Do not force monotonicity where the relationship is genuinely non-linear.

============================================================
PHASE 9 — MODEL TRAINING AND VALIDATION
============================================================

Design a rigorous ML pipeline.

All of the following must be performed inside the training pipeline / CV where applicable:

- imputation
- encoding
- transformations
- scaling
- feature selection

Prevent train-test leakage.

Use suitable cross-validation.

Because universities come from different geographical regions, include:

1. Standard cross-validation with appropriate stratification where possible.
2. Region-aware evaluation.
3. Leave-one-region-out evaluation if sample sizes allow.

The purpose is to test whether the model learned "good website quality" rather than characteristics specific to a region.

Report mean AND standard deviation across folds.

Compare the final model against meaningful baselines.

============================================================
PHASE 10 — FAIRNESS / BIAS ANALYSIS
============================================================

Because university context differs by:

- region
- country
- urban/rural setting
- public/private/local-government control
- university size
- international student proportion

test whether prediction errors or quality scores show systematic differences across these groups.

Do NOT automatically remove legitimate website-quality differences.

The goal is to determine whether the model is depending on contextual identity rather than website characteristics.

Run useful subgroup analyses such as:

- average predicted score per region
- prediction error per region if gold labels exist
- public vs private
- urban vs rural
- city-size groups

If a context feature produces substantial performance improvement, investigate whether it represents legitimate website-quality information or undesirable proxy bias.

============================================================
PHASE 11 — FINAL RANKING
============================================================

After selecting the best scientifically defensible model or scoring approach, generate:

predicted_quality_score
global_website_rank
regional_website_rank
country_website_rank

Higher quality score = better rank.

Use rank 1 as the best website.

If ties occur, define a transparent tie-breaking strategy.

Do NOT use QS/Webometrics ranking to determine our ranking.

After our ranking is produced independently, compare it with external ranking benchmarks using:

- Spearman correlation
- Kendall correlation where appropriate
- scatter plots
- top-k overlap
- qualitative differences

Explain that external academic rankings and website-quality rankings measure different constructs, so perfect correlation is neither expected nor necessarily desirable.

============================================================
PHASE 12 — EXPLAINABILITY
============================================================

For the selected model, explain WHY universities receive their scores.

Use SHAP and/or permutation importance.

Provide:

GLOBAL INTERPRETATION:
- most influential features
- direction of influence
- feature blocks that matter most

LOCAL INTERPRETATION:
For selected top, middle, and bottom-ranked universities show:
- features that increased their quality score
- features that reduced their quality score
- actionable website improvements

The final system should be capable of answering questions such as:

"Why is University A ranked above University B?"

and

"What should University B improve to move higher?"

============================================================
PHASE 13 — BLOCK-WISE ANALYSIS
============================================================

Where supported by the supplied schema, group features into conceptual blocks such as:

- navigation/usability
- content/information
- freshness/notices
- events
- accessibility
- interaction
- technical performance
- SEO/discoverability
- credibility

Run block-wise analysis and, where appropriate, ablation:

full model
versus
model without each block

This will show which dimensions genuinely contribute useful information.

============================================================
PHASE 14 — DELIVERABLES
============================================================

I want the final response to be a COMPLETE DATA-SCIENCE WORKFLOW, not generic advice.

Produce the outputs in this order:

1. Executive summary of what you found.
2. Dataset audit.
3. Problems/errors discovered.
4. Cleaning decisions.
5. Missing-value strategy feature by feature or feature type.
6. Features excluded immediately and why.
7. EDA findings.
8. Relevant plots and their interpretations.
9. Feature-engineering decisions.
10. Feature-selection analysis.
11. Final feature set.
12. Target/label assessment.
13. Baseline methodology.
14. ML model candidates.
15. Training/preprocessing pipeline.
16. Cross-validation strategy.
17. Model comparison.
18. Best model and why.
19. Ranking methodology.
20. Final global ranking.
21. Final regional rankings.
22. Final country rankings where enough universities exist.
23. External QS/Webometrics comparison.
24. SHAP/feature importance analysis.
25. Fairness/regional-bias analysis.
26. Important limitations.
27. Exact recommendations for improving the research methodology.
28. Final cleaned/model-ready dataset schema.

============================================================
CODE REQUIREMENT
============================================================

Give me complete, executable Python code suitable for Google Colab/Jupyter.

Prefer:

pandas
numpy
matplotlib
scikit-learn

and where justified:

LightGBM
XGBoost
CatBoost
SHAP

Do not give pseudocode where executable code is possible.

Organize the code into clear cells/sections.

Every major code block should:
- explain what it is doing
- execute one logical stage
- display useful outputs
- avoid leakage

For plots:

- give readable titles
- proper axis labels
- avoid unnecessarily plotting dozens of meaningless graphs
- prioritize the features that actually reveal something useful

Save useful artefacts where appropriate:

cleaned_dataset.csv
model_ready_dataset.csv
feature_selection_report.csv
global_university_website_ranking.csv
regional_university_website_ranking.csv
feature_importance.csv
model_evaluation.csv

============================================================
DECISION-MAKING REQUIREMENT
============================================================

Do NOT blindly follow my assumptions.

Act like a senior data scientist reviewing this as a research project.

If something in our existing methodology is statistically invalid, redundant, leaky, misleading, or technically weak:

1. identify it,
2. explain why it is a problem,
3. propose a better alternative,
4. then implement the better alternative.

Do not change the research objective.

The objective is still:

"Use objectively measurable university website-quality attributes to construct a defensible ML-based website-quality scoring and ranking framework."

============================================================
FINAL SCIENTIFIC PRINCIPLE
============================================================

The model should learn:

GOOD WEBSITE CHARACTERISTICS
        ↓
HIGH WEBSITE-QUALITY SCORE
        ↓
HIGH WEBSITE-QUALITY RANK

It should NOT learn:

FAMOUS UNIVERSITY / RICH COUNTRY / HIGH QS RANK
        ↓
HIGH WEBSITE RANK

Keep this distinction throughout the complete analysis.

Now first read:
- the dataset,
- the final attribute definition document,
- and the project report.

Do not immediately train a model.

Begin with the dataset/schema audit and tell me what you discover before deciding the final preprocessing and modelling strategy.