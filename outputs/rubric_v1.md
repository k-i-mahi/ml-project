# Rubric v1 — Expert Website-Quality Scoring

**Status: FROZEN.** This document is authored and committed *before* any score is computed
(`04_rubric_trackA.ipynb` reads these rules and asserts they match). Every rule can be disputed
individually by reading it, independently of the results it produces.

**What this is.** Track A: an explicit, auditable scoring function over the 69 extracted
attributes. It encodes judgement about *what matters and how it combines* — the part a correlation
matrix cannot supply.

**What this is not.** It is **not** ground truth and **not** the ML target. It is a deterministic
function of the features, so training a model to predict it would report a meaningless R². It
serves three purposes only: (1) a baseline the learned model must beat, (2) the stratifier for the
Track B sample, (3) one half of the Spearman(A, B) validation.

---

## 1. Design principle: quality is gated and hierarchical, not additive

A person landing on a university website does not add up features. They ask, in order:

1. Does this site work at all?
2. Can I do the thing I came to do?
3. Is this institution alive and maintained?
4. How deep is the service?
5. Can everyone use it, and is it technically sound?
6. Is it polished?

Failing an early question is not compensated by excelling at a later one. A site with no primary
navigation is not rescued by having social-media icons. This ordering is the rubric's spine.

**The empirical case for this design.** Rashida et al. (2021) scored content as `(count × 50)/25`
— an unweighted sum of 25 binary presence flags — and separately surveyed 1,820 students across the
same 22 universities. Their automated ranking and their own student ranking correlate at
**ρ = 0.102 (p = 0.67)**; the top-5 lists overlap on 1 of 5. Naive equal-weight presence-counting
does not predict human judgement. Every non-linearity and gate below exists to avoid that failure.

---

## 2. Gates (Tier 0)

Gates are **not scored**. They impose a ceiling on the achievable score regardless of everything
else, because they represent conditions under which a site cannot reasonably be called good.

| Gate | Condition | Ceiling | Justification |
|---|---|---|---|
| Primary navigation | `a02_primary_nav = 0` | **45 / 100** | With no primary menu there is no wayfinding. Every task-completion path is broken regardless of what content exists underneath. |
| Transport security | `a65_https = 0` | **60 / 100** | A university site handling applicant data over plain HTTP fails a baseline obligation. Capped rather than zeroed because the content may still be good. |

Gates are deliberately few and both are unambiguous failures. `a65_https` is at 98.9% prevalence,
so as a *scored feature* it would carry almost no information (audit F06) — as a gate it does the
one job it is suited for.

Ceilings compose multiplicatively-by-minimum: a site failing both is capped at 45.

---

## 3. Tiers

Features are assigned to exactly one tier. A tier receives a weight; features share their tier's
weight equally, subject to the persona boosts in §5. **Tiers, not blocks, are the unit of
weighting** — a person weights *needs*, not taxonomy branches. Block-level weights are derived
afterwards for reporting and compared against realised variance share.

### Tier 1 — Can I do what I came to do?
The heaviest tier. Failing any of these is a serious defect for the dominant audience
(prospective students, then current students).

`a37_programs_listing`, `a46_admissions_policy`, `a22_admission_notice`, `a43_contact_link`,
`a04_search_bar`, `a34_department_links`, `a35_faculty_link`, `a03_nav_item_count` (curved)

### Tier 2 — Is this institution alive?
A university site frozen for two years reads as institutional neglect. **Dated content outranks
undated content** — this is why `notice_evidence` replaces the raw `a16`/`a17` flags.

`notice_evidence`, `a20_news_events`, `event_evidence`, `a27_event_datetime`,
`a36_research_highlight`, `a21_calendar_link`

### Tier 3 — Depth of service
Valuable, not disqualifying.

`a39_library_link`, `a40_career_link`, `a41_alumni_link`, `a38_scholarship`, `a44_student_portal`,
`a42_faq_link`, `a45_prospectus`, `a32_vision_mission`, `a33_about_blurb`, `a47_footer_contact`,
`a48_footer_sitemap`, `a51_quick_links`

### Tier 4 — Accessibility and technical soundness
The dimension with the strongest support in the web-quality literature, and the only place with
genuine continuous measurement.

`a72_alt_text_pct`, `a73_accessible_design`, `a74_a11y_toggle`, `a53_contrast_ratio`,
`a63_mobile_score`, `a67_gzip`, `load_time`, `broken_links`, `a69_title_meta`,
`a71_sitemap_robots`, `a05_language_toggle`, `a06_breadcrumb`

### Tier 5 — Polish
Small positive contribution.

`a29_video_content`, `a30_image_gallery`, `a31_social_feed_embed`, `a25_event_images`,
`a26_event_captions`, `a28_contests`, `a58_live_chat`, `a59_feedback_form`, `a50_social_links`,
`a49_copyright_line`, `a57_logo_prominence`, `a70_favicon`, `a01_logo`, `a54_banner_carousel`

### Tier 6 — Self-promotion (near-zero weight)
Institution-asserted credibility signals. Trivially gamed, weak evidence of website quality, and
the route by which prestige would contaminate the score if weighted normally. **Retained rather
than deleted** — whether a site chooses to display a badge is a real design decision — but confined
to near-zero weight (audit F19).

`a07_qs_badge`, `a09_national_rank`, `a11_accreditation`, `a12_accred_count`, `a13_achievements`,
`a15_stat_item_count`, `a60_trust_seal`, `a61_testimonials`, `a75_bookmark`

---

## 4. Non-linear response curves

Applied in `02_features.ipynb`; restated here because they are rubric decisions, not data cleaning.
Each is piecewise-linear through explicit knots.

| Feature | Naive treatment | What this rubric asserts |
|---|---|---|
| `a03_nav_item_count` | more = better | **Inverted U.** 0–1 broken, 5–9 ideal, 15–20 an undifferentiated dump. Peak at 5–9. |
| `a24_event_count` | more = better | 0 = dead, 4–10 = active, 20 (the cap) = probably an undated dump, scored *below* 10. |
| `a15_stat_item_count` | more = better | Hard diminishing returns after ~6; beyond that it is marketing filler. |
| `a72_alt_text_pct` | linear | **Concave.** Steep 0→60%, gentle 60→100%. Partial coverage delivers most of the benefit. |
| `a53_contrast_ratio` | more = better | **Plateaus at 7:1** (WCAG AAA). 21:1 is not better than 12:1 — it is just black on white. |
| Notice board without timestamp | full credit | **Partial credit** (`notice_evidence` level 1 of 3). Undated notices are weak freshness evidence. |
| `a54_banner_carousel` | positive | **Negative direction.** Auto-rotating carousels are consistently poor for usability. |
| `load_time` | raw seconds | **Within-region percentile only.** Absolute values measure the collector (audit F02/F17). |

None of this comes out of a correlation matrix. It comes from what a website is *for*.

---

## 5. Rater personas

A single scorer has no measurable reliability, which is the standard objection to one-rater labels.
The rubric is therefore applied under **five personas**, each a defensible weighting of the same
gates and curves. Their disagreement is a reportable uncertainty measure, and their agreement
(ICC) is a reportable reliability statistic.

### Tier weights (%), by persona

| Tier | P1 Domestic student | P2 International student | P3 Accessibility specialist | P4 Researcher | P5 Literature-aligned |
|---|---|---|---|---|---|
| T1 Task completion | **40** | 34 | 26 | 28 | 30 |
| T2 Institutional life | 20 | 16 | 14 | **22** | 18 |
| T3 Service depth | 20 | 22 | 14 | 20 | 16 |
| T4 Accessibility/technical | 12 | 16 | **38** | 16 | 24 |
| T5 Polish | 6 | 8 | 6 | 8 | 10 |
| T6 Self-promotion | 2 | 4 | 2 | 6 | 2 |
| **Total** | 100 | 100 | 100 | 100 | 100 |

### Within-tier emphasis multipliers

Applied inside a tier, then renormalised so the tier's total weight is unchanged. This is what
makes the personas genuinely different rather than five rescalings of one ranking.

| Persona | Multipliers |
|---|---|
| P1 Domestic student | `a22_admission_notice` ×1.5, `a37_programs_listing` ×1.5, `a44_student_portal` ×1.5 |
| P2 International student | `a05_language_toggle` ×3.0, `a38_scholarship` ×2.0, `a45_prospectus` ×2.0, `a46_admissions_policy` ×1.5 |
| P3 Accessibility specialist | `a72_alt_text_pct` ×2.0, `a73_accessible_design` ×2.0, `a74_a11y_toggle` ×2.0, `a53_contrast_ratio` ×2.0 |
| P4 Researcher | `a36_research_highlight` ×2.5, `a35_faculty_link` ×2.0, `a39_library_link` ×2.0, `a34_department_links` ×1.5 |
| P5 Literature-aligned | `a53_contrast_ratio` ×1.5, `a57_logo_prominence` ×1.5, `a54_banner_carousel` ×1.5 (raises web appearance to ≈10%, per Saleh et al.) |

**P5 exists as a sensitivity check.** Saleh et al.'s SLR shows usability (17/24 studies) and
information quality (15/24) dominating, with web appearance mid-tier (8/24). P5 encodes that
literature emphasis so we can report whether the top-50 is stable against it. Citation frequency
measures what researchers chose to study, not what users need — which is why it informs one persona
rather than all five.

---

## 6. Scoring procedure

For persona *p*:

```
tier_score[t]  = Σ_f (m[p,f] · g[f])  /  Σ_f m[p,f]        for features f in tier t
raw_score[p]   = 100 · Σ_t (w[p,t]/100) · tier_score[t]
score[p]       = min(raw_score[p], gate_ceiling)
```

where `g[f] ∈ [0,1]` is the goodness value from `goodness_matrix.csv` (curves already applied) and
`m[p,f]` is the persona multiplier (default 1.0).

### Outputs

| Column | Meaning |
|---|---|
| `trackA_P1` … `trackA_P5` | Per-persona score, 0–100 |
| `trackA_consensus` | Mean of the five — the primary Track A label |
| `trackA_sd` | SD across the five — per-university label uncertainty |
| `trackA_min` / `trackA_max` | Range across personas |

### Reliability

**ICC(2,k)**, two-way random effects, absolute agreement, average of k=5 raters — the correct form
when the raters are the same five applied to every subject and the aggregate score is the quantity
of interest. Reported with its F-test and 95% CI.

---

## 7. Declared limitations of Track A

1. **It is a formula.** Any model trained on it will re-learn arithmetic. It is a baseline, never
   a target. This is stated wherever a Track A number is reported.
2. **It cannot see the websites.** It scores the same 69 extracted attributes the model sees. It
   adds judgement about combination, not new observation.
3. **The five personas share gates and curves**, so they are correlated by construction. ICC
   measures agreement on *weighting*, not independent observation, and is reported as such.
4. **Weights are asserted, not estimated.** Their realised influence is reported against
   `block_variance_report.csv`, because a nominal weight on a near-constant dimension buys nothing
   (measured: the technical block draws 9.1% nominal weight but 3.9% of realised ranking variance).

These limitations are precisely why Track B exists.
