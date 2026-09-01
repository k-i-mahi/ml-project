# Lab 3 report — deltas against the delivered work

The report was used for one thing: the **attribute schema** (Table 2/3, 69 attributes in blocks B1–B11), which was verified to map exactly onto the data and is used throughout. Everything below is a place where the report and the built system differ. This is a list of deltas, not a critique.

## 1. Scope: the title says Bangladeshi, the data is global

The report is framed around Bangladeshi university websites. The dataset covers **1,226 universities across 54 country labels and 6 regions**; Bangladesh accounts for 22 of them. All work here is global, per the stated requirement. §7 and §8 of the report read as an earlier Bangladesh-only draft and do not describe this dataset.

## 2. Row count: §4.1 says 1,200, §5.1 says 1,230

The raw file has **1,230** rows. Four domains appear twice — `unam.mx`, `tec.mx`, `ipn.mx`, `aun.edu.eg` — with the two crawls agreeing on 68 of 69 attributes and disagreeing on load time by up to 3.3×. Resolved to **1,226** by keeping the geographically correct row, taking the median load time, and fixing `country` on the Mexican pair (recorded as United States). Documented as audit finding F03.

## 3. The gold label (L3) was not achievable and was replaced

The report specifies `label_expert_score`: 180 sites rated by 3 trained human raters. That was not available. Substituting a composite of the same features and training on it would have produced a high R² that measures nothing — the published error in Biyyapu et al. (98.2% accuracy predicting a lookup table from its own inputs).

**Replaced by a two-track design:** an explicit rubric applied in code to all 1,226 (Track A, the baseline), and 900 blind pairwise judgments over 200 universities fitted with Bradley–Terry (Track B, the target). Track B is not a closed-form function of the features, which is what makes it a legitimate learning target. ρ(A,B) = 0.790 is then a real empirical result rather than an artefact.

The substitution is a **weakening** and is labelled as one: these are LLM-elicited judgments over extracted profiles, not three humans looking at live websites.

## 4. Six attributes in the schema are intentionally absent

`a19`, `a52`, `a55`, `a56`, `a64`, `a68` do not appear in the data. This is **not** scraper loss — report §4.6 documents 75 − 6 = 69, and the 69 present map exactly onto B1–B11. No imputation was attempted and none was needed.

## 5. Two prestige columns are unusable

`a10_webometrics_value` is 99.35% null and `a08_qs_value` is 94.96% null. Both are dropped — unusable *and* leaky, since a ranking of website quality that reads an existing prestige ranking is circular. The **presence flags** `a07_qs_badge` and `a09_national_rank` are kept as genuine site features (does the page display a credibility signal), at near-zero rubric weight. SHAP confirms the block does almost nothing: B2_rankings_recog drives 1.0% of the ranking.

## 6. Internal inconsistencies in the notice and event fields

382 rows carry a notice date with `a16_notice_board = 0`; 156 have a board and no date. 327 rows flag events with `a24_event_count = 0`; 158 count events with the flag off. Rather than averaging these away, `notice_evidence` and `event_evidence` reconcile them into ordinals that record the contradiction as its own level. Audit findings F15/F16.

## 7. The confound the report does not mention

`member` (collector) and `region` are **perfectly 1:1** — each of the 6 collectors covered exactly one region. Load-time medians run 8.07 s (M2/Western Europe) to 12.94 s (M6/Latin America & Africa), and that gradient cannot be attributed to websites rather than to collectors. Load time is therefore used only in region-standardised form, every ranking is produced with and without it (Spearman between the two: 0.9943), and the confound is reported as unresolvable rather than adjusted for.

## 8. Where the report's weighting intuitions did not survive measurement

| block | declared in rubric_v1 | realised (SHAP) | ratio |
|---|---|---|---|
| B11_accessibility | 6.4% | 14.8% | 2.30× |
| B7_visual_design | 3.4% | 6.6% | 1.94× |
| B6_footer | 5.4% | 6.4% | 1.19× |
| B5_page_content | 37.9% | 42.9% | 1.13× |
| B3_notices_updates | 12.7% | 10.2% | 0.80× |
| B1_header_nav | 11.5% | 8.0% | 0.70× |
| B8_service_interact | 1.8% | 1.1% | 0.59× |
| B10_seo_metadata | 3.6% | 1.9% | 0.51× |
| B4_events_media | 8.9% | 4.3% | 0.49× |
| B2_rankings_recog | 2.1% | 1.0% | 0.48× |
| B9_technical_perf | 6.2% | 2.8% | 0.45× |

Accessibility does more than twice its declared share; events, SEO and technical performance do about half theirs. This is the project's central empirical claim and it is only visible by measuring realised influence — no amount of deliberation about weights would have surfaced it.
