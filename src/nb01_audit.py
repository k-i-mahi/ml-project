# %% [markdown]
# # 01 — Forensic Dataset Audit
#
# Full audit of `all_universities.csv` against the attribute schema (Lab 3 report, Tables 2–3).
# Every finding is recorded in `EXPECTED / ACTUAL / PROBLEM / FIX` form. **Nothing is silently
# corrected** — each applied fix is logged with the exact action taken.
#
# **Outputs:** `outputs/audit_report.csv`, `outputs/cleaned_dataset.csv`, `outputs/assumptions.md`

# %%
import pathlib, warnings, json
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 60)

ROOT = pathlib.Path.cwd()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

RAW = ROOT / "all_universities.csv.xls"
raw = pd.read_csv(RAW, encoding="utf-8-sig")
print(f"Loaded {RAW.name}: {raw.shape[0]:,} rows x {raw.shape[1]} columns")

# %% [markdown]
# ## 1. Audit ledger
#
# Findings accumulate here. `severity`: **critical** (blocks modelling), **major** (biases
# results), **minor** (documentation / hygiene).

# %%
FINDINGS = []

def finding(fid, severity, category, scope, expected, actual, problem, fix, action="pending"):
    FINDINGS.append(dict(id=fid, severity=severity, category=category, scope=scope,
                         expected=expected, actual=actual, problem=problem,
                         fix=fix, action=action))

# %% [markdown]
# ## 2. Column groups
#
# `basic.md` requires the seven groups be kept conceptually separate before any modelling.

# %%
ID_COLS      = ["uni_id", "name", "url", "final_url"]
CONTEXT_COLS = ["member", "region", "country", "page_lang", "native_lang", "switched_to_english"]
CRAWL_COLS   = ["http_status", "ok", "fetched_at", "extractor_version", "load_time_s", "render_error"]
ATTR_COLS    = [c for c in raw.columns if c[:1] == "a" and c[1:3].isdigit()]
EXTERNAL_COLS = ["a08_qs_value", "a10_webometrics_value"]   # prestige benchmark values

groups = pd.DataFrame([
    ("C  Identification",        len(ID_COLS),       ", ".join(ID_COLS)),
    ("B  University / context",  len(CONTEXT_COLS),  ", ".join(CONTEXT_COLS)),
    ("D  Crawl / extraction",    len(CRAWL_COLS),    ", ".join(CRAWL_COLS)),
    ("A  Website-quality attrs", len(ATTR_COLS),     f"a01...a75 ({len(ATTR_COLS)} retained)"),
    ("E  External benchmark",    len(EXTERNAL_COLS), ", ".join(EXTERNAL_COLS)),
    ("F  Target / label",        0,                  "NONE - no label column exists (built in nb04/nb05)"),
    ("G  Model outputs",         0,                  "not yet produced"),
], columns=["group", "n", "columns"])
print(groups.to_string(index=False))
print(f"\nTotal accounted: {len(ID_COLS)+len(CONTEXT_COLS)+len(CRAWL_COLS)+len(ATTR_COLS)} of {raw.shape[1]}")

# %% [markdown]
# ### Attribute block partition (report §6.2, blocks B1–B11)

# %%
BLOCK_PREFIX = {
    "B1_header_nav":       ["a01", "a02", "a03", "a04", "a05", "a06"],
    "B2_rankings_recog":   ["a07", "a08", "a09", "a10", "a11", "a12", "a13", "a14", "a15"],
    "B3_notices_updates":  ["a16", "a17", "a18", "a20", "a21", "a22"],
    "B4_events_media":     ["a23", "a24", "a25", "a26", "a27", "a28", "a29", "a30", "a31"],
    "B5_page_content":     [f"a{i:02d}" for i in range(32, 47)],
    "B6_footer":           ["a47", "a48", "a49", "a50", "a51"],
    "B7_visual_design":    ["a53", "a54", "a57"],
    "B8_service_interact": ["a58", "a59", "a60", "a61"],
    "B9_technical_perf":   ["a62", "a63", "a65", "a66", "a67"],
    "B10_seo_metadata":    ["a69", "a70", "a71"],
    "B11_accessibility":   ["a72", "a73", "a74", "a75"],
}
COL_BLOCK, block_rows = {}, []
for blk, prefs in BLOCK_PREFIX.items():
    cols = [c for c in ATTR_COLS if c.split("_")[0] in prefs]
    for c in cols:
        COL_BLOCK[c] = blk
    block_rows.append((blk, len(prefs), len(cols)))
bt = pd.DataFrame(block_rows, columns=["block", "spec_n", "found_n"])
bt["ok"] = bt.spec_n == bt.found_n
print(bt.to_string(index=False))
print(f"\nTotal mapped {bt.found_n.sum()} / {len(ATTR_COLS)} attributes - all blocks complete: {bt.ok.all()}")

# Crosswalk to the Saleh et al. (2022) six SLR factors, for literature framing.
SLR_CROSSWALK = {
    "B1_header_nav": "Usability", "B7_visual_design": "Web appearance",
    "B3_notices_updates": "Information quality", "B5_page_content": "Specific content",
    "B4_events_media": "Specific content", "B6_footer": "Usability",
    "B8_service_interact": "Service interaction quality", "B9_technical_perf": "Functionality",
    "B10_seo_metadata": "Functionality", "B11_accessibility": "Functionality",
    "B2_rankings_recog": "Information quality",
}
json.dump({"block_prefix": BLOCK_PREFIX, "col_block": COL_BLOCK, "slr": SLR_CROSSWALK},
          open(OUT / "block_map.json", "w"), indent=2)
print(f"\nWrote {OUT/'block_map.json'}")

# %% [markdown]
# ## 3. Missing attribute numbers — intentional or scraper loss?
#
# The schema numbers attributes a01–a75, but only 69 are present. The report (§4.6) states six
# attributes were **excluded from the original 75-attribute specification**, which confirms the
# gaps are a design decision rather than extraction failure.

# %%
present_nums = sorted(int(c[1:3]) for c in ATTR_COLS)
gaps = [n for n in range(1, 76) if n not in present_nums]
print(f"Attributes present: {len(present_nums)}   Gap numbers: {gaps}")
print(f"75 - {len(gaps)} = {75-len(gaps)}  -> matches retained count: {75-len(gaps)==len(ATTR_COLS)}")
finding("F01", "minor", "schema", f"a{gaps}",
        "75-attribute specification", f"69 attributes present; {gaps} absent",
        "Gaps could indicate silent scraper loss.",
        "Report §4.6 documents six intentional exclusions (75-6=69). No action needed.",
        "resolved-no-action")

# %% [markdown]
# ## 4. Duplicates

# %%
domain = raw.url.str.replace(r"^https?://(www\.)?", "", regex=True).str.split("/").str[0].str.lower()
raw = raw.assign(_domain=domain)

print(f"Duplicate uni_id : {raw.uni_id.duplicated().sum()}")
print(f"Duplicate name   : {raw.name.duplicated().sum()}")
print(f"Duplicate domain : {raw._domain.duplicated().sum()}")

dup_domains = raw._domain.value_counts()[lambda s: s > 1].index.tolist()
dups = raw[raw._domain.isin(dup_domains)].sort_values("_domain")
print("\n--- Duplicated institutions ---")
print(dups[["uni_id", "member", "region", "country", "name", "load_time_s", "_domain"]].to_string(index=False))

# %% [markdown]
# ### Inter-collector agreement — an unplanned reliability experiment
#
# Three of these pairs were crawled independently by **two different collectors**, which makes them
# a free test–retest of the extractor.

# %%
agree_rows = []
for dm, g in dups.groupby("_domain"):
    disagree = [c for c in ATTR_COLS if g[c].nunique(dropna=False) > 1]
    agree_rows.append((dm, g.member.nunique(), len(ATTR_COLS) - len(disagree), len(ATTR_COLS), disagree))
agree = pd.DataFrame(agree_rows, columns=["domain", "n_collectors", "n_agree", "n_attrs", "disagreements"])
print(agree.to_string(index=False))
cross = agree[agree.n_collectors > 1]
AGREE_PCT = 100 * cross.n_agree.sum() / cross.n_attrs.sum()
print(f"\nCross-collector pairs: {len(cross)}")
print(f"Attribute agreement on cross-collector re-crawls: "
      f"{cross.n_agree.sum()}/{cross.n_attrs.sum()} = {AGREE_PCT:.1f}%")
print("The ONLY disagreeing attribute:", sorted({c for d in agree.disagreements for c in d}))

lt = dups.groupby("_domain").load_time_s.agg(["min", "max"])
lt["ratio"] = (lt["max"] / lt["min"]).round(2)
print("\nLoad-time spread on identical websites:")
print(lt.to_string())

finding("F02", "major", "reliability", "load_time_s / a62_load_speed_s",
        "Same website -> same measured load time",
        f"Re-crawls disagree by up to {lt.ratio.max():.1f}x (UNAM 13.46s vs 4.07s) while agreeing "
        f"on {AGREE_PCT:.0f}% of all other attributes",
        "Load time measures the collector's network conditions, not the website. It is also "
        "perfectly confounded with region (one collector per region).",
        "Region-standardise (z-score within region); report all rankings with AND without it.",
        "deferred-to-nb02")

# %% [markdown]
# ### Duplicate resolution rule
#
# Three Mexican universities (UNAM, IPN, Tec de Monterrey) were collected twice: once correctly by
# M6 (Latin America, `country=Mexico`) and once by M1 with **`country` mislabelled as United
# States**. Assiut University was entered twice by the same collector under two URL forms.
#
# **Rule:** keep the geographically correct row; set `load_time_s` to the median of the pair
# (neither measurement is privileged); log every drop.

# %%
TLD_COUNTRY = {"mx": "Mexico", "eg": "Egypt & North Africa"}
drop_ids, resolution_log = [], []
for dm, g in dups.groupby("_domain"):
    tld = dm.rsplit(".", 1)[-1]
    expected_country = TLD_COUNTRY.get(tld)
    keep = g[g.country == expected_country] if expected_country is not None else g
    keep_id = int(keep.uni_id.min()) if len(keep) else int(g.uni_id.min())
    med = float(g.load_time_s.median())
    drop_ids += [int(x) for x in g.uni_id if int(x) != keep_id]
    mislabelled = expected_country is not None and bool((g.country != expected_country).any())
    resolution_log.append(dict(
        domain=dm, kept_uni_id=keep_id,
        dropped=[int(x) for x in g.uni_id if int(x) != keep_id],
        reason="mislabelled country on dropped row" if mislabelled else "same-collector duplicate entry",
        load_time_median=round(med, 3)))
print(json.dumps(resolution_log, indent=2))

finding("F03", "critical", "duplicates", "4 domains / 8 rows",
        "One row per university", f"{len(dup_domains)} domains appear twice ({len(dups)} rows)",
        "Duplicated institutions double-weight those universities and would appear twice in the "
        "final ranking. Three carry a mislabelled country (Mexican universities filed as US).",
        "Keep the geographically correct row, median the load time, drop the other 4 rows.",
        "applied")

# %% [markdown]
# ## 5. Constant, near-constant and redundant columns

# %%
nun = raw.nunique(dropna=False)
constants = [c for c in raw.columns if nun[c] <= 1 and c != "_domain"]
print("Constant / empty columns:", constants)
for c in constants:
    actual = (f"single value: {raw[c].dropna().unique()[:1]}" if raw[c].notna().any() else "100% null")
    finding(f"F04.{c}", "minor", "no-variance", c, "Discriminative column", actual,
            "Zero variance - carries no information for any model.", "Drop.", "applied")

identical = bool((raw.load_time_s.fillna(-1) == raw.a62_load_speed_s.fillna(-1)).all())
print(f"\nload_time_s identical to a62_load_speed_s in all rows: {identical}")
finding("F05", "major", "redundancy", "a62_load_speed_s vs load_time_s",
        "Distinct measurements", "Byte-identical in all 1,230 rows",
        "Perfectly collinear duplicate. It inflates the technical block's apparent width and would "
        "double-count load time under any block-equal weighting.",
        "Drop a62_load_speed_s; retain load_time_s (region-standardised in nb02).", "applied")

binaries = [c for c in ATTR_COLS if set(pd.unique(raw[c].dropna())) <= {0, 1}]
prev = raw[binaries].mean().sort_values()
near_const = prev[(prev < 0.03) | (prev > 0.97)]
print(f"\nBinary attributes: {len(binaries)}   Near-constant (<3% or >97%):")
print((near_const * 100).round(1).to_string())
finding("F06", "minor", "near-constant", ", ".join(near_const.index),
        "Attributes that discriminate between universities",
        "; ".join(f"{k}={v*100:.1f}%" for k, v in near_const.items()),
        "Almost no discriminative information. a65_https at 98.9% is a hygiene floor, not a "
        "differentiator - weighting it normally overstates its contribution.",
        "Retain in the dataset (trees are unharmed) but FLAG; a65_https becomes a rubric GATE "
        "rather than a scored feature.", "flagged")

# %% [markdown]
# ## 6. Missing values — and what a null actually means
#
# `basic.md`: **NULL IS NOT ZERO**. Treatment is decided per feature, never by `fillna(0)`.

# %%
miss = pd.DataFrame({"n_missing": raw.isna().sum(), "pct": (raw.isna().mean() * 100).round(2)})
miss = miss[miss.n_missing > 0].sort_values("pct", ascending=False)
print(miss.to_string())

finding("F07", "critical", "missingness", "a10_webometrics_value",
        "Integer 1-30000 where displayed",
        f"{raw.a10_webometrics_value.isna().mean()*100:.2f}% null "
        f"({raw.a10_webometrics_value.notna().sum()} usable values)",
        "Unusable: 8 values cannot support any analysis. Also an external prestige benchmark, "
        "excluded from the primary model by design.", "Drop from the modelling dataset.", "applied")
finding("F08", "critical", "missingness", "a08_qs_value",
        "Integer 1-2000 where displayed", f"{raw.a08_qs_value.isna().mean()*100:.2f}% null",
        "Unusable AND prestige-leaking: the project's core claim requires ranking on website "
        "quality independently of institutional reputation.", "Drop from the modelling dataset.",
        "applied")
finding("F09", "major", "missingness", "a18_recent_notice_date",
        "ISO-8601 date or null", f"{raw.a18_recent_notice_date.isna().mean()*100:.2f}% null",
        "Missingness is INFORMATIVE - no dated notice found is itself a freshness signal, not an "
        "absent measurement. Imputing a date would erase that signal.",
        "Keep null; derive notice_recency_days plus an explicit a18_missing indicator.",
        "deferred-to-nb02")

# %% [markdown]
# ## 7. Range and type validation against the schema
#
# Schema ranges taken from report Table 2 / Table 3.

# %%
SCHEMA_RANGE = {
    "a03_nav_item_count": (0, 20, "int"), "a08_qs_value": (1, 2000, "int-nullable"),
    "a10_webometrics_value": (1, 30000, "int-nullable"), "a12_accred_count": (0, 10, "int"),
    "a15_stat_item_count": (0, 10, "int"), "a24_event_count": (0, 20, "int"),
    "a53_contrast_ratio": (1, 21, "float"), "a63_mobile_score": (0, 100, "float"),
    "a66_broken_links": (0, np.inf, "int"), "a72_alt_text_pct": (0, 100, "float"),
}
rows = []
for c, (lo, hi, typ) in SCHEMA_RANGE.items():
    s = pd.to_numeric(raw[c], errors="coerce")
    rows.append(dict(column=c, schema=f"[{lo}, {hi}] {typ}", n_missing=int(s.isna().sum()),
                     observed_min=s.min(), observed_max=s.max(),
                     n_below=int((s < lo).sum()), n_above=int((s > hi).sum())))
rng = pd.DataFrame(rows)
print(rng.to_string(index=False))
viol = rng[(rng.n_below > 0) | (rng.n_above > 0)]
print(f"\nRange violations: {len(viol)} column(s)"
      + ("" if len(viol) else " - all attributes lie within schema bounds."))

for c in binaries:
    bad = set(pd.unique(raw[c].dropna())) - {0, 1}
    if bad:
        finding(f"F10.{c}", "major", "range", c, "{0,1}", f"also contains {bad}",
                "Non-binary values in a Boolean attribute.", "Investigate before use.", "flagged")

# %% [markdown]
# ### Caps that are extractor artefacts, not true values
#
# Several counts pile up exactly on their schema maximum. A value sitting on the cap means
# "at least N", not "exactly N", and must not be read as a continuous measurement.

# %%
for c, cap in [("a03_nav_item_count", 20), ("a15_stat_item_count", 10), ("a24_event_count", 20)]:
    n_at = int((raw[c] == cap).sum())
    print(f"{c:24s} at cap {cap:>3}: {n_at:>4} rows ({100*n_at/len(raw):.1f}%)")
finding("F11", "major", "censoring", "a03_nav_item_count, a15_stat_item_count, a24_event_count",
        "Uncensored counts", "Mass piles up exactly on the schema maximum (20 / 10 / 20)",
        "Right-censored at the extractor cap. Treating the cap as a true count makes 'more' look "
        "monotonically better when the cap actually signals an undifferentiated dump.",
        "Apply non-linear (inverted-U / diminishing-return) rubric curves rather than linear "
        "scaling; document the cap in the feature dictionary.", "deferred-to-nb02/04")

# %% [markdown]
# ### `a63_mobile_score` is an ordinal heuristic, not a Lighthouse score

# %%
print(raw.a63_mobile_score.value_counts().sort_index().to_string())
top3 = raw.a63_mobile_score.value_counts(normalize=True).head(3)
print(f"\nTop-3 values cover {top3.sum()*100:.1f}% of rows: {list(top3.index)}")
finding("F12", "major", "measurement", "a63_mobile_score",
        "Continuous 0-100 responsiveness score",
        f"{raw.a63_mobile_score.nunique()} distinct values; {top3.sum()*100:.1f}% fall on just "
        f"{list(top3.index)}",
        "Presented as continuous but actually a coarse rule-based heuristic. Scaling it as "
        "continuous implies precision the measurement does not have.",
        "Recode as a 6-level ORDINAL feature.", "deferred-to-nb02")

# %% [markdown]
# ### `a66_broken_links` — the zero-vs-null question
#
# This governs 81% of rows, and with no denominator column a rate cannot be computed.

# %%
b = raw.a66_broken_links
print(f"zeros: {(b==0).sum()} ({(b==0).mean()*100:.1f}%)   max: {b.max()}   "
      f"2nd max: {b.nlargest(2).iloc[-1]}   nulls: {b.isna().sum()}")
print(f"Crawl succeeded everywhere?  ok all True: {raw.ok.all()};  "
      f"render_error all null: {raw.render_error.isna().all()}")
finding("F13", "critical", "semantics", "a66_broken_links",
        "Count of broken links, with a denominator to form a rate",
        f"{(b==0).mean()*100:.1f}% zeros, no total-links column, one extreme value of {int(b.max())} "
        f"against a second-highest of {int(b.nlargest(2).iloc[-1])}",
        "Zero is ambiguous between 'checked, none found' and 'checker did not run'. Without a "
        "denominator, 2 broken links on a 20-link page cannot be distinguished from 2 on a "
        "500-link page.",
        "Read 0 as verified-absence (ok=True and render_error=null for EVERY row - no evidence of "
        "checker failure anywhere in the crawl metadata). Use log1p + winsorisation at p99 plus a "
        "broken_links_present binary. Absence of a denominator is a permanent documented "
        "limitation.", "assumption-documented")

# %% [markdown]
# ## 8. Impossible dates

# %%
notice = pd.to_datetime(raw.a18_recent_notice_date, errors="coerce")
fetched = pd.to_datetime(raw.fetched_at, errors="coerce", utc=True).dt.tz_localize(None)
future = notice > fetched
print(f"Parsed notice dates : {notice.notna().sum()}")
print(f"After the crawl date: {int(future.sum())}  "
      f"({future.sum()/notice.notna().sum()*100:.1f}% of dated rows)")
print(f"Latest notice date  : {notice.max().date()}   Latest crawl: {fetched.max().date()}")
finding("F14", "critical", "impossible-value", "a18_recent_notice_date",
        "Notice date <= crawl date",
        f"{int(future.sum())} dates after the crawl, up to {notice.max().date()}",
        "A notice cannot be posted after it was observed. Left uncorrected these produce negative "
        "recency, i.e. universities scoring as fresher than physically possible.",
        "CENSOR to recency=0 and flag with notice_date_future - do not delete the rows; the "
        "presence of a dated notice is still real information.", "deferred-to-nb02")

# %% [markdown]
# ## 9. Internally inconsistent field combinations

# %%
d_present = notice.notna()
inc = pd.DataFrame([
    ("notice date present but a16_notice_board=0", int(((raw.a16_notice_board == 0) & d_present).sum())),
    ("a16_notice_board=1 but no notice date",      int(((raw.a16_notice_board == 1) & ~d_present).sum())),
    ("a17_notice_timestamp=1 but no notice date",  int(((raw.a17_notice_timestamp == 1) & ~d_present).sum())),
    ("a23_upcoming_events=1 but a24_event_count=0", int(((raw.a23_upcoming_events == 1) & (raw.a24_event_count == 0)).sum())),
    ("a23_upcoming_events=0 but a24_event_count>0", int(((raw.a23_upcoming_events == 0) & (raw.a24_event_count > 0)).sum())),
], columns=["inconsistency", "n_rows"])
print(inc.to_string(index=False))
finding("F15", "major", "inconsistency", "a16 / a17 / a18",
        "Notice-board flag, timestamp flag and date agree",
        f"{int(((raw.a16_notice_board==0)&d_present).sum())} rows have a date with the board flag "
        f"off; {int(((raw.a16_notice_board==1)&~d_present).sum())} have the board flag on with no date",
        "The three notice fields disagree on whether a notice board exists. Using them as three "
        "independent binaries triple-counts a contradictory signal.",
        "Collapse into ONE ordinal notice_evidence (0 none / 1 undated board / 2 dated board / "
        "3 dated and recent) that reconciles the contradiction explicitly.", "deferred-to-nb02")
finding("F16", "major", "inconsistency", "a23 / a24",
        "Events flag agrees with the event count",
        f"{int(((raw.a23_upcoming_events==1)&(raw.a24_event_count==0)).sum())} rows flag events "
        f"with count 0; {int(((raw.a23_upcoming_events==0)&(raw.a24_event_count>0)).sum())} count "
        "events with the flag off",
        "Same double-counting problem as the notice fields.",
        "Collapse into one ordinal event_evidence.", "deferred-to-nb02")

# %% [markdown]
# ## 10. The largest bias risk: collector is perfectly confounded with region

# %%
print(pd.crosstab(raw.member, raw.region).to_string())
one_to_one = bool((raw.groupby("member").region.nunique() == 1).all()
                  and (raw.groupby("region").member.nunique() == 1).all())
print(f"\nEach collector maps to exactly one region and vice versa: {one_to_one}")
print("\nMedian load time by collector / region:")
print(raw.groupby(["member", "region"]).load_time_s.median().round(2).to_string())
finding("F17", "critical", "confounding", "member x region",
        "Collectors overlap across regions so their effects can be separated",
        "Perfect 1:1 mapping - zero overlap between collectors and regions",
        "'African websites are slower' and 'M6 had a slower connection' are statistically "
        "indistinguishable in this design. No post-hoc method can separate them.",
        "Report as an UNRESOLVABLE limitation. Mitigate with region-standardised load time and "
        "leave-one-region-out validation; never claim a causal regional difference in any "
        "timing-derived measure.", "documented-limitation")

# %% [markdown]
# ## 11. `country` mixes countries with multi-country buckets

# %%
BUCKETS = ["Sub-Saharan Africa", "Egypt & North Africa", "Belgium/Ireland/Iceland",
           "Switzerland/Austria", "Peru & Andes", "Balkans", "Central Asia",
           "Bulgaria & Slovakia", "Baltics", "Other SE/E Asia"]
is_bucket = raw.country.isin(BUCKETS)
print(f"Distinct country labels: {raw.country.nunique()}   Bucket labels: {len(BUCKETS)}  "
      f"covering {int(is_bucket.sum())} rows ({is_bucket.mean()*100:.1f}%)")
vc = raw.country.value_counts()
eligible = [c for c in vc[vc >= 20].index if c not in BUCKETS]
print(f"\nCountries with >=20 universities AND not a bucket: {len(eligible)}")
print(eligible)
finding("F18", "major", "categorical", "country",
        "One country per row",
        f"{len(BUCKETS)} labels are multi-country buckets covering {int(is_bucket.sum())} rows",
        "A 'country ranking' over a bucket such as Sub-Saharan Africa (42 universities across many "
        "countries) is not a country ranking at all.",
        f"Add country_is_bucket; publish country-level rankings ONLY for the {len(eligible)} real "
        "countries with >=20 universities.", "applied")

# %% [markdown]
# ## 12. Leakage scan

# %%
leak = pd.DataFrame([
    ("a08_qs_value", "prestige", "Displayed QS rank - institutional reputation, not website quality", "dropped"),
    ("a10_webometrics_value", "prestige", "Displayed Webometrics rank - external benchmark", "dropped"),
    ("a07_qs_badge", "none", "Whether the site DISPLAYS a badge - a design choice, not a rank", "kept, lowest tier"),
    ("a09_national_rank", "none", "Whether the site DISPLAYS a national-rank mention", "kept, lowest tier"),
    ("uni_id / name / url / final_url", "identity", "Row identifiers", "never predictors"),
    ("member", "collector identity", "Confounded with region; a model could learn the collector", "context only"),
    ("region / country", "geography", "Legitimate for stratification and fairness testing", "context only"),
], columns=["column", "leakage_type", "reasoning", "decision"])
print(leak.to_string(index=False))
finding("F19", "critical", "leakage", "a07_qs_badge, a09_national_rank",
        "Prestige variables excluded from the quality model",
        "Both are binary PRESENCE flags, not rank values",
        "Easy to over-correct: these measure whether the site displays a credibility signal, which "
        "is a genuine (if weak and trivially gamed) website attribute. Dropping them discards real "
        "website information; weighting them normally smuggles in prestige.",
        "KEEP as website features but assign the lowest rubric tier (near-zero weight).",
        "kept-lowest-tier")

# %% [markdown]
# ## 13. Apply the fixes → `cleaned_dataset.csv`

# %%
clean = raw.copy()
n0 = len(clean)

# (a) duplicate resolution: median the load time onto the kept row, drop the redundant rows
for entry in resolution_log:
    clean.loc[clean.uni_id == entry["kept_uni_id"], "load_time_s"] = entry["load_time_median"]
clean = clean[~clean.uni_id.isin(drop_ids)].copy()
print(f"Duplicate rows dropped: {n0-len(clean)}  ->  {len(clean):,} rows")

# (b) drop constant, redundant and unusable/leaky columns
DROP_COLS = [c for c in constants + ["a62_load_speed_s"] + EXTERNAL_COLS + ["_domain"]
             if c in clean.columns]
clean = clean.drop(columns=DROP_COLS)
print(f"Columns dropped ({len(DROP_COLS)}): {DROP_COLS}")

# (c) audit flags carried forward (derived features themselves are built in nb02)
_notice = pd.to_datetime(clean.a18_recent_notice_date, errors="coerce")
_fetch = pd.to_datetime(clean.fetched_at, errors="coerce", utc=True).dt.tz_localize(None)
clean["notice_date_future"] = (_notice > _fetch).fillna(False).astype(int)
clean["a18_missing"] = _notice.isna().astype(int)
clean["country_is_bucket"] = clean.country.isin(BUCKETS).astype(int)
_vc = clean.country.value_counts()
clean["country_rank_eligible"] = (clean.country.map(_vc).ge(20)
                                  & ~clean.country.isin(BUCKETS)).astype(int)

n_attr_clean = len([c for c in clean.columns if c[:1] == "a" and c[1:3].isdigit()])
print(f"\nCleaned dataset: {clean.shape[0]:,} rows x {clean.shape[1]} columns")
print(f"Attributes retained: {n_attr_clean}")
print(f"Country-rank-eligible universities: {int(clean.country_rank_eligible.sum())} "
      f"across {clean.loc[clean.country_rank_eligible==1,'country'].nunique()} countries")

clean.to_csv(OUT / "cleaned_dataset.csv", index=False)
print(f"\nWrote {OUT/'cleaned_dataset.csv'}")

# %% [markdown]
# ## 14. Audit report

# %%
audit = pd.DataFrame(FINDINGS)
audit = audit.sort_values("severity", key=lambda s: s.map({"critical": 0, "major": 1, "minor": 2}))
audit = audit.reset_index(drop=True)
audit.to_csv(OUT / "audit_report.csv", index=False)
print(audit.severity.value_counts().to_string())
print(f"\nWrote {OUT/'audit_report.csv'} ({len(audit)} findings)\n")
print(audit[["id", "severity", "category", "scope", "action"]].to_string(index=False))

# %% [markdown]
# ## 15. Documented assumptions
#
# Every inference made where the schema is silent, together with the risk if it is wrong.

# %%
ASSUMPTIONS = f"""# Documented Assumptions

Generated by `01_audit.ipynb`. Each entry records an inference made where the schema
(Lab 3 report, Tables 2-3) is silent, together with the risk if the inference is wrong.

## A1 - `a66_broken_links = 0` means "checked, none found"

**Basis.** `ok` is True and `render_error` is null for all {n0:,} rows. No field anywhere in the
crawl metadata records a link-checker failure, and no row shows any other symptom of a partial
crawl.

**Risk if wrong.** If some zeros are silent checker failures, {(b==0).mean()*100:.1f}% of rows
carry a false "perfect reliability" signal, biasing those universities upward.

**Mitigation.** `a66` enters only via `log1p` plus a `broken_links_present` binary, never as a
large linear term, so the cost of the assumption is bounded.

## A2 - No denominator exists for broken links

`a66` is an absolute count with no total-link column, so a broken-link *rate* cannot be computed.
Two broken links on a 20-link page and on a 500-link page are indistinguishable. **Permanent
limitation** - not repairable without re-crawling.

## A3 - Load time measures the collector, not the website

Cross-collector re-crawls of identical sites disagree by up to {lt.ratio.max():.1f}x while agreeing
on {AGREE_PCT:.0f}% of all other attributes. Load time is therefore treated as instrumentation:
region-standardised, with every ranking reported both with and without it.

## A4 - Count attributes are right-censored at their extractor cap

Values sitting exactly on 20 / 10 / 20 mean "at least N". Non-linear rubric curves replace linear
scaling so that hitting the cap does not read as maximum quality.

## A5 - Missing `a18_recent_notice_date` is informative

{raw.a18_recent_notice_date.isna().mean()*100:.1f}% null. Absence of a dated notice is evidence
about the website (nothing dated was found), not an absent measurement. Kept null and represented
by an explicit indicator; never imputed.

## A6 - The six absent attribute numbers are intentional

`{gaps}` - report §4.6 documents six exclusions from the original 75-attribute specification
(75 - 6 = 69). Not scraper loss; no action taken.

## A7 - Displayed-badge attributes are website features, not prestige

`a07_qs_badge` and `a09_national_rank` record whether the site *displays* a credibility signal,
which is a design choice. Retained but confined to the lowest rubric tier. The prestige *values*
(`a08`, `a10`) are dropped outright.

## A8 - Collector and region cannot be separated

Perfect 1:1 mapping. Any apparent regional difference in a timing-derived measure is
uninterpretable. Reported as an unresolvable limitation of the collection design.
"""
(OUT / "assumptions.md").write_text(ASSUMPTIONS, encoding="utf-8")
print(ASSUMPTIONS)

# %% [markdown]
# ## Summary
#
# **Still no label column.** Nothing in this audit changes that — the target is built in nb04/nb05.

# %%
print(f"rows      : {n0:,} -> {len(clean):,}")
print(f"columns   : {raw.shape[1]-1} -> {clean.shape[1]}")
print(f"attributes: {len(ATTR_COLS)} -> {n_attr_clean}")
print("\nArtefacts written:")
for f in ["cleaned_dataset.csv", "audit_report.csv", "assumptions.md", "block_map.json"]:
    print(f"  outputs/{f}  ({(OUT/f).stat().st_size:,} bytes)")
