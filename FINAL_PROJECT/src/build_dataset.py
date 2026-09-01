"""
build_dataset.py
================
Turns the raw crawl (`all_universities.csv.xls`, 1,230 rows x 85 columns) into the
modelling dataset used by the project:

    data/university_website_scores.csv   all universities, features + score + grade
    data/train.csv                       80% split
    data/test.csv                        20% split
    data/data_dictionary.csv             what every column means
    data/weka/*.arff                     the same splits for Weka

Run:  python src/build_dataset.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT.parent / "all_universities.csv.xls"
DATA = ROOT / "data"
(DATA / "weka").mkdir(parents=True, exist_ok=True)

RNG_SEED = 42

# ======================================================================================
# 1. LOAD AND CLEAN
# ======================================================================================

df = pd.read_csv(RAW)
audit: list[tuple[str, str]] = []
audit.append(("raw shape", f"{df.shape[0]} rows x {df.shape[1]} columns"))

# ---- drop columns that carry no usable information ------------------------------------
# `ok` is True everywhere, `extractor_version` is constant, `render_error` is entirely null.
DEAD = [c for c in ["ok", "extractor_version", "render_error"] if c in df.columns]
df = df.drop(columns=DEAD)
audit.append(("constant / empty columns dropped", ", ".join(DEAD)))

# `a62_load_speed_s` duplicates `load_time_s` in every row.
if {"load_time_s", "a62_load_speed_s"} <= set(df.columns):
    assert (df.load_time_s == df.a62_load_speed_s).all()
    df = df.drop(columns=["a62_load_speed_s"])
    audit.append(("exact duplicate column dropped", "a62_load_speed_s == load_time_s"))

# External ranking *values* are 95-99% missing and describe institutional prestige, not the
# website. They are dropped. The *badge* flags (a07, a09) are kept: whether a site displays a
# ranking badge is a property of the page.
PRESTIGE = [c for c in ["a08_qs_value", "a10_webometrics_value"] if c in df.columns]
nulls = {c: f"{df[c].isna().mean():.1%} null" for c in PRESTIGE}
df = df.drop(columns=PRESTIGE)
audit.append(("prestige value columns dropped", "; ".join(f"{k} ({v})" for k, v in nulls.items())))

# ---- duplicate domains -----------------------------------------------------------------
# Five universities were reached through two different entry URLs that resolve to the same
# page. Deduplication is done on the RESOLVED url, so `aun.edu.eg` and `aun.edu.eg/main` are
# recognised as one site. One row is kept; load time is set to the median of the pair.
_resolved = df.final_url.fillna(df.url)
df["_domain"] = (_resolved.str.replace(r"^https?://(www\.)?", "", regex=True)
                 .str.rstrip("/").str.lower())
dupes = df[df._domain.duplicated(keep=False)].sort_values("_domain")
if len(dupes):
    med = dupes.groupby("_domain").load_time_s.median()
    df.loc[df._domain.isin(med.index), "load_time_s"] = df._domain.map(med)
    df = df.drop_duplicates(subset="_domain", keep="first")
    audit.append(("duplicate domains merged", f"{len(dupes)} rows -> {len(dupes)//2} universities"))
df = df.drop(columns="_domain").reset_index(drop=True)

# ---- impossible dates ------------------------------------------------------------------
# 285 notice dates fall after the crawl date. They are censored to the crawl date rather than
# deleted, and the fact is recorded.
crawl = pd.to_datetime(df.fetched_at, errors="coerce", utc=True).dt.tz_localize(None)
notice = pd.to_datetime(df.a18_recent_notice_date, errors="coerce")
future = (notice > crawl).fillna(False)
notice = notice.where(~future, crawl)
df["notice_date_future"] = future.astype(int)
audit.append(("notice dates after the crawl date", f"{int(future.sum())} censored to the crawl date"))

# ---- country buckets --------------------------------------------------------------------
# Ten `country` values name a region rather than a country. They are either composite
# ("Egypt & North Africa", "Switzerland/Austria") or a named region.
NAMED_REGIONS = {"Sub-Saharan Africa", "Balkans", "Baltics", "Central Asia"}
df["country_is_bucket"] = (df.country.isin(NAMED_REGIONS)
                           | df.country.str.contains(r"[&/]", regex=True, na=False)).astype(int)
audit.append(("country field mixes levels", f"{int(df.country_is_bucket.sum())} rows carry a regional bucket label"))

audit.append(("clean shape", f"{df.shape[0]} rows x {df.shape[1]} columns"))

# ======================================================================================
# 2. MISSING VALUES: absence has a direction
# ======================================================================================
# Median imputation is wrong for these three. The median of the notice gap is 1 day, so
# median-filling would describe a site with NO dated notice as having posted yesterday --
# handing the strongest freshness signal to the sites that earned it least. Where absence has
# a known meaning, the feature is filled at the worst defensible value instead.

df["notice_recency_days"] = (crawl - notice).dt.days.clip(lower=0)

MISSING_POLICY = {
    "notice_recency_days": (3650.0, "no dated notice exists anywhere -> treat as ten years stale"),
    "a72_alt_text_pct":    (0.0,    "no image carries a text alternative -> 0% labelled"),
    "a53_contrast_ratio":  (1.0,    "no readable text block was found -> worst possible contrast"),
}
policy_log = {}
for col, (fill, why) in MISSING_POLICY.items():
    n = int(df[col].isna().sum())
    df[f"{col}_was_missing"] = df[col].isna().astype(int)   # computed BEFORE the fill
    policy_log[col] = {"n_filled": n, "fill_value": fill,
                       "median_would_have_been": round(float(df[col].median()), 2), "reason": why}
    df[col] = df[col].fillna(fill)
    audit.append((f"missing: {col}", f"{n} rows filled at {fill:g} (median would be {policy_log[col]['median_would_have_been']:g})"))

(DATA / "missing_value_policy.json").write_text(json.dumps(policy_log, indent=2), encoding="utf-8")

# ======================================================================================
# 3. ENGINEERED FEATURES
# ======================================================================================

def unit(x, lo, hi):
    """Linear rescale into [0, 1], clipped."""
    return np.clip((np.asarray(x, float) - lo) / (hi - lo), 0, 1)


# --- navigation breadth: an inverted U -----------------------------------------------
# A menu with 0-2 items cannot lead anywhere; 5-9 items is a well-organised site; 15+ items
# is a wall of links. Neither "more is better" nor "less is better" describes this, which is
# why it cannot be captured by a linear term.
def nav_breadth_curve(n):
    n = np.asarray(n, float)
    return np.where(n <= 0, 0.00,
           np.where(n <= 2, 0.30,
           np.where(n <= 4, 0.75,
           np.where(n <= 9, 1.00,
           np.where(n <= 12, 0.90,
           np.where(n <= 15, 0.75, 0.55))))))


# --- freshness: recent posts count, stale ones do not --------------------------------
def recency_curve(days):
    d = np.asarray(days, float)
    return np.where(d <= 7,   1.00,
           np.where(d <= 30,  0.90,
           np.where(d <= 90,  0.70,
           np.where(d <= 180, 0.45,
           np.where(d <= 365, 0.20, 0.00)))))


# --- events: a few dated events beat twenty undated ones -----------------------------
def event_curve(count, has_datetime):
    c = np.asarray(count, float)
    base = np.where(c <= 0, 0.00,
           np.where(c <= 2, 0.55,
           np.where(c <= 12, 1.00,
           np.where(c < 20, 0.85, 0.70))))     # 20 is the cap value: an undated dump
    return base * np.where(np.asarray(has_datetime, float) > 0, 1.0, 0.6)


# --- contrast: a plateau at the accessibility threshold ------------------------------
# WCAG AAA is 7:1. Beyond that, more contrast is not better -- 21:1 is not a better reading
# experience than 12:1 -- so the curve flattens instead of continuing to reward.
def contrast_curve(ratio):
    r = np.asarray(ratio, float)
    return np.where(r < 3.0, unit(r, 1.0, 3.0) * 0.30,
           np.where(r < 4.5, 0.30 + unit(r, 3.0, 4.5) * 0.30,
           np.where(r < 7.0, 0.60 + unit(r, 4.5, 7.0) * 0.40, 1.00)))


# --- alt text: concave, because the first images matter most -------------------------
def alt_text_curve(pct):
    return np.sqrt(unit(pct, 0, 100))


# --- broken links: tolerance for one, not for forty ----------------------------------
def broken_curve(n):
    n = np.asarray(n, float)
    return np.where(n <= 0, 1.00,
           np.where(n <= 2, 0.85,
           np.where(n <= 5, 0.65,
           np.where(n <= 15, 0.40,
           np.where(n <= 40, 0.15, 0.00)))))


# --- load time: judged against the same region ---------------------------------------
# Raw load time is confounded with who collected the data (each of the six collectors covered
# exactly one region, and regional medians span 8.1s to 12.9s). Standardising within region
# removes the part of the difference that is measurement rather than website.
df["load_time_pct_region"] = df.groupby("region").load_time_s.rank(pct=True)
df["load_time_z_region"] = df.groupby("region").load_time_s.transform(lambda s: (s - s.mean()) / (s.std() + 1e-9))
speed_score = 1.0 - df.load_time_pct_region     # faster than peers = better

# --- ordinal evidence features --------------------------------------------------------
# The raw flags contradict each other (382 rows carry a notice date but no board flag, 156 the
# reverse). Rather than leave the contradiction in, the three signals are reconciled into one
# ordinal: 0 = nothing, 1 = a board, 2 = a board with a timestamp, 3 = a dated recent post.
df["notice_evidence"] = (
    df.a16_notice_board.fillna(0).astype(int)
    + df.a17_notice_timestamp.fillna(0).astype(int)
    + (df.notice_recency_days <= 90).astype(int)
).clip(0, 3)

df["event_evidence"] = (
    df.a23_upcoming_events.fillna(0).astype(int)
    + (df.a24_event_count.fillna(0) > 0).astype(int)
    + df.a27_event_datetime.fillna(0).astype(int)
).clip(0, 3)

df["nav_quality"] = nav_breadth_curve(df.a03_nav_item_count.fillna(0))
df["broken_links_log"] = np.log1p(df.a66_broken_links.fillna(0))

ENGINEERED = ["notice_recency_days", "notice_evidence", "event_evidence", "nav_quality",
              "load_time_z_region", "load_time_pct_region", "broken_links_log",
              "notice_date_future", "country_is_bucket",
              "notice_recency_days_was_missing", "a72_alt_text_pct_was_missing",
              "a53_contrast_ratio_was_missing"]

# ======================================================================================
# 4. THE SCORING MODEL
# ======================================================================================
# Seven dimensions of what a prospective student needs from a university website. Each is a
# 0-1 sub-score built from named attributes; the final score is their weighted sum, subject
# to two gates. Weights sum to 100 and were fixed before any score was computed.

g = lambda c: df[c].fillna(0).astype(float).clip(0, 1)     # a binary attribute as 0/1

# ---- D1  Academic information (28) ----------------------------------------------------
D1 = (0.30 * g("a37_programs_listing")
      + 0.25 * g("a34_department_links")
      + 0.15 * g("a35_faculty_link")
      + 0.12 * g("a39_library_link")
      + 0.10 * g("a36_research_highlight")
      + 0.08 * g("a40_career_link"))

# ---- D2  Admission support (22) -------------------------------------------------------
D2 = (0.28 * g("a46_admissions_policy")
      + 0.22 * g("a22_admission_notice")
      + 0.20 * g("a38_scholarship")
      + 0.17 * g("a43_contact_link")
      + 0.13 * g("a45_prospectus"))

# ---- D3  Currency and activity (15) ---------------------------------------------------
D3 = (0.40 * recency_curve(df.notice_recency_days)
      + 0.25 * event_curve(df.a24_event_count.fillna(0), df.a27_event_datetime.fillna(0))
      + 0.20 * g("a20_news_events")
      + 0.15 * g("a21_calendar_link"))

# ---- D4  Navigation and findability (15) ----------------------------------------------
D4 = (0.25 * g("a02_primary_nav")
      + 0.25 * df.nav_quality
      + 0.20 * g("a04_search_bar")
      + 0.10 * g("a06_breadcrumb")
      + 0.10 * g("a51_quick_links")
      + 0.10 * g("a48_footer_sitemap"))

# ---- D5  Usability and accessibility (10) ---------------------------------------------
D5 = (0.35 * unit(df.a63_mobile_score.fillna(0), 0, 100)
      + 0.25 * contrast_curve(df.a53_contrast_ratio)
      + 0.20 * alt_text_curve(df.a72_alt_text_pct)
      + 0.10 * g("a73_accessible_design")
      + 0.10 * g("a05_language_toggle"))

# ---- D6  Technical quality (7) ---------------------------------------------------------
D6 = (0.35 * g("a65_https")
      + 0.35 * broken_curve(df.a66_broken_links.fillna(0))
      + 0.30 * speed_score)

# ---- D7  Institutional transparency (3) ------------------------------------------------
D7 = (0.30 * g("a32_vision_mission")
      + 0.20 * g("a33_about_blurb")
      + 0.20 * g("a47_footer_contact")
      + 0.15 * g("a11_accreditation")
      + 0.15 * g("a44_student_portal"))

WEIGHTS = {"D1_academic_information": 28, "D2_admission_support": 22,
           "D3_currency_activity": 15, "D4_navigation_findability": 15,
           "D5_usability_accessibility": 10, "D6_technical_quality": 7,
           "D7_institutional_transparency": 3}
assert sum(WEIGHTS.values()) == 100

for name, sub in zip(WEIGHTS, [D1, D2, D3, D4, D5, D6, D7]):
    df[name] = np.clip(sub, 0, 1)

raw_score = sum(WEIGHTS[k] * df[k] for k in WEIGHTS)

# ---- gates -----------------------------------------------------------------------------
# Two failures are severe enough to cap the achievable score no matter what else is present.
# A site with no main menu cannot be navigated; a site without HTTPS should not be trusted
# with an application form. Gates are what make the scoring function non-linear at the top.
cap = pd.Series(100.0, index=df.index)
cap = np.where(g("a02_primary_nav") == 0, np.minimum(cap, 45), cap)
cap = np.where(g("a65_https") == 0, np.minimum(cap, 60), cap)
df["gate_applied"] = ((g("a02_primary_nav") == 0) | (g("a65_https") == 0)).astype(int)

df["website_score"] = np.round(np.minimum(raw_score, cap), 2)

# Bands chosen so each grade names a distinguishable population rather than lumping
# 60% of sites into "A". Fixed before scoring, and stated in the report.
GRADES = [(85, "A+"), (75, "A"), (65, "B"), (50, "C"), (35, "D"), (-1, "F")]
df["grade"] = [next(g_ for lo, g_ in GRADES if s >= lo) for s in df.website_score]

audit.append(("score range", f"{df.website_score.min():.1f} to {df.website_score.max():.1f}"))
audit.append(("gates applied", f"{int(df.gate_applied.sum())} universities capped"))

# ======================================================================================
# 5. FINAL COLUMN SET AND SPLIT
# ======================================================================================

META = ["uni_id", "name", "url", "country", "region"]
LEAK = list(WEIGHTS) + ["website_score", "grade", "gate_applied"]
# Columns withheld from the model.
#
# The second group matters most. `nav_quality`, `notice_evidence`, `event_evidence` and
# `load_time_pct_region` are the *outputs of the scoring curves* -- handing them to a model
# would be handing it the answer. The model is given the RAW observable attributes instead
# (`a03_nav_item_count`, `notice_recency_days`, `a24_event_count`, ...) and has to recover
# the threshold, saturation and gate behaviour on its own. That is the learning task.
DROP = [
    # identifiers and crawl metadata -- not properties of the website
    "member", "final_url", "http_status", "fetched_at", "page_lang", "native_lang",
    "switched_to_english", "a18_recent_notice_date", "notice_date_future",
    "country_is_bucket",
    # confounded with the data collector (each collector covered exactly one region)
    "load_time_s",
    # intermediate outputs of the scoring curves -- would leak the scoring function
    "nav_quality", "notice_evidence", "event_evidence", "load_time_pct_region",
]

FEATURES = [c for c in df.columns
            if c not in META + LEAK + DROP and df[c].dtype != object]
FEATURES = sorted(FEATURES)

out = df[META + FEATURES + ["website_score", "grade"]].copy()
out = out.sort_values("website_score", ascending=False).reset_index(drop=True)
out.insert(5, "rank", np.arange(1, len(out) + 1))

# ranks within region and (real) country
out["regional_rank"] = out.groupby("region").website_score.rank(ascending=False, method="min").astype(int)
eligible = df.set_index("uni_id").country_is_bucket
out["_bucket"] = out.uni_id.map(eligible)
big = out[out._bucket == 0].country.value_counts()
out["country_rank"] = np.where(
    out.country.isin(big[big >= 20].index) & (out._bucket == 0),
    out.groupby("country").website_score.rank(ascending=False, method="min"), np.nan)
out = out.drop(columns="_bucket")

out.to_csv(DATA / "university_website_scores.csv", index=False)

# ---- 80 / 20 stratified split ----------------------------------------------------------
band = pd.cut(out.website_score, [-1, 35, 50, 65, 75, 85, 101], labels=False)
rng = np.random.default_rng(RNG_SEED)
test_idx = []
for b in sorted(band.unique()):
    idx = out.index[band == b].to_numpy()
    rng.shuffle(idx)
    test_idx.extend(idx[: int(round(0.20 * len(idx)))])
test_mask = out.index.isin(test_idx)

train, test = out[~test_mask].copy(), out[test_mask].copy()
COLS = META + FEATURES + ["website_score", "grade"]
train[COLS].to_csv(DATA / "train.csv", index=False)
test[COLS].to_csv(DATA / "test.csv", index=False)

audit.append(("train / test split", f"{len(train)} / {len(test)}  ({len(test)/len(out):.0%} test, stratified on score band, seed {RNG_SEED})"))

# ======================================================================================
# 6. DATA DICTIONARY
# ======================================================================================

DIM_OF = {}
for dim, cols in {
    "D1 academic information": ["a37_programs_listing", "a34_department_links", "a35_faculty_link",
                                 "a39_library_link", "a36_research_highlight", "a40_career_link"],
    "D2 admission support":    ["a46_admissions_policy", "a22_admission_notice", "a38_scholarship",
                                 "a43_contact_link", "a45_prospectus"],
    "D3 currency & activity":  ["notice_recency_days", "notice_evidence", "event_evidence",
                                 "a20_news_events", "a21_calendar_link", "a27_event_datetime"],
    "D4 navigation":           ["a02_primary_nav", "a03_nav_item_count", "nav_quality", "a04_search_bar",
                                 "a06_breadcrumb", "a51_quick_links", "a48_footer_sitemap"],
    "D5 usability & access":   ["a63_mobile_score", "a53_contrast_ratio", "a72_alt_text_pct",
                                 "a73_accessible_design", "a05_language_toggle", "a74_a11y_toggle"],
    "D6 technical quality":    ["a65_https", "a66_broken_links", "broken_links_log",
                                 "load_time_z_region", "load_time_pct_region", "a67_gzip"],
    "D7 transparency":         ["a32_vision_mission", "a33_about_blurb", "a47_footer_contact",
                                 "a11_accreditation", "a44_student_portal"],
}.items():
    for c in cols:
        DIM_OF[c] = dim

rows = []
for c in FEATURES:
    s = out[c]
    rows.append(dict(
        column=c,
        role="feature",
        dimension=DIM_OF.get(c, "not used by the scoring model"),
        dtype=str(s.dtype),
        min=round(float(s.min()), 3), max=round(float(s.max()), 3),
        mean=round(float(s.mean()), 3),
        engineered=int(c in ENGINEERED),
    ))
for c, role in [("website_score", "TARGET (0-100)"), ("grade", "target, banded"),
                ("rank", "output"), ("regional_rank", "output"), ("country_rank", "output")]:
    rows.append(dict(column=c, role=role, dimension="", dtype=str(out[c].dtype),
                     min="", max="", mean="", engineered=0))
pd.DataFrame(rows).to_csv(DATA / "data_dictionary.csv", index=False)

# ======================================================================================
# 7. ARFF EXPORT FOR WEKA
# ======================================================================================

def arff(path: Path, frame: pd.DataFrame, relation: str, target: str, target_type: str) -> int:
    cols = [c for c in FEATURES] + [target]
    f = frame[cols]
    lines = [f"% University Website Quality - {relation}",
             f"% {len(f)} instances, {len(FEATURES)} predictive attributes",
             f"% Target attribute: {target}", "", f"@relation {relation}", ""]
    for c in FEATURES:
        lines.append(f"@attribute {c} numeric")
    lines.append(f"@attribute {target} {target_type}")
    lines += ["", "@data"]
    for _, r in f.iterrows():
        vals = [f"{r[c]:g}" if pd.notna(r[c]) else "?" for c in FEATURES]
        vals.append(f"{r[target]:g}" if target_type == "numeric" else str(r[target]))
        lines.append(",".join(vals))
    path.write_text("\n".join(lines), encoding="utf-8")
    return len(f)


W = DATA / "weka"
GRADE_SET = "{A+,A,B,C,D,F}"
arff(W / "train.arff", train, "website_quality_train", "website_score", "numeric")
arff(W / "test.arff", test, "website_quality_test", "website_score", "numeric")
arff(W / "train_classification.arff", train, "website_grade_train", "grade", GRADE_SET)
arff(W / "test_classification.arff", test, "website_grade_test", "grade", GRADE_SET)
arff(W / "all_universities.arff", out, "website_quality_all", "website_score", "numeric")

# ======================================================================================
# 8. SUMMARY
# ======================================================================================

summary = dict(
    n_universities=int(len(out)),
    n_features=len(FEATURES),
    n_train=int(len(train)), n_test=int(len(test)),
    score_min=float(out.website_score.min()), score_max=float(out.website_score.max()),
    score_mean=float(out.website_score.mean()), score_sd=float(out.website_score.std()),
    grade_counts={k: int(v) for k, v in out.grade.value_counts().items()},
    dimension_weights=WEIGHTS,
    n_gated=int(df.gate_applied.sum()),
    n_countries_ranked=int(out.country_rank.notna().sum()),
    seed=RNG_SEED,
)
(DATA / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

print("=" * 78)
print("DATA PREPARATION LOG")
print("=" * 78)
for k, v in audit:
    print(f"  {k:<38} {v}")
print("=" * 78)
print(f"  features retained                      {len(FEATURES)}")
print(f"  score  mean {out.website_score.mean():.1f}  sd {out.website_score.std():.1f}"
      f"  range {out.website_score.min():.1f}-{out.website_score.max():.1f}")
print(f"  grades  {dict(out.grade.value_counts())}")
print("=" * 78)
print("written to data/:")
for p in sorted(DATA.rglob("*")):
    if p.is_file():
        print(f"  {p.relative_to(DATA)}")
