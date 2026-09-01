"""Two corrections, applied together.

(1) MISSING = ABSENT, NOT AVERAGE.
    A NaN in this dataset does not mean "we failed to measure it". It means the website
    does not have the thing. Median-imputing such a value scores an absence as if it were
    typical, which is exactly backwards. Every NaN is therefore replaced with the explicit
    worst-case value, and the accompanying _missing flag is kept so the model can still
    tell "absent" apart from "present but bad".

(2) RE-RENDER THE PROFILE CARDS FROM A PROSPECTIVE STUDENT'S POINT OF VIEW.
    The original cards led with alt-text percentage and contrast ratio, and 38% of the
    900 judgments ended up citing alt-text -- a property a sighted applicant cannot
    perceive. The new cards are organised around the questions someone actually asks when
    deciding where to apply, and demote the invisible technical metrics to a single line.

The 200 universities and the 900 pairs are deliberately UNCHANGED, so the old and new
labels can be compared directly and the effect of the persona change can be measured.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
BATCH = OUT / "trackB_batches_v2"
BATCH.mkdir(exist_ok=True)

# ══════════════════════════════════════════════════ 1. missing-value semantics
data = pd.read_csv(OUT / "model_ready_dataset.csv")

MISSING_RULE = {
    # column                  value  why
    "notice_recency_days":  (3650, "no dated notice exists at all -> treated as 10 years stale, "
                                   "the worst freshness on the scale (a18_missing keeps it distinguishable)"),
    "a72_alt_text_pct":     (0.0,  "no image carries a text alternative -> 0% labelled"),
    "a53_contrast_ratio":   (1.0,  "contrast could not be measured because there is no readable "
                                   "text block -> worst possible contrast"),
}

print("MISSING-VALUE SEMANTICS — 'absent' now scores as absent, not as average\n")
print(f"{'column':<24}{'n NaN':>7}  {'was':<12}{'now':<8} meaning")
print("-" * 100)
for col, (val, why) in MISSING_RULE.items():
    n = int(data[col].isna().sum())
    med = data[col].median()
    print(f"{col:<24}{n:>7}  median={med:<5.4g} {val:<8g} {why[:58]}")
    data[col] = data[col].fillna(val)

# every _missing flag must still exist so the two cases stay separable
for f in ["a18_missing", "a72_missing", "a53_missing"]:
    assert f in data.columns, f"{f} flag lost"
assert data[list(MISSING_RULE)].isna().sum().sum() == 0

remaining = data.select_dtypes(include=[np.number]).isna().sum()
remaining = remaining[remaining > 0]
print(f"\nnumeric columns still containing NaN: {len(remaining)}")
print("  -> no imputation is needed anywhere downstream; absence is now encoded in the value")

data.to_csv(OUT / "model_ready_v2.csv", index=False)
(OUT / "missing_value_policy.json").write_text(json.dumps(
    {k: {"fill_value": v[0], "rationale": v[1],
         "n_rows_affected": int(pd.read_csv(OUT / "model_ready_dataset.csv")[k].isna().sum())}
     for k, v in MISSING_RULE.items()}, indent=2), encoding="utf-8")
print(f"\nwrote model_ready_v2.csv {data.shape}")

# ══════════════════════════════════════════ 2. student-perspective profile cards
key = pd.read_csv(OUT / "trackB_key.csv")
pairs = pd.read_csv(OUT / "trackB_pairs.csv")
d = data.set_index("uni_id")

IDENTITY = {"name", "url", "member", "region", "country", "uni_id", "page_lang",
            "switched_to_english", "fetched_at", "trackA_consensus", "country_is_bucket",
            "country_rank_eligible"}

def yn(v):
    return "YES" if v else "no"


def freshness(row):
    if row.a18_missing == 1:
        return "no dated post anywhere on the site"
    dys = int(row.notice_recency_days)
    if dys >= 3650:
        return "no dated post anywhere on the site"
    if dys == 0:
        return "latest post is from today"
    if dys == 1:
        return "latest post is from yesterday"
    if dys <= 14:
        return f"latest post {dys} days ago"
    if dys <= 60:
        return f"latest post {dys} days ago"
    if dys <= 365:
        return f"latest post {dys} days ago — going quiet"
    return f"latest post {dys} days ago — looks abandoned"


def events(row):
    n = int(row.a24_event_count)
    if row.a23_upcoming_events == 0 and n == 0:
        return "nothing listed"
    if n == 0:
        return "an events section exists but nothing is listed in it"
    timed = " with dates and times" if row.a27_event_datetime else ", but no dates given"
    return f"{n} upcoming event{'s' if n != 1 else ''}{timed}"


def readability(row):
    c = row.a53_contrast_ratio
    if row.a53_missing == 1 or c <= 1.5:
        return "text is essentially unreadable against its background"
    if c < 3.0:
        return f"text is very hard to read (contrast {c:.1f}:1)"
    if c < 4.5:
        return f"text is uncomfortable to read (contrast {c:.1f}:1)"
    if c < 7.0:
        return f"text is readable (contrast {c:.1f}:1)"
    return f"text is easy to read (contrast {c:.1f}:1)"


def phone(row):
    m = int(row.a63_mobile_score)
    return {100: "works well on a phone (100/100)", 90: "works on a phone (90/100)",
            75: "usable on a phone (75/100)", 50: "poor on a phone (50/100)",
            20: "barely usable on a phone (20/100)", 0: "unusable on a phone (0/100)"}.get(
        m, f"phone score {m}/100")


def speed(row):
    p = int(round(row.load_time_pct_region * 100)) if row.load_time_pct_region <= 1 else int(row.load_time_pct_region)
    if p >= 80:
        return f"loads quickly (faster than {p}% of comparable sites)"
    if p >= 40:
        return f"loads at an average speed ({p}th percentile)"
    if p >= 15:
        return f"loads slowly (slower than {100-p}% of comparable sites)"
    return f"loads very slowly (slower than {100-p}% of comparable sites)"


def nav(row):
    n = int(row.a03_nav_item_count)
    if row.a02_primary_nav == 0 or n == 0:
        return "NO MAIN MENU AT ALL"
    if n <= 2:
        return f"a {n}-item menu — very thin"
    if n <= 9:
        return f"a {n}-item menu"
    if n <= 14:
        return f"a large {n}-item menu"
    return f"a cluttered {n}-item menu"


def render(row):
    """A blind profile card written the way a prospective student reads a website."""
    contact = []
    if row.a43_contact_link:
        contact.append("contact page")
    if row.a47_footer_contact:
        contact.append("footer details")
    contact = " + ".join(contact) if contact else "NO WAY TO CONTACT THEM"

    footer_bits = [b for b, f in [("quick-links", row.a51_quick_links),
                                  ("sitemap", row.a48_footer_sitemap)] if f]
    depth = [b for b, f in [("student-portal", row.a44_student_portal), ("careers", row.a40_career_link),
                            ("alumni", row.a41_alumni_link), ("FAQ", row.a42_faq_link),
                            ("gallery", row.a30_image_gallery), ("video", row.a29_video_content),
                            ("clubs", row.a28_contests), ("social", row.a50_social_links),
                            ("live-chat", row.a58_live_chat), ("feedback-form", row.a59_feedback_form)] if f]
    marketing = [b for b, f in [("intl-ranking-badge", row.a07_qs_badge),
                                ("national-rank-claim", row.a09_national_rank),
                                ("accreditation", row.a11_accreditation),
                                ("awards", row.a13_achievements),
                                ("testimonials", row.a61_testimonials)] if f]
    a11y = []
    if row.a72_missing == 1 or row.a72_alt_text_pct < 25:
        a11y.append("images unlabelled for screen readers")
    elif row.a72_alt_text_pct >= 80:
        a11y.append("images labelled for screen readers")
    if row.a74_a11y_toggle:
        a11y.append("text-size/contrast control")
    nb = int(row.a66_broken_links)

    L = [
        f"APPLY  programmes {yn(row.a37_programs_listing)} | requirements {yn(row.a46_admissions_policy)}"
        f" | notices+deadlines {yn(row.a22_admission_notice)} | scholarships {yn(row.a38_scholarship)}"
        f" | prospectus {yn(row.a45_prospectus)} | reach a human: {contact}",

        f"STUDY  departments {yn(row.a34_department_links)} | faculty {yn(row.a35_faculty_link)}"
        f" | library {yn(row.a39_library_link)} | research {yn(row.a36_research_highlight)}",

        f"ALIVE  notice board: {freshness(row)} | news {yn(row.a20_news_events)}"
        f" | events: {events(row)} | calendar {yn(row.a21_calendar_link)}",

        f"NAV    {nav(row)} | search {yn(row.a04_search_bar)} | breadcrumb {yn(row.a06_breadcrumb)}"
        f" | footer: {', '.join(footer_bits) if footer_bits else 'nothing'}"
        f" | languages {yn(row.a05_language_toggle)}",

        f"WORKS  {phone(row)} | {speed(row)} | {readability(row)}"
        f" | broken links: {'none' if nb == 0 else nb} | https {yn(row.a65_https)}",

        f"EXTRA  {', '.join(depth) if depth else 'nothing beyond the basics'}"
        f" | about/mission {yn(row.a32_vision_mission or row.a33_about_blurb)}",

        f"ADVERT {', '.join(marketing) if marketing else 'nothing in particular'}",
        f"A11Y   {'; '.join(a11y) if a11y else 'nothing notable'}",
    ]
    return chr(10).join(L)


# blinding: the renderer must not be able to touch an identity column
src = render.__doc__ or ""
body = open(__file__, encoding="utf-8").read()
fn_src = body[body.index("def render(row):"):body.index("# blinding:")]
for col in IDENTITY:
    assert col not in fn_src, f"render() references identity column {col}"
print("\nblinding: render() references no identity column")

cards = {}
for sid, uid in zip(key.sid, key.uni_id):
    cards[sid] = render(d.loc[uid])

wl = [len(c.split()) for c in cards.values()]
print(f"rendered {len(cards)} cards, mean {np.mean(wl):.0f} words")

# closed-vocabulary blinding check
names = set()
for _, r in key.iterrows():
    for tok in re.split(r"[^A-Za-z]+", f"{r['name']} {r.country} {r.region}"):
        if len(tok) > 3:
            names.add(tok.lower())
blob = " ".join(cards.values()).lower()
hits = sorted(n for n in names if re.search(rf"\b{re.escape(n)}\b", blob))
ALLOWED_OVERLAP = {"international", "national", "university", "technology", "science", "college",
                   "institute", "research", "america", "asia", "europe", "africa", "central",
                   "south", "north", "east", "west", "southeast", "northern", "western", "eastern",
                   "southern", "latin", "oceania",
                   # legitimate card vocabulary that also appears inside some university names
                   "faculty", "engineering", "medical", "agriculture", "education", "management"}
leaks = [h for h in hits if h not in ALLOWED_OVERLAP]
print(f"blinding: {len(leaks)} identity tokens leaked into the cards -> {leaks}")
assert not leaks

(OUT / "trackB_profiles_v2.md").write_text(
    "# Track B v2 — blind profile cards, written from a prospective student's point of view\n\n"
    + "\n\n".join(f"## {sid}\n```\n{c}\n```" for sid, c in cards.items()), encoding="utf-8")

# ══════════════════════════════════════════════════════ 3. rebuild the batches
HEADER = """# Track B v2 — judging batch {b} of {nb}

**You are a prospective student deciding where to apply.** You have never heard of either
of these two universities. All you have is what their website shows you.

For each pair answer one question: **which of these two universities would I feel more
confident applying to, based on the website alone?**

What should weigh heavily:
* can I find the programmes, the admission requirements, the deadlines, and the fees
* can I reach a human being if I have a question
* is there evidence the institution is active and the information is current
* can I find my way around the site, on a laptop and on a phone
* can I actually read the page

What should weigh lightly: marketing badges, photo galleries, videos, social-media links,
and anything a visitor would never notice.

Record one row per pair in `trackB_judgments_v2.csv`:

`pair_id,winner,confidence,reason`

* `winner` — `left` or `right` (or `tie` only when genuinely inseparable)
* `confidence` — `clear` / `slight` / `toss-up`
* `reason` — one short line naming the deciding factor

---

"""

for b, g in pairs.groupby("batch"):
    parts = [HEADER.format(b=b, nb=pairs.batch.nunique())]
    for _, r in g.iterrows():
        parts.append(f"## {r.pair_id}\n\n**LEFT**\n```\n{r.left_sid}\n{cards[r.left_sid]}\n```\n\n"
                     f"**RIGHT**\n```\n{r.right_sid}\n{cards[r.right_sid]}\n```\n")
    (BATCH / f"batch_{b:02d}.md").write_text("\n".join(parts), encoding="utf-8")

print(f"wrote {pairs.batch.nunique()} batches to {BATCH.name}/ "
      f"(same 200 universities, same {len(pairs)} pairs as v1)")

hdr = "pair_id,winner,confidence,reason\n"
p = OUT / "trackB_judgments_v2.csv"
if not p.exists():
    p.write_text(hdr, encoding="utf-8")
print(f"judgment file ready: {p.name}")
