# %% [markdown]
# # 05 — Track B: Blind Pairwise Judgement Materials
#
# Track A is a formula. A model trained on it re-learns arithmetic — the published failure of
# Biyyapu et al. (98.2% accuracy predicting a lookup table from its own inputs). Track B exists to
# produce a target that is **not a closed-form function of the features**.
#
# Method: 200 universities are rendered as **blind profile cards** — no name, no country, no
# region, no URL, no collector, and no Track A score — and judged in ~900 pairs on the holistic
# question *"which of these two would I rather land on?"*. A Bradley–Terry model (nb06) recovers a
# latent quality score from the win/loss outcomes.
#
# Blinding is what guarantees no prestige and no rubric-anchoring leaks into the labels. An
# anonymous feature profile of Harvard and of an unknown Polish technical university are judged on
# identical terms.
#
# **Outputs:** `trackB_sample.csv`, `trackB_profiles.md`, `trackB_pairs.csv`,
# `trackB_batches/batch_NN.md`, `trackB_judgments.csv` (empty template)

# %%
import pathlib, json, shutil, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)
SEED = 20260901
rng = np.random.default_rng(SEED)

ROOT = pathlib.Path.cwd()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
OUT = ROOT / "outputs"
BATCH_DIR = OUT / "trackB_batches"
BATCH_DIR.mkdir(parents=True, exist_ok=True)
# Clear stale batch files individually — OneDrive keeps a handle on the directory itself, so
# rmtree on the folder fails with WinError 5.
for _f in BATCH_DIR.glob("*.md"):
    _f.unlink()

df = pd.read_csv(OUT / "model_ready_dataset.csv")
A = pd.read_csv(OUT / "expert_labels_trackA.csv")
d = df.merge(A[["uni_id", "trackA_consensus"]], on="uni_id")
print(f"Population: {len(d):,} universities across {d.region.nunique()} regions")

# %% [markdown]
# ## 1. Stratified sample of 200
#
# Stratified by region, and **within region across the full Track A range**, so the sample is not
# concentrated in the middle where judgements are least informative. Track A is used here purely as
# a stratifier — it never appears on a profile card.

# %%
N_TARGET, N_STRATA = 200, 5
per_region = {r: N_TARGET // d.region.nunique() for r in sorted(d.region.unique())}
for r in list(per_region)[:N_TARGET - sum(per_region.values())]:
    per_region[r] += 1

picks = []
for region, n in per_region.items():
    sub = d[d.region == region].copy()
    sub["stratum"] = pd.qcut(sub.trackA_consensus, N_STRATA, labels=False, duplicates="drop")
    base, extra = n // N_STRATA, n % N_STRATA
    for s, g in sub.groupby("stratum"):
        k = base + (1 if s < extra else 0)
        picks.append(g.sample(min(k, len(g)), random_state=SEED + int(s)))
sample = pd.concat(picks).reset_index(drop=True)
print(f"Sampled {len(sample)} universities")
print(pd.crosstab(sample.region, pd.qcut(sample.trackA_consensus, 5,
                                         labels=["Q1 low", "Q2", "Q3", "Q4", "Q5 high"])).to_string())
print(f"\nTrack A range in sample: {sample.trackA_consensus.min():.1f}–"
      f"{sample.trackA_consensus.max():.1f}  (population "
      f"{d.trackA_consensus.min():.1f}–{d.trackA_consensus.max():.1f})")

# non-ordinal blind IDs: nothing about the ID encodes rank, region or population order
sids = rng.choice(np.arange(1000, 9999), size=len(sample), replace=False)
sample["sid"] = [f"S-{s}" for s in sids]
sample = sample.sample(frac=1, random_state=SEED).reset_index(drop=True)

# %% [markdown]
# ## 2. Blind profile renderer
#
# Attributes are grouped by **what a visitor needs**, not by the block taxonomy, and stated in
# plain language. Deliberately withheld: any aggregate count of features present, which would
# reintroduce the presence-counting the design exists to avoid.

# %%
def _yn(row, spec):
    """Split a list of (column, label) into present / absent label lists."""
    yes = [lab for col, lab in spec if row.get(col, 0) == 1]
    no = [lab for col, lab in spec if row.get(col, 0) == 0]
    return yes, no

def _line(tag, yes, no):
    if not yes and not no:
        return None
    parts = ", ".join(yes) if yes else "none"
    if no:
        parts += " | NO " + ", ".join(no)
    return f"{tag} {parts}"

def render(row):
    L = [row.sid]

    nav_yes, nav_no = _yn(row, [("a01_logo", "logo"), ("a04_search_bar", "search"),
                                ("a06_breadcrumb", "breadcrumb"),
                                ("a05_language_toggle", "language-toggle")])
    menu = (f"menu({int(row.a03_nav_item_count)} items)" if row.a02_primary_nav == 1
            else "NO PRIMARY MENU")
    L.append(_line("NAV", [menu] + nav_yes, nav_no))

    L.append(_line("TASK", *_yn(row, [
        ("a37_programs_listing", "programs"), ("a46_admissions_policy", "admissions-policy"),
        ("a22_admission_notice", "admission-notices"), ("a43_contact_link", "contact"),
        ("a34_department_links", "departments"), ("a35_faculty_link", "faculty")])))

    life = []
    if row.notice_evidence == 3:
        life.append(f"notice board, dated, latest {int(row.notice_recency_days)}d ago")
    elif row.notice_evidence == 2:
        life.append(f"notice board, dated, latest {int(row.notice_recency_days)}d ago (stale)")
    elif row.notice_evidence == 1:
        life.append("notice board but NO dates shown")
    else:
        life.append("no notice board")
    if row.event_evidence >= 2:
        life.append(f"{int(row.a24_event_count)} upcoming events"
                    + (" with date/time" if row.a27_event_datetime == 1 else ", no times given"))
    elif row.event_evidence == 1:
        life.append("events section but nothing listed")
    else:
        life.append("no events")
    ly, ln = _yn(row, [("a20_news_events", "news"), ("a21_calendar_link", "calendar"),
                       ("a36_research_highlight", "research-highlights")])
    L.append(_line("LIFE", life + ly, ln))

    L.append(_line("DEPTH", *_yn(row, [
        ("a39_library_link", "library"), ("a40_career_link", "careers"),
        ("a41_alumni_link", "alumni"), ("a38_scholarship", "scholarships"),
        ("a44_student_portal", "student-portal"), ("a42_faq_link", "faq"),
        ("a45_prospectus", "prospectus"), ("a32_vision_mission", "vision/mission"),
        ("a33_about_blurb", "about"), ("a47_footer_contact", "footer-contact"),
        ("a48_footer_sitemap", "footer-sitemap"), ("a51_quick_links", "quick-links")])))

    alt = "alt-text not measured" if pd.isna(row.a72_alt_text_pct) else f"alt-text {row.a72_alt_text_pct:.0f}%"
    con = "contrast not measured" if pd.isna(row.a53_contrast_ratio) else f"contrast {row.a53_contrast_ratio:.1f}:1"
    ay, an = _yn(row, [("a73_accessible_design", "accessible-design"),
                       ("a74_a11y_toggle", "text-size/contrast-toggle")])
    L.append(_line("A11Y", [alt, con] + ay, an))

    tech = [f"mobile {int(row.a63_mobile_score)}/100",
            f"{int(row.a66_broken_links)} broken links",
            f"loads faster than {100*(1-row.load_time_pct_region):.0f}% of peers"]
    ty, tn = _yn(row, [("a65_https", "https"), ("a67_gzip", "compression"),
                       ("a69_title_meta", "page-title/meta"), ("a71_sitemap_robots", "sitemap/robots"),
                       ("a70_favicon", "favicon")])
    L.append(_line("TECH", tech + ty, tn))

    L.append(_line("EXTRAS", *_yn(row, [
        ("a29_video_content", "video"), ("a30_image_gallery", "gallery"),
        ("a54_banner_carousel", "rotating-carousel"), ("a31_social_feed_embed", "social-feed"),
        ("a25_event_images", "event-images"), ("a26_event_captions", "event-captions"),
        ("a28_contests", "contests"), ("a58_live_chat", "live-chat"),
        ("a59_feedback_form", "feedback-form"), ("a50_social_links", "social-links"),
        ("a49_copyright_line", "copyright"), ("a57_logo_prominence", "prominent-logo")])))

    L.append(_line("CLAIMS", *_yn(row, [
        ("a07_qs_badge", "QS-badge"), ("a09_national_rank", "national-rank-mention"),
        ("a11_accreditation", "accreditation"), ("a13_achievements", "awards"),
        ("a14_stats_block", "stats-block"), ("a60_trust_seal", "trust-seal"),
        ("a61_testimonials", "testimonials"), ("a75_bookmark", "bookmark-widget")])))
    return "\n".join(x for x in L if x)

print(render(sample.iloc[0]))
print("\n" + "-" * 78 + "\n")
print(render(sample.iloc[1]))

# %% [markdown]
# ## 3. Blinding verification
#
# Nothing that could identify the institution, and nothing that could anchor to Track A, may appear
# on a card. Verified by exhaustive search rather than by inspection.

# %%
import re

cards = {r.sid: render(r) for _, r in sample.iterrows()}

# Closed-vocabulary check. The renderer may emit ONLY fixed label strings and numbers, so the set
# of alphabetic tokens across all 200 cards must be a subset of this declared vocabulary. This is
# stricter than searching for known names: it catches any word the renderer was never meant to
# produce, including a name it has never seen. (Substring search over names gives false positives -
# "national" is legitimate vocabulary via `national-rank-mention`.)
ALLOWED = set("""s nav task life depth a11y a y tech extras claims
logo search breadcrumb language toggle menu items no primary none
programs admissions policy admission notices contact departments faculty
notice board dated latest ago stale but dates shown upcoming events with date time times given
section nothing listed news calendar research highlights d
library careers alumni scholarships student portal faq prospectus vision mission about
footer sitemap quick links contact
alt text not measured contrast accessible design size
mobile broken loads faster than of peers https compression page title meta robots favicon
video gallery rotating carousel social feed event images captions contests live chat feedback form
copyright prominent qs badge national rank mention accreditation awards stats block trust seal
testimonials bookmark widget""".split())

tokens = set()
for c in cards.values():
    tokens |= set(re.findall(r"[a-zA-Z]+", c.lower()))
unexpected = tokens - ALLOWED

print(f"Cards rendered       : {len(cards)}")
print(f"Mean card length     : {np.mean([len(c.split()) for c in cards.values()]):.0f} words")
print(f"Distinct word tokens : {len(tokens)}  (closed vocabulary of {len(ALLOWED)})")
print(f"Unexpected tokens    : {len(unexpected)}  {sorted(unexpected) if unexpected else ''}")
assert not unexpected, f"blinding failed - renderer emitted unexpected tokens: {sorted(unexpected)}"

# Belt and braces: confirm the renderer structurally cannot reach the identity columns.
import inspect
src = inspect.getsource(render)
forbidden = [c for c in ["name", "url", "final_url", "region", "country", "member",
                         "trackA_consensus", "uni_id"] if re.search(rf"\brow\.{c}\b|['\"]{c}['\"]", src)]
print(f"Identity columns referenced by render(): {forbidden or 'none'}")
assert not forbidden, f"render() touches identity columns: {forbidden}"

print("\nBlinding verified two ways: the renderer never references an identity column, and every "
      "word across all 200 cards comes from the closed attribute vocabulary. No name, country, "
      "region, URL or Track A anchor can appear.")

# %% [markdown]
# ## 4. Pair design
#
# | Type | n | Purpose |
# |---|---|---|
# | random | 500 | Connectivity of the comparison graph; wide quality gaps anchor the scale |
# | close | 340 | Track A gap ≤ 5 points — the hard cases, where the ordering is actually informative |
# | repeat | 60 | The same pair re-presented **with sides swapped**, ≥10 batches later |
#
# The repeats give an **intra-rater self-consistency statistic** — the standard reviewer objection
# to single-rater labels, answered with a number instead of an assurance.

# %%
N_RANDOM, N_CLOSE, N_REPEAT, BATCH_SIZE = 500, 340, 60, 25
sids = sample.sid.tolist()
score = dict(zip(sample.sid, sample.trackA_consensus))

all_pairs = [(a, b) for i, a in enumerate(sids) for b in sids[i + 1:]]
gaps = np.array([abs(score[a] - score[b]) for a, b in all_pairs])
print(f"Candidate pairs: {len(all_pairs):,}   with Track A gap <= 5: {(gaps<=5).sum():,}")

# The 500 "random" pairs are built as 5 rounds of a random perfect matching over the 200 items.
# Each round pairs every item exactly once, so every university is guaranteed 5 comparisons before
# the close pairs are added. Pure uniform sampling left some items with a single appearance, which
# gives Bradley-Terry essentially no information about them.
random_pairs, seen = [], set()
for _round in range(N_RANDOM // (len(sids) // 2)):
    shuffled = list(rng.permutation(sids))
    for a, b in zip(shuffled[::2], shuffled[1::2]):
        key = tuple(sorted((a, b)))
        if key not in seen:
            seen.add(key)
            random_pairs.append(key)
while len(random_pairs) < N_RANDOM:                      # top up any collisions
    a, b = rng.choice(sids, 2, replace=False)
    key = tuple(sorted((a, b)))
    if key not in seen:
        seen.add(key); random_pairs.append(key)

close_pool = [all_pairs[i] for i in np.flatnonzero(gaps <= 5)]
close_pool = [p for p in close_pool if tuple(sorted(p)) not in seen]
close_idx = rng.choice(len(close_pool), N_CLOSE, replace=False)
close = [close_pool[i] for i in close_idx]

cov = pd.Series([s for p in random_pairs for s in p]).value_counts()
print(f"Random pairs from {N_RANDOM//(len(sids)//2)} perfect matchings: {len(random_pairs)}   "
      f"appearances per item min {cov.min()}, max {cov.max()}")

rows = []
for (a, b), typ in [(p, "random") for p in random_pairs] + [(p, "close") for p in close]:
    if rng.random() < .5:                      # randomise which side each item appears on
        a, b = b, a
    rows.append(dict(left_sid=a, right_sid=b, pair_type=typ, repeat_of=""))
pairs = pd.DataFrame(rows).sample(frac=1, random_state=SEED).reset_index(drop=True)

n_unique_batches = int(np.ceil(len(pairs) / BATCH_SIZE))
pairs["batch"] = pairs.index // BATCH_SIZE + 1

# repeats: drawn from early batches, re-presented SWAPPED in the final batches
early = pairs[pairs.batch <= 20].sample(N_REPEAT, random_state=SEED)
rep = pd.DataFrame(dict(
    left_sid=early.right_sid.values, right_sid=early.left_sid.values,   # sides swapped
    pair_type="repeat", repeat_of=early.index.map(lambda i: f"P{i+1:04d}").values))
rep["batch"] = n_unique_batches + np.arange(len(rep)) // BATCH_SIZE + 1

pairs = pd.concat([pairs, rep], ignore_index=True)
pairs.insert(0, "pair_id", [f"P{i+1:04d}" for i in range(len(pairs))])
pairs["trackA_gap"] = [abs(score[a] - score[b]) for a, b in zip(pairs.left_sid, pairs.right_sid)]

print(f"\nTotal pairs: {len(pairs)}   batches: {pairs.batch.max()}")
print(pairs.pair_type.value_counts().to_string())
appear = pd.concat([pairs.left_sid, pairs.right_sid]).value_counts()
print(f"\nComparisons per university: min {appear.min()}, median {appear.median():.0f}, "
      f"max {appear.max()}  (all 200 appear: {appear.size == 200})")
print(f"Track A gap — random: {pairs[pairs.pair_type=='random'].trackA_gap.mean():.1f} mean | "
      f"close: {pairs[pairs.pair_type=='close'].trackA_gap.mean():.1f} mean")

# %% [markdown]
# ## 5. Emit judging batches
#
# One file per batch, each pair rendered with both cards inline so judging needs no cross-reference.

# %%
HEADER = """# Track B — judging batch {b} of {tot}

For each pair answer the single question: **which of these two websites would I rather land on?**
Judge holistically, as a visitor, not by counting features. Weigh what a real person would weigh:
can I do what I came for, is the place alive, can everyone use it.

Record one row per pair in `trackB_judgments.csv`:

`pair_id,winner,confidence,reason`

* `winner` — `left` or `right` (or `tie` only when genuinely inseparable)
* `confidence` — `clear` / `slight` / `toss-up`
* `reason` — one short line naming the deciding factor

---
"""
for b, g in pairs.groupby("batch"):
    parts = [HEADER.format(b=b, tot=pairs.batch.max())]
    for _, r in g.iterrows():
        parts.append(f"## {r.pair_id}\n\n**LEFT**\n```\n{cards[r.left_sid]}\n```\n\n"
                     f"**RIGHT**\n```\n{cards[r.right_sid]}\n```\n")
    (BATCH_DIR / f"batch_{b:02d}.md").write_text("\n".join(parts), encoding="utf-8")

sizes = [f.stat().st_size for f in BATCH_DIR.glob("*.md")]
print(f"Wrote {len(sizes)} batch files to outputs/trackB_batches/")
print(f"  mean size {np.mean(sizes)/1024:.1f} KB, total {sum(sizes)/1024:.0f} KB")

# %% [markdown]
# ## 6. Artefacts

# %%
(OUT / "trackB_profiles.md").write_text(
    "# Track B — blind profile cards\n\n"
    "No name, country, region, URL or Track A score appears here (verified in §3).\n\n"
    + "\n\n".join(f"```\n{c}\n```" for c in cards.values()), encoding="utf-8")

pairs.to_csv(OUT / "trackB_pairs.csv", index=False)
sample[["sid", "uni_id", "name", "region", "country", "trackA_consensus"]].to_csv(
    OUT / "trackB_key.csv", index=False)          # the unblinding key, kept OUT of the batches

jpath = OUT / "trackB_judgments.csv"
if not jpath.exists():
    pd.DataFrame(columns=["pair_id", "winner", "confidence", "reason"]).to_csv(jpath, index=False)
    print("Created empty trackB_judgments.csv (append-only, checkpointed across sessions)")
else:
    print(f"trackB_judgments.csv already has {len(pd.read_csv(jpath))} rows — preserved")

for f in ["trackB_profiles.md", "trackB_pairs.csv", "trackB_key.csv", "trackB_judgments.csv"]:
    print(f"  outputs/{f}  ({(OUT/f).stat().st_size:,} bytes)")

# %% [markdown]
# ## Next step — the judging stage
#
# The judgements are made **conversationally, not by a script**. A formula here would reintroduce
# exactly the circularity Track B exists to break: the label would once again be a closed-form
# function of the features, and the model would once again learn arithmetic.
#
# Each batch file is read and judged, with results appended to `trackB_judgments.csv`. The file is
# append-only and checkpointed so the work is resumable. nb06 fits Bradley–Terry once judging is
# complete.
