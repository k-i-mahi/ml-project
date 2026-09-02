"""
attribute_reference.py
======================
The human-readable specification of every landing-page attribute, carried forward from
the collection instrument, so that the report and the code describe the same schema in
the same words.

Each entry is:

    column      the name of the column as it appears in the dataset
    no          the attribute number used by the collection instrument (1-75)
    name        the human-readable attribute name
    dtype       Boolean / Integer / Float / Derived
    domain      the value range or format
    why         the justification for collecting it
    src         provenance, using the reference keys:
                  P1 = Mukanda, Mbuguah & Wabwoba (2022)  -- technical & usability criteria
                  P2 = Rashida et al. (2021)              -- key strings & questionnaire
                  P3 = Saleh et al. (2022)                -- six-factor SLR structure
                  New = added by this project from observation of real landing pages
                  Derived = engineered here; see Section 4.5

Six attributes of the original 75-attribute specification (a19, a52, a55, a56, a64, a68) are
absent from the collected data, and two prestige-value columns (a08, a10) were dropped in
cleaning, so 71 columns remain after feature engineering.
"""
from __future__ import annotations

# column, no, name, dtype, domain, why, src
_ROWS = [
    # ---------------------------------------------------------------- B1 header & navigation
    ("a01_logo", 1, "Logo / institutional branding", "Boolean", "{0,1}",
     "Institutional identity", "P2"),
    ("a02_primary_nav", 2, "Primary navigation menu presence", "Boolean", "{0,1}",
     "Core wayfinding; a page without one cannot be navigated at all", "P1; P2"),
    ("a03_nav_item_count", 3, "Number of top-level navigation items", "Integer", "[0-20]",
     "Navigation complexity: too few leads nowhere, too many is a wall of links", "P3"),
    ("a04_search_bar", 4, "Search bar", "Boolean", "{0,1}",
     "Fast task completion", "P1"),
    ("a05_language_toggle", 5, "Language toggle (native / English)", "Boolean", "{0,1}",
     "Multilingual accessibility", "P2; P3"),
    ("a06_breadcrumb", 6, "Breadcrumb / current-location cue", "Boolean", "{0,1}",
     "Recognition over recall", "P1"),

    # ---------------------------------------------------------------- B2 rankings & recognition
    ("a07_qs_badge", 7, "QS World Ranking badge / mention", "Boolean", "{0,1}",
     "Global prestige signal displayed on the page", "New"),
    ("a09_national_rank", 9, "National / local ranking mention", "Boolean", "{0,1}",
     "Local prestige signal", "New; P1"),
    ("a11_accreditation", 11, "Accreditation badges", "Boolean", "{0,1}",
     "Credibility signal", "New"),
    ("a12_accred_count", 12, "Accreditation badge count", "Integer", "[0-10]",
     "Credibility depth", "New"),
    ("a13_achievements", 13, "Recent achievements / awards section", "Boolean", "{0,1}",
     "Institutional reputation", "New"),
    ("a14_stats_block", 14, "Impact / highlight statistics block", "Boolean", "{0,1}",
     "Quantified credibility", "New"),
    ("a15_stat_item_count", 15, "Number of highlight statistics shown", "Integer", "[0-10]",
     "Richness of proof points", "New"),

    # ---------------------------------------------------------------- B3 notices & updates
    ("a16_notice_board", 16, "Notice board presence", "Boolean", "{0,1}",
     "Core recurring information need", "P2"),
    ("a17_notice_timestamp", 17, "Notice timestamp shown", "Boolean", "{0,1}",
     "Signals content freshness", "P2"),
    ("notice_recency_days", 18, "Days since the most recent dated notice", "Derived",
     "[0-3650]",
     "Turns attribute 18 (an ISO date) into a staleness measure comparable across a crawl "
     "that spans time", "Derived"),
    ("a20_news_events", 20, "News & events section", "Boolean", "{0,1}",
     "Institutional activity visibility", "P2"),
    ("a21_calendar_link", 21, "Events / academic calendar link", "Boolean", "{0,1}",
     "Time-sensitive planning information", "P1; P2"),
    ("a22_admission_notice", 22, "Admission announcements / deadlines", "Boolean", "{0,1}",
     "High-frequency prospective-student query", "P2"),

    # ---------------------------------------------------------------- B4 events & media
    ("a23_upcoming_events", 23, "Upcoming events section", "Boolean", "{0,1}",
     "Engagement and currency", "New"),
    ("a24_event_count", 24, "Number of upcoming events listed", "Integer", "[0-20]",
     "Activity richness", "New"),
    ("a25_event_images", 25, "Event images present", "Boolean", "{0,1}",
     "Visual engagement", "New; P3"),
    ("a26_event_captions", 26, "Event captions / descriptions present", "Boolean", "{0,1}",
     "Informativeness", "New"),
    ("a27_event_datetime", 27, "Event date & time displayed", "Boolean", "{0,1}",
     "Actionability: an undated event cannot be attended", "New"),
    ("a28_contests", 28, "Forthcoming contests or competitions", "Boolean", "{0,1}",
     "Student engagement signal", "New"),
    ("a29_video_content", 29, "Video content on landing page", "Boolean", "{0,1}",
     "Rich media engagement", "P3"),
    ("a30_image_gallery", 30, "Photo / image gallery", "Boolean", "{0,1}",
     "Visual storytelling", "P3"),
    ("a31_social_feed_embed", 31, "Social media feed embed", "Boolean", "{0,1}",
     "Real-time engagement", "P2"),

    # ---------------------------------------------------------------- B5 landing-page content
    ("a32_vision_mission", 32, "Vision & mission statement", "Boolean", "{0,1}",
     "Institutional identity signal", "P2"),
    ("a33_about_blurb", 33, "About / overview blurb", "Boolean", "{0,1}",
     "First-impression orientation", "P2"),
    ("a34_department_links", 34, "Department list / quick links", "Boolean", "{0,1}",
     "Primary navigation need for an applicant", "P2"),
    ("a35_faculty_link", 35, "Faculty highlight or link", "Boolean", "{0,1}",
     "Academic credibility cue", "P2"),
    ("a36_research_highlight", 36, "Research / publication highlight", "Boolean", "{0,1}",
     "Academic reputation signal", "P2; P3"),
    ("a37_programs_listing", 37, "Academic programmes listing", "Boolean", "{0,1}",
     "The decision-critical information a prospective student came for", "P2"),
    ("a38_scholarship", 38, "Scholarship information / link", "Boolean", "{0,1}",
     "Strong student interest area (53% in the P2 survey)", "P2"),
    ("a39_library_link", 39, "Library access / link", "Boolean", "{0,1}",
     "Common student resource", "P2"),
    ("a40_career_link", 40, "Career / job portal link", "Boolean", "{0,1}",
     "Student utility feature", "P2"),
    ("a41_alumni_link", 41, "Alumni section / link", "Boolean", "{0,1}",
     "Institutional trust signal", "P2"),
    ("a42_faq_link", 42, "FAQ link", "Boolean", "{0,1}",
     "Self-service reduces friction", "P2"),
    ("a43_contact_link", 43, "Contact information / webmail link", "Boolean", "{0,1}",
     "Direct communication channel", "P2"),
    ("a44_student_portal", 44, "Student portal link", "Boolean", "{0,1}",
     "Core functional access", "P3"),
    ("a45_prospectus", 45, "Prospectus / brochure download", "Boolean", "{0,1}",
     "Decision-support material", "P3"),
    ("a46_admissions_policy", 46, "Admissions policy / procedure link", "Boolean", "{0,1}",
     "Decision-critical information", "P3"),

    # ---------------------------------------------------------------- B6 footer
    ("a47_footer_contact", 47, "Footer contact details", "Boolean", "{0,1}",
     "Standard usability expectation", "P2"),
    ("a48_footer_sitemap", 48, "Footer sitemap link", "Boolean", "{0,1}",
     "Redundant navigation path", "P1"),
    ("a49_copyright_line", 49, "Copyright / last-updated line", "Boolean", "{0,1}",
     "Currency and credibility cue", "P1"),
    ("a50_social_links", 50, "Social media links", "Boolean", "{0,1}",
     "Engagement / communication channel", "P2"),
    ("a51_quick_links", 51, "Quick-links block", "Boolean", "{0,1}",
     "Reduces clicks to key pages", "P2"),

    # ---------------------------------------------------------------- B7 visual design
    ("a53_contrast_ratio", 53, "Colour contrast ratio", "Float", "[1-21] WCAG scale",
     "Readability and accessibility", "P1"),
    ("a54_banner_carousel", 54, "Banner / carousel content", "Boolean", "{0,1}",
     "First-impression engagement", "P2"),
    ("a57_logo_prominence", 57, "Prominent logo placement", "Boolean", "{0,1}",
     "Brand recall", "P3"),

    # ---------------------------------------------------------------- B8 service interaction
    ("a58_live_chat", 58, "Live chat / chatbot presence", "Boolean", "{0,1}",
     "Real-time support", "P3"),
    ("a59_feedback_form", 59, "Feedback / contact form on landing page", "Boolean", "{0,1}",
     "Two-way interaction", "P3"),
    ("a60_trust_seal", 60, "Trust / security seal displayed", "Boolean", "{0,1}",
     "Perceived trust", "P3"),
    ("a61_testimonials", 61, "Testimonials / student stories section", "Boolean", "{0,1}",
     "Sense of community", "P3"),

    # ---------------------------------------------------------------- B9 technical performance
    ("load_time_z_region", 62, "Page load speed, standardised within region", "Derived",
     "z-score",
     "Raw load time is confounded with the collector, since each of six collectors covered "
     "exactly one region; only the within-region form is used", "Derived"),
    ("a63_mobile_score", 63, "Mobile responsiveness score", "Float", "[0-100]",
     "Multi-device access need", "P1"),
    ("a65_https", 65, "HTTPS / SSL presence", "Boolean", "{0,1}",
     "Security and user trust; an unencrypted site should not receive an application form",
     "P2"),
    ("a66_broken_links", 66, "Broken links on landing page", "Integer", "[0-N]",
     "Reliability and credibility", "P1"),
    ("broken_links_log", 66, "Broken links, log-compressed", "Derived", "log(1+n)",
     "The raw count is heavily right-skewed; the log keeps 0 vs 3 from being swamped by "
     "40 vs 200", "Derived"),
    ("a67_gzip", 67, "Page compression enabled (Gzip etc.)", "Boolean", "{0,1}",
     "Faster, lighter load", "P1"),

    # ---------------------------------------------------------------- B10 SEO & metadata
    ("a69_title_meta", 69, "Page title & meta description present", "Boolean", "{0,1}",
     "Search-engine visibility", "P1"),
    ("a70_favicon", 70, "Favicon presence", "Boolean", "{0,1}",
     "Brand recognition in-browser", "General convention"),
    ("a71_sitemap_robots", 71, "Sitemap / robots.txt reference", "Boolean", "{0,1}",
     "Crawlability and indexing", "P1"),

    # ---------------------------------------------------------------- B11 accessibility
    ("a72_alt_text_pct", 72, "Alt-text coverage on images", "Float", "[0-100] %",
     "Screen-reader accessibility", "P1"),
    ("a73_accessible_design", 73, "Accessible design across devices", "Boolean", "{0,1}",
     "Inclusive access for all", "P1"),
    ("a74_a11y_toggle", 74, "Text-size / contrast accessibility toggle", "Boolean", "{0,1}",
     "Inclusive design option", "P3"),
    ("a75_bookmark", 75, "Bookmark / save-page facility", "Boolean", "{0,1}",
     "Convenience functionality", "P3"),

    # ---------------------------------------------------------------- B12 measurement quality
    ("notice_recency_days_was_missing", 18, "Notice date was never found", "Derived",
     "{0,1}",
     "Records that attribute 18 could not be measured, so 'never measured' stays "
     "distinguishable from 'measured as stale'", "Derived"),
    ("a72_alt_text_pct_was_missing", 72, "Alt-text coverage was never measured", "Derived",
     "{0,1}", "As above, for attribute 72", "Derived"),
    ("a53_contrast_ratio_was_missing", 53, "Contrast ratio was never measured", "Derived",
     "{0,1}", "As above, for attribute 53", "Derived"),
]

FIELDS = ("no", "name", "dtype", "domain", "why", "src")
ATTRIBUTE_REFERENCE = {r[0]: dict(zip(FIELDS, r[1:])) for r in _ROWS}

# Block names, in schema order, mapped to the source groups used by build_dataset.py.
BLOCKS = [
    ("B1",  "Header & navigation",   "Header & navigation",    "Wayfinding and structural usability"),
    ("B2",  "Rankings & recognition","Rankings & recognition",  "Externally conferred prestige signals"),
    ("B3",  "Notices & updates",     "Notices & updates",       "Content currency and maintenance"),
    ("B4",  "Events & media",        "Events & media",          "Engagement and multimedia richness"),
    ("B5",  "Landing-page content",  "Page content",            "Information completeness"),
    ("B6",  "Footer details",        "Footer",                  "Secondary navigation and credibility"),
    ("B7",  "Visual design",         "Visual design",           "Appearance and readability"),
    ("B8",  "Service interaction",   "Service & interaction",   "Two-way interaction and trust"),
    ("B9",  "Technical performance", "Technical performance",   "Speed, security, reliability"),
    ("B10", "SEO & metadata",        "SEO & metadata",          "Discoverability"),
    ("B11", "Accessibility",         "Accessibility",           "Inclusive access"),
    ("B12", "Measurement quality",   "Measurement quality",     "Whether the attribute could be measured at all"),
]
BLOCK_OF_GROUP = {grp: code for code, _, grp, _ in BLOCKS}
BLOCK_TITLE = {code: title for code, title, _, _ in BLOCKS}

if __name__ == "__main__":
    print(f"{len(ATTRIBUTE_REFERENCE)} attributes documented across {len(BLOCKS)} blocks")
    for code, title, grp, sig in BLOCKS:
        n = sum(1 for c, d in ATTRIBUTE_REFERENCE.items())
        print(f"  {code:<4} {title:<24} {sig}")
