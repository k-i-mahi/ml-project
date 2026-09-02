# Weka attribute reference

71 predictive attributes: **60 nominal `{0,1}` flags** and **11 numeric**. Types are identical in every ARFF file.

| # | attribute | ARFF type | range | dimension it scores |
|---|---|---|---|---|
| 1 | `a01_logo` | `{0,1}` | 0 / 1 | D7 identity & transparency |
| 2 | `a02_primary_nav` | `{0,1}` | 0 / 1 | D4 navigation & findability |
| 3 | `a03_nav_item_count` | `numeric` | 0 - 20 | D4 navigation & findability |
| 4 | `a04_search_bar` | `{0,1}` | 0 / 1 | D4 navigation & findability |
| 5 | `a05_language_toggle` | `{0,1}` | 0 / 1 | D5 usability & accessibility |
| 6 | `a06_breadcrumb` | `{0,1}` | 0 / 1 | D4 navigation & findability |
| 7 | `a07_qs_badge` | `{0,1}` | 0 / 1 | not used by the scoring model |
| 8 | `a09_national_rank` | `{0,1}` | 0 / 1 | not used by the scoring model |
| 9 | `a11_accreditation` | `{0,1}` | 0 / 1 | D7 identity & transparency |
| 10 | `a12_accred_count` | `numeric` | 0 - 4 | not used by the scoring model |
| 11 | `a13_achievements` | `{0,1}` | 0 / 1 | D7 identity & transparency |
| 12 | `a14_stats_block` | `{0,1}` | 0 / 1 | not used by the scoring model |
| 13 | `a15_stat_item_count` | `numeric` | 0 - 10 | not used by the scoring model |
| 14 | `a16_notice_board` | `{0,1}` | 0 / 1 | D3 currency & activity |
| 15 | `a17_notice_timestamp` | `{0,1}` | 0 / 1 | D3 currency & activity |
| 16 | `a20_news_events` | `{0,1}` | 0 / 1 | D3 currency & activity |
| 17 | `a21_calendar_link` | `{0,1}` | 0 / 1 | D3 currency & activity |
| 18 | `a22_admission_notice` | `{0,1}` | 0 / 1 | D2 admission support |
| 19 | `a23_upcoming_events` | `{0,1}` | 0 / 1 | D3 currency & activity |
| 20 | `a24_event_count` | `numeric` | 0 - 20 | D3 currency & activity |
| 21 | `a25_event_images` | `{0,1}` | 0 / 1 | not used by the scoring model |
| 22 | `a26_event_captions` | `{0,1}` | 0 / 1 | not used by the scoring model |
| 23 | `a27_event_datetime` | `{0,1}` | 0 / 1 | D3 currency & activity |
| 24 | `a28_contests` | `{0,1}` | 0 / 1 | not used by the scoring model |
| 25 | `a29_video_content` | `{0,1}` | 0 / 1 | not used by the scoring model |
| 26 | `a30_image_gallery` | `{0,1}` | 0 / 1 | not used by the scoring model |
| 27 | `a31_social_feed_embed` | `{0,1}` | 0 / 1 | not used by the scoring model |
| 28 | `a32_vision_mission` | `{0,1}` | 0 / 1 | D7 identity & transparency |
| 29 | `a33_about_blurb` | `{0,1}` | 0 / 1 | D7 identity & transparency |
| 30 | `a34_department_links` | `{0,1}` | 0 / 1 | D1 academic information |
| 31 | `a35_faculty_link` | `{0,1}` | 0 / 1 | D1 academic information |
| 32 | `a36_research_highlight` | `{0,1}` | 0 / 1 | D1 academic information |
| 33 | `a37_programs_listing` | `{0,1}` | 0 / 1 | D1 academic information |
| 34 | `a38_scholarship` | `{0,1}` | 0 / 1 | D2 admission support |
| 35 | `a39_library_link` | `{0,1}` | 0 / 1 | D1 academic information |
| 36 | `a40_career_link` | `{0,1}` | 0 / 1 | D1 academic information |
| 37 | `a41_alumni_link` | `{0,1}` | 0 / 1 | D7 identity & transparency |
| 38 | `a42_faq_link` | `{0,1}` | 0 / 1 | D2 admission support |
| 39 | `a43_contact_link` | `{0,1}` | 0 / 1 | D2 admission support |
| 40 | `a44_student_portal` | `{0,1}` | 0 / 1 | D7 identity & transparency |
| 41 | `a45_prospectus` | `{0,1}` | 0 / 1 | D2 admission support |
| 42 | `a46_admissions_policy` | `{0,1}` | 0 / 1 | D2 admission support |
| 43 | `a47_footer_contact` | `{0,1}` | 0 / 1 | D7 identity & transparency |
| 44 | `a48_footer_sitemap` | `{0,1}` | 0 / 1 | D4 navigation & findability |
| 45 | `a49_copyright_line` | `{0,1}` | 0 / 1 | D3 currency & activity |
| 46 | `a50_social_links` | `{0,1}` | 0 / 1 | D7 identity & transparency |
| 47 | `a51_quick_links` | `{0,1}` | 0 / 1 | D4 navigation & findability |
| 48 | `a53_contrast_ratio` | `numeric` | 1 - 21 | D5 usability & accessibility |
| 49 | `a53_contrast_ratio_was_missing` | `{0,1}` | 0 / 1 | not used by the scoring model |
| 50 | `a54_banner_carousel` | `{0,1}` | 0 / 1 | not used by the scoring model |
| 51 | `a57_logo_prominence` | `{0,1}` | 0 / 1 | not used by the scoring model |
| 52 | `a58_live_chat` | `{0,1}` | 0 / 1 | D2 admission support |
| 53 | `a59_feedback_form` | `{0,1}` | 0 / 1 | not used by the scoring model |
| 54 | `a60_trust_seal` | `{0,1}` | 0 / 1 | D7 identity & transparency |
| 55 | `a61_testimonials` | `{0,1}` | 0 / 1 | not used by the scoring model |
| 56 | `a63_mobile_score` | `numeric` | 0 - 100 | D5 usability & accessibility |
| 57 | `a65_https` | `{0,1}` | 0 / 1 | D6 technical quality & discoverability |
| 58 | `a66_broken_links` | `numeric` | 0 - 1129 | D6 technical quality & discoverability |
| 59 | `a67_gzip` | `{0,1}` | 0 / 1 | D6 technical quality & discoverability |
| 60 | `a69_title_meta` | `{0,1}` | 0 / 1 | D6 technical quality & discoverability |
| 61 | `a70_favicon` | `{0,1}` | 0 / 1 | D6 technical quality & discoverability |
| 62 | `a71_sitemap_robots` | `{0,1}` | 0 / 1 | D6 technical quality & discoverability |
| 63 | `a72_alt_text_pct` | `numeric` | 0 - 100 | D5 usability & accessibility |
| 64 | `a72_alt_text_pct_was_missing` | `{0,1}` | 0 / 1 | not used by the scoring model |
| 65 | `a73_accessible_design` | `{0,1}` | 0 / 1 | D5 usability & accessibility |
| 66 | `a74_a11y_toggle` | `{0,1}` | 0 / 1 | D5 usability & accessibility |
| 67 | `a75_bookmark` | `{0,1}` | 0 / 1 | not used by the scoring model |
| 68 | `broken_links_log` | `numeric` | 0 - 7.02997 | not used by the scoring model |
| 69 | `load_time_z_region` | `numeric` | -1.87492 - 5.32945 | D6 technical quality & discoverability |
| 70 | `notice_recency_days` | `numeric` | 0 - 3650 | D3 currency & activity |
| 71 | `notice_recency_days_was_missing` | `{0,1}` | 0 / 1 | not used by the scoring model |

| target | ARFF type | values |
|---|---|---|
| `website_score` | `numeric` | 7.59 - 94.94 |
| `grade` | `{F,D,C,B,A,'A+'}` | worst to best |
