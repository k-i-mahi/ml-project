"""
make_tables.py
==============
Emits every data-driven table in report/report.tex as a standalone .tex fragment under
report/tables/, straight from the CSVs in data/ and results/.

The report \\input{}s these fragments, so a number can never disagree between the code and
the document: rerun `build_dataset.py` -> `make_figures.py` -> `make_tables.py` and the
report re-typesets with the new values.

Run:  python src/make_tables.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from attribute_reference import ATTRIBUTE_REFERENCE, BLOCKS, BLOCK_OF_GROUP

ROOT = Path(__file__).resolve().parent.parent
DATA, RES = ROOT / "data", ROOT / "results"
TAB = ROOT / "report" / "tables"
TAB.mkdir(parents=True, exist_ok=True)


def tex(s) -> str:
    """Escape a value for LaTeX."""
    s = str(s)
    for a, b in [("\\", r"\textbackslash "), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"),
                 ("#", r"\#"), ("$", r"\$"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde "),
                 ("^", r"\textasciicircum ")]:
        s = s.replace(a, b)
    return s


def mono(s) -> str:
    return r"\texttt{" + tex(s) + "}"


def write(name: str, lines: list[str]) -> None:
    (TAB / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  {name}")


cat = pd.read_csv(DATA / "feature_catalog.csv")
strat = pd.read_csv(DATA / "split_stratification.csv")
summary = json.loads((DATA / "dataset_summary.json").read_text(encoding="utf-8"))
nums = json.loads((RES / "report_numbers.json").read_text(encoding="utf-8"))
comp = pd.read_csv(RES / "model_comparison_report.csv")
gate = pd.read_csv(RES / "error_by_gate_report.csv")
imp = pd.read_csv(RES / "feature_importance_report.csv")
bd = pd.read_csv(RES / "bangladesh_predictions_report.csv")
prof = pd.read_csv(RES / "bangladesh_dimension_profile_report.csv")
sens = pd.read_csv(RES / "weight_sensitivity_report.csv")
oat = pd.read_csv(RES / "weight_sensitivity_oat_report.csv")
abl = pd.read_csv(RES / "block_ablation_report.csv")

print("writing report/tables/:")

# ===================================================================== 1. feature catalogue
# The attribute catalogue -- attribute numbering, names, data types,
# value domains and justifications -- with two columns added: the scoring dimension that
# consumes the attribute, and the points it is worth at its maximum.
GRP_ORDER = [g for _, _, g, _ in BLOCKS]
REF = ATTRIBUTE_REFERENCE

HDR = (r"\textbf{\#} & \textbf{Attribute} & \textbf{Type} & \textbf{Domain} & "
       r"\textbf{Justification} & \textbf{Dim} & \textbf{Pts} \\")
L = [r"\begin{longtable}{@{}r@{\hspace{4pt}}p{4.15cm}p{1.15cm}p{1.5cm}p{4.85cm}c r@{}}",
     r"\caption{Landing-page attributes used for university website evaluation. Attribute "
     r"numbers, names, data types, value domains and justifications are as specified for the "
     r"collection instrument; the dataset column name is printed beneath each "
     r"attribute name. \emph{Dim} is the scoring dimension of Section~\ref{sec:label} that "
     r"consumes the attribute and \emph{Pts} the points it contributes at its maximum; a dash "
     r"marks an attribute that is observed and supplied to every model but does not enter the "
     r"label.}\label{tab:catalog}\\",
     r"\toprule", HDR, r"\midrule\endfirsthead",
     r"\multicolumn{7}{@{}l}{\small\itshape Table \ref{tab:catalog}, continued from previous page}\\",
     r"\toprule", HDR, r"\midrule\endhead",
     r"\midrule\multicolumn{7}{@{}r}{\small\itshape continued on next page}\\\endfoot",
     r"\bottomrule\endlastfoot"]

for code, title, grp, signal in BLOCKS:
    sub = cat[cat.source_group == grp].copy()
    if not len(sub):
        continue
    sub["_no"] = [REF[f]["no"] for f in sub.feature]
    sub = sub.sort_values("_no")
    n_s = int((sub.scored_by != "--").sum())
    L.append(r"\multicolumn{7}{@{}l}{\cellcolor{softgrey}\textbf{" + code + ". " + tex(title) +
             r"} \small --- " + tex(signal) + r" \ (" + f"{len(sub)} attributes, {n_s} scored"
             + r")}\\\addlinespace[1pt]")
    for _, r in sub.iterrows():
        d = REF[r.feature]
        scored = r.scored_by != "--"
        why = r"\small " + tex(d["why"])
        if not scored:
            why += r" \emph{\footnotesize (excluded: " + tex(r["transform"]) + ")}"
        if scored and float(r.weight_in_dimension) == 0.0:
            pts = r"\emph{mod.}"
        elif scored:
            pts = f"{r.points_at_maximum:.2f}"
        else:
            pts = "--"
        nm, col, no = tex(d["name"]), tex(r.feature), d["no"]
        dt, dom = tex(d["dtype"]), tex(d["domain"])
        dm = tex(r.scored_by) if scored else "--"
        L.append(f"{no} & {nm} " + r"\texttt{\scriptsize " + col + "} & "
                 + f"{dt} & {dom} & {why} & {dm} & {pts} " + r"\\")
L.append(r"\end{longtable}")
write("tab_catalog.tex", L)

# ---- provenance key ---------------------------------------------------------------------
L = [r"\begin{tabular}{ll}", r"\toprule",
     r"\textbf{Key} & \textbf{Provenance} \\", r"\midrule",
     r"P1 & Mukanda, Mbuguah \& Wabwoba (2022) --- technical and usability criteria \\",
     r"P2 & Rashida \emph{et al.} (2021) --- landing-page key strings and questionnaire \\",
     r"P3 & Saleh \emph{et al.} (2022) --- six-factor systematic review structure \\",
     r"New & Added by this project from observation of live landing pages \\",
     r"Derived & Engineered in \texttt{build\_dataset.py}; see Section~\ref{sec:engineering} \\",
     r"\bottomrule", r"\end{tabular}"]
write("tab_provenance.tex", L)

# ---- feature blocks B1..B12 -------------------------------------------------------------
L = [r"\begin{tabular}{llcp{5.0cm}l}", r"\toprule",
     r"\textbf{\#} & \textbf{Feature block} & \textbf{Attrs} & \textbf{Nature of the signal} "
     r"& \textbf{Dimensions} \\", r"\midrule"]
tot = 0
for code, title, grp, signal in BLOCKS:
    sub = cat[cat.source_group == grp]
    if not len(sub):
        continue
    tot += len(sub)
    dims = sorted({d for d in sub.scored_by if d != "--"})
    L.append(f"{code} & {tex(title)} & {len(sub)} & \\small {tex(signal)} & "
             f"{', '.join('$' + d + '$' for d in dims) if dims else '---'} \\\\")
L += [r"\midrule", f"& \\textbf{{Total}} & \\textbf{{{tot}}} & & \\\\",
      r"\bottomrule", r"\end{tabular}"]
write("tab_blocks.tex", L)

# ---- exclusion register: why each unscored attribute is unscored -------------------------
BUCKET = {
    "prestige display; excluded to prevent prestige leakage": "Prestige leakage",
    "count depth of a badge already scored by presence": "Depth count of a scored presence",
    "count depth of a marketing block": "Depth count of a scored presence",
    "marketing block; serves no applicant information need": "No applicant information need",
    "engagement signal; serves no applicant information need": "No applicant information need",
    "marketing; serves no applicant information need": "No applicant information need",
    "media richness; presence is measurable, quality is not": "Quality not judgeable from presence",
    "media richness; present on 5.6% of sites": "Quality not judgeable from presence",
    "visual presentation the dataset cannot assess for quality": "Quality not judgeable from presence",
    "present on 1.9% of sites; too rare to carry weight": "Too rare to carry weight",
    "present on 1.0% of sites; a browser feature, not a page property": "Too rare to carry weight",
    "redundant transform of a66, which is scored": "Redundant with a scored attribute",
    "measurement metadata, not a property of the website": "Measurement metadata",
}
ORDER = ["Prestige leakage", "Quality not judgeable from presence",
         "No applicant information need", "Depth count of a scored presence",
         "Too rare to carry weight", "Redundant with a scored attribute",
         "Measurement metadata"]
unsc = cat[cat.scored_by == "--"].copy()
unsc["bucket"] = [BUCKET.get(t, "Other") for t in unsc["transform"]]
L = [r"\begin{tabular}{p{4.4cm}cp{8.9cm}}", r"\toprule",
     r"\textbf{Reason for exclusion} & \textbf{Attrs} & \textbf{Attributes} \\", r"\midrule"]
for _b in ORDER:
    _sub = unsc[unsc.bucket == _b].sort_values("feature")
    if not len(_sub):
        continue
    _names = ", ".join(mono(f) for f in _sub.feature)
    L.append(f"{tex(_b)} & {len(_sub)} & \\small {_names} \\\\[2pt]")
L += [r"\midrule",
      f"\\textbf{{Total excluded}} & \\textbf{{{len(unsc)}}} & "
      f"\\small of {len(cat)} features; the other {len(cat) - len(unsc)} enter the label \\\\",
      r"\bottomrule", r"\end{tabular}"]
write("tab_exclusions.tex", L)

# ---- literature review comparison table --------------------------------------------------
LIT = [
    (r"Rashida \emph{et al.} (2021), \emph{Computers} 10(5):57",
     "A multi-method framework for evaluating university websites in Bangladesh. The "
     "automated tool crawls with Selenium and evaluates three criteria: content of "
     "information (25 key strings), website performance and loading time (20 browser timing "
     "variables). A questionnaire study compares the survey ranking with the automated one.",
     "The principal baseline. Most of our content attributes --- vision and mission, faculty, "
     "notice board, library, contact --- originate in or extend its content criteria. We "
     "extend its automated evaluation from a national to a global corpus, and replace its "
     "fixed weighting with a documented, sensitivity-tested one."),
    (r"Saleh \emph{et al.} (2022), \emph{Indonesian J.\ of Electrical Engineering and "
     "Computer Science} 25(1):511--520",
     "A systematic literature review of university website quality evaluation. It extracts 79 "
     "quality factors from prior work and consolidates them into six: information quality, "
     "specific content, usability, web appearance, service interaction quality and "
     "functionality. It reports the absence of any universally accepted model.",
     "The six-factor structure is the theoretical foundation for grouping and justifying our "
     "attribute blocks. Its finding that no accepted model exists is what obliges us to "
     r"construct a label and to state its construction in full (Section~\ref{sec:label})."),
    (r"Mukanda, Mbuguah \& Wabwoba (2022), \emph{Int.\ J.\ of Computer Trends and "
     "Technology} 70(2):1--9",
     "Evaluates the usability of the top five Kenyan university websites using automated "
     "tools (Nibbler, Silktide, Semrush, Ahrefs), covering responsiveness, colour contrast, "
     "broken links, speed, compression, SEO, and accessibility.",
     "The source of our technical and accessibility attributes: mobile responsiveness, "
     "contrast ratio, broken links, load speed, compression, SEO metadata and alt-text "
     "coverage. Where it evaluates five universities, we evaluate 1,225."),
    (r"Biyyapu \emph{et al.} (2023), \emph{Computers} 12(9):181",
     "Applies machine learning to website quality classification from automatically "
     "extracted completeness attributes (missing URLs, images, videos, PDFs, form fields), "
     "comparing an MLP against SVM and other classifiers.",
     "Motivates the move from a fixed rule-based score to a learned model, and the practice "
     "of comparing several algorithms on one feature vector. Our feature space is "
     r"substantially broader, and Section~\ref{sec:why} goes beyond reporting which family "
     "wins to identifying the structural reason it wins."),
]
L = [r"\begin{longtable}{@{}p{0.4cm}p{3.1cm}p{5.4cm}p{6.2cm}@{}}",
     r"\caption{Summary of the literature most relevant to this framework.}\label{tab:lit}\\",
     r"\toprule",
     r"\textbf{\#} & \textbf{Reference} & \textbf{Focus / method} & \textbf{What we reuse} \\",
     r"\midrule\endfirsthead",
     r"\multicolumn{4}{@{}l}{\small\itshape Table \ref{tab:lit}, continued}\\",
     r"\toprule",
     r"\textbf{\#} & \textbf{Reference} & \textbf{Focus / method} & \textbf{What we reuse} \\",
     r"\midrule\endhead",
     r"\bottomrule\endlastfoot"]
for i, (ref, focus, reuse) in enumerate(LIT, 1):
    L.append(f"{i} & \\small {ref} & \\small {focus} & \\small {reuse} \\\\" + r"\addlinespace[3pt]")
L.append(r"\end{longtable}")
write("tab_litreview.tex", L)

# ===================================================================== 2. catalogue summary
piv = pd.crosstab(cat.source_group, cat.scored_by != "--")
piv.columns = ["unscored", "scored"] if False in piv.columns else piv.columns
piv = pd.crosstab(cat.source_group, np.where(cat.scored_by != "--", "scored", "unscored"))
for c in ("scored", "unscored"):
    if c not in piv:
        piv[c] = 0
piv["total"] = piv.scored + piv.unscored
piv = piv.reindex([g for g in GRP_ORDER if g in piv.index])
L = [r"\begin{tabular}{lccc}", r"\toprule",
     r"\textbf{Source group} & \textbf{Enters the score} & \textbf{Observed only} & "
     r"\textbf{Total} \\", r"\midrule"]
for g, r in piv.iterrows():
    L.append(f"{tex(g)} & {r.scored} & {r.unscored} & {r.total} \\\\")
L += [r"\midrule",
      f"\\textbf{{Total}} & \\textbf{{{piv.scored.sum()}}} & \\textbf{{{piv.unscored.sum()}}} & "
      f"\\textbf{{{piv.total.sum()}}} \\\\", r"\bottomrule", r"\end{tabular}"]
write("tab_catalog_summary.tex", L)

mt = cat.measurement_type.value_counts()
L = [r"\begin{tabular}{lc}", r"\toprule",
     r"\textbf{Measurement type} & \textbf{Features} \\", r"\midrule"]
for k, v in mt.items():
    L.append(f"{tex(k)} & {v} \\\\")
L += [r"\midrule", f"\\textbf{{Total}} & \\textbf{{{mt.sum()}}} \\\\", r"\bottomrule",
      r"\end{tabular}"]
write("tab_measurement_types.tex", L)

# ===================================================================== 3. dimension terms
DIMW = {"D1": 28, "D2": 22, "D3": 15, "D4": 15, "D5": 10, "D6": 7, "D7": 3}
DIMSHORT = {"D1": "Academic info.", "D2": "Admission support", "D3": "Currency",
            "D4": "Navigation", "D5": "Usability", "D6": "Technical",
            "D7": "Identity"}
DIMTITLE = {"D1": "Academic information", "D2": "Admission support",
            "D3": "Currency and activity", "D4": "Navigation and findability",
            "D5": "Usability and accessibility", "D6": "Technical quality and discoverability",
            "D7": "Institutional identity and transparency"}
HDR2 = (r"& \textbf{Attribute} & \textbf{$\lambda_{k,j}$} & \textbf{Points} & "
        r"\textbf{Transform} \\")
L = [r"\begin{longtable}{@{}llrrp{4.6cm}@{}}",
     r"\caption{Every term of every dimension: the attribute, its weight "
     r"$\lambda_{k,j}$ inside the dimension, the points it is worth at its maximum, and "
     r"the transform $\phi_{k,j}$ applied before weighting. \emph{mod.} marks "
     r"\texttt{a27\_event\_datetime}, which is not an additive term but the multiplier "
     r"inside $f_{\text{evt}}$.}\label{tab:terms}\\",
     r"\toprule", HDR2, r"\midrule\endfirsthead",
     r"\multicolumn{5}{@{}l}{\small\itshape Table \ref{tab:terms}, continued from previous page}\\",
     r"\toprule", HDR2, r"\midrule\endhead",
     r"\midrule\multicolumn{5}{@{}r}{\small\itshape continued on next page}\\\endfoot",
     r"\bottomrule\endlastfoot"]
for d in DIMW:
    sub = cat[cat.scored_by == d].sort_values("weight_in_dimension", ascending=False)
    L.append(r"\multicolumn{5}{@{}l}{\cellcolor{softgrey}\textbf{$" + d +
             r"$ --- " + tex(DIMTITLE[d]) + r"} \quad $w_{" + d[1] + r"} = " +
             str(DIMW[d]) + r"$ points}\\")
    for _, r in sub.iterrows():
        mod = float(r.weight_in_dimension) == 0.0
        lam = r"\emph{mod.}" if mod else f"{r.weight_in_dimension:.2f}"
        pts = r"\emph{mod.}" if mod else f"{r.points_at_maximum:.2f}"
        L.append(f"& {mono(r.feature)} & {lam} & {pts} & "
                 f"\\small {tex(r["transform"])} \\\\")
    L.append(f"& \\emph{{sum}} & {sub.weight_in_dimension.sum():.2f} & "
             f"{sub.points_at_maximum.sum():.2f} & \\\\\\addlinespace[3pt]")
L += [r"\midrule",
      f"& \\textbf{{All seven dimensions}} & & \\textbf{{{cat.points_at_maximum.sum():.0f}}} & \\\\"]
L.append(r"\end{longtable}")
write("tab_dimension_terms.tex", L)

# ===================================================================== 4. stratification
L = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
     r"\textbf{Score band} & \textbf{All} & \textbf{Train} & \textbf{Test} & "
     r"\textbf{All \%} & \textbf{Test \%} & \textbf{Drift (pp)} \\", r"\midrule"]
for _, r in strat.iterrows():
    L.append(f"{tex(r.band)} & {r['all']} & {r.train} & {r.test} & "
             f"{r.all_pct:.1f} & {r.test_pct:.1f} & {r.drift_pp:+.1f} \\\\")
L += [r"\midrule",
      f"\\textbf{{Total}} & \\textbf{{{strat['all'].sum()}}} & \\textbf{{{strat.train.sum()}}} & "
      f"\\textbf{{{strat.test.sum()}}} & 100.0 & 100.0 & \\\\",
      r"\bottomrule", r"\end{tabular}"]
write("tab_stratification.tex", L)

# ===================================================================== 5. model comparison
L = [r"\begin{tabular}{rlcccccc}", r"\toprule",
     r"& \textbf{Model} & \textbf{Family} & $\boldsymbol{R^2}$ & \textbf{MAE} & "
     r"\textbf{RMSE} & \textbf{Spearman} & \textbf{CV }$\boldsymbol{R^2}$ \\", r"\midrule"]
for i, r in comp.iterrows():
    bold = (lambda v: r"\textbf{" + v + "}") if i == 0 else (lambda v: v)
    sp = "---" if not np.isfinite(r.Spearman) else bold(f"{r.Spearman:.3f}")
    L.append(f"{i+1} & {bold(tex(r.model))} & {tex(r.family)} & {bold(f'{r.R2:.3f}')} & "
             f"{bold(f'{r.MAE:.2f}')} & {bold(f'{r.RMSE:.2f}')} & {sp} & "
             f"${r.cv_mean:.3f} \\pm {r.cv_sd:.3f}$ \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
write("tab_comparison.tex", L)

# ===================================================================== 6. gate error
FAM = dict(zip(comp.model, comp.family))
L = [r"\begin{tabular}{llccc}", r"\toprule",
     r"\textbf{Model} & \textbf{Family} & \textbf{MAE gated} & \textbf{MAE ungated} & "
     r"\textbf{ratio} \\", r"\midrule"]
lin = gate[gate.model.isin(["Linear Regression", "Ridge", "Lasso", "SVR (RBF kernel)",
                            "Neural Net (MLP)"])]
tre = gate[~gate.model.isin(lin.model)]
for blk, rows_ in [("a", lin), ("b", tre)]:
    if blk == "b":
        L.append(r"\midrule")
    for _, r in rows_.iterrows():
        b = (lambda v: r"\textbf{" + v + "}") if r.model == "LightGBM" else (lambda v: v)
        L.append(f"{b(tex(r.model))} & {tex(FAM.get(r.model, ''))} & {b(f'{r.gated:.2f}')} & "
                 f"{b(f'{r.ungated:.2f}')} & ${r.ratio:.2f}\\times$ \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
write("tab_gate.tex", L)

# ===================================================================== 7. importance
# Realised importance is grouped by the D-code prefix so that renaming a dimension in
# build_dataset.py cannot silently zero a row here.
imp["dcode"] = imp.dimension.str.extract(r"^(D\d)")[0].fillna("--")
bycode = imp.groupby("dcode").importance.sum()
unscored_share = float(bycode.get("--", 0.0))
L = [r"\begin{tabular}{lrrr}", r"\toprule",
     r"\textbf{Dimension} & \textbf{Declared weight} & \textbf{Realised importance} & "
     r"\textbf{Ratio} \\", r"\midrule"]
for d in DIMW:
    got = float(bycode.get(d, 0.0))
    L.append(f"${d}$ {tex(DIMTITLE[d])} & {DIMW[d]} & {got:.1f}\\% & "
             f"{got / DIMW[d]:.2f} \\\\")
L += [r"\midrule",
      f"not used by the label & 0 & {unscored_share:.1f}\\% & --- \\\\",
      r"\bottomrule", r"\end{tabular}"]
write("tab_importance_dim.tex", L)

L = [r"\begin{tabular}{rlrl}", r"\toprule",
     r"\# & \textbf{Feature} & \textbf{Gain \%} & \textbf{Dimension} \\", r"\midrule"]
for i, r in imp.head(15).iterrows():
    L.append(f"{i+1} & {mono(r.feature)} & {r.importance:.2f} & {tex(r.dimension)} \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
write("tab_importance_top.tex", L)

# ===================================================================== 8. Bangladesh
L = [r"\begin{tabular}{rlrrrrrc}", r"\toprule",
     r"\textbf{\#} & \textbf{University} & \textbf{World} & \textbf{Actual} & "
     r"\textbf{Predicted} & \textbf{Error} & \textbf{Pred.\ \#} & \textbf{Grade} \\",
     r"\midrule"]
for _, r in bd.iterrows():
    mark = r"\textbf{" if "KUET" in str(r.short) else ""
    end = "}" if mark else ""
    L.append(f"{mark}{r.actual_rank_in_bd}{end} & {mark}{tex(r.short)}{end} & "
             f"{mark}{int(r['rank'])}{end} & {mark}{r.actual:.2f}{end} & "
             f"{mark}{r.predicted:.2f}{end} & {r.error:+.2f} & "
             f"{r.predicted_rank_in_bd} & {tex(r.grade)} \\\\")
L += [r"\midrule",
      f"\\multicolumn{{3}}{{l}}{{\\emph{{mean}}}} & {bd.actual.mean():.2f} & "
      f"{bd.predicted.mean():.2f} & "
      f"\\multicolumn{{3}}{{l}}{{MAE {bd.error.abs().mean():.2f}}} \\\\",
      r"\bottomrule", r"\end{tabular}"]
write("tab_bangladesh.tex", L)

L = [r"\begin{tabular}{lcccrr}", r"\toprule",
     r"\textbf{Dimension} & \textbf{BD} & \textbf{World} & \textbf{Top 100} & "
     r"\textbf{vs world} & \textbf{vs top 100} \\", r"\midrule"]
for _, r in prof.iterrows():
    d = r.dimension[:2]
    L.append(f"${d}$ {tex(DIMSHORT[d])} ($w={int(r.weight)}$) & {r.bangladesh:.3f} & "
             f"{r.world:.3f} & {r.top100:.3f} & {r.gap_vs_world_pts:+.2f} & "
             f"{r.gap_vs_top100_pts:+.2f} \\\\")
L += [r"\midrule",
      f"\\textbf{{Total (points)}} & & & & \\textbf{{{prof.gap_vs_world_pts.sum():+.2f}}} & "
      f"\\textbf{{{prof.gap_vs_top100_pts.sum():+.2f}}} \\\\",
      r"\bottomrule", r"\end{tabular}"]
write("tab_bd_profile.tex", L)

# ===================================================================== 9. sensitivity
L = [r"\begin{tabular}{lcccc}", r"\toprule",
     r"\textbf{Perturbation} & \textbf{Spearman $\rho$} & \textbf{Worst of 300} & "
     r"\textbf{Top 10 kept} & \textbf{Top 50 kept} \\", r"\midrule"]
for _, r in sens.iterrows():
    L.append(f"{tex(r.perturbation)} & {r.spearman_mean:.4f} & {r.spearman_min:.4f} & "
             f"{r.top10_overlap:.0%} & {r.top50_overlap:.0%} \\\\".replace("%", r"\%"))
L += [r"\bottomrule", r"\end{tabular}"]
write("tab_sensitivity.tex", L)

rem = oat[oat.change == "removed"]
L = [r"\begin{tabular}{lcccc}", r"\toprule",
     r"\textbf{Dimension deleted} & \textbf{Weight was} & \textbf{Spearman $\rho$} & "
     r"\textbf{Top 10 kept} & \textbf{Median rank shift} \\", r"\midrule"]
for _, r in rem.iterrows():
    L.append(f"${r.dimension}$ {tex(DIMSHORT[r.dimension])} & {int(r.weight_from)} & "
             f"{r.spearman:.4f} & {r.top10_overlap:.0%} & {r.median_rank_shift:.0f} \\\\"
             .replace("%", r"\%"))
L += [r"\bottomrule", r"\end{tabular}"]
write("tab_sensitivity_oat.tex", L)

# ===================================================================== 10. ablation
L = [r"\begin{tabular}{llrccc}", r"\toprule",
     r"\textbf{Block removed} & \textbf{Kind} & \textbf{Features} & $\boldsymbol{R^2}$ & "
     r"\textbf{MAE} & $\boldsymbol{\Delta}$\textbf{MAE} \\", r"\midrule"]
for _, r in abl.sort_values("dMAE", ascending=False).iterrows():
    L.append(f"{tex(r.block)} & \\small {tex(r.kind)} & {r.n_removed} & {r.R2:.3f} & "
             f"{r.MAE:.2f} & {r.dMAE:+.2f} \\\\")
L += [r"\midrule",
      f"\\textbf{{nothing removed}} & & {len(cat)} & \\textbf{{{nums['base_r2']:.3f}}} & "
      f"\\textbf{{{nums['base_mae']:.2f}}} & --- \\\\",
      r"\bottomrule", r"\end{tabular}"]
write("tab_ablation.tex", L)

# ===================================================================== 11. macros
m = [r"% auto-generated numeric macros -- do not edit by hand",
     r"\newcommand{\nUni}{" + f"{summary['n_universities']:,}" + "}",
     r"\newcommand{\nFeat}{" + str(summary["n_features"]) + "}",
     r"\newcommand{\nScored}{" + str(summary["n_features_scored"]) + "}",
     r"\newcommand{\nUnscored}{" + str(summary["n_features_unscored"]) + "}",
     r"\newcommand{\nTrain}{" + str(summary["n_train"]) + "}",
     r"\newcommand{\nTest}{" + str(summary["n_test"]) + "}",
     r"\newcommand{\nGated}{" + str(summary["n_gated"]) + "}",
     r"\newcommand{\pctGated}{" + f"{100*summary['n_gated']/summary['n_universities']:.1f}" + r"\%}",
     r"\newcommand{\nGatedTest}{" + str(nums["n_gated_test"]) + "}",
     r"\newcommand{\nBD}{" + str(summary["n_holdout"]) + "}",
     r"\newcommand{\maxDrift}{" + f"{summary['max_band_drift_pp']:.1f}" + "}",
     r"\newcommand{\tR}{" + f"{nums['r2']:.4f}" + "}",
     r"\newcommand{\tMAE}{" + f"{nums['mae']:.2f}" + "}",
     r"\newcommand{\tRMSE}{" + f"{nums['rmse']:.2f}" + "}",
     r"\newcommand{\tSpear}{" + f"{nums['spearman']:.4f}" + "}",
     r"\newcommand{\tKendall}{" + f"{nums['kendall']:.4f}" + "}",
     r"\newcommand{\wOne}{" + f"{100*nums['within1']:.1f}" + r"\%}",
     r"\newcommand{\wTwo}{" + f"{100*nums['within2']:.1f}" + r"\%}",
     r"\newcommand{\wFive}{" + f"{100*nums['within5']:.1f}" + r"\%}",
     r"\newcommand{\bestCV}{" + f"{nums['best_cv']:.4f}" + "}",
     r"\newcommand{\nGrid}{" + str(nums["n_grid"]) + "}",
     r"\newcommand{\cvAgree}{" + f"{nums['cv_test_agreement']:.3f}" + "}",
     r"\newcommand{\bdMAE}{" + f"{nums['bd']['mae']:.2f}" + "}",
     r"\newcommand{\bdRestMAE}{" + f"{nums['bd']['rest_mae']:.2f}" + "}",
     r"\newcommand{\bdR}{" + f"{nums['bd']['r2']:.4f}" + "}",
     r"\newcommand{\bdSpear}{" + f"{nums['bd']['spearman']:.4f}" + "}",
     r"\newcommand{\bdMaxErr}{" + f"{nums['bd']['max_abs_error']:.2f}" + "}",
     r"\newcommand{\bdMean}{" + f"{nums['bd']['mean_score']:.1f}" + "}",
     r"\newcommand{\worldMean}{" + f"{nums['bd']['world_mean']:.1f}" + "}",
     r"\newcommand{\bdExact}{" + str(int((bd.actual_rank_in_bd == bd.predicted_rank_in_bd).sum())) + "}",
     r"\newcommand{\sensRho}{" + f"{nums['sens_rho_30pct']:.4f}" + "}",
     r"\newcommand{\sensTop}{" + f"{100*nums['sens_top10_30pct']:.0f}" + r"\%}",
     r"\newcommand{\ablDfour}{" + f"{abl[abl.block=='D4'].dMAE.iloc[0]:.2f}" + "}",
     r"\newcommand{\ablDone}{" + f"{abl[abl.block=='D1'].dMAE.iloc[0]:.2f}" + "}",
     r"\newcommand{\impNav}{" + f"{imp[imp.feature=='a02_primary_nav'].importance.iloc[0]:.1f}" + r"\%}",
     r"\newcommand{\impProg}{" + f"{imp[imp.feature=='a37_programs_listing'].importance.iloc[0]:.1f}" + r"\%}",
     r"\newcommand{\linGated}{" + f"{gate[gate.model=='Linear Regression'].gated.iloc[0]:.2f}" + "}",
     r"\newcommand{\lgbGated}{" + f"{gate[gate.model=='LightGBM'].gated.iloc[0]:.2f}" + "}",
     r"\newcommand{\linRatio}{" + f"{gate[gate.model=='Linear Regression'].ratio.iloc[0]:.2f}" + "}",
     r"\newcommand{\scoreMean}{" + f"{summary['score_mean']:.1f}" + "}",
     r"\newcommand{\scoreSD}{" + f"{summary['score_sd']:.1f}" + "}",
     r"\newcommand{\scoreMin}{" + f"{summary['score_min']:.1f}" + "}",
     r"\newcommand{\scoreMax}{" + f"{summary['score_max']:.1f}" + "}",
     ]
for g_, n_ in summary["grade_counts"].items():
    m.append(r"\newcommand{\grade" + g_.replace("+", "P") + "}{" + str(n_) + "}")
write("macros.tex", m)

# ===================================================================== 12. data-driven tables
full = pd.read_csv(DATA / "university_website_scores.csv")
dims = pd.read_csv(DATA / "dimension_scores.csv")
DIMCOLS = [c for c in dims.columns if c[0] == "D" and c[1].isdigit()]

ISO = {"Australia": "AU", "Portugal": "PT", "Hungary": "HU", "France": "FR",
       "United Kingdom": "UK", "India": "IN", "Singapore": "SG", "Germany": "DE",
       "Japan": "JP", "United States": "US", "Italy": "IT", "Spain": "ES",
       "Netherlands": "NL", "Canada": "CA", "China": "CN", "Brazil": "BR",
       "Poland": "PL", "South Korea": "KR", "Belgium": "BE", "Sweden": "SE",
       "Switzerland": "CH", "Austria": "AT", "Norway": "NO", "Denmark": "DK",
       "Finland": "FI", "Ireland": "IE", "New Zealand": "NZ", "Mexico": "MX",
       "Turkey": "TR", "Greece": "GR", "Czechia": "CZ", "Czech Republic": "CZ",
       "Russia": "RU", "South Africa": "ZA", "Egypt": "EG", "Pakistan": "PK",
       "Bangladesh": "BD", "Malaysia": "MY", "Indonesia": "ID", "Thailand": "TH",
       "Vietnam": "VN", "Philippines": "PH", "Saudi Arabia": "SA", "Israel": "IL",
       "Chile": "CL", "Argentina": "AR", "Colombia": "CO", "Peru": "PE",
       "Romania": "RO", "Ukraine": "UA", "Nigeria": "NG", "Kenya": "KE"}


def shorten(n, k=34):
    n = (n.replace("University of ", "Univ. of ").replace(" University", " Univ.")
          .replace("Universitat", "Univ.").replace("Technological", "Tech."))
    return n if len(n) <= k else n[: k - 1] + "."


top = full.nsmallest(10, "rank")
L = [r"\begin{tabular}{rlr}", r"\toprule", r"\# & University & Score \\", r"\midrule"]
for _, r in top.iterrows():
    cc = ISO.get(str(r.country).strip(), "")
    L.append(f"{int(r['rank'])} & {tex(shorten(r['name']))}"
             + (f" ({cc})" if cc else "") + f" & {r.website_score:.1f} " + r"\\")
L += [r"\bottomrule", r"\end{tabular}"]
write("tab_top10.tex", L)

BD_SHORT = dict(zip(bd.name, bd.short))
bdt = full[full.country == "Bangladesh"].nsmallest(8, "rank")
L = [r"\begin{tabular}{rlrr}", r"\toprule",
     r"\# & University & Global & Score \\", r"\midrule"]
for i, (_, r) in enumerate(bdt.iterrows(), 1):
    nm = tex(BD_SHORT.get(r["name"], shorten(r["name"], 24)))
    hl = "KUET" in nm
    row = (f"\\textbf{{{i}}} & \\textbf{{{nm}}} & \\textbf{{{int(r['rank'])}}} & "
           f"\\textbf{{{r.website_score:.1f}}}" if hl else
           f"{i} & {nm} & {int(r['rank'])} & {r.website_score:.1f}")
    L.append(row + r" \\")
L += [r"\bottomrule", r"\end{tabular}"]
write("tab_bdtop.tex", L)

# ---- worked example ---------------------------------------------------------------------
DIMTITLE_FULL = {"D1": "Academic information", "D2": "Admission support",
                 "D3": "Currency and activity", "D4": "Navigation and findability",
                 "D5": "Usability and accessibility",
                 "D6": "Technical quality and discoverability",
                 "D7": "Institutional identity and transparency"}
ex = dims[dims.name.str.contains("Khulna University of Engineering", na=False)].iloc[0]
exr = full[full.name == ex["name"]].iloc[0]
L = [r"\begin{tabular}{clccr}", r"\toprule",
     r"& \textbf{Dimension} & \textbf{Sub-score $D_k$} & \textbf{Weight $w_k$} & "
     r"\textbf{Points} \\", r"\midrule"]
for c in DIMCOLS:
    d = c[:2]
    L.append(f"${d}$ & {tex(DIMTITLE_FULL[d])} & {ex[c]:.3f} & {DIMW[d]} & "
             f"{ex['pts_' + d]:.2f} " + r"\\")
L += [r"\midrule",
      r"& \multicolumn{3}{l}{Uncapped total $\sum_k w_k D_k$} & "
      + f"{sum(ex['pts_' + c[:2]] for c in DIMCOLS):.2f} " + r"\\",
      r"& \multicolumn{3}{l}{Cap in force} & " + f"{ex.cap:.2f} " + r"\\",
      r"& \multicolumn{3}{l}{\textbf{Published score}} & \textbf{"
      + f"{ex.website_score:.2f}" + r"} \\",
      r"& \multicolumn{3}{l}{Grade / global rank / national rank} & \textbf{"
      + f"{exr.grade}" + r"} / " + f"{int(exr['rank'])} / {int(exr.country_rank)} " + r"\\",
      r"\bottomrule", r"\end{tabular}"]
write("tab_worked_example.tex", L)

# ---- dimension spread -------------------------------------------------------------------
q = dims[DIMCOLS].describe()
L = [r"\begin{tabular}{l" + "c" * len(DIMCOLS) + "}", r"\toprule",
     "& " + " & ".join(f"$D_{c[1]}$" for c in DIMCOLS) + r" \\", r"\midrule",
     "Weight & " + " & ".join(str(DIMW[c[:2]]) for c in DIMCOLS) + r" \\",
     "Mean & " + " & ".join(f"{q.loc['mean', c]:.3f}" for c in DIMCOLS) + r" \\",
     "Std.\\ dev. & " + " & ".join(f"{q.loc['std', c]:.3f}" for c in DIMCOLS) + r" \\",
     "Inter-quartile range & "
     + " & ".join(f"{q.loc['75%', c] - q.loc['25%', c]:.3f}" for c in DIMCOLS) + r" \\",
     r"\bottomrule", r"\end{tabular}"]
write("tab_dimspread.tex", L)

# ---- extra macros -----------------------------------------------------------------------
gate_te_n = int(nums["n_gated_test"])
bdd = dims[dims.country == "Bangladesh"]
band_lo, band_hi = bd.actual.nsmallest(len(bd)).iloc[-17], bd.actual.max()
extra = [
    r"\newcommand{\rTwoMin}{" + f"{comp[comp.family != 'baseline'].R2.min():.3f}" + "}",
    r"\newcommand{\rTwoMax}{" + f"{comp.R2.max():.3f}" + "}",
    r"\newcommand{\nUngatedTest}{" + str(nums["n_test"] - gate_te_n) + "}",
    r"\newcommand{\nGateNav}{" + str(int(dims.gate_nav.sum())) + "}",
    r"\newcommand{\nGateHttps}{" + str(int(dims.gate_https.sum())) + "}",
    r"\newcommand{\nGateBoth}{"
    + str(int(((dims.gate_nav == 1) & (dims.gate_https == 1)).sum())) + "}",
    r"\newcommand{\ptsLost}{" + f"{dims.points_lost_to_gate.sum():,.0f}" + "}",
    r"\newcommand{\ptsLostMean}{"
    + f"{dims[dims.points_lost_to_gate > 0].points_lost_to_gate.mean():.1f}" + "}",
    r"\newcommand{\httpsShare}{"
    + f"{100 * (1 - dims.gate_https.mean()):.1f}" + r"\%}",
    r"\newcommand{\nBinary}{"
    + str(int((cat.measurement_type == "binary flag").sum())) + "}",
    r"\newcommand{\bdBandN}{17}",
    r"\newcommand{\bdBandLo}{" + f"{band_lo:.0f}" + "}",
    r"\newcommand{\bdBandHi}{" + f"{band_hi:.0f}" + "}",
    r"\newcommand{\bdGapTop}{" + f"{-prof.gap_vs_top100_pts.sum():.1f}" + "}",
    r"\newcommand{\bdGapNavAdm}{"
    + f"{-(prof[prof.dimension.str.startswith(('D2', 'D4'))].gap_vs_top100_pts.sum()):.2f}" + "}",
    r"\newcommand{\bdDone}{" + f"{prof[prof.dimension.str.startswith('D1')].bangladesh.iloc[0]:.3f}" + "}",
    r"\newcommand{\worldDone}{" + f"{prof[prof.dimension.str.startswith('D1')].world.iloc[0]:.3f}" + "}",
    r"\newcommand{\bdDfour}{" + f"{prof[prof.dimension.str.startswith('D4')].bangladesh.iloc[0]:.3f}" + "}",
    r"\newcommand{\worldDfour}{" + f"{prof[prof.dimension.str.startswith('D4')].world.iloc[0]:.3f}" + "}",
    r"\newcommand{\bdGapDtwo}{" + f"{prof[prof.dimension.str.startswith('D2')].gap_vs_top100_pts.iloc[0]:.2f}" + "}",
    r"\newcommand{\bdGapDfour}{" + f"{prof[prof.dimension.str.startswith('D4')].gap_vs_top100_pts.iloc[0]:.2f}" + "}",
    r"\newcommand{\nTopTwoHundred}{" + str(int(nums["bd"]["n_top200"])) + "}",
    r"\newcommand{\buetRank}{"
    + str(int(bd[bd.short == "BUET"].actual_rank_in_bd.iloc[0])) + "}",
    r"\newcommand{\buetScore}{" + f"{bd[bd.short == 'BUET'].actual.iloc[0]:.2f}" + "}",
    r"\newcommand{\buetGrade}{" + str(bd[bd.short == "BUET"].grade.iloc[0]) + "}",
    r"\newcommand{\nCountryRanked}{" + str(int(full.country_rank.notna().sum())) + "}",
    r"\newcommand{\nCountries}{"
    + str(int(full[full.country_rank.notna()].country.nunique())) + "}",
]
gated_bd = bdd[bdd.gate_applied == 1].sort_values("points_lost_to_gate")
extra.append(r"\newcommand{\bdGateLoss}{"
             + ", ".join(f"{v:.1f}" for v in gated_bd.points_lost_to_gate) + "}")
(TAB / "macros.tex").write_text(
    (TAB / "macros.tex").read_text(encoding="utf-8") + "\n".join(extra) + "\n",
    encoding="utf-8")
print("  macros.tex (extended)")

print(f"\n{len(list(TAB.glob('*.tex')))} fragments written to {TAB}")
