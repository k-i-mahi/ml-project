# -*- coding: utf-8 -*-
"""
Build the Weka (ARFF) view of the dataset.

    python src/make_weka.py

Reads the CSV files that build_dataset.py already wrote and re-expresses them as ARFF with
*correct attribute types*. The earlier export declared every attribute `numeric`, including
the 60 attributes that are 0/1 presence flags. Weka accepts that, but it silently disables a
large part of the tool: the entropy-based attribute evaluators (InfoGain, GainRatio,
ChiSquared) reject numeric attributes, NaiveBayes fits a Gaussian density to a coin flip, and
every tree prints `a01_logo <= 0.5` instead of `a01_logo = 0`.

Types are inferred from all 1225 rows, never from one split, so every file declares an
identical header and Weka's "Supplied test set" accepts any pairing.

Writes to data/weka/:
    train.arff  test.arff  all_universities.arff        target website_score (numeric)
    train_classification.arff  test_classification.arff  target grade (nominal)
    bangladesh.arff  bangladesh_classification.arff      the 22 held-out universities
    *_named.arff                                         same rows, university name attached
    index_*.csv                                          instance number -> university
    ATTRIBUTES.md                                        the attribute type legend

The *_named.arff files carry a leading `university` string attribute so Weka can print the
university beside every prediction instead of a bare instance number. A string attribute
cannot be fed to a classifier, so those files are used through FilteredClassifier with
Remove -R 1, which drops the name before training; the guide gives the click path.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
W = DATA / "weka"
W.mkdir(parents=True, exist_ok=True)

META = ["uni_id", "name", "url", "country", "region"]
TARGETS = ["website_score", "grade", "rank", "regional_rank", "country_rank"]

# Worst -> best, so the confusion matrix reads as an ordinal scale and a one-band error sits
# on a cell adjacent to the diagonal.
GRADE_ORDER = ["F", "D", "C", "B", "A", "A+"]

all_df = pd.read_csv(DATA / "university_website_scores.csv")
train = pd.read_csv(DATA / "train.csv")
test = pd.read_csv(DATA / "test.csv")
bd = pd.read_csv(DATA / "test_bangladesh.csv")

FEATURES = [c for c in train.columns if c not in META + TARGETS]
assert FEATURES == [c for c in test.columns if c not in META + TARGETS]
assert all(c in all_df.columns for c in FEATURES)


def quote(v: str) -> str:
    """ARFF nominal values need quoting unless they are plain alphanumerics."""
    s = str(v)
    return s if s.replace("_", "").isalnum() else "'" + s.replace("'", r"\'") + "'"


# ----------------------------------------------------------------------------------------
# Attribute typing, decided once over all 1225 rows.
# ----------------------------------------------------------------------------------------
def attr_type(col: str) -> str:
    vals = set(pd.unique(all_df[col].dropna()))
    if vals <= {0, 1} and len(vals) > 0:
        return "{0,1}"          # presence flag / missingness indicator
    return "numeric"            # count, percentage, ratio or z-score


TYPE_OF = {c: attr_type(c) for c in FEATURES}
N_NOMINAL = sum(1 for t in TYPE_OF.values() if t != "numeric")
GRADE_SET = "{" + ",".join(quote(g) for g in GRADE_ORDER) + "}"

HEADER_NOTE = [
    "% University Website Quality - CSE 4112 Machine Learning Laboratory, KUET",
    "%",
    "% 1225 university landing pages, 71 predictive attributes.",
    f"% {N_NOMINAL} attributes are nominal {{0,1}} presence flags; "
    f"{len(FEATURES) - N_NOMINAL} are numeric counts, percentages, ratios or z-scores.",
    "% Identifier columns (uni_id, name, url, country, region) are excluded on purpose so no",
    "% model can memorise a university; index_*.csv maps instance number back to the name.",
    "% All 22 Bangladeshi universities are in the test split and none is in train.",
    "% There are no missing values: they were imputed during dataset construction and every",
    "% imputed column carries a companion *_was_missing flag.",
    "%",
]


def write_arff(path: Path, frame: pd.DataFrame, relation: str, target: str,
               named: bool = False) -> int:
    numeric_target = target == "website_score"
    f = frame

    out = list(HEADER_NOTE)
    out += [f"% Relation: {relation}",
            f"% Instances: {len(f)}",
            f"% Target attribute: {target} "
            + ("(numeric, 0-100)" if numeric_target
               else "(nominal, worst to best: " + " < ".join(GRADE_ORDER) + ")")]
    if named:
        out += ["%",
                "% Attribute 1 is the university name, present only so Weka can label its",
                "% predictions. It must NOT reach the model: wrap the scheme in",
                "%   weka.classifiers.meta.FilteredClassifier",
                "%     -F \"weka.filters.unsupervised.attribute.Remove -R 1\"",
                "% and set 'Output predictions' to also output attribute 1."]
    out += ["", f"@relation {relation}", ""]
    if named:
        out.append("@attribute university string")
    for c in FEATURES:
        out.append(f"@attribute {c} {TYPE_OF[c]}")
    out.append(f"@attribute {target} " + ("numeric" if numeric_target else GRADE_SET))
    out += ["", "@data"]

    for _, r in f.iterrows():
        vals = [quote(r["name"])] if named else []
        for c in FEATURES:
            v = r[c]
            if pd.isna(v):
                vals.append("?")
            elif TYPE_OF[c] == "numeric":
                vals.append(f"{v:.6g}")
            else:
                vals.append(str(int(v)))
        tv = r[target]
        vals.append("?" if pd.isna(tv)
                    else (f"{tv:.6g}" if numeric_target else quote(tv)))
        out.append(",".join(vals))

    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return len(f)


def write_index(path: Path, frame: pd.DataFrame) -> None:
    """Weka's prediction output identifies rows by position only; this is the key back."""
    idx = frame[[c for c in META if c in frame.columns] + ["website_score", "grade"]].copy()
    idx.insert(0, "weka_instance", range(1, len(idx) + 1))
    idx.to_csv(path, index=False)


SETS = [
    ("train", train, "website_quality_train"),
    ("test", test, "website_quality_test"),
    ("all_universities", all_df, "website_quality_all"),
    ("bangladesh", bd, "website_quality_bangladesh"),
]

counts = {}
for stem, frame, relation in SETS:
    counts[stem] = write_arff(W / f"{stem}.arff", frame, relation, "website_score")
    write_arff(W / f"{stem}_named.arff", frame, relation + "_named", "website_score", named=True)
    write_index(W / f"index_{stem}.csv", frame)

for stem, frame, relation in SETS:
    if stem == "all_universities":
        name, rel = "all_universities_classification", "website_grade_all"
    else:
        name, rel = f"{stem}_classification", relation.replace("quality", "grade")
    write_arff(W / f"{name}.arff", frame, rel, "grade")
    write_arff(W / f"{name}_named.arff", frame, rel + "_named", "grade", named=True)

# ----------------------------------------------------------------------------------------
# Attribute legend, so the types are documented next to the data rather than only in code.
# ----------------------------------------------------------------------------------------
dd = pd.read_csv(DATA / "data_dictionary.csv").set_index("column")
rows = ["# Weka attribute reference", "",
        f"{len(FEATURES)} predictive attributes: **{N_NOMINAL} nominal `{{0,1}}` flags** and "
        f"**{len(FEATURES) - N_NOMINAL} numeric**. Types are identical in every ARFF file.", "",
        "| # | attribute | ARFF type | range | dimension it scores |",
        "|---|---|---|---|---|"]
for i, c in enumerate(FEATURES, 1):
    t = TYPE_OF[c]
    lo, hi = all_df[c].min(), all_df[c].max()
    rng = "0 / 1" if t != "numeric" else f"{lo:g} - {hi:g}"
    dim = dd.loc[c, "dimension"] if c in dd.index else ""
    rows.append(f"| {i} | `{c}` | `{t}` | {rng} | {dim} |")
rows += ["", "| target | ARFF type | values |", "|---|---|---|",
         "| `website_score` | `numeric` | "
         f"{all_df.website_score.min():g} - {all_df.website_score.max():g} |",
         f"| `grade` | `{GRADE_SET}` | worst to best |"]
(W / "ATTRIBUTES.md").write_text("\n".join(rows) + "\n", encoding="utf-8")

print(f"wrote {len(SETS) * 4} ARFF files to {W}")
print(f"  attributes : {len(FEATURES)} ({N_NOMINAL} nominal, "
      f"{len(FEATURES) - N_NOMINAL} numeric) + target")
print(f"  instances  : " + ", ".join(f"{k}={v}" for k, v in counts.items()))
