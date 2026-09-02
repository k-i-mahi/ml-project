# -*- coding: utf-8 -*-
r"""
Run the Weka schemes quoted in WEKA_GUIDE.md and record what Weka actually prints.

    python src/run_weka.py                    # uses the default Weka location
    python src/run_weka.py "D:\Weka\weka.jar"

Nothing in the guide is typed by hand: every number in its tables comes from
results/weka_results.csv and results/weka_confusion.json, which this script writes.

Weka is driven through weka.classifiers.Evaluation, the same evaluation code the Explorer
calls, so the CLI numbers here are the numbers the GUI shows for the same scheme and options.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
W = ROOT / "data" / "weka"
RESULTS = ROOT / "results"
JAR = sys.argv[1] if len(sys.argv) > 1 else r"C:\Program Files\Weka-3-8-7\weka.jar"

GRADES = ["F", "D", "C", "B", "A", "A+"]

# Schemes that need MTJ (LinearRegression, M5P, GaussianProcesses) are deliberately absent:
# several Weka 3.8.7 builds ship a weka.jar with the MTJ classes stripped and those schemes
# then fail with NoClassDefFoundError. SMOreg is the linear-model stand-in that always runs.
REGRESSION = [
    ("ZeroR", "weka.classifiers.rules.ZeroR", []),
    ("SMOreg", "weka.classifiers.functions.SMOreg", []),
    ("REPTree", "weka.classifiers.trees.REPTree", []),
    ("RandomTree", "weka.classifiers.trees.RandomTree", []),
    ("MultilayerPerceptron", "weka.classifiers.functions.MultilayerPerceptron", []),
    ("Bagging(REPTree)", "weka.classifiers.meta.Bagging", []),
    ("RandomForest(100)", "weka.classifiers.trees.RandomForest", ["-I", "100"]),
]
CLASSIFICATION = [
    ("ZeroR", "weka.classifiers.rules.ZeroR", []),
    ("NaiveBayes", "weka.classifiers.bayes.NaiveBayes", []),
    ("J48", "weka.classifiers.trees.J48", []),
    ("IBk (k=5)", "weka.classifiers.lazy.IBk", ["-K", "5"]),
    ("RandomForest(100)", "weka.classifiers.trees.RandomForest", ["-I", "100"]),
]
# 10-fold CV over 1225 rows is minutes per fold for these two; they are measured on the
# held-out split only.
SKIP_IN_CV = {"MultilayerPerceptron", "SMOreg"}


def weka(scheme, args, train, test=None, folds=None) -> str:
    cmd = ["java", "-Xmx3g", "-cp", JAR, scheme, "-t", str(W / train)]
    if test:
        cmd += ["-T", str(W / test)]
    if folds:
        cmd += ["-x", str(folds)]
    p = subprocess.run(cmd + list(args), capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    out = p.stdout + p.stderr
    if "NoClassDefFoundError" in out:
        raise SystemExit(f"{scheme} cannot run on this Weka build:\n"
                         + "\n".join(out.splitlines()[:3]))
    return out


def after(text, marker):
    i = text.find(marker)
    return text[i:] if i >= 0 else ""


def value(text, label):
    m = re.search(re.escape(label) + r"\s+([-\d.]+)", text)
    return float(m.group(1)) if m else None


rows = []
PROTOCOLS = [
    ("train -> test (244)", "train.arff", "test.arff", None),
    ("10-fold CV (1225)", "all_universities.arff", None, 10),
    ("train -> Bangladesh (22)", "train.arff", "bangladesh.arff", None),
]

for protocol, train, test, folds in PROTOCOLS:
    for label, scheme, args in REGRESSION:
        if folds and label in SKIP_IN_CV:
            continue
        block = after(weka(scheme, args, train, test, folds),
                      "=== Cross-validation ===" if folds else "=== Error on test data ===")
        rows.append(dict(task="regression", protocol=protocol, scheme=label,
                         correlation=value(block, "Correlation coefficient"),
                         mae=value(block, "Mean absolute error"),
                         rmse=value(block, "Root mean squared error"),
                         accuracy_pct=None, kappa=None))
        print(f"{protocol:26} {label:22} MAE {rows[-1]['mae']}", flush=True)

confusion = {}
for label, scheme, args in CLASSIFICATION:
    out = weka(scheme, args, "train_classification.arff", "test_classification.arff")
    block = after(out, "=== Error on test data ===")
    hit = re.search(r"Correctly Classified Instances\s+(\d+)\s+([\d.]+)", block)
    rows.append(dict(task="classification", protocol="train -> test (244)", scheme=label,
                     correlation=None, mae=None, rmse=None,
                     accuracy_pct=float(hit.group(2)) if hit else None,
                     kappa=value(block, "Kappa statistic")))

    matrix = [[int(x) for x in m.group(1).split()]
              for m in (re.match(r"^\s*((?:\d+\s+)+)\|", ln)
                        for ln in after(block, "=== Confusion Matrix ===").splitlines()) if m]
    n = sum(sum(r) for r in matrix)
    k = len(matrix)
    confusion[label] = dict(
        n=n,
        exact_pct=round(100 * sum(matrix[i][i] for i in range(k)) / n, 1),
        within_one_band_pct=round(
            100 * sum(matrix[i][j] for i in range(k) for j in range(k) if abs(i - j) <= 1) / n, 1),
        errors_more_than_one_band=[
            dict(actual=GRADES[i], predicted=GRADES[j], count=matrix[i][j])
            for i in range(k) for j in range(k) if abs(i - j) > 1 and matrix[i][j]],
        bands=GRADES, matrix=matrix)
    print(f"{'classification':26} {label:22} "
          f"{confusion[label]['exact_pct']}% exact, "
          f"{confusion[label]['within_one_band_pct']}% within one band", flush=True)

import csv

with open(RESULTS / "weka_results.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=["task", "protocol", "scheme", "correlation", "mae",
                                       "rmse", "accuracy_pct", "kappa"])
    w.writeheader()
    w.writerows(rows)
(RESULTS / "weka_confusion.json").write_text(json.dumps(confusion, indent=2), encoding="utf-8")
print(f"\nwrote {RESULTS / 'weka_results.csv'} and {RESULTS / 'weka_confusion.json'}")
