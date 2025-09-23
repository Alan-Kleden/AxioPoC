# -*- coding: utf-8 -*-
import csv, os, argparse
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

p = argparse.ArgumentParser()
p.add_argument("--feat", default=os.environ.get("FEAT", r"artifacts_benchB_msg\features.csv"))
p.add_argument("--raw",  default=os.environ.get("RAW",  r"data\convokit\cmv\cmv.csv"))
args = p.parse_args()

FEAT, RAW = args.feat, args.raw

# 1) outcome par thread
outcome_by_tid = {}
with open(RAW, newline="", encoding="utf-8") as f:
    r = csv.DictReader(f)
    for row in r:
        tid = row["thread_id"]
        if tid not in outcome_by_tid:
            outcome_by_tid[tid] = row["outcome"]

# 2) charge features et joint outcome
X, y, tids = [], [], []
with open(FEAT, newline="", encoding="utf-8") as f:
    r = csv.DictReader(f)
    for row in r:
        tid = row["thread_id"]
        if tid in outcome_by_tid:
            y.append(int(outcome_by_tid[tid]))
            X.append([float(row["R_mean"]), float(row["R_last"]), float(row["R_slope"])])
            tids.append(tid)

X = np.array(X); y = np.array(y)
print(f"Threads: {len(y)} | Positives: {y.sum()} ({y.mean():.1%}) | FEAT={FEAT} | RAW={RAW}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
aucs, accs = [], []
for tr, te in cv.split(X, y):
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X[tr], y[tr])
    p = clf.predict_proba(X[te])[:,1]
    aucs.append(roc_auc_score(y[te], p))
    accs.append(accuracy_score(y[te], (p>=0.5).astype(int)))

print(f"AUC mean±sd: {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
print(f"ACC mean±sd: {np.mean(accs):.3f} ± {np.std(accs):.3f}")
