#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import sys
import numpy as np

from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline

# --- charge X (features) ---
def load_X(path, cols):
    tids, X = [], []
    with open(path, encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            tids.append(row["thread_id"])
            X.append([float(row[c]) for c in cols])
    return np.array(tids), np.array(X, dtype=float)

# --- charge y (labels) ---
def load_y(path):
    ymap = {}
    with open(path, encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            # outcome doit être 0/1; on garde le 1er vu par thread_id
            tid = row["thread_id"]
            if tid not in ymap and row.get("outcome", "") != "":
                ymap[tid] = int(row["outcome"])
    return ymap

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--feat", required=True, help="CSV des features (doit contenir thread_id + colonnes)")
    p.add_argument("--raw",  required=True, help="CSV brut (thread_id,text,timestamp,outcome)")
    p.add_argument("--cols", required=True, help="Colonnes de features, séparées par des virgules")
    p.add_argument("--model", choices=["logreg", "rf", "hgb"], default="rf")
    p.add_argument("--reps", type=int, default=5, help="Nb de répétitions (seeds) du K-Fold")
    p.add_argument("--cv",   type=int, default=5, help="Nb de folds (StratifiedKFold)")
    p.add_argument("--shuffle-labels", action="store_true", help="Permutation aléatoire des labels (sanity check)")
    p.add_argument("--scale", action="store_true", help="Forcer un StandardScaler (utile surtout pour logreg)")
    args = p.parse_args()

    cols = [c.strip() for c in args.cols.split(",") if c.strip()]
    tids, X = load_X(args.feat, cols)
    ymap = load_y(args.raw)

    # aligne X et y (filtre si besoin)
    y = np.array([ymap[t] for t in tids], dtype=int)

    if args.shuffle_labels:
        rng = np.random.RandomState(123)
        y = rng.permutation(y)

    print(f"Threads: {len(y)} | Positives: {int(y.sum())} ({100.0 * y.mean():.1f}%)")
    print(f"Cols: {cols}")

    # sélection du modèle
    if args.model == "logreg":
        from sklearn.linear_model import LogisticRegression
        if args.scale:
            clf_factory = lambda seed: make_pipeline(
                __import__("sklearn.preprocessing").preprocessing.StandardScaler(),
                LogisticRegression(max_iter=1000, random_state=seed)
            )
        else:
            clf_factory = lambda seed: LogisticRegression(max_iter=1000, random_state=seed)
        label = "LOGREG"
    elif args.model == "rf":
        from sklearn.ensemble import RandomForestClassifier
        # baseline raisonnable (celle qu’on a utilisée)
        clf_factory = lambda seed: RandomForestClassifier(
            n_estimators=400, max_depth=None, min_samples_leaf=5,
            max_features="sqrt", random_state=seed
        )
        label = "RF"
    else:
        # HistGradientBoosting (optionnel)
        from sklearn.ensemble import HistGradientBoostingClassifier
        clf_factory = lambda seed: HistGradientBoostingClassifier(random_state=seed)
        label = "HGB"

    aucs, accs = [], []
    # CV répété (seeds = 0..reps-1)
    for seed in range(args.reps):
        skf = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=seed)
        for tr, te in skf.split(X, y):
            clf = clf_factory(seed)
            clf.fit(X[tr], y[tr])

            # proba si dispo, sinon decision_function/predict
            if hasattr(clf, "predict_proba"):
                p = clf.predict_proba(X[te])[:, 1]
            elif hasattr(clf, "decision_function"):
                # on mappe décision en pseudo-proba monotone
                s = clf.decision_function(X[te])
                # squashing (min-max) pour rester dans [0,1]
                smin, smax = np.min(s), np.max(s)
                p = (s - smin) / (smax - smin + 1e-9)
            else:
                # fallback binaire
                pred = clf.predict(X[te])
                p = pred.astype(float)

            yhat = (p >= 0.5).astype(int)
            aucs.append(roc_auc_score(y[te], p))
            accs.append(accuracy_score(y[te], yhat))

    print(f"{label} | AUC mean±sd: {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
    print(f"{label} | ACC mean±sd: {np.mean(accs):.3f} ± {np.std(accs):.3f}")

if __name__ == "__main__":
    sys.exit(main())
