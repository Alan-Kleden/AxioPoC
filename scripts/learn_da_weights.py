# -*- coding: utf-8 -*-
"""
learn_da_weights.py
Apprend des poids diag(w) par régression logistique (source) pour pondérer D_a.
Usage:
  python scripts/learn_da_weights.py --feat FEATURES_AXIO.csv --raw LABELS.csv --out WEIGHTS.pkl [--cols a,b,c]
Notes:
  - On apprend un modèle LogReg sur X -> y et on prend |coef_| comme importance.
  - On normalise w pour que mean(w)=1 (échelle neutre).
"""
import argparse, joblib, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression

EXCLUDE = {"thread_id","label"}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--feat", required=True)
    p.add_argument("--raw",  required=True)
    p.add_argument("--out",  required=True)
    p.add_argument("--cols", default="")
    args = p.parse_args()

    dfX = pd.read_csv(args.feat)
    if "label" in dfX.columns:
        df = dfX.copy()
    else:
        dfY = pd.read_csv(args.raw)[["thread_id","label"]]
        df = dfX.merge(dfY, on="thread_id", how="inner")

    # Colonnes à utiliser
    if args.cols.strip():
        cols = [c.strip() for c in args.cols.split(",")]
    else:
        cols = [c for c in df.columns if c not in EXCLUDE]

    X = df[cols].to_numpy(dtype=float)
    y = df["label"].to_numpy(dtype=int)

    # LogReg linéaire, pas de pénalisation forte (C grand) pour dégager des poids
    clf = LogisticRegression(max_iter=2000, C=10.0, solver="lbfgs")
    clf.fit(X, y)

    coef = np.abs(clf.coef_.ravel())
    # fallback si tous zéros (très rare): w=1
    if not np.isfinite(coef).all() or coef.sum() == 0:
        w = np.ones_like(coef)
    else:
        w = coef / (coef.mean() if coef.mean() != 0 else 1.0)  # mean(w)=1

    joblib.dump({"cols": cols, "w": w}, args.out)
    print(f"Saved weights for {len(cols)} dims → {args.out}; mean(w)={w.mean():.3f}")

if __name__ == "__main__":
    main()
