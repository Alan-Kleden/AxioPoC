# -*- coding: utf-8 -*-
"""
make_signature.py
Calcule la signature de communauté: moyenne (mu) et covariance (cov) sur les features axio.
Usage:
  python scripts/make_signature.py --feat FEATURES_AXIO.csv --out SIGNATURE.pkl [--cols a,b,c]
"""
import argparse, joblib, numpy as np, pandas as pd

EXCLUDE = {"thread_id","label"}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--feat", required=True)
    p.add_argument("--out",  required=True)
    p.add_argument("--cols", default="")
    args = p.parse_args()

    df = pd.read_csv(args.feat)
    if args.cols.strip():
        cols = [c.strip() for c in args.cols.split(",")]
    else:
        cols = [c for c in df.columns if c not in EXCLUDE]

    X = df[cols].to_numpy(dtype=float)
    mu = X.mean(axis=0)
    # Regularisation numérique légère sur cov
    cov = np.cov(X, rowvar=False)
    # Sauvegarde
    joblib.dump({"cols": cols, "mu": mu, "cov": cov}, args.out)
    print(f"Saved signature with {len(cols)} dims → {args.out}")

if __name__ == "__main__":
    main()
