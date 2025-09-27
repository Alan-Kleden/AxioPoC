# -*- coding: utf-8 -*-
"""
eval_da.py (with optional diag weights)
Évalue D_a intra-communauté (LogReg sur D_a uniquement).
Usage:
  python scripts/eval_da.py --feat FEATURES_AXIO.csv --raw LABELS.csv --signature SIGNATURE.pkl --metric sigma_inv|diag --cv 5 --reps 3 [--weights WEIGHTS.pkl]
"""
import argparse, joblib, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
Da = np.nan_to_num(Da, nan=0.0, posinf=0.0, neginf=0.0)


EXCLUDE = {"thread_id","label"}

def compute_da_matrix(X: np.ndarray, mu: np.ndarray, cov: np.ndarray, metric: str="sigma_inv",
                      eps: float=1e-6, w: np.ndarray|None=None, cols=None, w_cols=None) -> np.ndarray:
    """
    D_a(x;C) = sqrt( (x-mu)^T M (x-mu) )
    - 'sigma_inv': M = (cov + eps I)^(-1)
    - 'diag'     : M = diag(w) si w fourni, sinon I
    """
    V = X - mu
    if metric == "sigma_inv":
        d = cov.shape[0]
        M = np.linalg.inv(cov + eps * np.eye(d))
        return np.sqrt(np.einsum("ij,jk,ik->i", V, M, V))
    elif metric == "diag":
        d = X.shape[1]
        if w is None:
            return np.sqrt(np.sum(V*V, axis=1))
        # aligner w sur l'ordre des 'cols' si w_cols donné
        if w_cols is not None and cols is not None and w.shape[0] == len(w_cols):
            w_map = {c: w[i] for i,c in enumerate(w_cols)}
            w_vec = np.array([w_map.get(c,1.0) for c in cols], dtype=float)
        else:
            w_vec = w
        return np.sqrt(np.sum((V*V) * w_vec, axis=1))
    else:
        raise SystemExit(f"Unsupported metric '{metric}'")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--feat", required=True)
    p.add_argument("--raw",  required=True)
    p.add_argument("--signature", required=True)
    p.add_argument("--metric", choices=["sigma_inv","diag"], default="sigma_inv")
    p.add_argument("--cv", type=int, default=5)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--weights", default="")  # optional weights.pkl for diag
    args = p.parse_args()

    df_feat = pd.read_csv(args.feat)
    if "label" in df_feat.columns:
        df = df_feat.copy()
    else:
        df_raw = pd.read_csv(args.raw)[["thread_id","label"]]
        df = df_feat.merge(df_raw, on="thread_id", how="inner")

    if "label" not in df.columns:
        if "label_y" in df.columns: df = df.rename(columns={"label_y":"label"})
        elif "label_x" in df.columns: df = df.rename(columns={"label_x":"label"})
        else: raise SystemExit("No 'label' column available after loading/merge.")

    sig = joblib.load(args.signature)
    cols = sig["cols"]
    mu   = np.asarray(sig["mu"], dtype=float)
    cov  = np.asarray(sig["cov"], dtype=float)

    X = df[cols].to_numpy(dtype=float)
    y = df["label"].to_numpy(dtype=int)

    w, w_cols = None, None
    if args.weights:
        w_obj = joblib.load(args.weights)
        w, w_cols = np.asarray(w_obj["w"], dtype=float), list(w_obj["cols"])

    Da = compute_da_matrix(X, mu, cov, metric=args.metric, w=w, cols=cols, w_cols=w_cols).reshape(-1,1)

    aucs, accs = [], []
    skf = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=42)
    for _ in range(args.reps):
        for tr, te in skf.split(Da, y):
            m = LogisticRegression(max_iter=2000).fit(Da[tr], y[tr])
            p1 = m.predict_proba(Da[te])[:,1]
            aucs.append(roc_auc_score(y[te], p1))
            accs.append(accuracy_score(y[te], m.predict(Da[te])))

    print(f"Intra {args.metric}{' [diag(w)]' if args.weights else ''} | AUC mean±sd: {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
    print(f"Intra {args.metric}{' [diag(w)]' if args.weights else ''} | ACC mean±sd: {np.mean(accs):.3f} ± {np.std(accs):.3f}")

if __name__ == "__main__":
    main()
