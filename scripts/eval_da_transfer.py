# -*- coding: utf-8 -*-
"""
eval_da_transfer.py (with optional diag weights)
Transfert: entraîne un logit sur D_a(source), évalue sur D_a(cible) (signature cible).
Usage:
  python scripts/eval_da_transfer.py --feat-src ... --raw-src ... --signature-src ... --feat-tgt ... --raw-tgt ... --signature-tgt ... --metric sigma_inv|diag [--weights WEIGHTS.pkl]
Note:
  - Si --weights fourni et metric=diag, on applique diag(w) des poids appris sur la source.
  - On aligne w sur les colonnes de chaque signature par nom.
"""
import argparse, joblib, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

def compute_da(X, mu, cov, metric="sigma_inv", eps=1e-6, w=None, cols=None, w_cols=None):
    V = X - mu
    if metric == "sigma_inv":
        d = cov.shape[0]
        M = np.linalg.inv(cov + eps*np.eye(d))
        return np.sqrt(np.einsum("ij,jk,ik->i", V, M, V))
    elif metric == "diag":
        if w is None:
            return np.sqrt(np.sum(V*V, axis=1))
        # aligner w par nom de colonne
        w_map = {c: w[i] for i,c in enumerate(w_cols)}
        w_vec = np.array([w_map.get(c,1.0) for c in cols], dtype=float)
        return np.sqrt(np.sum((V*V) * w_vec, axis=1))
    else:
        raise SystemExit(f"Unsupported metric '{metric}'")

def load_xy(feat_path, raw_path, cols):
    df_feat = pd.read_csv(feat_path)
    if "label" in df_feat.columns:
        df = df_feat.copy()
    else:
        raw = pd.read_csv(raw_path)[["thread_id","label"]]
        df = df_feat.merge(raw, on="thread_id", how="inner")

    if "label" not in df.columns:
        if "label_y" in df.columns: df = df.rename(columns={"label_y":"label"})
        elif "label_x" in df.columns: df = df.rename(columns={"label_x":"label"})
        else: raise SystemExit("No 'label' column available after loading/merge.")

    X = df[cols].to_numpy(dtype=float)
    y = df["label"].to_numpy(dtype=int)
    return X, y

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--feat-src", required=True)
    p.add_argument("--raw-src", required=True)
    p.add_argument("--signature-src", required=True)
    p.add_argument("--feat-tgt", required=True)
    p.add_argument("--raw-tgt", required=True)
    p.add_argument("--signature-tgt", required=True)
    p.add_argument("--metric", choices=["sigma_inv","diag"], default="sigma_inv")
    p.add_argument("--weights", default="")  # weights learned on source (optional, diag only)
    args = p.parse_args()

    sig_s = joblib.load(args.signature_src)
    sig_t = joblib.load(args.signature_tgt)
    cols_s, mu_s, cov_s = sig_s["cols"], np.asarray(sig_s["mu"]), np.asarray(sig_s["cov"])
    cols_t, mu_t, cov_t = sig_t["cols"], np.asarray(sig_t["mu"]), np.asarray(sig_t["cov"])

    Xs, ys = load_xy(args.feat_src, args.raw_src, cols_s)
    Xt, yt = load_xy(args.feat_tgt, args.raw_tgt, cols_t)

    w, w_cols = None, None
    if args.weights:
        w_obj = joblib.load(args.weights)
        w, w_cols = np.asarray(w_obj["w"], dtype=float), list(w_obj["cols"])

    Das = compute_da(Xs, mu_s, cov_s, metric=args.metric, w=w, cols=cols_s, w_cols=w_cols).reshape(-1,1)
    Dat = compute_da(Xt, mu_t, cov_t, metric=args.metric, w=w, cols=cols_t, w_cols=w_cols).reshape(-1,1)

    clf = LogisticRegression(max_iter=2000).fit(Das, ys)
    p1 = clf.predict_proba(Dat)[:,1]
    auc = roc_auc_score(yt, p1)
    acc = accuracy_score(yt, (p1 >= 0.5).astype(int))

    tag = f"{args.metric}{' [diag(w_src)]' if args.weights else ''}"
    print(f"Transfer {tag} Eval: n_src={len(ys)}, n_tgt={len(yt)}, AUC {auc:.6f}, ACC {acc:.6f}")

if __name__ == "__main__":
    main()
