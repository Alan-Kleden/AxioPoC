# -*- coding: utf-8 -*-
"""
eval_thread_holdout.py  —  Hold-out au niveau thread avec axe télotopique

Fonctions :
- Charge les embeddings message-level (Parquet) et l’axe (npy)
- Projette chaque message sur l’axe (cosine ou dot)
- Agrège par thread (mean ou max)
- Stratified hold-out split (au niveau thread)
- Choisit un seuil sur le set train (Youden) et mesure AUC/ACC sur test
- (Option) Permutation test sur le test pour p-value

Entrées minimales :
  --emb, --axis, --labels, --texts, --outdir

Paramètres utiles :
  --min-msgs       (>=5 conseillé)
  --score          (cos|dot)           [default: cos]
  --agg            (mean|max)          [default: mean]
  --test-size      (0<.. <1)           [default: 0.2]
  --seed           (int)               [default: 0]
  --perm-iters     (int)               [default: 0 ;  e.g. 1000 pour SAFE]
  --thread-col     [default: thread_id]
  --label-col      [default: label]
  --emb-prefix     [default: e]        (colonnes e0,e1,...)
  --msg-id-col     [default: __msg_id__] (optionnel)

Sorties :
  outdir/summary.json
  outdir/preds_test.csv (scores + label)
  outdir/preds_train.csv (scores + label)
  outdir/perm_test.json (si --perm-iters>0)
"""

from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit

# ---------- Utils JSON (caster numpy -> python) ----------
def to_py(obj):
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return obj

# ---------- Chargements ----------
def load_embeddings_parquet(path: str, emb_prefix="e", thread_col="thread_id"):
    import pyarrow.parquet as pq
    t = pq.read_table(path)
    df = t.to_pandas()
    # colonnes embedding
    emcols = [c for c in df.columns if isinstance(c, str) and c.startswith(emb_prefix)]
    if not emcols:
        raise ValueError(f"Aucune colonne embedding avec préfixe '{emb_prefix}' dans {path}")
    if thread_col not in df.columns:
        raise ValueError(f"Colonne '{thread_col}' absente de {path}")
    # gardons seulement ce qui est utile
    cols = [thread_col] + emcols
    # msg_id optionnel
    if "__msg_id__" in df.columns:
        cols = ["__msg_id__"] + cols
    df = df[cols].copy()
    # types
    df[thread_col] = pd.to_numeric(df[thread_col], errors="coerce").astype("Int64")
    return df, emcols

def load_labels_csv(path: str, thread_col="thread_id", label_col="label"):
    lab = pd.read_csv(path, usecols=[thread_col, label_col])
    lab[thread_col] = pd.to_numeric(lab[thread_col], errors="coerce").astype("Int64")
    lab[label_col] = pd.to_numeric(lab[label_col], errors="coerce").astype(int)
    # Sanity : labels binaires
    u = sorted(lab[label_col].unique().tolist())
    if not set(u).issubset({0,1}):
        raise ValueError(f"Labels non binaires trouvés: {u}")
    return lab

def load_axis_npy(path: str):
    u = np.load(path)
    if u.ndim != 1:
        raise ValueError(f"Axe attendu 1D, trouvé shape={u.shape}")
    return u

# ---------- Scoring ----------
def normalize_rows(M: np.ndarray, eps=1e-12):
    # norme L2 par ligne
    nrm = np.linalg.norm(M, axis=1, keepdims=True)
    nrm = np.maximum(nrm, eps)
    return M / nrm

def project_scores(E: np.ndarray, axis: np.ndarray, mode="cos"):
    if mode == "cos":
        a = axis / max(np.linalg.norm(axis), 1e-12)
        E_ = normalize_rows(E)
        s = E_.dot(a)
        return s
    elif mode == "dot":
        return E.dot(axis)
    else:
        raise ValueError(f"mode inconnu: {mode}")

def aggregate_thread_scores(df_msg: pd.DataFrame, score_col: str, thread_col="thread_id", agg="mean"):
    if agg == "mean":
        aggser = df_msg.groupby(thread_col)[score_col].mean()
    elif agg == "max":
        aggser = df_msg.groupby(thread_col)[score_col].max()
    else:
        raise ValueError(f"agg inconnu: {agg}")
    out = aggser.reset_index().rename(columns={score_col:"score"})
    return out

# ---------- Thresholding (Youden) ----------
def best_threshold_youden(y_true, scores):
    fpr, tpr, thr = roc_curve(y_true, scores)
    youden = tpr - fpr
    k = int(np.argmax(youden))
    return float(thr[k])

# ---------- Permutation test ----------
def permutation_pvalue(y_true, scores, n_iters=1000, seed=0):
    if n_iters <= 0:
        return None
    rng = np.random.default_rng(seed)
    auc_real = roc_auc_score(y_true, scores)
    null = []
    y = np.array(y_true)
    for _ in range(n_iters):
        y_perm = y.copy()
        rng.shuffle(y_perm)
        null.append(roc_auc_score(y_perm, scores))
    null = np.array(null, dtype=float)
    p = float((np.sum(null >= auc_real) + 1) / (len(null) + 1))  # right-tail
    return {"auc_real": auc_real, "null_mean": float(null.mean()), "null_sd": float(null.std(ddof=1)), "p": p}

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True, help="Parquet embeddings (messages)")
    ap.add_argument("--axis", required=True, help="Axe npy")
    ap.add_argument("--labels", required=True, help="CSV labels (thread_id,label)")
    ap.add_argument("--texts", required=True, help="CSV textes (pour traçabilité; non obligatoire au calcul)")
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--thread-col", default="thread_id")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--msg-id-col", default="__msg_id__")
    ap.add_argument("--emb-prefix", default="e")

    ap.add_argument("--min-msgs", type=int, default=5)
    ap.add_argument("--score", choices=["cos","dot"], default="cos")
    ap.add_argument("--agg", choices=["mean","max"], default="mean")

    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--perm-iters", type=int, default=0)

    args = ap.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # Chargements
    emb_df, emcols = load_embeddings_parquet(args.emb, emb_prefix=args.emb_prefix, thread_col=args.thread_col)
    axis = load_axis_npy(args.axis)
    labels = load_labels_csv(args.labels, thread_col=args.thread_col, label_col=args.label_col)

    # Projection message-level
    E = emb_df[emcols].to_numpy(dtype=np.float32)
    s = project_scores(E, axis.astype(np.float32), mode=args.score)
    emb_df = emb_df.assign(_score=s)

    # Agrégation par thread
    # 1) on filtre threads avec >= min-msgs
    vc = emb_df[args.thread_col].value_counts()
    keep_threads = set(vc[vc >= args.min_msgs].index.astype(int))
    df_kept = emb_df[emb_df[args.thread_col].isin(keep_threads)].copy()

    # 2) join labels et drop NA
    thr_scores = aggregate_thread_scores(df_kept, "_score", thread_col=args.thread_col, agg=args.agg)
    thr = thr_scores.merge(labels, on=args.thread_col, how="inner").dropna(subset=[args.label_col])
    thr[args.thread_col] = thr[args.thread_col].astype(int)
    thr[args.label_col]  = thr[args.label_col].astype(int)

    # Sanity
    n_threads = len(thr)
    lab_counts = dict(Counter(thr[args.label_col].tolist()))
    if len(lab_counts) < 2:
        raise ValueError(f"Stratification impossible: une seule classe dans le set (counts={lab_counts}).")

    # Hold-out split (fix du bug : prendre next(...) sur l’itérateur)
    y_all = thr[args.label_col].to_numpy()
    sss = StratifiedShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
    train_idx, test_idx = next(sss.split(np.zeros(n_threads), y_all))
    thr_train = thr.iloc[train_idx].reset_index(drop=True)
    thr_test  = thr.iloc[test_idx].reset_index(drop=True)

    # AUCs
    auc_train = roc_auc_score(thr_train[args.label_col], thr_train["score"])
    auc_test  = roc_auc_score(thr_test[args.label_col],  thr_test["score"])

    # Seuil choisi sur train (Youden)
    thr_opt = best_threshold_youden(thr_train[args.label_col].to_numpy(), thr_train["score"].to_numpy())
    yhat_train = (thr_train["score"].to_numpy() >= thr_opt).astype(int)
    yhat_test  = (thr_test["score"].to_numpy()  >= thr_opt).astype(int)

    acc_train = accuracy_score(thr_train[args.label_col], yhat_train)
    acc_test  = accuracy_score(thr_test[args.label_col],  yhat_test)

    # Permutation SAFE (optionnel) — sur test uniquement
    perm = None
    if args.perm_iters and args.perm_iters > 0:
        perm = permutation_pvalue(thr_test[args.label_col].to_numpy(), thr_test["score"].to_numpy(),
                                  n_iters=int(args.perm_iters), seed=args.seed)

    # Sauvegardes
    # Preds CSV
    thr_train.assign(pred=yhat_train).to_csv(outdir / "preds_train.csv", index=False)
    thr_test.assign(pred=yhat_test).to_csv(outdir / "preds_test.csv", index=False)

    # Summary JSON
    summary = {
        "emb_path": str(Path(args.emb).resolve()),
        "axis_path": str(Path(args.axis).resolve()),
        "labels_path": str(Path(args.labels).resolve()),
        "texts_path": str(Path(args.texts).resolve()),
        "thread_col": args.thread_col,
        "label_col": args.label_col,
        "emb_prefix": args.emb_prefix,
        "min_msgs": int(args.min_msgs),
        "score_mode": args.score,
        "agg_mode": args.agg,
        "test_size": float(args.test_size),
        "seed": int(args.seed),

        "n_threads_kept": int(n_threads),
        "label_counts": {str(k): int(v) for k, v in lab_counts.items()},

        "auc_train": to_py(auc_train),
        "auc_test":  to_py(auc_test),
        "acc_train": to_py(acc_train),
        "acc_test":  to_py(acc_test),
        "thr_opt_youden": to_py(thr_opt),
    }
    if perm is not None:
        summary["perm_test"] = {k: to_py(v) for k, v in perm.items()}

    (outdir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # Log console
    print(f"[OK] Hold-out @thread | n={n_threads} | split train={len(train_idx)} test={len(test_idx)}")
    print(f"     AUC train={auc_train:.3f} | AUC test={auc_test:.3f} | ACC test={acc_test:.3f} (thr={thr_opt:.4f})")
    if perm is not None:
        print(f"     Permutation: auc_real={perm['auc_real']:.3f} | null≈{perm['null_mean']:.3f}±{perm['null_sd']:.3f} | p={perm['p']:.4f}")

if __name__ == "__main__":
    main()
