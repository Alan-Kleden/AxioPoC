#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse, os, math, json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt

def exp_weights(n, lam):
    # indices 0..n-1, poids ~ exp(-lam * (n-1-i)) -> poids récents plus forts
    idx = np.arange(n, dtype=float)
    w = np.exp(-lam * ( (n-1) - idx ))
    s = w.sum()
    return w / s if s != 0 else np.ones(n)/n

def time_weighted_mean(x, lam):
    if len(x) == 0:
        return np.nan
    w = exp_weights(len(x), lam)
    return float(np.dot(w, x))

def telos_slope(x):
    if len(x) < 2:
        return 0.0
    # slope sur l'indice temporel (0..n-1)
    t = np.arange(len(x), dtype=float)
    m, b = np.polyfit(t, x, 1)
    return float(m)

def telos_vol(x):
    if len(x) < 2:
        return 0.0
    dx = np.diff(x)
    return float(dx.std(ddof=0))

def project_telos(emb_mat, axis_vec):
    # emb_mat: (n_msg, d), axis_vec: (d,)
    # normalisation de l'axe pour cos-like projection
    v = axis_vec.astype(np.float32)
    v = v / (np.linalg.norm(v) + 1e-12)
    return emb_mat @ v

def load_embeddings(emb_path, cols_prefix="e"):
    # lit uniquement thread_id, temps, et vecteurs e*
    table = pq.read_table(emb_path)
    df = table.to_pandas()
    # colonnes embedding
    ecols = [c for c in df.columns if str(c).startswith(cols_prefix)]
    if "__t__" in df.columns:
        tcol = "__t__"
    elif "created_utc" in df.columns:
        tcol = "created_utc"
    elif "time" in df.columns:
        tcol = "time"
    elif "timestamp" in df.columns:
        tcol = "timestamp"
    else:
        # fallback: ordre d'apparition
        df["__t__"] = np.arange(len(df))
        tcol = "__t__"
    return df[["thread_id", tcol] + ecols].rename(columns={tcol: "__t__"}), ecols

def main():
    ap = argparse.ArgumentParser(description="Export per-thread temporal summary (telos) from message embeddings + axis.")
    ap.add_argument("--emb", required=True, help="Parquet des embeddings (message-level) avec 'thread_id', '__t__' (ou temps), et colonnes e0..eN")
    ap.add_argument("--axis", required=True, help="Fichier .npy de l'axe télotopique (dim = embeddings)")
    ap.add_argument("--labels", required=True, help="CSV labels avec colonnes: thread_id,label (thread_id castable en int)")
    ap.add_argument("--outcsv", required=True, help="CSV de sortie: résumé par thread (telos_wd_l*, telos_slope, telos_vol)")
    ap.add_argument("--min-msgs", type=int, default=5, help="Seuil minimum de messages par thread à garder")
    ap.add_argument("--sample-per-class", type=int, default=0, help="Si >0, échantillonne jusqu'à N par classe (accélère)")
    ap.add_argument("--plots-dir", default="", help="Si fourni, écrit quelques figures (distributions/trajectoires)")
    ap.add_argument("--lam02", type=float, default=0.02, help="lambda pour telos_wd_l0.02")
    ap.add_argument("--lam05", type=float, default=0.05, help="lambda pour telos_wd_l0.05")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.outcsv) or ".", exist_ok=True)
    if args.plots_dir:
        os.makedirs(args.plots_dir, exist_ok=True)

    print("[load] embeddings…")
    df_emb, ecols = load_embeddings(args.emb)
    print(f"   rows={len(df_emb)} | dim={len(ecols)} | tcol='__t__'")

    print("[load] axis…")
    axis = np.load(args.axis)
    if axis.ndim != 1:
        axis = axis.reshape(-1)
    if len(ecols) != axis.shape[0]:
        raise ValueError(f"Axis dim {axis.shape[0]} != emb dim {len(ecols)}")

    print("[load] labels…")
    df_lab = pd.read_csv(args.labels, usecols=["thread_id","label"])
    # Force int pour join robuste
    df_lab["thread_id"] = df_lab["thread_id"].astype(int)

    # Projection telos par message
    print("[proj] project telos for all messages…")
    E = df_emb[ecols].to_numpy(dtype=np.float32, copy=False)
    tel = project_telos(E, axis).astype(np.float32)
    df_emb = df_emb[["thread_id","__t__"]].copy()
    df_emb["telos"] = tel
    # tri temporel intra-thread
    df_emb = df_emb.sort_values(["thread_id","__t__"])

    # comptages
    vc = df_emb["thread_id"].value_counts()
    keep = set(vc[vc >= args.min_msgs].index)
    print(f"[filter] min-msgs={args.min_msgs} -> threads kept={len(keep)}")

    # jonction labels
    df_thr = pd.DataFrame({"thread_id": list(keep)})
    df_thr = df_thr.merge(df_lab, on="thread_id", how="inner")
    print(f"[labels] labeled threads kept={len(df_thr)}")

    # échantillonnage (optionnel) pour accélérer
    if args.sample_per_class and args.sample_per_class > 0:
        dfs = []
        for y in [0,1]:
            sub = df_thr[df_thr["label"]==y]
            if len(sub) > args.sample_per_class:
                sub = sub.sample(args.sample_per_class, random_state=1337)
            dfs.append(sub)
        df_thr = pd.concat(dfs, ignore_index=True)
        print(f"[sample] sample-per-class={args.sample_per_class} -> {len(df_thr)} threads")

    # calcul résumé par thread
    rows = []
    # pre-split to speed
    g = df_emb.groupby("thread_id", sort=False)
    for tid, y in df_thr[["thread_id","label"]].itertuples(index=False):
        s = g.get_group(tid)["telos"].to_numpy(dtype=np.float32, copy=False)
        r = {
            "thread_id": int(tid),
            "label": int(y),
            "n_msgs": int(len(s)),
            f"telos_wd_l{args.lam02}": time_weighted_mean(s, args.lam02),
            f"telos_wd_l{args.lam05}": time_weighted_mean(s, args.lam05),
            "telos_slope": telos_slope(s),
            "telos_vol": telos_vol(s),
        }
        rows.append(r)

    df_out = pd.DataFrame(rows)
    df_out.to_csv(args.outcsv, index=False, encoding="utf-8")
    print(f"[OK] summary -> {args.outcsv} | rows={len(df_out)}")

    # quelques figures optionnelles
    if args.plots_dir:
        for col, lab in [("telos_slope","slope"), ("telos_vol","vol")]:
            for cls, name in [(0,"neg"), (1,"pos")]:
                x = df_out.loc[df_out["label"]==cls, col].to_numpy()
                plt.figure()
                plt.hist(x, bins=40)
                plt.title(f"dist {lab} (label={cls})")
                plt.tight_layout()
                fn = os.path.join(args.plots_dir, f"dist_{lab}_{name}_min{args.min_msgs}.png")
                plt.savefig(fn, dpi=130)
                plt.close()
        print(f"[OK] plots -> {args.plots_dir}")

if __name__ == "__main__":
    main()
