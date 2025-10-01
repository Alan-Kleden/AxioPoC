# scripts/explore_temporal_patterns.py
# -*- coding: utf-8 -*-
"""
Exploration visuelle des trajectoires telos(t) sur AfD.
- Echantillonne N fils positifs et N fils négatifs (label=1/0) avec ≥ min_msgs
- Calcule telos(t) (= projection embedding . axis) par ordre temporel par fil
- Sauve: figures PNG + CSV récapitulatif (mean_w, slope, vol).
"""

import argparse, os
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt

def load_axis(path: Path) -> np.ndarray:
    a = np.load(path)
    a = a / (np.linalg.norm(a) + 1e-12)
    return a

def load_embedded_msgs(parquet_path: Path):
    # On lit thread_id, time, et vecteurs e0..e*
    table = pq.read_table(parquet_path)
    df = table.to_pandas()
    # time: essaye colonnes connues
    time_col = None
    for c in ["__t__", "created_utc", "time", "timestamp"]:
        if c in df.columns:
            time_col = c; break
    if time_col is None:
        raise ValueError("Aucune colonne temps trouvée (__t__/created_utc/time/timestamp).")
    emb_cols = [c for c in df.columns if c.startswith("e")]
    return df[["thread_id", time_col] + emb_cols].rename(columns={time_col:"t"}), emb_cols

def telos_scores(df_emb: pd.DataFrame, emb_cols, axis_vec: np.ndarray) -> pd.DataFrame:
    # projection telos = E . axis
    V = df_emb[emb_cols].to_numpy(dtype=np.float32)
    s = V.dot(axis_vec.astype(np.float32))
    out = df_emb[["thread_id","t"]].copy()
    out["telos"] = s
    return out

def exp_wmean(x, lam=0.02):
    # poids croissant vers la fin (index 0..n-1)
    n = len(x)
    w = np.exp(lam * np.arange(n, dtype=np.float32))
    w /= w.sum() if w.sum() != 0 else 1.0
    return float((x * w).sum())

def fit_slope(x):
    # pente linéaire (index comme temps)
    if len(x) < 2: return 0.0
    t = np.arange(len(x), dtype=np.float32)
    t = (t - t.mean()) / (t.std() + 1e-12)
    x = (x - x.mean()) / (x.std() + 1e-12)
    # corrélation ~ pente normalisée
    return float((t * x).mean())

def volatility(x):
    if len(x) < 2: return 0.0
    return float(np.std(np.diff(x), ddof=1))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True, help="Parquet embeddings messages (avec e0..e*, thread_id, temps)")
    ap.add_argument("--axis", required=True, help="Axe télotopique .npy (dimension = dim embedding)")
    ap.add_argument("--labels", required=True, help="CSV labels AfD (thread_id,label)")
    ap.add_argument("--min-msgs", type=int, default=5)
    ap.add_argument("--n-per-class", type=int, default=50)
    ap.add_argument("--outdir", default="REPORTS/temporal_explore")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    axis = load_axis(Path(args.axis))
    df_emb, emb_cols = load_embedded_msgs(Path(args.emb))
    df_emb = df_emb.dropna(subset=["thread_id","t"]).copy()
    # ints
    df_emb["thread_id"] = pd.to_numeric(df_emb["thread_id"], errors="coerce").astype("Int64")
    df_emb = df_emb.dropna(subset=["thread_id"]).copy()
    df_emb["thread_id"] = df_emb["thread_id"].astype(int)

    # proj telos
    df_tel = telos_scores(df_emb, emb_cols, axis)
    # ordre chrono par fil
    df_tel = df_tel.sort_values(["thread_id","t"]).reset_index(drop=True)

    # compte msgs/thread
    vc = df_tel["thread_id"].value_counts()
    keep = set(vc[vc >= args.min_msgs].index)
    df_tel = df_tel[df_tel.thread_id.isin(keep)]

    # labels
    lab = pd.read_csv(args.labels, usecols=["thread_id","label"])
    lab["thread_id"] = pd.to_numeric(lab["thread_id"], errors="coerce").astype("Int64")
    lab = lab.dropna(subset=["thread_id"]).copy()
    lab["thread_id"] = lab["thread_id"].astype(int)
    lab = lab[lab.thread_id.isin(keep)]

    # échantillons
    pos_ids = lab.loc[lab.label==1, "thread_id"].sample(min(args.n_per_class, (lab.label==1).sum()), random_state=42).tolist()
    neg_ids = lab.loc[lab.label==0, "thread_id"].sample(min(args.n_per_class, (lab.label==0).sum()), random_state=42).tolist()

    def summarize_thread(thid):
        s = df_tel.loc[df_tel.thread_id==thid, "telos"].to_numpy()
        return {
            "thread_id": thid,
            "n_msgs": int(len(s)),
            "telos_mean_w_l002": exp_wmean(s, lam=0.02),
            "telos_mean_w_l005": exp_wmean(s, lam=0.05),
            "telos_slope": fit_slope(s),
            "telos_vol": volatility(s),
        }

    # Figures de trajectoires
    def plot_trajectories(ids, title, fname):
        plt.figure(figsize=(9,6))
        for th in ids:
            s = df_tel.loc[df_tel.thread_id==th, "telos"].to_numpy()
            if len(s)==0: continue
            plt.plot(range(len(s)), s, alpha=0.25)
        plt.title(title)
        plt.xlabel("index dans le fil")
        plt.ylabel("telos(t) = proj(E(t), axis)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(outdir / fname, dpi=150)
        plt.close()

    plot_trajectories(pos_ids, f"AfD ≥{args.min_msgs} — Traj. telos(t) (label=1)", f"traj_pos_min{args.min_msgs}.png")
    plot_trajectories(neg_ids, f"AfD ≥{args.min_msgs} — Traj. telos(t) (label=0)", f"traj_neg_min{args.min_msgs}.png")

    # Distributions slope/vol
    def dist_plot(metric, ids, title, fname):
        vals = []
        for th in ids:
            s = df_tel.loc[df_tel.thread_id==th, "telos"].to_numpy()
            if len(s)==0: continue
            if metric=="slope": vals.append(fit_slope(s))
            elif metric=="vol": vals.append(volatility(s))
        if len(vals)==0: return
        plt.figure(figsize=(7,5))
        plt.hist(vals, bins=30)
        plt.title(title)
        plt.xlabel(metric)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(outdir / fname, dpi=150)
        plt.close()

    dist_plot("slope", pos_ids, f"Distribution telos_slope (label=1)", f"dist_slope_pos_min{args.min_msgs}.png")
    dist_plot("slope", neg_ids, f"Distribution telos_slope (label=0)", f"dist_slope_neg_min{args.min_msgs}.png")
    dist_plot("vol", pos_ids, f"Distribution telos_vol (label=1)", f"dist_vol_pos_min{args.min_msgs}.png")
    dist_plot("vol", neg_ids, f"Distribution telos_vol (label=0)", f"dist_vol_neg_min{args.min_msgs}.png")

    # CSV résumé pour ces échantillons
    recs = []
    take_ids = pos_ids + neg_ids
    lab_small = lab[lab.thread_id.isin(take_ids)].set_index("thread_id")["label"].to_dict()
    for th in take_ids:
        r = summarize_thread(th)
        r["label"] = int(lab_small.get(th, -1))
        recs.append(r)
    pd.DataFrame(recs).to_csv(outdir / f"sample_threads_min{args.min_msgs}.csv", index=False)

    print(f"[OK] Figures + CSV -> {outdir}")

if __name__ == "__main__":
    main()
