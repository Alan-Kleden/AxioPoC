#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Bootstrap the stability of a telotopic axis (μ+ - μ-) built at the THREAD level.

Usage (example):
  python scripts/bootstrap_axis_stability.py ^
    --emb H:\AxioPoC\artifacts_xfer\afd_msg_emb_PARTIAL.parquet ^
    --labels C:\AxioPoC\data\wikipedia\afd\afd_eval.csv ^
    --B 100 --sample-frac 0.7 ^
    --min-msgs 0 ^
    --out C:\AxioPoC\REPORTS\axis_stability.json ^
    --plot C:\AxioPoC\REPORTS\axis_stability_hist.png
"""

import argparse
import json
import math
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# pyarrow is faster/robust to read parquet
try:
    import pyarrow.parquet as pq
except ImportError as e:
    raise SystemExit("pyarrow is required. pip install pyarrow") from e

try:
    from tqdm import tqdm
    TQDM_KW = dict(leave=False)
except ImportError:
    def tqdm(x, **k): return x
    TQDM_KW = {}

def parse_args():
    ap = argparse.ArgumentParser(description="Bootstrap stability of μ+-μ- axis (thread-level).")
    ap.add_argument("--emb", required=True, help="Parquet with message embeddings (cols: thread_id, e0..eD, maybe __t__).")
    ap.add_argument("--labels", required=True, help="CSV with thread-level labels (cols: thread_id, label).")
    ap.add_argument("--thread-col", default="thread_id", help="Thread id column name (both files).")
    ap.add_argument("--label-col", default="label", help="Label column name (labels CSV).")
    ap.add_argument("--emb-prefix", default="e", help="Prefix for embedding columns (default: 'e').")
    ap.add_argument("--min-msgs", type=int, default=0, help="Keep only threads with >= min-msgs messages (0 = no filter).")
    ap.add_argument("--B", type=int, default=100, help="Number of bootstrap replicates.")
    ap.add_argument("--sample-frac", type=float, default=0.7, help="Fraction of threads per bootstrap sample (0<frac<=1).")
    ap.add_argument("--seed", type=int, default=42, help="Random seed.")
    ap.add_argument("--out", required=True, help="Output JSON summary.")
    ap.add_argument("--plot", default=None, help="Optional: path to save histogram PNG.")
    return ap.parse_args()

def cosine(u, v, eps=1e-12):
    un = np.linalg.norm(u)
    vn = np.linalg.norm(v)
    if un < eps or vn < eps:
        return np.nan
    return float(np.dot(u, v) / (un * vn))

def load_thread_embeddings(emb_path, thread_col="thread_id", emb_prefix="e", min_msgs=0):
    # Read only needed columns: thread_id + embedding cols (auto-detect)
    table = pq.read_table(emb_path)
    df = table.to_pandas()
    if thread_col not in df.columns:
        raise ValueError(f"'{thread_col}' not in embeddings parquet columns.")

    emb_cols = [c for c in df.columns if isinstance(c, str) and c.startswith(emb_prefix)]
    if not emb_cols:
        raise ValueError(f"No embedding columns found with prefix '{emb_prefix}'.")

    # count messages per thread & aggregate mean at thread level
    counts = df.groupby(thread_col, observed=True)[emb_cols[0]].size().rename("__n_msgs__")
    grp = df.groupby(thread_col, observed=True)[emb_cols].mean()
    grp = grp.join(counts)

    if min_msgs > 0:
        grp = grp[grp["__n_msgs__"] >= min_msgs]

    return grp.drop(columns="__n_msgs__"), emb_cols  # DataFrame indexed by thread_id

def load_labels(labels_path, thread_col="thread_id", label_col="label"):
    lab = pd.read_csv(labels_path, usecols=[thread_col, label_col])
    # Cast thread ids to int if possible; otherwise leave as object and align types later
    # Try int -> fall back to string
    try:
        lab[thread_col] = lab[thread_col].astype(int)
    except Exception:
        lab[thread_col] = lab[thread_col].astype(str)
    return lab

def align_types(left_index, right_series):
    """
    Ensure comparable dtypes between thread_index (from embeddings) and labels.thread_id.
    Returns labels with coerced dtype matching left_index dtype.
    """
    if left_index.dtype == 'int64':
        try:
            return right_series.astype(int)
        except Exception:
            # fallback: cast left to string if labels cannot be int
            return right_series.astype(str)
    else:
        # cast to string on both sides
        return right_series.astype(str)

def compute_axis_mu_plus_minus(X, y):
    """
    X: np.ndarray (n_threads, dim)
    y: np.ndarray in {0,1}
    Returns: normalized axis u = (mean_pos - mean_neg), shape (dim,)
    """
    if X.size == 0 or y.size == 0:
        return np.zeros(X.shape[1], dtype=np.float32)

    pos = X[y == 1]
    neg = X[y == 0]

    if pos.shape[0] == 0 or neg.shape[0] == 0:
        return np.zeros(X.shape[1], dtype=np.float32)

    u = pos.mean(axis=0) - neg.mean(axis=0)
    n = np.linalg.norm(u)
    return (u / (n + 1e-12)).astype(np.float32)

def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1) Load thread-level embeddings
    thr_emb_df, emb_cols = load_thread_embeddings(
        args.emb, thread_col=args.thread_col, emb_prefix=args.emb_prefix, min_msgs=args.min_msgs
    )

    # 2) Load labels and align types
    lab = load_labels(args.labels, thread_col=args.thread_col, label_col=args.label_col)
    # Align types between thr_emb_df.index and lab[thread_col]
    lab[args.thread_col] = align_types(thr_emb_df.index.to_series().astype(str), lab[args.thread_col])

    # Try to align embeddings index dtype to labels dtype
    # Convert both to string keys to be safe, then map back
    thr_index_as_str = thr_emb_df.index.astype(str)
    thr_emb_df = thr_emb_df.set_index(thr_index_as_str, drop=True)

    lab = lab.drop_duplicates(subset=[args.thread_col])
    lab = lab.set_index(args.thread_col)

    # 3) Inner join on thread_id
    joined = thr_emb_df.join(lab, how="inner")
    if joined.empty:
        raise SystemExit("No overlap between embeddings threads and labels after join.")

    # Extract np arrays
    X = joined[emb_cols].to_numpy(dtype=np.float32)
    y = joined[args.label_col].to_numpy()
    # Force binary {0,1} if booleans
    if y.dtype == bool:
        y = y.astype(np.int8)

    n_threads, dim = X.shape

    # 4) Full-axis
    u_full = compute_axis_mu_plus_minus(X, y)

    # 5) Bootstrap
    B = int(args.B)
    frac = float(args.sample_frac)
    n_sample = max(1, int(math.ceil(frac * n_threads)))

    cos_list = []
    skipped = 0

    for _ in tqdm(range(B), desc="bootstrap", **TQDM_KW):
        idx = rng.choice(n_threads, size=n_sample, replace=True)
        Xb = X[idx]
        yb = y[idx]
        # must contain both classes
        if (yb == 1).sum() == 0 or (yb == 0).sum() == 0:
            skipped += 1
            continue
        ub = compute_axis_mu_plus_minus(Xb, yb)
        cs = cosine(ub, u_full)
        if not (np.isnan(cs) or np.isinf(cs)):
            cos_list.append(float(cs))
        else:
            skipped += 1

    cos_arr = np.array(cos_list, dtype=np.float32)
    if cos_arr.size == 0:
        raise SystemExit("All bootstrap replicates were skipped (class imbalance?). Try increasing sample-frac or B.")

    # 6) Summary stats
    summary = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "emb_path": str(args.emb),
        "labels_path": str(args.labels),
        "thread_col": args.thread_col,
        "label_col": args.label_col,
        "emb_prefix": args.emb_prefix,
        "min_msgs": int(args.min_msgs),
        "n_threads_used": int(n_threads),
        "dim": int(dim),
        "B_requested": int(B),
        "B_effective": int(cos_arr.size),
        "skipped": int(skipped),
        "sample_frac": frac,
        "cos_mean": float(np.mean(cos_arr)),
        "cos_median": float(np.median(cos_arr)),
        "cos_min": float(np.min(cos_arr)),
        "cos_max": float(np.max(cos_arr)),
        "cos_p05": float(np.quantile(cos_arr, 0.05)),
        "cos_p25": float(np.quantile(cos_arr, 0.25)),
        "cos_p75": float(np.quantile(cos_arr, 0.75)),
        "cos_p95": float(np.quantile(cos_arr, 0.95)),
        "threshold_pass_median_ge_0.90": bool(np.median(cos_arr) >= 0.90)
    }

    # 7) Save JSON
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[OK] Stability summary -> {out_path}")

    # 8) Optional plot
    if args.plot:
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(6,4), dpi=140)
            plt.hist(cos_arr, bins=30)
            plt.axvline(summary["cos_median"], linestyle="--")
            plt.title("Bootstrap cosine(u_b, u_full)")
            plt.xlabel("cosine similarity")
            plt.ylabel("count")
            plot_path = Path(args.plot)
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            plt.tight_layout()
            plt.savefig(plot_path)
            print(f"[OK] Histogram -> {plot_path}")
        except Exception as e:
            print(f"[WARN] Plot skipped: {e}")

if __name__ == "__main__":
    main()
