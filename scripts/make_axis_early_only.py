#!/usr/bin/env python
import argparse, os, sys, math, pathlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

def norm(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

def main():
    ap = argparse.ArgumentParser(description="Build telos axis from early-only messages per thread.")
    ap.add_argument("--emb", required=True, help="Parquet embeddings with cols: thread_id, __t__, e0..eD-1")
    ap.add_argument("--labels", required=True, help="CSV with thread_id,label (0/1)")
    ap.add_argument("--min-msgs", type=int, default=5, help="Min messages per thread to be eligible")
    ap.add_argument("--early-frac", type=float, default=0.3, help="Fraction of earliest messages to use per thread")
    ap.add_argument("--emb-prefix", default="e", help="Embedding column prefix (default: e)")
    ap.add_argument("--time-col", default="__t__", help="Time column name (default: __t__)")
    ap.add_argument("--thread-col", default="thread_id", help="Thread id column")
    ap.add_argument("--label-col", default="label", help="Label column in labels CSV")
    ap.add_argument("--out", required=True, help="Path to save axis .npy")
    args = ap.parse_args()

    # Load labels
    lab = pd.read_csv(args.labels, usecols=[args.thread_col, args.label_col])
    lab[args.thread_col] = lab[args.thread_col].astype(int)

    # Load embeddings (only needed columns)
    tbl = pq.read_table(args.emb)
    df = tbl.to_pandas()
    # identify embedding columns
    ecols = [c for c in df.columns if isinstance(c, str) and c.startswith(args.emb_prefix)]
    for c in (args.thread_col, args.time_col):
        if c not in df.columns:
            print(f"[ERR] missing column in embeddings: {c}", file=sys.stderr); sys.exit(2)

    # keep labeled threads
    df[args.thread_col] = df[args.thread_col].astype(int)
    df = df.merge(lab, on=args.thread_col, how="inner")

    # min msgs filter per thread
    vc = df[args.thread_col].value_counts()
    keep_threads = set(vc[vc >= args.min_msgs].index)
    df = df[df[args.thread_col].isin(keep_threads)]

    # early-only slice per thread
    df = df.sort_values([args.thread_col, args.time_col])
    parts = []
    for tid, grp in df.groupby(args.thread_col, sort=False):
        k = max(1, math.ceil(len(grp) * args.early_frac))
        parts.append(grp.iloc[:k])
    early = pd.concat(parts, ignore_index=True)

    # class means in embedding space
    X = early[ecols].to_numpy(dtype=np.float32)
    y = early[args.label_col].to_numpy()
    if X.ndim != 2 or X.shape[0] == 0:
        print("[ERR] no data after early-only selection.", file=sys.stderr); sys.exit(3)

    mu_pos = X[y == 1].mean(axis=0) if (y == 1).any() else np.zeros(X.shape[1], dtype=np.float32)
    mu_neg = X[y == 0].mean(axis=0) if (y == 0).any() else np.zeros(X.shape[1], dtype=np.float32)
    u = mu_pos - mu_neg
    u = norm(u)

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, u)

    # print summary
    print(f"[OK] axis saved -> {args.out}")
    print(f"  threads eligible      : {len(keep_threads)}")
    print(f"  early_frac            : {args.early_frac}")
    print(f"  rows used (early-only): {len(early)}")
    print(f"  dim                   : {u.shape[0]}")
    print(f"  ||mu_pos||,||mu_neg|| : {np.linalg.norm(mu_pos):.3f}, {np.linalg.norm(mu_neg):.3f}")

if __name__ == "__main__":
    main()
