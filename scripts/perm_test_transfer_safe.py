#!/usr/bin/env python3
"""
Permutation test (SAFE) for AfD -> CMV transfer (or any source -> target):
- Permutes TARGET labels in BOTH raw and features to avoid leakage.
- Calls your existing eval_da_transfer.py to compute AUC each iteration.
- Saves a JSON with auc_real, null_mean, null_sd, p(one-sided), iters, and null_samples.

Assumptions:
- Both target features and raw contain a common ID column 'thread_id'.
- Target raw has a column 'label'.
- Target features may or may not have a 'label' column; if it has one, it will be replaced by the permuted labels.
"""

import argparse
import json
import os
import random
import shutil
import statistics as st
import subprocess
import sys
import tempfile

import pandas as pd


def _run_eval_auc(feat_src, raw_src, sig_src, feat_tgt, raw_tgt, sig_tgt, metric="diag"):
    """
    Runs eval_da_transfer.py and parses "AUC ... , ACC ..." line.
    Returns float(AUC).
    """
    cmd = [
        sys.executable, "./scripts/eval_da_transfer.py",
        "--feat-src", feat_src, "--raw-src", raw_src, "--signature-src", sig_src,
        "--feat-tgt", feat_tgt, "--raw-tgt",  raw_tgt, "--signature-tgt",  sig_tgt,
        "--metric", metric,
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    out = (p.stdout or "") + ("\n" + p.stderr if p.stderr else "")
    if p.returncode != 0:
        raise SystemExit(f"[eval_da_transfer.py ERROR]\n{out}")

    # Expected line format observed earlier:
    # "Transfer diag Eval: n_src=279636, n_tgt=640, AUC 0.525475, ACC 0.487500"
    for line in out.splitlines():
        if "AUC" in line and "ACC" in line:
            parts = line.replace(",", " ").split()
            try:
                auc_idx = parts.index("AUC") + 1
                return float(parts[auc_idx])
            except Exception:
                pass
    raise SystemExit(f"[parse ERROR] Could not find 'AUC ... ACC ...' line in output:\n{out}")


def _ensure_cols(feat_tgt, raw_tgt):
    if "thread_id" not in feat_tgt.columns or "thread_id" not in raw_tgt.columns:
        raise SystemExit("Missing 'thread_id' in target features or target raw.")
    if "label" not in raw_tgt.columns:
        raise SystemExit("Missing 'label' in target raw.")
    # no return; will raise if invalid


def _make_permuted_pair(ft, rt, rng, tmpdir, idx):
    """
    Returns paths (feat_perm_csv, raw_perm_csv) with permuted labels aligned by thread_id.
    - If features had 'label', it is replaced with permuted labels.
    - If not, features are kept identical (no label column).
    """
    # Permute labels in RAW (shuffle rows' label)
    rt_perm = rt.copy()
    # pandas sample with fixed random_state from Python RNG (bridge an int)
    rt_perm["label"] = rt_perm["label"].sample(
        frac=1.0, random_state=rng.randint(0, 10**9)
    ).values

    # Write permuted RAW
    raw_perm_path = os.path.join(tmpdir, f"raw_perm_{idx}.csv")
    rt_perm.to_csv(raw_perm_path, index=False)

    # If features contain 'label', overwrite it from permuted raw via join on thread_id
    if "label" in ft.columns:
        ft_no_lab = ft.drop(columns=["label"])
        ft_perm = ft_no_lab.merge(rt_perm[["thread_id", "label"]], on="thread_id", how="left")
    else:
        ft_perm = ft.copy()

    feat_perm_path = os.path.join(tmpdir, f"feat_perm_{idx}.csv")
    ft_perm.to_csv(feat_perm_path, index=False)

    return feat_perm_path, raw_perm_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat-src", required=True, help="Source features CSV")
    ap.add_argument("--raw-src",  required=True, help="Source raw labels CSV")
    ap.add_argument("--sig-src",  required=True, help="Source signature pickle")

    ap.add_argument("--feat-tgt", required=True, help="Target features CSV")
    ap.add_argument("--raw-tgt",  required=True, help="Target raw labels CSV")
    ap.add_argument("--sig-tgt",  required=True, help="Target signature pickle")

    ap.add_argument("--iters", type=int, default=1000, help="Number of permutations (e.g., 1000)")
    ap.add_argument("--seed",  type=int, default=123, help="Random seed for reproducibility")
    ap.add_argument("--metric", default="diag", choices=["diag", "sigma_inv"], help="Distance metric")
    ap.add_argument("--outjson", required=True, help="Output JSON path (will include null_samples)")

    args = ap.parse_args()

    # Load target data once
    ft = pd.read_csv(args.feat_tgt)
    rt = pd.read_csv(args.raw_tgt)
    _ensure_cols(ft, rt)

    rng = random.Random(args.seed)

    # Real AUC with original files
    auc_real = _run_eval_auc(
        args.feat_src, args.raw_src, args.sig_src,
        args.feat_tgt, args.raw_tgt, args.sig_tgt,
        metric=args.metric,
    )
    print(f"AUC_real={auc_real:.6f}")

    # Permutations
    tmpdir = tempfile.mkdtemp(prefix="perm_safe_")
    null_aucs = []
    try:
        for i in range(args.iters):
            feat_perm_csv, raw_perm_csv = _make_permuted_pair(ft, rt, rng, tmpdir, i)
            auc_i = _run_eval_auc(
                args.feat_src, args.raw_src, args.sig_src,
                feat_perm_csv, raw_perm_csv, args.sig_tgt,
                metric=args.metric,
            )
            null_aucs.append(auc_i)
            # Progress ping every 100 iters
            if (i + 1) % 100 == 0:
                print(f"[perm] {i+1}/{args.iters} | last AUC={auc_i:.4f}")

        p_one_sided = sum(1 for x in null_aucs if x >= auc_real) / len(null_aucs)
        null_mean = st.mean(null_aucs)
        null_sd   = st.pstdev(null_aucs) if len(null_aucs) > 1 else 0.0

        payload = {
            "setting": "transfer",
            "direction": "afd->cmv",
            "signature": "Da13_diag",
            "metric": args.metric,
            "iters": args.iters,
            "seed": args.seed,
            "auc_real": auc_real,
            "null_mean": null_mean,
            "null_sd": null_sd,
            "p_value_one_sided_auc>=real": p_one_sided,
            "null_samples": null_aucs,  # full list, for histogram/percentiles
        }

        os.makedirs(os.path.dirname(args.outjson), exist_ok=True)
        with open(args.outjson, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        print(f"[OK] Permutation SAFE -> {args.outjson}")
        print(f"AUC_real={auc_real:.6f} | null_mean={null_mean:.6f} ± {null_sd:.6f} | p={p_one_sided:.4f}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
