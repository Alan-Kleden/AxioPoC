#!/usr/bin/env python3
"""
Plot histogram of permutation null AUCs vs the real AUC.

Input JSON (from perm_test_transfer_safe.py) is expected to contain:
- "auc_real": float
- "null_samples": list[float]  # full list of null AUCs (preferred)
- optionally "null_mean", "null_sd": floats (used as fallback for visualization)

Usage:
python scripts/plot_perm_hist_vs_real.py \
  --permjson artifacts_xfer/permtestSAFE_afd2cmv_Da13_diag_1000.json \
  --title "AfD→CMV (Da13 diag) — Permutation null vs real" \
  --out REPORTS/img/perm_hist_vs_real_afd2cmv_Da13.png
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--permjson", required=True, help="Path to permutation JSON with null_samples/auc_real")
    ap.add_argument("--title", default="Permutation null vs real (AUC)", help="Figure title")
    ap.add_argument("--out", required=True, help="Output PNG path")
    args = ap.parse_args()

    with open(args.permjson, encoding="utf-8") as f:
        d = json.load(f)

    auc_real = float(d["auc_real"])
    samples = d.get("null_samples", None)
    null_mean = float(d.get("null_mean", 0.5))
    null_sd = float(d.get("null_sd", 0.0))

    plt.figure(figsize=(7, 4))

    if samples:
        arr = np.array(samples, dtype=float)
        # histogram normalized to a density
        plt.hist(arr, bins=30, alpha=0.75, density=True)
    else:
        # Fallback: normal-ish band around mean±2*sd for visualization only
        xs = np.linspace(0.3, 0.7, 400)
        if null_sd > 0:
            pdf = (1 / np.sqrt(2 * np.pi * null_sd ** 2)) * np.exp(-(xs - null_mean) ** 2 / (2 * null_sd ** 2))
            plt.plot(xs, pdf)
        plt.axvspan(null_mean - 2 * null_sd, null_mean + 2 * null_sd, alpha=0.15)

    # Real AUC as dashed vertical line
    plt.axvline(auc_real, linestyle="--", label=f"AUC real = {auc_real:.3f}")

    plt.xlabel("AUC")
    plt.ylabel("Density")
    plt.title(args.title)
    plt.legend(loc="upper left")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"[OK] Figure -> {args.out}")


if __name__ == "__main__":
    main()
