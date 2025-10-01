#!/usr/bin/env python3
import argparse, json, matplotlib.pyplot as plt, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--permjson", required=True, help="permtest_*.json")
    ap.add_argument("--title", default="AfD→CMV — Permutation null vs real")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.permjson, encoding="utf-8") as f:
        d = json.load(f)
    auc_real = d["auc_real"]
    null_mean = d["null_mean"]
    ci = d.get("null_ci_95", None)

    plt.figure(figsize=(6,4))
    plt.bar([0], [null_mean])
    if ci:
        plt.errorbar([0], [null_mean], 
                     yerr=[[null_mean-ci[0]],[ci[1]-null_mean]], fmt='o', capsize=5)
    plt.axhline(auc_real, linestyle="--")
    plt.xticks([0], ["Null (mean ± CI)"])
    plt.ylabel("AUC")
    plt.title(args.title)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"[OK] Figure -> {args.out}")

if __name__ == "__main__":
    main()
