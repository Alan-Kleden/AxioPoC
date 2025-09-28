#!/usr/bin/env python3
import argparse, pandas as pd, numpy as np, matplotlib.pyplot as plt, os
from sklearn.metrics import roc_curve, auc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores_csv", required=True, help="CSV avec y_true,score")
    ap.add_argument("--title", default="ROC")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.scores_csv)
    if not {"y_true","score"}.issubset(df.columns):
        raise SystemExit("scores_csv doit contenir y_true et score")
    fpr, tpr, _ = roc_curve(df["y_true"].values, df["score"].values)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0,1], [0,1], linestyle="--")
    plt.xlabel("FPR"), plt.ylabel("TPR"), plt.title(args.title)
    plt.legend(loc="lower right")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.tight_layout(); plt.savefig(args.out, dpi=150)
    print(f"[OK] ROC -> {args.out} (AUC={roc_auc:.3f})")

if __name__ == "__main__":
    main()
