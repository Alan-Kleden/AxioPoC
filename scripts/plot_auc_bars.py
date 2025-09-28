#!/usr/bin/env python3
import argparse, pandas as pd, numpy as np, matplotlib.pyplot as plt, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV consolidé (depuis collect_metrics.py)")
    ap.add_argument("--filter_signature", default="Da13", help="Contient ce motif (ex: Da13)")
    ap.add_argument("--filter_metric", default="diag", help="Motif dans 'signature' (ex: diag)")
    ap.add_argument("--title", default="AUC (mean ± sd)")
    ap.add_argument("--out", required=True, help="PNG de sortie")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    # filtrage souple sur signature (ex: 'Da13' et 'diag')
    m = df["signature"].astype(str).str.contains(args.filter_signature, case=False, na=False) & \
        df["signature"].astype(str).str.contains(args.filter_metric, case=False, na=False)
    df = df[m].copy()

    # enlever AUC==0 ou NaN
    df = df[(df["auc_mean"].fillna(0) > 0)]
    if not len(df):
        print("[INFO] Rien à tracer après filtrage.")
        return

    # etiquette compacte
    df["label"] = df.apply(lambda r: f"{r['setting']}|{r['direction'] or 'intra'}", axis=1)
    # group-by pour éviter doublons (on garde le meilleur AUC si multiples)
    df.sort_values("auc_mean", ascending=False, inplace=True)
    df = df.drop_duplicates(subset=["label","dataset","signature","model"])

    x = np.arange(len(df))
    y = df["auc_mean"].values.astype(float)
    yerr = df["auc_sd"].fillna(0).values.astype(float)

    plt.figure(figsize=(10,5))
    plt.bar(x, y, yerr=yerr, capsize=4)
    plt.axhline(0.5, linestyle="--")
    plt.xticks(x, df["label"].tolist(), rotation=30, ha="right")
    plt.ylabel("AUC")
    plt.title(args.title)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"[OK] Figure -> {args.out}")

if __name__ == "__main__":
    main()
