#!/usr/bin/env python3
import argparse, pandas as pd, numpy as np, matplotlib.pyplot as plt, os

def parse_ylim(s):
    if not s: return None
    try:
        a,b = s.split(",")
        return float(a), float(b)
    except Exception:
        raise SystemExit("--ylim doit être comme '0,1' ou '0.45,0.55'")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--filter_signature", default="Da13")
    ap.add_argument("--filter_metric", default="diag")
    ap.add_argument("--title", default="AUC (mean ± sd)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ylim", default="", help="ex: 0,1 ou 0.45,0.55")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

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
    # éviter doublons
    df.sort_values("auc_mean", ascending=False, inplace=True)
    df = df.drop_duplicates(subset=["label","dataset","signature","model"])

    x = np.arange(len(df))
    y = df["auc_mean"].astype(float).values
    yerr = df["auc_sd"].astype(float).fillna(0).values

    plt.figure(figsize=(10,5))
    bars = plt.bar(x, y, yerr=yerr, capsize=4)
    # ligne hasard
    plt.axhline(0.5, linestyle="--", zorder=0)

    # annotations de valeur
    for xi, yi in zip(x, y):
        plt.text(xi, yi + 0.005, f"{yi:.3f}", ha="center", va="bottom", fontsize=10)

    # ticks
    plt.xticks(x, df["label"].tolist(), rotation=30, ha="right")
    plt.ylabel("AUC")
    plt.title(args.title)

    # gestion des limites Y
    ylim = parse_ylim(args.ylim)
    if ylim:
        plt.ylim(*ylim)
    else:
        y_min, y_max = float(np.min(y)), float(np.max(y))
        if y_min >= 0.45 and y_max <= 0.55:
            plt.ylim(0.45, 0.55)     # zoom utile autour du hasard
        else:
            plt.ylim(0.0, 1.0)       # échelle standard

    # marge X si une seule barre
    if len(x) == 1:
        plt.xlim(-0.6, 0.6)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"[OK] Figure -> {args.out}")

if __name__ == "__main__":
    main()
