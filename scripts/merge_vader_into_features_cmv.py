# -*- coding: utf-8 -*-
"""
Convertit les thread_id numériques de cmv_vader_by_thread.csv en t3_<base36>,
puis fusionne dans features_axio_cmv.csv (ajoute/écrase vader_pos_mean / vader_neu_mean / vader_neg_mean).
"""

import argparse
import pandas as pd
from pathlib import Path

def to_base36(n: int) -> str:
    if n == 0: return "0"
    digits = "0123456789abcdefghijklmnopqrstuvwxyz"
    out = []
    x = int(n)
    while x > 0:
        x, r = divmod(x, 36)
        out.append(digits[r])
    return "".join(reversed(out))

def to_t3(s):
    # s peut être '71816829', 71816829, 't3_2e4b5f', etc.
    s = "" if pd.isna(s) else str(s)
    if s.startswith("t3_"):
        return s
    # s numérique ?
    try:
        n = int(float(s))
        return "t3_" + to_base36(n)
    except Exception:
        return s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat-in",  required=True, help="features_axio_cmv.csv (input)")
    ap.add_argument("--vader",    required=True, help="cmv_vader_by_thread.csv (input)")
    ap.add_argument("--feat-out", required=True, help="features_axio_cmv.csv (output)")
    args = ap.parse_args()

    ax = pd.read_csv(args.feat_in)
    vd = pd.read_csv(args.vader)

    # Normalise colonnes attendues
    need = {"thread_id", "vader_pos_mean", "vader_neu_mean", "vader_neg_mean"}
    if "thread_id" not in vd.columns:
        raise SystemExit("Le fichier VADER doit contenir 'thread_id'")
    # s'il s'agit de colonnes pos/neu/neg par message: renomme avant d’appeler ce script
    for c_old, c_new in [("vader_pos","vader_pos_mean"),
                         ("vader_neu","vader_neu_mean"),
                         ("vader_neg","vader_neg_mean")]:
        if c_old in vd.columns and c_new not in vd.columns:
            vd = vd.rename(columns={c_old: c_new})

    # Convertit IDs VADER -> t3_<base36>
    vd["thread_id"] = vd["thread_id"].map(to_t3)

    # Type-strict pour la jointure
    ax["thread_id"] = ax["thread_id"].astype(str)
    vd["thread_id"] = vd["thread_id"].astype(str)

    # Merge (left sur features)
    keep_cols = ["thread_id", "vader_pos_mean", "vader_neu_mean", "vader_neg_mean"]
    vd_keep = vd[[c for c in keep_cols if c in vd.columns]].drop_duplicates("thread_id")
    merged = ax.merge(vd_keep, on="thread_id", how="left", suffixes=("", "_vd"))

    # Si features contient déjà des colonnes VADER, on écrase par celles du merge
    for c in ["vader_pos_mean", "vader_neu_mean", "vader_neg_mean"]:
        if c + "_vd" in merged.columns:
            merged[c] = merged[c + "_vd"].fillna(merged.get(c, 0)).fillna(0.0)
            merged.drop(columns=[c + "_vd"], inplace=True)
        else:
            # si la colonne n’existait pas côté features, on la crée
            if c not in merged.columns:
                merged[c] = 0.0
            merged[c] = merged[c].fillna(0.0)

    merged.to_csv(args.feat_out, index=False)

    nonzero = {
        "vader_pos_mean": float((merged["vader_pos_mean"].ne(0)).mean() * 100),
        "vader_neu_mean": float((merged["vader_neu_mean"].ne(0)).mean() * 100),
        "vader_neg_mean": float((merged["vader_neg_mean"].ne(0)).mean() * 100),
    }
    print(f"UPDATED {args.feat_out} rows={len(merged)} nonzero%={nonzero}")

if __name__ == "__main__":
    main()
