# -*- coding: utf-8 -*-
"""
make_features_axio.py
Harmonise des colonnes "axio-affectives" communes entre CMV et AfD.
Usage:
  python scripts/make_features_axio.py --feat IN_FEATURES.csv --raw IN_LABELS.csv --out OUT_AXIO.csv
"""
import argparse, sys
import pandas as pd
import numpy as np

REQ_COLS = [
    "thread_id","len_mean","qmark_ratio",
    "polite_ratio","hedge_ratio","you_i_ratio","agree_markers","neg_markers",
    "R_proxy","R_last_proxy","R_slope_proxy","R_iqr_proxy",
    "vader_pos_mean","vader_neg_mean","vader_neu_mean",
    "label"
]

def pick(df, name_list, default=0):
    """Retourne la première colonne existante (par nom) sinon un scalaire default."""
    for n in name_list:
        if n in df.columns:
            return df[n]
    return default

def make_axio(feat_path: str, raw_path: str) -> pd.DataFrame:
    dfx = pd.read_csv(feat_path)
    dfr = pd.read_csv(raw_path)
    if "thread_id" not in dfx or "thread_id" not in dfr:
        raise SystemExit("Both --feat and --raw must contain a 'thread_id' column.")
    if "label" not in dfr:
        raise SystemExit("--raw must contain a 'label' column.")

    df = dfx.merge(dfr[["thread_id","label"]], on="thread_id", how="inner")

    # Structure
    out = pd.DataFrame()
    out["thread_id"]  = df["thread_id"]
    out["len_mean"]   = pick(df, ["len_mean"], 0)
    out["qmark_ratio"]= pick(df, ["qmark_ratio","qmark_ratio_fix"], 0)

    # Pragmatique
    out["polite_ratio"] = pick(df, ["polite_ratio"], 0)
    out["hedge_ratio"]  = pick(df, ["hedge_ratio"], 0)
    out["you_i_ratio"]  = pick(df, ["you_i_ratio","you_i_ratio_fix"], 0)
    out["agree_markers"]= pick(df, ["agree_markers","agree_markers_fix"], 0)
    out["neg_markers"]  = pick(df, ["neg_markers","neg_markers_fix"], 0)

    # Dynamiques R (proxies harmonisés)
    # CMV: R_mean/R_last/R_slope/R_iqr
    # AfD: R_mean_w20/R_last_w20/R_slope_w20/R_iqr_w20 (si absent -> 0)
    out["R_proxy"]       = pick(df, ["R_mean","R_mean_w20"], 0)
    out["R_last_proxy"]  = pick(df, ["R_last","R_last_w20"], 0)
    out["R_slope_proxy"] = pick(df, ["R_slope","R_slope_w20"], 0)
    out["R_iqr_proxy"]   = pick(df, ["R_iqr","R_iqr_w20"], 0)

    # Affects VADER (si présents, sinon 0)
    out["vader_pos_mean"] = pick(df, ["vader_pos_mean","vader_pos"], 0)
    out["vader_neg_mean"] = pick(df, ["vader_neg_mean","vader_neg"], 0)
    out["vader_neu_mean"] = pick(df, ["vader_neu_mean","vader_neu"], 0)

    # Label
    out["label"] = df["label"].astype(int)

    # Assure présence de toutes les colonnes requises (remplissage 0 si manquantes)
    for c in REQ_COLS:
        if c not in out.columns:
            out[c] = 0

    # Ordre de colonnes
    out = out[REQ_COLS]
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--feat", required=True)
    p.add_argument("--raw",  required=True)
    p.add_argument("--out",  required=True)
    args = p.parse_args()

    axio = make_axio(args.feat, args.raw)
    axio.to_csv(args.out, index=False)
    # Log entête
    print(",".join(axio.columns.tolist()))

if __name__ == "__main__":
    main()
