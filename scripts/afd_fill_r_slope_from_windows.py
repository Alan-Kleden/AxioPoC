# -*- coding: utf-8 -*-
"""
afd_fill_r_slope_from_windows.py  (version corrigée)
Remplit/complète R_slope_proxy dans features_axio_afd.csv à partir de features_eval_plus.csv :
  1) R_slope_w20 (priorité : côté PLUS si dispo, sinon côté AXIO)
  2) (R_last_w20 - R_last_w10)/10 sinon si dispo (depuis PLUS)
  3) (R_mean_w20 - R_mean_w10)/10 sinon si dispo (depuis PLUS)
  4) 0 sinon
Usage:
  python scripts/afd_fill_r_slope_from_windows.py ^
    --plus C:\AxioPoC\artifacts_benchAfd_final_vader\features_eval_plus.csv ^
    --axio-in  C:\AxioPoC\artifacts_xfer\features_axio_afd.csv ^
    --axio-out C:\AxioPoC\artifacts_xfer\features_axio_afd.csv
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def safe_num(s):
    try: return pd.to_numeric(s)
    except: return pd.Series([np.nan]*len(s))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plus", required=True)
    ap.add_argument("--axio-in", required=True)
    ap.add_argument("--axio-out", required=True)
    args = ap.parse_args()

    ax  = pd.read_csv(args.axio_in)
    pls = pd.read_csv(args.plus)

    # Restreindre aux colonnes utiles si elles existent
    need = ["thread_id","R_slope_w20","R_last_w20","R_last_w10","R_mean_w20","R_mean_w10"]
    cols = [c for c in need if c in pls.columns]
    pls  = pls[cols].copy()

    # Convertir en numériques quand présent
    for c in cols:
        if c != "thread_id":
            pls[c] = safe_num(pls[c])

    # Merge avec suffixe pour éviter collisions
    m = ax.merge(pls, on="thread_id", how="left", suffixes=("", "_plus"))

    # Prépare la colonne cible
    if "R_slope_proxy" not in m.columns:
        m["R_slope_proxy"] = 0.0

    def getcol(df, name):
        """Renvoie la série prioritaire: name_plus si existe, sinon name si existe, sinon NaN."""
        if name + "_plus" in df.columns:
            return df[name + "_plus"]
        if name in df.columns:
            return df[name]
        return pd.Series(np.nan, index=df.index)

    slope_w20   = getcol(m, "R_slope_w20")                    # priorité aux valeurs du PLUS
    last_w20    = getcol(m, "R_last_w20")
    last_w10    = getcol(m, "R_last_w10")
    mean_w20    = getcol(m, "R_mean_w20")
    mean_w10    = getcol(m, "R_mean_w10")

    slope_from_last = (last_w20 - last_w10) / 10.0 if (last_w20.notna().any() and last_w10.notna().any()) else pd.Series(np.nan, index=m.index)
    slope_from_mean = (mean_w20 - mean_w10) / 10.0 if (mean_w20.notna().any() and mean_w10.notna().any()) else pd.Series(np.nan, index=m.index)

    # Construire le "meilleur disponible"
    best = slope_w20.copy()
    best = best.where(best.notna(), slope_from_last)
    best = best.where(best.notna(), slope_from_mean)

    before_nonzero = (m["R_slope_proxy"].ne(0)).sum()
    mask = (m["R_slope_proxy"].fillna(0)==0) & best.notna()
    m.loc[mask, "R_slope_proxy"] = best[mask]
    after_nonzero = (m["R_slope_proxy"].ne(0)).sum()
    filled = int(after_nonzero - before_nonzero)

    # Sauver (mêmes colonnes/ordre que AXIO d’origine si possible)
    out_cols = list(ax.columns)
    if "R_slope_proxy" not in out_cols:
        out_cols.append("R_slope_proxy")
    out = m[out_cols]

    Path(args.axio_out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.axio_out, index=False)
    print(f"Filled R_slope_proxy via windows: +{filled} rows (non-zero now = {after_nonzero}) -> {args.axio_out}")

if __name__ == "__main__":
    main()
