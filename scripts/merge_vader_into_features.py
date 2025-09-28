# -*- coding: utf-8 -*-
"""
Merge VADER-by-thread into an existing features CSV on `thread_id`,
en harmonisant les types pour éviter object vs int64.
Usage:
  python scripts/merge_vader_into_features.py \
    --feat-in  artifacts_xfer/features_axio_cmv.csv \
    --vader    artifacts_xfer/cmv_vader_by_thread.csv \
    --feat-out artifacts_xfer/features_axio_cmv.csv
"""
import argparse
import pandas as pd

def coerce_thread_id(df, col="thread_id"):
    # On tente int64 d'abord ; si ça casse, on passe à string.
    try:
        df[col] = pd.to_numeric(df[col], errors="raise").astype("int64")
        return "int64"
    except Exception:
        df[col] = df[col].astype(str)
        return "str"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--feat-in",  required=True)
    p.add_argument("--vader",    required=True)
    p.add_argument("--feat-out", required=True)
    args = p.parse_args()

    ax = pd.read_csv(args.feat_in)
    vd = pd.read_csv(args.vader)

    if "thread_id" not in ax.columns or "thread_id" not in vd.columns:
        raise SystemExit("Les deux fichiers doivent contenir une colonne 'thread_id'.")

    t_ax = coerce_thread_id(ax, "thread_id")
    t_vd = coerce_thread_id(vd, "thread_id")

    if t_ax != t_vd:
        # Harmonise sur string si les types diffèrent encore
        ax["thread_id"] = ax["thread_id"].astype(str)
        vd["thread_id"] = vd["thread_id"].astype(str)
        t_ax = t_vd = "str"

    # Déduire les noms VADER attendus et combler les NaN à 0 après fusion
    vader_cols = [c for c in vd.columns if c != "thread_id"]
    if not vader_cols:
        raise SystemExit("Aucune colonne VADER trouvée dans le fichier --vader.")

    merged = ax.merge(vd, on="thread_id", how="left")
    for c in vader_cols:
        if c in merged.columns:
            merged[c] = merged[c].fillna(0.0)

    merged.to_csv(args.feat_out, index=False)
    nonzero = {c: float((merged[c].fillna(0) != 0).mean()*100)
               for c in vader_cols if c in merged.columns}
    print(f"UPDATED {args.feat_out} rows={len(merged)} dtypes(ax/vd)={t_ax}/{t_vd} nonzero%={nonzero}")

if __name__ == "__main__":
    main()
