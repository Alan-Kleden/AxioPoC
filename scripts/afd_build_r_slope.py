# -*- coding: utf-8 -*-
"""
afd_build_r_slope.py (robuste)
- Lit rt_series AfD (format long OU large, noms de colonnes libres).
- Calcule une pente sur les 20 DERNIERS points non nuls par thread.
- Fusionne dans features_axio_afd.csv pour remplir R_slope_proxy si 0.
- Fallback: si un features_eval_plus.csv est fourni et contient R_slope_w20,
  on l'utilise là où R_slope_proxy est encore 0.
Usage:
  python scripts/afd_build_r_slope.py ^
    --series "C:\...\rt_series.csv" ^
    --axio-in "C:\...\features_axio_afd.csv" ^
    --axio-out "C:\...\features_axio_afd.csv" ^
    [--features-plus "C:\AxioPoC\artifacts_benchAfd_final_vader\features_eval_plus.csv"]
"""
import argparse, re
from pathlib import Path
import pandas as pd
import numpy as np

def slope_last_k(y, k=20):
    y = [v for v in y if pd.notnull(v)]
    if len(y) < 2:
        return 0.0
    yk = np.array(y[-k:], dtype=float)
    x = np.arange(len(yk), dtype=float)
    vx = x - x.mean()
    vy = yk - yk.mean()
    denom = (vx**2).sum()
    if denom <= 0:
        return 0.0
    return float((vx*vy).sum() / denom)

def try_long(df):
    cols = {c.lower(): c for c in df.columns}
    # heuristique: thread_id, + 1 colonne "valeur" (r|score|value|intensity...), + 1 colonne temps (t|time|step|idx…)
    id_col = cols.get("thread_id")
    if not id_col:
        return None
    # candidates valeur
    val_candidates = [c for c in df.columns if c.lower() in ("r","score","value","val","intensity","y")]
    # candidates temps
    t_candidates = [c for c in df.columns if c.lower() in ("t","time","step","idx","index","k")]
    if val_candidates and t_candidates:
        return ("long", id_col, t_candidates[0], val_candidates[0])
    # autre heuristique: exactement 3 colonnes avec thread_id + 2 inconnues -> assume long
    if len(df.columns) == 3 and id_col:
        other = [c for c in df.columns if c != id_col]
        return ("long", id_col, other[0], other[1])
    return None

def natural_key(cname: str):
    # extrait un entier final s'il existe, pour trier r_0, r_1, ..., r_20
    m = re.search(r"(\d+)$", cname)
    return (int(m.group(1)) if m else float('inf'))

def try_wide(df):
    if "thread_id" not in df.columns:
        return None
    series_cols = [c for c in df.columns if c != "thread_id"]
    if len(series_cols) < 2:
        return None
    # si aucune idée d'ordre, on garde l'ordre du fichier; sinon on trie par entier final
    sorted_cols = sorted(series_cols, key=natural_key)
    return ("wide", "thread_id", sorted_cols)

def read_series_any(path: str):
    df = pd.read_csv(path)
    guess = try_long(df)
    if guess is not None:
        fmt, id_col, t_col, v_col = guess
        dfl = df[[id_col, t_col, v_col]].rename(columns={id_col:"thread_id", t_col:"t", v_col:"r"})
        dfl = dfl.sort_values(["thread_id","t"])
        return "long", dfl
    guess = try_wide(df)
    if guess is not None:
        fmt, id_col, series_cols = guess
        rows = []
        for _, row in df[[id_col]+series_cols].iterrows():
            vals = [row[c] for c in series_cols]
            rows.append({"thread_id": row[id_col], "series": vals})
        return "wide", pd.DataFrame(rows)
    raise SystemExit("Schéma inconnu pour rt_series.csv (ni long, ni large).")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", required=True)
    ap.add_argument("--axio-in", required=True)
    ap.add_argument("--axio-out", required=True)
    ap.add_argument("--features-plus", default="")  # optionnel: features_eval_plus.csv (pour fallback R_slope_w20)
    args = ap.parse_args()

    fmt, S = read_series_any(args.series)

    out = []
    if fmt == "long":
        for tid, grp in S.groupby("thread_id", sort=False):
            y = grp["r"].tolist()
            out.append((tid, slope_last_k(y, 20)))
    else:
        for _, row in S.iterrows():
            y = row["series"]
            out.append((row["thread_id"], slope_last_k(y, 20)))
    df_slope = pd.DataFrame(out, columns=["thread_id", "R_slope_w20_from_series"])

    ax = pd.read_csv(args.axio_in)
    ax = ax.merge(df_slope, on="thread_id", how="left")
    if "R_slope_proxy" not in ax.columns:
        ax["R_slope_proxy"] = 0.0
    mask_series = (ax["R_slope_proxy"]==0) & ax["R_slope_w20_from_series"].notna()
    ax.loc[mask_series, "R_slope_proxy"] = ax.loc[mask_series, "R_slope_w20_from_series"].astype(float)

    # Fallback depuis features_eval_plus.csv si dispo: R_slope_w20
    filled_plus = 0
    if args.features_plus:
        try:
            plus = pd.read_csv(args.features_plus, usecols=["thread_id","R_slope_w20"])
            ax = ax.merge(plus, on="thread_id", how="left")
            mask_plus = (ax["R_slope_proxy"]==0) & ax["R_slope_w20"].notna()
            filled_plus = int(mask_plus.sum())
            ax.loc[mask_plus, "R_slope_proxy"] = ax.loc[mask_plus, "R_slope_w20"].astype(float)
        except Exception as e:
            print(f"[INFO] Pas de fallback features_plus (ignorer si normal): {e}")

    Path(args.axio_out).parent.mkdir(parents=True, exist_ok=True)
    ax.to_csv(args.axio_out, index=False)

    print(f"Updated {args.axio_out} | filled from series: {int(mask_series.sum())} | filled from features_plus: {filled_plus}")

if __name__ == "__main__":
    main()
