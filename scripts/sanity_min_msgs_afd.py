# scripts/sanity_min_msgs_afd.py
# -*- coding: utf-8 -*-
"""
Sanity check AfD : combien de threads ont ≥N messages ? et
quelle couverture / répartition de labels après intersection ?

Exemples d’usage (PowerShell) :
  python .\scripts\sanity_min_msgs_afd.py ^
    --emb H:\AxioPoC\artifacts_xfer\afd_msg_emb_PARTIAL.parquet ^
    --labels C:\AxioPoC\data\wikipedia\afd\afd_eval.csv ^
    --min-msgs 5 ^
    --out C:\AxioPoC\REPORTS\sanity_minmsgs_afd.csv
"""

import argparse
from pathlib import Path
import pandas as pd

def read_labels(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["thread_id", "label"])
    # normalise dtype (les IDs AfD sont des entiers)
    df["thread_id"] = pd.to_numeric(df["thread_id"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["thread_id"]).copy()
    df["thread_id"] = df["thread_id"].astype(int)
    return df

def read_thread_ids_from_parquet(path: Path) -> pd.Series:
    # lecture paresseuse de la seule colonne utile
    import pyarrow.parquet as pq
    table = pq.read_table(path, columns=["thread_id"])
    s = table.to_pandas()["thread_id"]
    # normalise dtype
    s = pd.to_numeric(s, errors="coerce").dropna().astype(int)
    return s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True, help="Parquet messages embeddings (avec colonne thread_id)")
    ap.add_argument("--labels", required=True, help="CSV labels AfD (thread_id, label)")
    ap.add_argument("--min-msgs", type=int, default=5, help="Seuil min de messages par thread (def=5)")
    ap.add_argument("--out", default="", help="CSV de sortie (optionnel)")
    args = ap.parse_args()

    emb_p = Path(args.emb)
    lab_p = Path(args.labels)

    if not emb_p.exists():
        raise FileNotFoundError(f"Embeddings parquet introuvable : {emb_p}")
    if not lab_p.exists():
        raise FileNotFoundError(f"Labels CSV introuvable : {lab_p}")

    # 1) charge
    lab = read_labels(lab_p)
    s_threads = read_thread_ids_from_parquet(emb_p)

    # 2) stats globales messages/thread
    vc = s_threads.value_counts()
    n_threads_total = int(vc.shape[0])
    n_msgs_total = int(s_threads.shape[0])

    # 3) filtres ≥3 et ≥5
    keep3 = set(vc[vc >= 3].index)
    keep5 = set(vc[vc >= 5].index)

    # 4) intersection avec labels
    lab3 = lab[lab.thread_id.isin(keep3)]
    lab5 = lab[lab.thread_id.isin(keep5)]

    def pack(df, tag):
        n = int(df.shape[0])
        pos = int((df["label"] == 1).sum())
        neg = int((df["label"] == 0).sum())
        return {
            "set": tag,
            "n_labeled_threads": n,
            "pos": pos,
            "neg": neg,
            "pos_ratio": round(pos / n, 4) if n else 0.0,
        }

    rows = [
        {"set": "ALL_threads_in_emb", "n_threads": n_threads_total, "n_msgs": n_msgs_total},
        {"set": ">=3_threads_in_emb", "n_threads": int((vc >= 3).sum()), "n_msgs": int(vc[vc >= 3].sum())},
        {"set": ">=5_threads_in_emb", "n_threads": int((vc >= 5).sum()), "n_msgs": int(vc[vc >= 5].sum())},
    ]
    summary = pd.DataFrame(rows)

    cover = pd.DataFrame([pack(lab3, "labels∩(>=3)"), pack(lab5, "labels∩(>=5)")])

    # 5) affichage console
    print("=== AfD sanity (embeddings + labels) ===")
    print(f"Emb threads total    : {n_threads_total}")
    print(f"Emb msgs total       : {n_msgs_total}")
    print(f"Threads ≥3 (emb)     : {(vc>=3).sum()} | msgs in those threads: {int(vc[vc>=3].sum())}")
    print(f"Threads ≥5 (emb)     : {(vc>=5).sum()} | msgs in those threads: {int(vc[vc>=5].sum())}")
    print("\n--- Couverture labels après intersection ---")
    if not cover.empty:
        print(cover.to_string(index=False))
    else:
        print("(aucune intersection)")

    # 6) export optionnel
    if args.out:
        out_p = Path(args.out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        # On écrit les deux tables dans un même CSV empilé avec un champ 'table'
        summary2 = summary.assign(table="emb_summary")
        cover2 = cover.assign(table="labels_coverage")
        pd.concat([summary2, cover2], ignore_index=True).to_csv(out_p, index=False)
        print(f"\n[OK] Écrit → {out_p}")

if __name__ == "__main__":
    main()
