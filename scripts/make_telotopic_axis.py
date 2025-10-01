#!/usr/bin/env python3
from __future__ import annotations
import argparse, os
import numpy as np
import pandas as pd

# ---------------- Auto-détection de colonnes ----------------
THREAD_CANDS = ["thread_id", "conversation_id", "thread", "conv_id", "t3_id"]

def pick_first_present(df: pd.DataFrame, candidates, required=False, what=""):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise SystemExit(f"Impossible de trouver une colonne pour {what}. Cherché: {candidates}")
    return ""
# ------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True, help="CSV/Parquet des embeddings message (e0..)")
    ap.add_argument("--raw_labels", required=True, help="CSV labels par thread (thread_id,label)")
    ap.add_argument("--thread_col", default="", help="Nom thread_id (auto si vide)")
    ap.add_argument("--out_axis", required=True, help="Fichier .npy pour l'axe normalisé")
    args = ap.parse_args()

    emb = pd.read_parquet(args.emb) if args.emb.lower().endswith(".parquet") else pd.read_csv(args.emb)
    lbl = pd.read_csv(args.raw_labels)

    thread_col = args.thread_col or pick_first_present(emb, THREAD_CANDS, required=True, what="thread_id")
    if thread_col not in lbl.columns:
        lbl_thread = pick_first_present(lbl, [thread_col] + THREAD_CANDS, required=True, what="thread_id (labels)")
        if lbl_thread != thread_col:
            lbl = lbl.rename(columns={lbl_thread: thread_col})

    if "label" not in lbl.columns:
        raise SystemExit("Labels: colonne 'label' manquante.")

    ecols = [c for c in emb.columns if c.startswith("e")]
    if not ecols:
        raise SystemExit("Aucune colonne d'embedding (e0,e1,...)")

    thr_emb = emb.groupby(thread_col)[ecols].mean().reset_index()
    df = thr_emb.merge(lbl[[thread_col, "label"]], on=thread_col, how="inner")
    if df["label"].nunique() < 2:
        raise SystemExit("Il faut au moins deux classes (label 0/1).")

    mu_pos = df[df["label"] == 1][ecols].mean().to_numpy(dtype=np.float64)
    mu_neg = df[df["label"] == 0][ecols].mean().to_numpy(dtype=np.float64)
    u = mu_pos - mu_neg

    norm = np.linalg.norm(u)
    if norm > 0:
        u = u / norm

    os.makedirs(os.path.dirname(args.out_axis) or ".", exist_ok=True)
    np.save(args.out_axis, u.astype(np.float32))
    print(f"[OK] Axis saved -> {args.out_axis} | dim={u.shape[0]} | norm=1.0 | thread_col={thread_col}")

if __name__ == "__main__":
    main()
