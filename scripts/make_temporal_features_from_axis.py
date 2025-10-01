#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Annule & remplace — make_temporal_features_from_axis.py (self-contained)

Lit un Parquet d'embeddings messages (cols: thread_id, time + e0..e{d-1}),
charge un axe télotopique (npy, vecteur dim d), projette chaque message
p_i = <e_i, u> puis agrège par thread en métriques temporelles:

- telos_wd_l0.02 : moyenne pondérée (poids exp) avec λ=0.02
- telos_wd_l0.05 : moyenne pondérée (poids exp) avec λ=0.05
- telos_klast5   : moyenne des 5 dernières projections
- telos_delta5   : moyenne des 5 dernières - moyenne des 5 premières

Sortie: CSV (thread_id + 4 features)
"""

from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
import os
from typing import Iterable

# ---------- Helpers (autonomes) ----------

def _pick_first(cols: Iterable[str], candidates: Iterable[str], required: bool, what: str) -> str:
    s = set(cols)
    for c in candidates:
        if c in s:
            return c
    if required:
        raise ValueError(f"Colonne requise pour {what} introuvable parmi {list(candidates)} dans {list(cols)[:20]}...")
    return ""

def _exp_weights(n: int, lam: float) -> np.ndarray:
    """
    Poids exponentiels croissants vers la fin de la série, stables numériquement.
    w_i ∝ exp(lam * (i - (n-1)))  pour i=0..n-1  puis normalisation.
    Ainsi, le plus grand exposant vaut 0 → pas d'overflow.
    """
    if n <= 0:
        return np.zeros((0,), dtype=np.float64)
    idx = np.arange(n, dtype=np.float64)
    z = lam * (idx - (n - 1))           # max(z) = 0
    w = np.exp(z)                        # ∈ (0, 1]
    s = w.sum(dtype=np.float64)
    if not np.isfinite(s) or s == 0.0:
        # fallback ultra-prudent (cas pathologique) : uniforme
        return np.full(n, 1.0 / n, dtype=np.float64)
    return (w / s).astype(np.float64)


def time_weighted_mean(series: np.ndarray, lam: float) -> float:
    """Moyenne pondérée exponentielle (croissante vers la fin de la série)."""
    if series.size == 0:
        return 0.0
    w = _exp_weights(series.size, lam)
    return float(np.dot(series, w))

def k_last_mean(series: np.ndarray, k: int = 5) -> float:
    """Moyenne des k derniers éléments (0 si série vide)."""
    if series.size == 0:
        return 0.0
    k = max(1, min(k, series.size))
    return float(series[-k:].mean())

def start_end_delta(series: np.ndarray, k: int = 5) -> float:
    """
    (moyenne des k derniers) - (moyenne des k premiers).
    """
    n = series.size
    if n == 0:
        return 0.0
    k = max(1, min(k, n))
    head = float(series[:k].mean())
    tail = float(series[-k:].mean())
    return float(tail - head)

def _is_emb_col(c: str) -> bool:
    # Embeddings exportés sous forme e0, e1, ...
    return c.startswith("e") and c[1:].isdigit()

# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True, help="Parquet d'embeddings (thread_id, time, e0..)")
    ap.add_argument("--axis", required=True, help="Fichier .npy d'un vecteur unitaire (dim = nb cols e*)")
    ap.add_argument("--out", required=True, help="CSV de sortie (thread_id + features)")
    ap.add_argument("--thread_col", default="", help="Nom explicite de la colonne thread_id (sinon auto)")
    ap.add_argument("--time_col", default="", help="Nom explicite de la colonne temps/ordre (sinon auto)")
    ap.add_argument("--k_last", type=int, default=5, help="k pour klast et delta")
    ap.add_argument("--lam_list", default="0.02,0.05", help="λ séparés par virgules pour les moyennes pondérées")
    args = ap.parse_args()

    # Lecture embeddings
    df = pd.read_parquet(args.emb)
    cols = df.columns.tolist()

    # Identifier colonnes
    thread_col = args.thread_col or _pick_first(
        cols, ["thread_id", "conversation_id", "thread", "conv_id", "t3_id"], True, "thread_id"
    )
    time_col = args.time_col or _pick_first(
        cols, ["__t__", "created_utc", "created", "timestamp", "time", "date"], True, "temps"
    )
    emb_cols = [c for c in cols if _is_emb_col(c)]
    if not emb_cols:
        raise ValueError("Aucune colonne d'embedding détectée (e0, e1, ...).")

    # Charger axe & normaliser
    u = np.load(args.axis)
    u = np.asarray(u, dtype=np.float32).reshape(-1)
    # Ordre des embeddings: e0..e{d-1}
    d = len(emb_cols)
    if u.shape[0] != d:
        raise ValueError(f"Dimension axe ({u.shape[0]}) != nb dims embeddings ({d}).")

    # Normalisation de l'axe (unitaire)
    nu = float(np.linalg.norm(u))
    if nu == 0.0:
        raise ValueError("Axe nul.")
    u = u / nu

    # Tri temporel intra-thread (important)
    df = df.sort_values([thread_col, time_col]).reset_index(drop=True)

    # Projections p = E·u (vectorisées par bloc pour limiter mémoire)
    # On évite de materialiser toute la matrice en float64.
    proj = np.zeros(len(df), dtype=np.float32)
    CHUNK = 200_000
    e_mat_cols = emb_cols  # ordre déjà correct
    for i in range(0, len(df), CHUNK):
        block = df[e_mat_cols].iloc[i:i+CHUNK].to_numpy(dtype=np.float32, copy=False)
        proj[i:i+CHUNK] = np.dot(block, u)

    df["_proj_"] = proj

    # Agrégations par thread
    lam_values = [float(x) for x in args.lam_list.split(",") if x.strip()]
    agg_rows = []
    for tid, g in df[[thread_col, "_proj_"]].groupby(thread_col, sort=False):
        s = g["_proj_"].to_numpy(dtype=np.float32)
        row = {thread_col: tid}
        # pondérés
        for lam in lam_values:
            row[f"telos_wd_l{lam}"] = time_weighted_mean(s, lam=lam)
        # k-last & delta
        row[f"telos_klast{args.k_last}"] = k_last_mean(s, k=args.k_last)
        row[f"telos_delta{args.k_last}"] = start_end_delta(s, k=args.k_last)
        agg_rows.append(row)

    out_df = pd.DataFrame(agg_rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"[OK] Features -> {args.out} | rows={len(out_df)} | cols={len(out_df.columns)}")

if __name__ == "__main__":
    main()
