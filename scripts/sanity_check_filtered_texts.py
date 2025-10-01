#!/usr/bin/env python
"""
Sanity check robuste pour afd_messages_filtered.csv
- Essaie engine='c', puis 'python', puis fallback en chunks
- Gère encodage via plusieurs essais + errors='replace'
- Résume: lignes / threads / vides / longueur moyenne + head
"""

from __future__ import annotations
import argparse, sys, io
from pathlib import Path
import pandas as pd

ENC_CANDIDATES = [
    ("utf-8", "replace"),
    ("utf-8-sig", "replace"),
    ("cp1252", "replace"),
    ("latin-1", "replace"),
]

USECOLS = ["thread_id", "text"]

def try_read_csv(path: Path, nrows: int | None, engine: str | None, encoding: str, errors: str):
    kw = dict(
        usecols=USECOLS,
        nrows=nrows,
        dtype={"thread_id": "Int64", "text": "string"},
        encoding=encoding,
        on_bad_lines="skip",
    )
    # moteur
    if engine is not None:
        kw["engine"] = engine
    # IMPORTANT: pas de low_memory avec engine="python"
    if engine == "c":
        kw["low_memory"] = False  # OK pour C-engine
    # encoding_errors dispo pandas>=1.5 ; sinon ignore
    try:
        return pd.read_csv(  # type: ignore[arg-type]
            path,
            encoding_errors=errors,  # newer pandas
            **kw,
        )
    except TypeError:
        # pandas plus ancien: pas de encoding_errors
        kw.pop("encoding", None)  # on va gérer via file wrapper ci-dessous
        with open(path, "rb") as fbin:
            text = io.TextIOWrapper(fbin, encoding=encoding, errors=errors)
            return pd.read_csv(text, **kw)

def read_csv_robust(path: Path, nrows: int | None = 200_000) -> pd.DataFrame:
    last_err: Exception | None = None

    # 1) engine='c' d'abord (plus rapide)
    for enc, err in ENC_CANDIDATES:
        try:
            return try_read_csv(path, nrows, engine="c", encoding=enc, errors=err)
        except Exception as e:
            last_err = e

    # 2) engine='python' (plus tolérant)
    for enc, err in ENC_CANDIDATES:
        try:
            return try_read_csv(path, nrows, engine="python", encoding=enc, errors=err)
        except Exception as e:
            last_err = e

    # 3) Fallback: lecture par chunks (engine='python')
    for enc, err in ENC_CANDIDATES:
        try:
            rows_left = nrows if nrows is not None else 200_000
            out = []
            # pas de low_memory ici
            for chunk in pd.read_csv(
                path,
                usecols=USECOLS,
                dtype={"thread_id": "Int64", "text": "string"},
                encoding=enc,
                on_bad_lines="skip",
                engine="python",
                chunksize=50_000,
            ):
                out.append(chunk)
                if rows_left is not None:
                    rows_left -= len(chunk)
                    if rows_left <= 0:
                        break
            if not out:
                raise RuntimeError("Aucun chunk lu.")
            return pd.concat(out, ignore_index=True)
        except Exception as e:
            last_err = e

    raise RuntimeError(f"Echec de lecture CSV robuste. Dernière erreur: {last_err}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True, help="Chemin du CSV filtré")
    ap.add_argument("--nrows", type=int, default=200_000, help="Nb max de lignes à lire (échantillon)")
    args = ap.parse_args()

    p = Path(args.inp)
    if not p.exists():
        print(f"[ERR] Fichier introuvable: {p}", file=sys.stderr)
        sys.exit(2)

    df = read_csv_robust(p, nrows=args.nrows)

    # Nettoyages légers
    df["text"] = df["text"].astype("string").fillna("").str.replace(r"\s+", " ", regex=True).str.strip()

    # Stats
    n_rows = len(df)
    n_threads = df["thread_id"].nunique(dropna=True)
    n_empty = int((df["text"] == "").sum())
    mean_len = float(df["text"].str.len().mean() or 0.0)
    p_empty = round(100.0 * n_empty / max(n_rows, 1), 2)

    print("=== SANITY (sample) ===")
    print(f"path           : {p}")
    print(f"rows(sample)   : {n_rows}")
    print(f"uniq_threads   : {n_threads}")
    print(f"empty_texts    : {n_empty} ({p_empty}%)")
    print(f"mean_len(chars): {mean_len:.1f}")
    print("\n--- HEAD(5) ---")
    print(df.head(5).to_string(index=False))

    # Warn colonnes manquantes
    missing_cols = [c for c in USECOLS if c not in df.columns]
    if missing_cols:
        print(f"[WARN] Colonnes manquantes: {missing_cols}")

if __name__ == "__main__":
    main()
